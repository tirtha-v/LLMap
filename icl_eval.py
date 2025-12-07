import json
import random
import argparse
from pathlib import Path
from typing import List, Dict

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


# -----------------------------
# Utility: load dataset
# -----------------------------
def load_jsonl(path: Path) -> List[Dict]:
    data = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            data.append(json.loads(line))
    return data


# -----------------------------
# Build label set
# -----------------------------
def get_label_set(train_data: List[Dict]) -> List[str]:
    labels = sorted({ex["label"] for ex in train_data})
    return labels


# -----------------------------
# Prompt construction
# -----------------------------
def build_prompt(
    raw_field: str,
    demos: List[Dict],
    label_list: List[str],
) -> str:
    header = (
        "You are an assistant that maps raw materials database field names "
        "to OPTIMADE schema fields.\n"
        "Return only the OPTIMADE field name.\n\n"
    )

    # List all valid labels explicitly
    label_str = ", ".join(label_list)
    header += f"Valid OPTIMADE fields: {label_str}.\n\n"

    # Add demonstrations
    if len(demos) > 0:
        header += "Here are some examples:\n"
        for i, ex in enumerate(demos, start=1):
            header += f"Example {i}:\n"
            header += f"Field: {ex['raw_field']} \u2192 {ex['label']}\n"
        header += "\n"

    # Query
    query = (
        "Now map the following field:\n"
        f"Field: {raw_field}\n"
    )

    # Some models behave better if we explicitly say "Answer:"
    footer = "Answer with only the OPTIMADE field name.\n"

    prompt = header + query + footer
    return prompt


# -----------------------------
# Output post-processing
# -----------------------------
def extract_label_from_output(
    output_text: str,
    label_list: List[str],
) -> str:
    """
    Simple heuristic: find the first label that appears in the output text,
    or fall back to the first token / best fuzzy match.
    """
    text_lower = output_text.strip().lower()

    # exact match on whole output
    for lab in label_list:
        if text_lower == lab.lower():
            return lab

    # substring search in the output
    for lab in label_list:
        if lab.lower() in text_lower:
            return lab

    # fallback: take first word and match to closest label by exact prefix
    first_token = text_lower.split()[0]
    for lab in label_list:
        if lab.lower().startswith(first_token):
            return lab

    # last resort: return a dummy label (will be counted as wrong)
    return label_list[0]


# -----------------------------
# Few-shot selector
# -----------------------------
def sample_demos(
    train_data: List[Dict],
    k: int,
) -> List[Dict]:
    if k <= 0:
        return []
    # Simple random sample; could be stratified by label
    return random.sample(train_data, k)


# -----------------------------
# Evaluate on a split
# -----------------------------
@torch.no_grad()
def evaluate_icl(
    model,
    tokenizer,
    data: List[Dict],
    train_data: List[Dict],
    label_list: List[str],
    k_shot: int,
    device: torch.device,
    max_new_tokens: int,
    log_path: Path,
    split_name: str,
) -> tuple[float, float]:
    correct = 0
    total = 0

    # confusion matrix
    label_to_idx = {lab: i for i, lab in enumerate(label_list)}
    n_labels = len(label_list)
    conf = [[0 for _ in range(n_labels)] for _ in range(n_labels)]

    # make sure directory exists
    log_path.parent.mkdir(parents=True, exist_ok=True)

    with log_path.open("w", encoding="utf-8") as log_f:
        for ex in data:
            raw_field = ex["raw_field"]
            gold_label = ex["label"]

            demos = sample_demos(train_data, k_shot)
            prompt = build_prompt(raw_field, demos, label_list)

            inputs = tokenizer(
                prompt,
                return_tensors="pt",
                truncation=True,
                max_length=512,
            ).to(device)

            output_ids = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                pad_token_id=tokenizer.pad_token_id,  # also suppresses pad_token spam
            )

            # only the generated continuation
            gen_ids = output_ids[0, inputs["input_ids"].shape[1]:]
            gen_text = tokenizer.decode(gen_ids, skip_special_tokens=True)

            pred_label = extract_label_from_output(gen_text, label_list)
            is_correct = (pred_label == gold_label)

            if is_correct:
                correct += 1
            total += 1

            # update confusion matrix
            if gold_label in label_to_idx and pred_label in label_to_idx:
                g = label_to_idx[gold_label]
                p = label_to_idx[pred_label]
                conf[g][p] += 1

            # log one JSON record
            record = {
                "split": split_name,
                "k_shot": k_shot,
                "raw_field": raw_field,
                "prompt": prompt,
                "generated": gen_text,
                "pred_label": pred_label,
                "gold_label": gold_label,
                "correct": is_correct,
            }
            log_f.write(json.dumps(record) + "\n")

    # accuracy
    acc = correct / max(total, 1)

    # macro-F1
    f1s = []
    for i in range(n_labels):
        tp = conf[i][i]
        fp = sum(conf[g][i] for g in range(n_labels) if g != i)
        fn = sum(conf[i][p] for p in range(n_labels) if p != i)

        if tp == 0 and fp == 0 and fn == 0:
            continue  # label never appears

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0

        if precision + recall == 0:
            f1 = 0.0
        else:
            f1 = 2 * precision * recall / (precision + recall)

        f1s.append(f1)

    macro_f1 = sum(f1s) / len(f1s) if f1s else 0.0

    return acc, macro_f1



# -----------------------------
# Main script
# -----------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str, required=True)
    parser.add_argument("--train_path", type=str, required=True)
    parser.add_argument("--val_path", type=str, required=True)
    parser.add_argument("--test_path", type=str, required=True)
    
    # NEW: optional OOD dataset
    parser.add_argument("--ood_path", type=str, default=None,
                        help="Optional OOD test set path (e.g., OMAT24)")

    parser.add_argument("--k_shot", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    random.seed(args.seed)
    log_dir = Path("logs_icl")
    
    train_data = load_jsonl(Path(args.train_path))
    val_data = load_jsonl(Path(args.val_path))
    test_data = load_jsonl(Path(args.test_path))

    # NEW
    ood_data = None
    if args.ood_path is not None:
        ood_data = load_jsonl(Path(args.ood_path))

    label_list = get_label_set(train_data)
    print("Labels:", label_list)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    print(f"Loading model: {args.model_name}")
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        args.model_name,
        torch_dtype=torch.float16,
        device_map="auto",
    )
    model.eval()

    # Validation
    print(f"Evaluating {args.k_shot}-shot ICL on validation set...")
    val_log_path = log_dir / f"val_k{args.k_shot}.jsonl"
    val_acc, val_macro_f1 = evaluate_icl(
        model=model,
        tokenizer=tokenizer,
        data=val_data,
        train_data=train_data,
        label_list=label_list,
        k_shot=args.k_shot,
        device=device,
        max_new_tokens=32,
        log_path=val_log_path,
        split_name="val",
    )
    print(f"Validation accuracy: {val_acc:.4f}")
    print(f"Validation macro-F1: {val_macro_f1:.4f}")

    # Test (ID)
    print(f"Evaluating {args.k_shot}-shot ICL on in-domain test set...")
    test_log_path = log_dir / f"test_id_k{args.k_shot}.jsonl"
    test_acc, test_macro_f1 = evaluate_icl(
        model=model,
        tokenizer=tokenizer,
        data=test_data,
        train_data=train_data,
        label_list=label_list,
        k_shot=args.k_shot,
        device=device,
        max_new_tokens=32,
        log_path=test_log_path,
        split_name="test_id",
    )
    print(f"Test (ID) accuracy: {test_acc:.4f}")
    print(f"Test (ID) macro-F1: {test_macro_f1:.4f}")

    # NEW: Test (OOD)
    if ood_data is not None:
        print(f"Evaluating {args.k_shot}-shot ICL on OOD test set...")
        ood_log_path = log_dir / f"test_ood_k{args.k_shot}.jsonl"
        ood_acc, ood_macro_f1 = evaluate_icl(
            model=model,
            tokenizer=tokenizer,
            data=ood_data,
            train_data=train_data,
            label_list=label_list,
            k_shot=args.k_shot,
            device=device,
            max_new_tokens=32,
            log_path=ood_log_path,
            split_name="test_ood",
        )
        print(f"Test (OOD) accuracy: {ood_acc:.4f}")
        print(f"Test (OOD) macro-F1: {ood_macro_f1:.4f}")


if __name__ == "__main__":
    main()
