import argparse
import json
import random
from pathlib import Path
from typing import Dict, List, Tuple

import torch
from datasets import Dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    TrainingArguments,
    Trainer,
)
from peft import LoraConfig, get_peft_model


# -----------------------------
# Data utilities
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


def get_label_set(train_data: List[Dict]) -> List[str]:
    return sorted({ex["label"] for ex in train_data})


# -----------------------------
# Text / prompt construction
# -----------------------------
def build_train_text(raw_field: str, label: str) -> str:
    """
    Training string: includes label at the end so the model learns to complete with the label.
    """
    prompt = (
        "Map the following materials database field name to one OPTIMADE schema field.\n"
        f"Field name: {raw_field}\n"
        "Answer with only the OPTIMADE field name.\n"
        f"{label}"
    )
    return prompt


def build_infer_prompt(raw_field: str, label_list: List[str]) -> str:
    """
    Inference-time prompt: no label at the end, model must generate it.
    """
    label_str = ", ".join(label_list)
    prompt = (
        "You are an assistant that maps raw materials database field names "
        "to OPTIMADE schema fields.\n"
        "Return only the OPTIMADE field name.\n\n"
        f"Valid OPTIMADE fields: {label_str}.\n\n"
        "Now map the following field:\n"
        f"Field: {raw_field}\n"
        "Answer with only the OPTIMADE field name.\n"
    )
    return prompt


def make_hf_dataset(examples: List[Dict]) -> Dataset:
    texts = [build_train_text(ex["raw_field"], ex["label"]) for ex in examples]
    return Dataset.from_dict({"text": texts})


# -----------------------------
# Output post-processing
# -----------------------------
def extract_label_from_output(output_text: str, label_list: List[str]) -> str:
    """
    Same heuristic as in icl_eval.py: try exact match, substring, then prefix.
    """
    text_lower = output_text.strip().lower()

    # exact match
    for lab in label_list:
        if text_lower == lab.lower():
            return lab

    # substring search
    for lab in label_list:
        if lab.lower() in text_lower:
            return lab

    # prefix match
    if text_lower:
        first_token = text_lower.split()[0]
        for lab in label_list:
            if lab.lower().startswith(first_token):
                return lab

    # fallback
    return label_list[0]


# -----------------------------
# Evaluation for one trained model
# -----------------------------
@torch.no_grad()
def evaluate_model(
    model,
    tokenizer,
    data: List[Dict],
    label_list: List[str],
    device: torch.device,
    max_new_tokens: int,
    log_path: Path,
    split_name: str,
    run_name: str,
) -> Tuple[float, float]:
    correct = 0
    total = 0

    label_to_idx = {lab: i for i, lab in enumerate(label_list)}
    n_labels = len(label_list)
    conf = [[0 for _ in range(n_labels)] for _ in range(n_labels)]

    log_path.parent.mkdir(parents=True, exist_ok=True)

    with log_path.open("w", encoding="utf-8") as log_f:
        for ex in data:
            raw_field = ex["raw_field"]
            gold_label = ex["label"]

            prompt = build_infer_prompt(raw_field, label_list)

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
                pad_token_id=tokenizer.pad_token_id,
            )

            gen_ids = output_ids[0, inputs["input_ids"].shape[1] :]
            gen_text = tokenizer.decode(gen_ids, skip_special_tokens=True)

            pred_label = extract_label_from_output(gen_text, label_list)
            is_correct = pred_label == gold_label

            if is_correct:
                correct += 1
            total += 1

            if gold_label in label_to_idx and pred_label in label_to_idx:
                g = label_to_idx[gold_label]
                p = label_to_idx[pred_label]
                conf[g][p] += 1

            record = {
                "run": run_name,
                "split": split_name,
                "raw_field": raw_field,
                "prompt": prompt,
                "generated": gen_text,
                "pred_label": pred_label,
                "gold_label": gold_label,
                "correct": is_correct,
            }
            log_f.write(json.dumps(record) + "\n")

    acc = correct / max(total, 1)

    # macro-F1
    f1s = []
    for i in range(n_labels):
        tp = conf[i][i]
        fp = sum(conf[g][i] for g in range(n_labels) if g != i)
        fn = sum(conf[i][p] for p in range(n_labels) if p != i)

        if tp == 0 and fp == 0 and fn == 0:
            continue

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
# Training + evaluation for one config
# -----------------------------
def train_and_eval_one_config(
    base_model_name: str,
    run_name: str,
    train_data: List[Dict],
    val_data: List[Dict],
    test_data: List[Dict],
    ood_data: List[Dict] | None,
    label_list: List[str],
    output_dir: Path,
    log_dir: Path,
    max_length: int,
    batch_size: int,
    num_epochs: int,
    learning_rate: float,
    lora_r: int,
    lora_alpha: int,
    lora_dropout: float,
    seed: int,
) -> Dict:
    torch.manual_seed(seed)
    random.seed(seed)

    train_ds = make_hf_dataset(train_data)
    val_ds = make_hf_dataset(val_data)

    print(f"[{run_name}] Train examples: {len(train_ds)}, Val examples: {len(val_ds)}")

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
    )

    print(f"[{run_name}] Loading base model: {base_model_name}")
    tokenizer = AutoTokenizer.from_pretrained(base_model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        base_model_name,
        quantization_config=bnb_config,
        device_map="auto",
    )
    model.config.use_cache = False

    lora_config = LoraConfig(
        r=lora_r,
        lora_alpha=lora_alpha,
        lora_dropout=lora_dropout,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=[
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
        ],
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    def tokenize_fn(batch):
        return tokenizer(
            batch["text"],
            truncation=True,
            max_length=max_length,
            padding="max_length",
        )

    train_tok = train_ds.map(tokenize_fn, batched=True, remove_columns=["text"])
    val_tok = val_ds.map(tokenize_fn, batched=True, remove_columns=["text"])

    train_tok = train_tok.map(lambda b: {"labels": b["input_ids"]}, batched=True)
    val_tok = val_tok.map(lambda b: {"labels": b["input_ids"]}, batched=True)

    training_args = TrainingArguments(
        output_dir=str(output_dir),
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        learning_rate=learning_rate,
        num_train_epochs=num_epochs,
        logging_steps=10,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_tok,
        eval_dataset=val_tok,
    )

    trainer.train()

    output_dir.mkdir(parents=True, exist_ok=True)
    trainer.save_model(output_dir)
    tokenizer.save_pretrained(output_dir)

    # ---- Evaluation ----
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()

    results = {}

    # Validation
    val_log_path = log_dir / f"{run_name}_val.jsonl"
    val_acc, val_f1 = evaluate_model(
        model=model,
        tokenizer=tokenizer,
        data=val_data,
        label_list=label_list,
        device=device,
        max_new_tokens=32,
        log_path=val_log_path,
        split_name="val",
        run_name=run_name,
    )
    results["val_acc"] = val_acc
    results["val_macro_f1"] = val_f1

    # Test (ID)
    test_log_path = log_dir / f"{run_name}_test_id.jsonl"
    test_acc, test_f1 = evaluate_model(
        model=model,
        tokenizer=tokenizer,
        data=test_data,
        label_list=label_list,
        device=device,
        max_new_tokens=32,
        log_path=test_log_path,
        split_name="test_id",
        run_name=run_name,
    )
    results["test_id_acc"] = test_acc
    results["test_id_macro_f1"] = test_f1

    # Test (OOD)
    if ood_data is not None:
        ood_log_path = log_dir / f"{run_name}_test_ood.jsonl"
        ood_acc, ood_f1 = evaluate_model(
            model=model,
            tokenizer=tokenizer,
            data=ood_data,
            label_list=label_list,
            device=device,
            max_new_tokens=32,
            log_path=ood_log_path,
            split_name="test_ood",
            run_name=run_name,
        )
        results["test_ood_acc"] = ood_acc
        results["test_ood_macro_f1"] = ood_f1

    return results


# -----------------------------
# Main: sequential hyperparameter search
# -----------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str, required=True)
    parser.add_argument("--train_path", type=str, required=True)
    parser.add_argument("--val_path", type=str, required=True)
    parser.add_argument("--test_path", type=str, required=True)
    parser.add_argument("--ood_path", type=str, default=None)
    parser.add_argument("--output_root", type=str, required=True)
    parser.add_argument("--log_root", type=str, required=True)
    parser.add_argument("--max_length", type=int, default=256)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    random.seed(args.seed)
    torch.manual_seed(args.seed)

    # -----------------------------
    # Load data
    # -----------------------------
    train_data = load_jsonl(Path(args.train_path))
    val_data = load_jsonl(Path(args.val_path))
    test_data = load_jsonl(Path(args.test_path))

    ood_data = None
    if args.ood_path is not None:
        ood_path = Path(args.ood_path)
        if ood_path.exists():
            ood_data = load_jsonl(ood_path)

    label_list = get_label_set(train_data)
    print("Labels:", label_list)

    output_root = Path(args.output_root)
    log_root = Path(args.log_root)
    output_root.mkdir(parents=True, exist_ok=True)
    log_root.mkdir(parents=True, exist_ok=True)

    base_name = args.model_name.split("/")[-1]

    # Summary will be written incrementally after EACH run
    summary_path = output_root / f"summary_{base_name}_sequential_search.json"
    summary: List[Dict] = []

    def append_and_flush(cfg_out: Dict):
        """Append one run's results to the in-memory summary and write to disk."""
        summary.append(cfg_out)
        with summary_path.open("w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2)

    # -----------------------------
    # Search spaces (2 values each)
    # -----------------------------
    # You can tweak these, but this is a reasonable starting point
    lr_candidates     = [5e-5, 1e-4]
    r_candidates      = [8, 16]
    epoch_candidates  = [3, 5]
    batch_candidates  = [4, 8]

    # Start from some defaults; will be overwritten by search
    current_lr      = lr_candidates[0]
    current_r       = r_candidates[0]
    current_epochs  = epoch_candidates[0]
    current_bs      = batch_candidates[0]

    # -----------------------------
    # 1) Tune learning rate
    # -----------------------------
    print("\n=== Tuning learning rate ===\n")
    best_val_f1 = -1.0
    best_lr = current_lr

    for lr in lr_candidates:
        run_name = f"{base_name}_LR{lr}_R{current_r}_EP{current_epochs}_BS{current_bs}"
        print(f"[LR search] Running: {run_name}")

        out_dir = output_root / run_name
        res = train_and_eval_one_config(
            base_model_name=args.model_name,
            run_name=run_name,
            train_data=train_data,
            val_data=val_data,
            test_data=test_data,
            ood_data=ood_data,
            label_list=label_list,
            output_dir=out_dir,
            log_dir=log_root,
            max_length=args.max_length,
            batch_size=current_bs,
            num_epochs=current_epochs,
            learning_rate=lr,
            lora_r=current_r,
            lora_alpha=32,
            lora_dropout=0.05,
            seed=args.seed,
        )
        cfg_out = {
            "stage": "lr_search",
            "run_name": run_name,
            "learning_rate": lr,
            "lora_r": current_r,
            "epochs": current_epochs,
            "batch_size": current_bs,
        }
        cfg_out.update(res)
        append_and_flush(cfg_out)

        if res.get("val_macro_f1", 0.0) > best_val_f1:
            best_val_f1 = res["val_macro_f1"]
            best_lr = lr

    current_lr = best_lr
    print(f"Best learning rate: {current_lr:.6g} (val_macro_f1={best_val_f1:.4f})")

    # -----------------------------
    # 2) Tune LoRA rank r
    # -----------------------------
    print("\n=== Tuning LoRA rank r ===\n")
    best_val_f1 = -1.0
    best_r = current_r

    for r in r_candidates:
        run_name = f"{base_name}_LR{current_lr}_R{r}_EP{current_epochs}_BS{current_bs}"
        print(f"[R search] Running: {run_name}")

        out_dir = output_root / run_name
        res = train_and_eval_one_config(
            base_model_name=args.model_name,
            run_name=run_name,
            train_data=train_data,
            val_data=val_data,
            test_data=test_data,
            ood_data=ood_data,
            label_list=label_list,
            output_dir=out_dir,
            log_dir=log_root,
            max_length=args.max_length,
            batch_size=current_bs,
            num_epochs=current_epochs,
            learning_rate=current_lr,
            lora_r=r,
            lora_alpha=32,
            lora_dropout=0.05,
            seed=args.seed,
        )
        cfg_out = {
            "stage": "r_search",
            "run_name": run_name,
            "learning_rate": current_lr,
            "lora_r": r,
            "epochs": current_epochs,
            "batch_size": current_bs,
        }
        cfg_out.update(res)
        append_and_flush(cfg_out)

        if res.get("val_macro_f1", 0.0) > best_val_f1:
            best_val_f1 = res["val_macro_f1"]
            best_r = r

    current_r = best_r
    print(f"Best LoRA rank: {current_r} (val_macro_f1={best_val_f1:.4f})")

    # -----------------------------
    # 3) Tune number of epochs
    # -----------------------------
    print("\n=== Tuning epochs ===\n")
    best_val_f1 = -1.0
    best_epochs = current_epochs

    for ep in epoch_candidates:
        run_name = f"{base_name}_LR{current_lr}_R{current_r}_EP{ep}_BS{current_bs}"
        print(f"[Epoch search] Running: {run_name}")

        out_dir = output_root / run_name
        res = train_and_eval_one_config(
            base_model_name=args.model_name,
            run_name=run_name,
            train_data=train_data,
            val_data=val_data,
            test_data=test_data,
            ood_data=ood_data,
            label_list=label_list,
            output_dir=out_dir,
            log_dir=log_root,
            max_length=args.max_length,
            batch_size=current_bs,
            num_epochs=ep,
            learning_rate=current_lr,
            lora_r=current_r,
            lora_alpha=32,
            lora_dropout=0.05,
            seed=args.seed,
        )
        cfg_out = {
            "stage": "epoch_search",
            "run_name": run_name,
            "learning_rate": current_lr,
            "lora_r": current_r,
            "epochs": ep,
            "batch_size": current_bs,
        }
        cfg_out.update(res)
        append_and_flush(cfg_out)

        if res.get("val_macro_f1", 0.0) > best_val_f1:
            best_val_f1 = res["val_macro_f1"]
            best_epochs = ep

    current_epochs = best_epochs
    print(f"Best epochs: {current_epochs} (val_macro_f1={best_val_f1:.4f})")

    # -----------------------------
    # 4) Tune batch size
    # -----------------------------
    print("\n=== Tuning batch size ===\n")
    best_val_f1 = -1.0
    best_bs = current_bs

    for bs in batch_candidates:
        run_name = f"{base_name}_LR{current_lr}_R{current_r}_EP{current_epochs}_BS{bs}"
        print(f"[Batch size search] Running: {run_name}")

        out_dir = output_root / run_name
        res = train_and_eval_one_config(
            base_model_name=args.model_name,
            run_name=run_name,
            train_data=train_data,
            val_data=val_data,
            test_data=test_data,
            ood_data=ood_data,
            label_list=label_list,
            output_dir=out_dir,
            log_dir=log_root,
            max_length=args.max_length,
            batch_size=bs,
            num_epochs=current_epochs,
            learning_rate=current_lr,
            lora_r=current_r,
            lora_alpha=32,
            lora_dropout=0.05,
            seed=args.seed,
        )
        cfg_out = {
            "stage": "batch_search",
            "run_name": run_name,
            "learning_rate": current_lr,
            "lora_r": current_r,
            "epochs": current_epochs,
            "batch_size": bs,
        }
        cfg_out.update(res)
        append_and_flush(cfg_out)

        if res.get("val_macro_f1", 0.0) > best_val_f1:
            best_val_f1 = res["val_macro_f1"]
            best_bs = bs

    current_bs = best_bs
    print(f"Best batch size: {current_bs} (val_macro_f1={best_val_f1:.4f})")

    # Final best config printed nicely
    print("\n=== Sequential search finished ===")
    print("Best config:")
    print(
        f"  lr={current_lr}, r={current_r}, epochs={current_epochs}, "
        f"batch_size={current_bs}"
    )
    print("Full summary written to:", summary_path)


if __name__ == "__main__":
    main()
