# LLMap
LLM-Based Field Mapping for Materials Datasets

## Abstract
We study field-level schema mapping for materials-science databases using large language models (LLMs). Given a raw field name from sources such as Materials Project, OQMD, Alexandria, JARVIS--DFT and AFLOW, the task is to predict a single target column from the OPTIMADE-compatible LeMaterial schema. We formulate this as multiclass text classification and compare two adaptation modes---(i) in-context learning (ICL) with Llama-3.1-8B and Mistral-7B, and (ii) parameter-efficient finetuning (PEFT, QLoRA) of Llama-3.1-8B and Phi-3-Mini---under both in-domain and out-of-domain (OMat24) evaluation. We further study a back-translation augmentation strategy that generates synthetic raw field names. Our main findings are: Mistral-7B is a strong ICL baseline, but Llama-3.1-8B with QLoRA clearly dominates after finetuning; small models like Phi-3-Mini underfit even with PEFT; and back-translation is beneficial only when combined with finetuning, substantially improving OOD macro-F1. 


---

## Environment Setup

```bash
conda create -n llmap python=3.12 -y
conda activate llmap
pip install -r requirements.txt
huggingface-cli login
```

Models used (you must accept licenses on Hugging Face):

* `meta-llama/Meta-Llama-3.1-8B-Instruct`
* `mistralai/Mistral-7B-Instruct-v0.3`
* `microsoft/Phi-3-mini-4k-instruct`

---

## Data

All required preprocessed splits are already included:

```
data/stp2_train.jsonl
data/stp2_val.jsonl
data/stp2_test_id.jsonl
data/stp2_train_bt_aug.jsonl    # includes back-translated synthetic samples
```

Each example follows this JSONL format:

```json
{"raw_field": "...", "label": "..."}
```

These files are sufficient to reproduce all reported results out-of-the-box.

---

## In-Context Learning (ICL)

Run ICL with frozen model weights:

```bash
python icl_eval.py \
  --model_name <hf_model_id> \
  --train_path data/stp2_train.jsonl \
  --val_path data/stp2_val.jsonl \
  --test_path data/stp2_test_id.jsonl \
  --k_shot <k>
```

Reproduce the full ICL curve for Llama-3.1-8B:

```bash
for k in {0..10}; do
  python icl_eval.py \
    --model_name meta-llama/Meta-Llama-3.1-8B-Instruct \
    --train_path data/stp2_train.jsonl \
    --val_path data/stp2_val.jsonl \
    --test_path data/stp2_test_id.jsonl \
    --k_shot $k
done
```

Use the same loop for Mistral-7B or Phi-3-Mini.  
ICL logs are written under:

```
logs_icl/
```

---

## PEFT (QLoRA) Fine-Tuning

Run full QLoRA fine-tuning + automatic hyperparameter search:

```bash
python peft_eval.py \
  --model_name <hf_model_id> \
  --train_path data/stp2_train.jsonl \
  --val_path data/stp2_val.jsonl \
  --test_path data/stp2_test_id.jsonl \
  --output_root runs/<tag> \
  --log_root logs_peft/<tag>
```

This script sweeps over:

* Learning rates
* LoRA ranks
* Epoch counts
* Batch sizes

A summary file will be written to:

```
runs/<tag>/summary_<model_name>_sequential_search.json
```

Select the run with the **highest validation macro-F1** to match the reported results.

To reproduce only the best configuration, edit the candidate lists inside `peft_eval.py`.

---

## 5. Back-Translation Experiments

The back-translation augmented training file is already included:

```
data/stp2_train_bt_aug.jsonl
```

Reproduce Llama-3.1-8B PEFT with augmentation:

```bash
python peft_eval.py \
  --model_name meta-llama/Meta-Llama-3.1-8B-Instruct \
  --train_path data/stp2_train_bt_aug.jsonl \
  --val_path data/stp2_val.jsonl \
  --test_path data/stp2_test_id.jsonl \
  --output_root runs/llama_bt \
  --log_root logs_peft/llama_bt
```

Repeat with the Phi-3-Mini model to reproduce its augmented results.

---

## Notes

* A GPU (≥ 24 GB recommended) is required for QLoRA on 8B models.
* Optional: OOD evaluation can be added using `--ood_path <jsonl>`, though no OOD file is included in this repo.

---
