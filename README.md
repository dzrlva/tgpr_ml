# TGPR-UCB

Tree-Guided Policy Refinement with UCB1-Tuned + Distributional PR Rewards.

## Setup

```bash
pip install -e .
```

## Data

```bash
python scripts/prepare_data.py
```

## Training

```bash
bash scripts/train_small.sh
bash scripts/train_7b.sh
bash scripts/train_14b.sh
bash scripts/train_20b.sh
```

## Evaluation on DS-1000

```bash
bash scripts/eval_ds1000.sh outputs/qwen_coder_3b/checkpoint-500 configs/qwen_coder_3b.yaml outputs/qwen_coder_3b/checkpoint-500
```

## Configs

- `qwen_coder_3b.yaml` — Qwen2.5-Coder-3B, LoRA adapter, vLLM, single GPU
- `qwen2.5_7b.yaml` — Qwen2.5-7B, full FT, vLLM
- `qwen2.5_14b.yaml` — Qwen2.5-14B, full FT, vLLM, TP=2
- `gpt_oss_20b.yaml` — gpt-oss-20b MoE, SGLang, TP=4
