#!/bin/bash
python -m tgpr_ucb.train \
    --config configs/qwen2.5_14b.yaml \
    --data_path data/train.jsonl \
    --output_dir outputs/qwen2.5_14b
