#!/bin/bash
python -m tgpr_ucb.train \
    --config configs/qwen_coder_3b.yaml \
    --data_path data/train.jsonl \
    --output_dir outputs/qwen_coder_3b
