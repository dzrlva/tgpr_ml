#!/bin/bash
python -m tgpr_ucb.train \
    --config configs/gpt_oss_20b.yaml \
    --data_path data/train.jsonl \
    --output_dir outputs/gpt_oss_20b
