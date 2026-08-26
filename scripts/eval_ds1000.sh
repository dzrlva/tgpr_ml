#!/bin/bash
# usage: bash scripts/eval_ds1000.sh <model_path> <config> [lora_path]

MODEL_PATH=$1
CONFIG=$2
LORA_PATH=${3:-""}

python -m tgpr_ucb.inference \
    --model_path $MODEL_PATH \
    --config $CONFIG \
    --dataset data/ds1000_test.parquet \
    --output results/ds1000_eval.jsonl \
    --lora_path $LORA_PATH
