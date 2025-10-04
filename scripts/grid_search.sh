#!/usr/bin/env bash
# Simple bash grid driver. Requires bash and python in PATH.
# Usage:
#   bash scripts/grid_search.sh \
#     --model /path/to/model \
#     --data_path data/train.json \
#     --eval_data_path data/test.json \
#     --task LM \
#     --output_base runs \
#     --deepspeed_config mllm/ds_config_zero2.json

set -euo pipefail

# Defaults
MODEL=""
DATA_PATH=""
EVAL_PATH=""
TASK="LM"
OUTPUT_BASE="runs"
DS_CONFIG=""
IMAGE_FOLDER=""

# Parse args
while [[ "$#" -gt 0 ]]; do
  case "$1" in
    --model) MODEL="$2"; shift 2;;
    --data_path) DATA_PATH="$2"; shift 2;;
    --eval_data_path) EVAL_PATH="$2"; shift 2;;
    --task) TASK="$2"; shift 2;;
    --output_base) OUTPUT_BASE="$2"; shift 2;;
    --deepspeed_config) DS_CONFIG="$2"; shift 2;;
    --image_folder) IMAGE_FOLDER="$2"; shift 2;;
    *) echo "Unknown arg: $1"; exit 1;;
  esac
done

if [[ -z "$MODEL" || -z "$DATA_PATH" ]]; then
  echo "--model and --data_path are required"
  exit 1
fi

TS=$(date +%Y%m%d_%H%M%S)
BASE_DIR="${OUTPUT_BASE}/grid_${TASK}_${TS}"
mkdir -p "$BASE_DIR"

run_one() {
  local NAME="$1"; shift
  local OUT_DIR="${BASE_DIR}/${NAME}"
  mkdir -p "$OUT_DIR"
  local CMD=( python mllm/finetune.py \
    --model_name_or_path "$MODEL" \
    --data_path "$DATA_PATH" \
    --task "$TASK" \
    --output_dir "$OUT_DIR" \
    --evaluation_strategy steps --eval_steps 500 \
    --save_steps 500 --logging_steps 50 --save_total_limit 2 \
    --gradient_checkpointing True )
  if [[ -n "$EVAL_PATH" ]]; then CMD+=( --eval_data_path "$EVAL_PATH" ); fi
  if [[ -n "$IMAGE_FOLDER" ]]; then CMD+=( --image_folder "$IMAGE_FOLDER" ); fi
  if [[ -n "$DS_CONFIG" ]]; then CMD+=( --deepspeed "$DS_CONFIG" ); fi
  # Append remaining flags
  CMD+=( "$@" )
  echo "[Run] ${CMD[*]}"
  "${CMD[@]}" |& tee "${OUT_DIR}/train.log"
}

# Example 5 runs; modify below to customize grid
run_one lora_bs1_lr2e-5_len2k_r64 \
  --use_lora True --tune_llm False --tune_vision False \
  --per_device_train_batch_size 1 --gradient_accumulation_steps 16 \
  --learning_rate 2e-5 --num_train_epochs 3 --model_max_length 2048 \
  --lora_r 64 --lora_alpha 64 --lora_dropout 0.05

run_one lora_bs2_lr1e-5_len2k_r128 \
  --use_lora True --tune_llm False --tune_vision True \
  --per_device_train_batch_size 2 --gradient_accumulation_steps 8 \
  --learning_rate 1e-5 --num_train_epochs 3 --model_max_length 2048 \
  --lora_r 128 --lora_alpha 128 --lora_dropout 0.05

run_one lora_bs1_lr5e-5_len1k_r64 \
  --use_lora True --tune_llm False --tune_vision False \
  --per_device_train_batch_size 1 --gradient_accumulation_steps 32 \
  --learning_rate 5e-5 --num_train_epochs 2 --model_max_length 1024 \
  --lora_r 64 --lora_alpha 32 --lora_dropout 0.1

run_one qlora_bs1_lr1e-4_len2k_r64 \
  --use_lora True --q_lora True --tune_llm False --tune_vision False \
  --per_device_train_batch_size 1 --gradient_accumulation_steps 16 \
  --learning_rate 1e-4 --num_train_epochs 3 --model_max_length 2048 \
  --lora_r 64 --lora_alpha 16 --lora_dropout 0.05

run_one lora_mild_bs1_lr5e-6_len1k_r32 \
  --use_lora True --tune_llm False --tune_vision True \
  --per_device_train_batch_size 1 --gradient_accumulation_steps 8 \
  --learning_rate 5e-6 --num_train_epochs 2 --model_max_length 1024 \
  --lora_r 32 --lora_alpha 16 --lora_dropout 0.05
