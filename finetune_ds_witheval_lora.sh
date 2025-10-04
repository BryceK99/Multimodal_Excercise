#!/usr/bin/env bash
# LoRA finetune with evaluation steps
# Usage example:
#   bash finetune_ds_witheval_lora.sh \
#     --model /path/to/model \
#     --data_path data/train.json \
#     --eval_data_path data/test.json \
#     --output_dir runs/lora_run \
#     --deepspeed mllm/ds_config_zero2.json

set -euo pipefail

export PYTHONPATH=$PYTHONPATH:`realpath .`
export CUDA_VISIBLE_DEVICES=4,5,6,7

GPUS_PER_NODE=4
NNODES=1
NODE_RANK=0
MASTER_ADDR=localhost
MASTER_PORT=6001

# Defaults
MODEL="./weights"
DATA_PATH="data/train.json"
EVAL_PATH="data/test.json"
OUTPUT_DIR="outputs/lora/"
DS_CONFIG=""
TASK="LM"
IMAGE_FOLDER=""

DISTRIBUTED_ARGS="
    --nproc_per_node $GPUS_PER_NODE \
    --nnodes $NNODES \
    --node_rank $NODE_RANK \
    --master_addr $MASTER_ADDR \
    --master_port $MASTER_PORT
"

# Training hparams (LoRA)
BATCH=1
ACC=16
LR=2e-5
EPOCHS=3
MAXLEN=2048
LOGSTEPS=50
EVALSTEPS=500
SAVESTEPS=500
SAVE_LIMIT=2

while [[ "$#" -gt 0 ]]; do
  case "$1" in
    --model) MODEL="$2"; shift 2;;
    --data_path) DATA_PATH="$2"; shift 2;;
    --eval_data_path) EVAL_PATH="$2"; shift 2;;
    --output_dir) OUTPUT_DIR="$2"; shift 2;;
    --deepspeed) DS_CONFIG="$2"; shift 2;;
    --task) TASK="$2"; shift 2;;
    --image_folder) IMAGE_FOLDER="$2"; shift 2;;
    --batch) BATCH="$2"; shift 2;;
    --acc) ACC="$2"; shift 2;;
    --lr) LR="$2"; shift 2;;
    --epochs) EPOCHS="$2"; shift 2;;
    --max_length) MAXLEN="$2"; shift 2;;
    *) echo "Unknown arg: $1"; exit 1;;
  esac
done

if [[ -z "$MODEL" || -z "$DATA_PATH" ]]; then
  echo "--model and --data_path are required"
  exit 1
fi

mkdir -p "$OUTPUT_DIR"
echo "[LoRA Finetune] Output: $OUTPUT_DIR"

CMD=( python mllm/finetune.py \
  --model_name_or_path "$MODEL" \
  --data_path "$DATA_PATH" \
  --task "$TASK" \
  --output_dir "$OUTPUT_DIR" \
  --per_device_train_batch_size "$BATCH" \
  --gradient_accumulation_steps "$ACC" \
  --learning_rate "$LR" \
  --num_train_epochs "$EPOCHS" \
  --model_max_length "$MAXLEN" \
  --gradient_checkpointing True \
  --evaluation_strategy steps \
  --eval_steps "$EVALSTEPS" \
  --save_steps "$SAVESTEPS" \
  --save_total_limit "$SAVE_LIMIT" \
  --logging_steps "$LOGSTEPS" \
  --use_lora True \
  --tune_llm False \
  --tune_vision False \
  --lora_r 64 \
  --lora_alpha 64 \
  --lora_dropout 0.05 )

if [[ -n "$EVAL_PATH" ]]; then CMD+=( --eval_data_path "$EVAL_PATH" ); fi
if [[ -n "$IMAGE_FOLDER" ]]; then CMD+=( --image_folder "$IMAGE_FOLDER" ); fi
if [[ -n "$DS_CONFIG" ]]; then CMD+=( --deepspeed "$DS_CONFIG" ); fi

echo "[Run] ${CMD[*]}"
"${CMD[@]}" |& tee "$OUTPUT_DIR/train.log"
