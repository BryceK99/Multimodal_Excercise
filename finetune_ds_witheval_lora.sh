#!/usr/bin/env bash
# LoRA finetune with evaluation steps (memory-friendly defaults)
# Usage example:
#   bash finetune_ds_witheval_lora.sh \
#     --model /path/to/model \
#     --data_path data/train.json \
#     --eval_data_path data/test.json \
#     --output_dir runs/lora_run \
#     --deepspeed mllm/ds_config_zero3.json \
#     --max_steps 200

set -euo pipefail

# 安全展开，避免未绑定变量导致 set -u 下报错
export PYTHONPATH="${PYTHONPATH:+$PYTHONPATH:}$(realpath .)"
export CUDA_VISIBLE_DEVICES=0
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-1}
export TOKENIZERS_PARALLELISM=false
# 缓解显存碎片
export PYTORCH_CUDA_ALLOC_CONF=${PYTORCH_CUDA_ALLOC_CONF:-"max_split_size_mb:128,expandable_segments:True"}

# 分布式占位（当前默认单卡，保留变量方便将来扩展）
GPUS_PER_NODE=1
NNODES=1
NODE_RANK=0
MASTER_ADDR=localhost
MASTER_PORT=6001

# Defaults
MODEL="HaoyeZhang/MLLM_Excercise_Model"
DATA_PATH="data/sft/train.json"
EVAL_PATH="data/sft/test.json"
OUTPUT_DIR="outputs/sft/"
DS_CONFIG=""
TASK="LM"
IMAGE_FOLDER="data/sft/images"

DISTRIBUTED_ARGS="
    --nproc_per_node $GPUS_PER_NODE \
    --nnodes $NNODES \
    --node_rank $NODE_RANK \
    --master_addr $MASTER_ADDR \
    --master_port $MASTER_PORT
"

# Training hparams (LoRA)
BATCH=1            # per_device_train_batch_size（已极小）
ACC=16               # 梯度累积，等效总batch=BATCH*ACC
LR=2e-5
EPOCHS=16
MAXLEN=1536          # 降低默认序列长度，缓解显存
LOGSTEPS=50
EVALSTEPS=1000       # 拉大评估间隔，避免频繁显存/I-O
SAVESTEPS=200     # 拉大保存间隔
SAVE_LIMIT=2
# 可选：限制总步数做“冒烟测试”，为空则不生效
STEPS=""

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
    --max_steps) STEPS="$2"; shift 2;;
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
  --bf16 True \
  --tf32 True \
  --dataloader_num_workers 2 \
  --eval_accumulation_steps 1 \
  --remove_unused_columns False \
  --report_to none \
  --disable_tqdm True \
  --evaluation_strategy steps \
  --eval_steps "$EVALSTEPS" \
  --save_steps "$SAVESTEPS" \
  --save_total_limit "$SAVE_LIMIT" \
  --logging_steps "$LOGSTEPS" \
  --use_lora True \
  --tune_llm False \
  --tune_vision False \
  --lora_r 8 \
  --lora_alpha 16 \
  --lora_dropout 0.05 )

if [[ -n "$EVAL_PATH" ]]; then CMD+=( --eval_data_path "$EVAL_PATH" ); fi
if [[ -n "$IMAGE_FOLDER" ]]; then CMD+=( --image_folder "$IMAGE_FOLDER" ); fi
if [[ -n "$DS_CONFIG" ]]; then CMD+=( --deepspeed "$DS_CONFIG" ); fi
if [[ -n "$STEPS" ]]; then CMD+=( --max_steps "$STEPS" ); fi

echo "[Run] ${CMD[*]}"
"${CMD[@]}" |& tee "$OUTPUT_DIR/train.log"