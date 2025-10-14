#!/usr/bin/env bash
# Visual Grounding finetune (defaults tuned for single GPU).
# Usage example:
#   bash finetune_grounding.sh \
#     --model HaoyeZhang/MLLM_Excercise_Model \
#     --data_path data/vg/all_vg_dataset.json \
#     --eval_data_path data/vg/CWB_flickr30k_eval.jsonl  # optional if converted to unified json
#     --output_dir outputs/grounding_run \
#     --deepspeed mllm/ds_config_zero2.json \
#     --max_steps 200

set -euo pipefail

export PYTHONPATH="${PYTHONPATH:+$PYTHONPATH:}$(realpath .)"
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-1}
export TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF=${PYTORCH_CUDA_ALLOC_CONF:-"max_split_size_mb:128,expandable_segments:True"}

# Distributed placeholders (single node/single GPU by default)
GPUS_PER_NODE=${GPUS_PER_NODE:-1}
NNODES=${NNODES:-1}
NODE_RANK=${NODE_RANK:-0}
MASTER_ADDR=${MASTER_ADDR:-localhost}
MASTER_PORT=${MASTER_PORT:-6002}

MODEL="HaoyeZhang/MLLM_Excercise_Model"
DATA_PATH="data/vg/all_vg_dataset.json"
EVAL_PATH=""
OUTPUT_DIR="outputs/grounding/"
DS_CONFIG=""
IMAGE_FOLDER=""  # unified json内已使用绝对/可解析路径可留空

# Training hparams
BATCH=1
ACC=16
LR=2e-5
EPOCHS=8
MAXLEN=1536
LOGSTEPS=50
EVALSTEPS=1000
SAVESTEPS=200
SAVE_LIMIT=2
STEPS=""    # 可选：限制步数快速冒烟

# Fine-tune knobs
TUNE_VISION=false
TUNE_LLM=false
USE_LORA=true

while [[ "$#" -gt 0 ]]; do
	case "$1" in
		--model) MODEL="$2"; shift 2;;
		--data_path) DATA_PATH="$2"; shift 2;;
		--eval_data_path) EVAL_PATH="$2"; shift 2;;
		--output_dir) OUTPUT_DIR="$2"; shift 2;;
		--deepspeed) DS_CONFIG="$2"; shift 2;;
		--image_folder) IMAGE_FOLDER="$2"; shift 2;;
		--batch) BATCH="$2"; shift 2;;
		--acc) ACC="$2"; shift 2;;
		--lr) LR="$2"; shift 2;;
		--epochs) EPOCHS="$2"; shift 2;;
		--max_length) MAXLEN="$2"; shift 2;;
		--max_steps) STEPS="$2"; shift 2;;
		--tune_vision) TUNE_VISION="$2"; shift 2;;
		--tune_llm) TUNE_LLM="$2"; shift 2;;
		--use_lora) USE_LORA="$2"; shift 2;;
		*) echo "Unknown arg: $1"; exit 1;;
	esac
done

mkdir -p "$OUTPUT_DIR"
echo "[Grounding Finetune] Output: $OUTPUT_DIR"

CMD=( python mllm/finetune.py \
	--model_name_or_path "$MODEL" \
	--data_path "$DATA_PATH" \
	--task Grounding \
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
	--use_lora "$USE_LORA" \
	--tune_llm "$TUNE_LLM" \
	--tune_vision "$TUNE_VISION" \
	--lora_r 8 \
    --lora_alpha 16 \
    --lora_dropout 0.05 )

if [[ -n "$EVAL_PATH" ]]; then CMD+=( --eval_data_path "$EVAL_PATH" ); fi
if [[ -n "$IMAGE_FOLDER" ]]; then CMD+=( --image_folder "$IMAGE_FOLDER" ); fi
if [[ -n "$DS_CONFIG" ]]; then CMD+=( --deepspeed "$DS_CONFIG" ); fi
if [[ -n "$STEPS" ]]; then CMD+=( --max_steps "$STEPS" ); fi

echo "[Run] ${CMD[*]}"
"${CMD[@]}" |& tee "$OUTPUT_DIR/train.log"