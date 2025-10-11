#!/bin/bash
export PYTHONPATH=$PYTHONPATH:`realpath .`

MODEL="HaoyeZhang/MLLM_Excercise_Model"
# 专门的 logp 缓存目录，避免与项目根目录混在一起
DATA_DIR="cache/logps"
mkdir -p "$DATA_DIR"
DATA="data/preference_train.json"
REF_NAME="reconstruct"

MODEL_MAX_Length=1024

export MALLOC_TRIM_THRESHOLD_=131072
export MALLOC_ARENA_MAX=2
# Use conservative allocator settings; expandable_segments triggered internal assert in your run.
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:64
export HF_HUB_DISABLE_TELEMETRY=1
export TRANSFORMERS_NO_ADVISORY_WARNINGS=1

deepspeed --master_port 29600 --include localhost:0 mllm/finetune.py \
    --model_name_or_path $MODEL \
    --data_path $DATA \
    --data_dir $DATA_DIR \
    --ref_name $REF_NAME \
    --remove_unused_columns false \
    --label_names "labels" \
    --prediction_loss_only false \
    --bf16 true \
    --bf16_full_eval true \
    --fp16 false \
    --fp16_full_eval false \
    --do_train \
    --tune_vision false \
    --tune_llm false \
    --model_max_length $MODEL_MAX_Length \
    --max_slice_nums 4 \
    --max_steps 2000 \
    --output_dir output/mllm_preference_training \
    --logging_dir output/mllm_preference_training/log \
    --logging_strategy "steps" \
    --per_device_train_batch_size 1 \
    --per_device_eval_batch_size 1 \
    --gradient_accumulation_steps 1 \
    --evaluation_strategy "no" \
    --save_strategy "steps" \
    --save_steps 300 \
    --save_total_limit 10 \
    --learning_rate 5e-5 \
    --weight_decay 0.01 \
    --warmup_ratio 0.05 \
    --lr_scheduler_type "cosine" \
    --logging_steps 1 \
    --gradient_checkpointing true \
    --deepspeed mllm/ds_pref_config_zero2_stage2_min.json \
    --report_to "tensorboard" \
    --dataloader_num_workers 1 \
    --dataloader_pin_memory false \
    --preference_use_average_logp False \
    --preference_beta 0.5 \
    --task Preference \
    --optim adamw_bnb_8bit