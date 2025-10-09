#!/bin/bash
# 偏好训练显存估算示例脚本
# 该脚本演示了不同配置下的显存估算结果

echo "================================================================================"
echo "示例1: 默认配置（finetune_preference.sh中的配置）"
echo "================================================================================"
echo "配置: 8B参数, 4 GPUs, ZeRO-2, bf16, 梯度检查点, batch_size=1, seq_len=2048"
echo ""
python estimate_preference_memory.py \
    --total_params 8.0 \
    --num_gpus 4 \
    --batch_size 1 \
    --sequence_length 2048 \
    --zero_stage 2 \
    --gradient_checkpointing \
    --dtype bf16

echo ""
echo ""
echo "================================================================================"
echo "示例2: 使用ZeRO-3进一步减少显存占用"
echo "================================================================================"
echo "配置: 8B参数, 4 GPUs, ZeRO-3, bf16, 梯度检查点, batch_size=1, seq_len=2048"
echo ""
python estimate_preference_memory.py \
    --total_params 8.0 \
    --num_gpus 4 \
    --batch_size 1 \
    --sequence_length 2048 \
    --zero_stage 3 \
    --gradient_checkpointing \
    --dtype bf16

echo ""
echo ""
echo "================================================================================"
echo "示例3: 减小序列长度以节省显存"
echo "================================================================================"
echo "配置: 8B参数, 4 GPUs, ZeRO-2, bf16, 梯度检查点, batch_size=1, seq_len=1024"
echo ""
python estimate_preference_memory.py \
    --total_params 8.0 \
    --num_gpus 4 \
    --batch_size 1 \
    --sequence_length 1024 \
    --zero_stage 2 \
    --gradient_checkpointing \
    --dtype bf16

echo ""
echo ""
echo "================================================================================"
echo "示例4: 不使用优化（对比基准）"
echo "================================================================================"
echo "配置: 8B参数, 4 GPUs, ZeRO-0, bf16, 无梯度检查点, batch_size=1, seq_len=2048"
echo ""
python estimate_preference_memory.py \
    --total_params 8.0 \
    --num_gpus 4 \
    --batch_size 1 \
    --sequence_length 2048 \
    --zero_stage 0 \
    --no_gradient_checkpointing \
    --dtype bf16

echo ""
echo ""
echo "================================================================================"
echo "示例5: 单GPU训练（小规模模型）"
echo "================================================================================"
echo "配置: 2B参数, 1 GPU, ZeRO-0, bf16, 梯度检查点, batch_size=1, seq_len=1024"
echo ""
python estimate_preference_memory.py \
    --total_params 2.0 \
    --num_gpus 1 \
    --batch_size 1 \
    --sequence_length 1024 \
    --zero_stage 0 \
    --gradient_checkpointing \
    --dtype bf16

echo ""
echo "================================================================================"
echo "显存估算示例结束"
echo "================================================================================"
