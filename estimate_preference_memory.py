#!/usr/bin/env python
# coding=utf-8
"""
估算偏好对齐训练（Preference Training）所需的显存

这个脚本用于估算在给定配置下，进行偏好对齐训练所需的GPU显存。
支持DeepSpeed ZeRO优化、混合精度训练等配置。
"""

import argparse
import json
from typing import Dict, Optional


def estimate_model_memory(
    num_params: int,
    dtype: str = "bf16",
    trainable_ratio: float = 1.0
) -> Dict[str, float]:
    """
    估算模型参数占用的显存
    
    Args:
        num_params: 模型总参数量
        dtype: 数据类型 (fp32, fp16, bf16)
        trainable_ratio: 可训练参数比例
        
    Returns:
        包含模型显存占用信息的字典 (单位: GB)
    """
    bytes_per_param = {"fp32": 4, "fp16": 2, "bf16": 2}[dtype]
    trainable_params = num_params * trainable_ratio
    frozen_params = num_params * (1 - trainable_ratio)
    
    # 模型参数显存
    model_memory = num_params * bytes_per_param / (1024 ** 3)
    
    return {
        "total_params": num_params,
        "trainable_params": trainable_params,
        "frozen_params": frozen_params,
        "model_memory_gb": model_memory,
    }


def estimate_optimizer_memory(
    trainable_params: int,
    optimizer: str = "adamw",
    dtype: str = "bf16",
    zero_stage: int = 2
) -> Dict[str, float]:
    """
    估算优化器状态占用的显存
    
    Args:
        trainable_params: 可训练参数量
        optimizer: 优化器类型 (adamw, sgd)
        dtype: 训练数据类型
        zero_stage: DeepSpeed ZeRO优化阶段 (0, 1, 2, 3)
        
    Returns:
        包含优化器显存占用信息的字典 (单位: GB)
    """
    # AdamW优化器: 需要存储momentum (fp32) 和 variance (fp32)
    # 每个可训练参数需要: 4 bytes (momentum) + 4 bytes (variance) = 8 bytes
    if optimizer.lower() == "adamw":
        optimizer_states_bytes = 8  # fp32的momentum和variance
    else:  # SGD等简单优化器
        optimizer_states_bytes = 4
    
    optimizer_memory = trainable_params * optimizer_states_bytes / (1024 ** 3)
    
    # DeepSpeed ZeRO会在不同GPU间分片优化器状态
    if zero_stage >= 1:
        # ZeRO-1: 分片优化器状态
        # ZeRO-2: 分片优化器状态和梯度
        # ZeRO-3: 分片优化器状态、梯度和模型参数
        # 这里我们假设在单GPU上估算，实际多GPU时会除以GPU数量
        pass
    
    return {
        "optimizer_memory_gb": optimizer_memory,
        "optimizer_states_bytes_per_param": optimizer_states_bytes,
    }


def estimate_gradient_memory(
    trainable_params: int,
    dtype: str = "bf16",
    zero_stage: int = 2
) -> Dict[str, float]:
    """
    估算梯度占用的显存
    
    Args:
        trainable_params: 可训练参数量
        dtype: 训练数据类型
        zero_stage: DeepSpeed ZeRO优化阶段
        
    Returns:
        包含梯度显存占用信息的字典 (单位: GB)
    """
    bytes_per_param = {"fp32": 4, "fp16": 2, "bf16": 2}[dtype]
    
    # 梯度通常以fp32存储以保证数值稳定性
    gradient_bytes = 4  # fp32
    gradient_memory = trainable_params * gradient_bytes / (1024 ** 3)
    
    return {
        "gradient_memory_gb": gradient_memory,
    }


def estimate_activation_memory(
    batch_size: int,
    sequence_length: int,
    hidden_size: int,
    num_layers: int,
    num_attention_heads: int,
    gradient_checkpointing: bool = True,
    dtype: str = "bf16"
) -> Dict[str, float]:
    """
    估算激活值占用的显存
    
    Args:
        batch_size: 批大小
        sequence_length: 序列长度
        hidden_size: 隐藏层维度
        num_layers: 层数
        num_attention_heads: 注意力头数
        gradient_checkpointing: 是否使用梯度检查点
        dtype: 数据类型
        
    Returns:
        包含激活值显存占用信息的字典 (单位: GB)
    """
    bytes_per_element = {"fp32": 4, "fp16": 2, "bf16": 2}[dtype]
    
    # 估算每层的激活值大小
    # 主要包括: 注意力权重、MLP中间态、LayerNorm等
    
    # 注意力权重: [batch, num_heads, seq_len, seq_len]
    attention_memory = (batch_size * num_attention_heads * 
                       sequence_length * sequence_length * 
                       bytes_per_element / (1024 ** 3))
    
    # 隐藏状态: [batch, seq_len, hidden_size]
    hidden_state_memory = (batch_size * sequence_length * 
                          hidden_size * bytes_per_element / (1024 ** 3))
    
    # MLP中间态 (通常是hidden_size的4倍)
    mlp_memory = (batch_size * sequence_length * hidden_size * 4 * 
                  bytes_per_element / (1024 ** 3))
    
    # 每层总激活值
    per_layer_memory = attention_memory + hidden_state_memory + mlp_memory
    
    if gradient_checkpointing:
        # 梯度检查点可以显著减少激活值显存，通常减少到O(sqrt(num_layers))
        # 这里我们采用保守估计，减少到原来的1/4
        activation_memory = per_layer_memory * num_layers * 0.25
    else:
        # 不使用梯度检查点时，需要保存所有层的激活值
        activation_memory = per_layer_memory * num_layers
    
    return {
        "activation_memory_gb": activation_memory,
        "per_layer_memory_gb": per_layer_memory,
        "attention_memory_per_layer_gb": attention_memory,
        "gradient_checkpointing": gradient_checkpointing,
    }


def estimate_batch_data_memory(
    batch_size: int,
    sequence_length: int,
    vocab_size: int,
    image_count: int = 1,
    image_channels: int = 3,
    image_size: int = 448,
    dtype: str = "bf16"
) -> Dict[str, float]:
    """
    估算批次数据占用的显存
    
    Args:
        batch_size: 批大小
        sequence_length: 序列长度
        vocab_size: 词表大小
        image_count: 每个样本的图片数量
        image_channels: 图片通道数
        image_size: 图片尺寸
        dtype: 数据类型
        
    Returns:
        包含批次数据显存占用信息的字典 (单位: GB)
    """
    bytes_per_element = {"fp32": 4, "fp16": 2, "bf16": 2}[dtype]
    
    # 输入token IDs: [batch, seq_len]
    input_ids_memory = (batch_size * sequence_length * 4 / (1024 ** 3))  # int32
    
    # 标签: [batch, seq_len]
    labels_memory = (batch_size * sequence_length * 4 / (1024 ** 3))  # int32
    
    # 图片数据: [batch, image_count, channels, height, width]
    image_memory = (batch_size * image_count * image_channels * 
                   image_size * image_size * bytes_per_element / (1024 ** 3))
    
    # Preference训练需要正负样本对，所以数据量翻倍
    # 在实现中，win和rej样本被拼接处理
    total_batch_memory = (input_ids_memory + labels_memory + image_memory) * 2
    
    return {
        "batch_data_memory_gb": total_batch_memory,
        "input_ids_memory_gb": input_ids_memory * 2,
        "labels_memory_gb": labels_memory * 2,
        "image_memory_gb": image_memory * 2,
    }


def estimate_preference_training_memory(
    # 模型参数
    total_params: int = 8_000_000_000,  # 8B参数
    trainable_ratio: float = 1.0,
    hidden_size: int = 4096,
    num_layers: int = 32,
    num_attention_heads: int = 32,
    # 训练参数
    batch_size: int = 1,
    gradient_accumulation_steps: int = 1,
    sequence_length: int = 2048,
    # 图像参数
    image_count: int = 1,
    image_size: int = 448,
    # 优化参数
    dtype: str = "bf16",
    optimizer: str = "adamw",
    gradient_checkpointing: bool = True,
    zero_stage: int = 2,
    num_gpus: int = 4,
) -> Dict[str, float]:
    """
    估算偏好对齐训练所需的总显存
    
    Args:
        total_params: 模型总参数量
        trainable_ratio: 可训练参数比例
        hidden_size: 隐藏层维度
        num_layers: 层数
        num_attention_heads: 注意力头数
        batch_size: 每个GPU的批大小
        gradient_accumulation_steps: 梯度累积步数
        sequence_length: 序列长度
        image_count: 每个样本的图片数量
        image_size: 图片尺寸
        dtype: 数据类型
        optimizer: 优化器类型
        gradient_checkpointing: 是否使用梯度检查点
        zero_stage: DeepSpeed ZeRO优化阶段
        num_gpus: GPU数量
        
    Returns:
        包含显存占用详细信息的字典 (单位: GB)
    """
    # 1. 模型参数显存
    model_info = estimate_model_memory(total_params, dtype, trainable_ratio)
    
    # 2. 优化器状态显存
    optimizer_info = estimate_optimizer_memory(
        model_info["trainable_params"], optimizer, dtype, zero_stage
    )
    
    # 3. 梯度显存
    gradient_info = estimate_gradient_memory(
        model_info["trainable_params"], dtype, zero_stage
    )
    
    # 4. 激活值显存
    activation_info = estimate_activation_memory(
        batch_size, sequence_length, hidden_size, num_layers,
        num_attention_heads, gradient_checkpointing, dtype
    )
    
    # 5. 批次数据显存
    batch_info = estimate_batch_data_memory(
        batch_size, sequence_length, 150000,  # 假设vocab_size
        image_count, 3, image_size, dtype
    )
    
    # 计算单GPU总显存
    single_gpu_memory = (
        model_info["model_memory_gb"] +
        optimizer_info["optimizer_memory_gb"] +
        gradient_info["gradient_memory_gb"] +
        activation_info["activation_memory_gb"] +
        batch_info["batch_data_memory_gb"]
    )
    
    # DeepSpeed ZeRO优化效果
    if zero_stage == 0:
        # 无优化
        per_gpu_memory = single_gpu_memory
    elif zero_stage == 1:
        # ZeRO-1: 分片优化器状态
        per_gpu_memory = (
            model_info["model_memory_gb"] +
            optimizer_info["optimizer_memory_gb"] / num_gpus +
            gradient_info["gradient_memory_gb"] +
            activation_info["activation_memory_gb"] +
            batch_info["batch_data_memory_gb"]
        )
    elif zero_stage == 2:
        # ZeRO-2: 分片优化器状态和梯度
        per_gpu_memory = (
            model_info["model_memory_gb"] +
            optimizer_info["optimizer_memory_gb"] / num_gpus +
            gradient_info["gradient_memory_gb"] / num_gpus +
            activation_info["activation_memory_gb"] +
            batch_info["batch_data_memory_gb"]
        )
    elif zero_stage == 3:
        # ZeRO-3: 分片优化器状态、梯度和模型参数
        per_gpu_memory = (
            model_info["model_memory_gb"] / num_gpus +
            optimizer_info["optimizer_memory_gb"] / num_gpus +
            gradient_info["gradient_memory_gb"] / num_gpus +
            activation_info["activation_memory_gb"] +
            batch_info["batch_data_memory_gb"]
        )
    else:
        per_gpu_memory = single_gpu_memory
    
    # 添加20%的安全边界（用于其他开销，如PyTorch框架本身、CUDA内核等）
    safety_margin = per_gpu_memory * 0.2
    per_gpu_memory_with_margin = per_gpu_memory + safety_margin
    
    return {
        "model_memory_gb": model_info["model_memory_gb"],
        "optimizer_memory_gb": optimizer_info["optimizer_memory_gb"],
        "gradient_memory_gb": gradient_info["gradient_memory_gb"],
        "activation_memory_gb": activation_info["activation_memory_gb"],
        "batch_data_memory_gb": batch_info["batch_data_memory_gb"],
        "single_gpu_total_gb": single_gpu_memory,
        "per_gpu_memory_gb": per_gpu_memory,
        "safety_margin_gb": safety_margin,
        "per_gpu_memory_with_margin_gb": per_gpu_memory_with_margin,
        "total_params": total_params,
        "trainable_params": model_info["trainable_params"],
        "num_gpus": num_gpus,
        "zero_stage": zero_stage,
        "gradient_checkpointing": gradient_checkpointing,
    }


def load_model_config(config_path: str) -> Dict:
    """从配置文件加载模型配置"""
    with open(config_path, 'r') as f:
        return json.load(f)


def print_memory_estimation(results: Dict[str, float]):
    """打印显存估算结果"""
    print("=" * 80)
    print("偏好对齐训练显存估算结果 (Preference Training Memory Estimation)")
    print("=" * 80)
    print()
    
    print(f"模型配置:")
    print(f"  总参数量:         {results['total_params']:,} ({results['total_params']/1e9:.2f}B)")
    print(f"  可训练参数量:     {results['trainable_params']:,} ({results['trainable_params']/1e9:.2f}B)")
    print(f"  GPU数量:          {results['num_gpus']}")
    print(f"  ZeRO优化阶段:     Stage {results['zero_stage']}")
    print(f"  梯度检查点:       {'启用' if results['gradient_checkpointing'] else '禁用'}")
    print()
    
    print(f"显存占用详情 (每个组件):")
    print(f"  模型参数:         {results['model_memory_gb']:>8.2f} GB")
    print(f"  优化器状态:       {results['optimizer_memory_gb']:>8.2f} GB")
    print(f"  梯度:             {results['gradient_memory_gb']:>8.2f} GB")
    print(f"  激活值:           {results['activation_memory_gb']:>8.2f} GB")
    print(f"  批次数据:         {results['batch_data_memory_gb']:>8.2f} GB")
    print(f"  ----------------")
    print(f"  单GPU总计(无分片): {results['single_gpu_total_gb']:>8.2f} GB")
    print()
    
    print(f"DeepSpeed ZeRO-{results['zero_stage']} 优化后:")
    print(f"  每GPU显存占用:    {results['per_gpu_memory_gb']:>8.2f} GB")
    print(f"  安全边界 (20%):   {results['safety_margin_gb']:>8.2f} GB")
    print(f"  ================")
    print(f"  推荐显存 (含边界): {results['per_gpu_memory_with_margin_gb']:>8.2f} GB")
    print()
    
    # 推荐GPU型号
    required_memory = results['per_gpu_memory_with_margin_gb']
    print(f"GPU推荐:")
    if required_memory <= 16:
        print(f"  ✓ 可以使用 16GB 显存的GPU (如 Tesla V100 16GB)")
    if required_memory <= 24:
        print(f"  ✓ 可以使用 24GB 显存的GPU (如 RTX 3090, RTX 4090)")
    if required_memory <= 32:
        print(f"  ✓ 可以使用 32GB 显存的GPU (如 Tesla V100 32GB)")
    if required_memory <= 40:
        print(f"  ✓ 可以使用 40GB 显存的GPU (如 A100 40GB)")
    if required_memory <= 80:
        print(f"  ✓ 可以使用 80GB 显存的GPU (如 A100 80GB, H100)")
    if required_memory > 80:
        print(f"  ⚠ 需要更大显存的GPU或增加GPU数量")
    
    print()
    print("=" * 80)


def main():
    parser = argparse.ArgumentParser(
        description="估算偏好对齐训练所需的GPU显存",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    # 模型参数
    parser.add_argument("--total_params", type=float, default=8.0,
                       help="模型总参数量 (单位: B, 即十亿)")
    parser.add_argument("--trainable_ratio", type=float, default=1.0,
                       help="可训练参数比例 (0.0-1.0)")
    parser.add_argument("--hidden_size", type=int, default=4096,
                       help="隐藏层维度")
    parser.add_argument("--num_layers", type=int, default=32,
                       help="模型层数")
    parser.add_argument("--num_attention_heads", type=int, default=32,
                       help="注意力头数")
    
    # 训练参数
    parser.add_argument("--batch_size", type=int, default=1,
                       help="每个GPU的批大小")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1,
                       help="梯度累积步数")
    parser.add_argument("--sequence_length", type=int, default=2048,
                       help="序列长度")
    
    # 图像参数
    parser.add_argument("--image_count", type=int, default=1,
                       help="每个样本的图片数量")
    parser.add_argument("--image_size", type=int, default=448,
                       help="图片尺寸")
    
    # 优化参数
    parser.add_argument("--dtype", type=str, default="bf16",
                       choices=["fp32", "fp16", "bf16"],
                       help="训练数据类型")
    parser.add_argument("--optimizer", type=str, default="adamw",
                       choices=["adamw", "sgd"],
                       help="优化器类型")
    parser.add_argument("--gradient_checkpointing", action="store_true", default=True,
                       help="是否使用梯度检查点")
    parser.add_argument("--no_gradient_checkpointing", action="store_false",
                       dest="gradient_checkpointing",
                       help="禁用梯度检查点")
    parser.add_argument("--zero_stage", type=int, default=2,
                       choices=[0, 1, 2, 3],
                       help="DeepSpeed ZeRO优化阶段")
    parser.add_argument("--num_gpus", type=int, default=4,
                       help="GPU数量")
    
    args = parser.parse_args()
    
    # 将参数量从B转换为实际数字
    total_params = int(args.total_params * 1e9)
    
    # 估算显存
    results = estimate_preference_training_memory(
        total_params=total_params,
        trainable_ratio=args.trainable_ratio,
        hidden_size=args.hidden_size,
        num_layers=args.num_layers,
        num_attention_heads=args.num_attention_heads,
        batch_size=args.batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        sequence_length=args.sequence_length,
        image_count=args.image_count,
        image_size=args.image_size,
        dtype=args.dtype,
        optimizer=args.optimizer,
        gradient_checkpointing=args.gradient_checkpointing,
        zero_stage=args.zero_stage,
        num_gpus=args.num_gpus,
    )
    
    # 打印结果
    print_memory_estimation(results)


if __name__ == "__main__":
    main()
