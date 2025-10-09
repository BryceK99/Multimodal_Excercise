#!/usr/bin/env python
# coding=utf-8
"""
显存优化策略对比工具

该脚本对比不同优化策略下的显存占用，帮助用户选择最佳配置
"""

import sys
import os

# 添加当前目录到路径
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from estimate_preference_memory import estimate_preference_training_memory


def compare_optimization_strategies():
    """对比不同优化策略的显存占用"""
    
    # 基础配置
    base_config = {
        "total_params": 8_000_000_000,
        "trainable_ratio": 1.0,
        "hidden_size": 4096,
        "num_layers": 32,
        "num_attention_heads": 32,
        "batch_size": 1,
        "sequence_length": 2048,
        "image_count": 1,
        "image_size": 448,
        "dtype": "bf16",
        "optimizer": "adamw",
        "num_gpus": 4,
    }
    
    strategies = [
        {
            "name": "基础配置 (无优化)",
            "config": {**base_config, "gradient_checkpointing": False, "zero_stage": 0}
        },
        {
            "name": "启用梯度检查点",
            "config": {**base_config, "gradient_checkpointing": True, "zero_stage": 0}
        },
        {
            "name": "ZeRO-1 (分片优化器)",
            "config": {**base_config, "gradient_checkpointing": True, "zero_stage": 1}
        },
        {
            "name": "ZeRO-2 (分片优化器+梯度) [推荐]",
            "config": {**base_config, "gradient_checkpointing": True, "zero_stage": 2}
        },
        {
            "name": "ZeRO-3 (分片所有)",
            "config": {**base_config, "gradient_checkpointing": True, "zero_stage": 3}
        },
        {
            "name": "ZeRO-2 + 减小序列长度",
            "config": {**base_config, "gradient_checkpointing": True, "zero_stage": 2, "sequence_length": 1024}
        },
        {
            "name": "ZeRO-2 + 8个GPU",
            "config": {**base_config, "gradient_checkpointing": True, "zero_stage": 2, "num_gpus": 8}
        },
    ]
    
    print("=" * 100)
    print("偏好对齐训练显存优化策略对比")
    print("=" * 100)
    print()
    print(f"基础配置: 8B参数, batch_size=1, seq_len=2048, bf16")
    print()
    print("-" * 100)
    print(f"{'策略':<35} {'单GPU总计':<15} {'优化后/GPU':<15} {'推荐显存':<15} {'说明':<20}")
    print("-" * 100)
    
    results = []
    for strategy in strategies:
        result = estimate_preference_training_memory(**strategy["config"])
        results.append({
            "name": strategy["name"],
            "result": result
        })
        
        print(f"{strategy['name']:<35} "
              f"{result['single_gpu_total_gb']:>8.2f} GB    "
              f"{result['per_gpu_memory_gb']:>8.2f} GB    "
              f"{result['per_gpu_memory_with_margin_gb']:>8.2f} GB    ", end="")
        
        # 添加说明
        mem = result['per_gpu_memory_with_margin_gb']
        if mem <= 24:
            print("RTX 3090/4090")
        elif mem <= 40:
            print("A100 40GB")
        elif mem <= 80:
            print("A100 80GB")
        else:
            print("需要更大显存")
    
    print("-" * 100)
    print()
    
    # 显示节省的显存
    print("显存节省效果对比:")
    print("-" * 100)
    baseline = results[0]["result"]["per_gpu_memory_with_margin_gb"]
    for i, r in enumerate(results):
        if i == 0:
            print(f"{r['name']:<35} 基线")
        else:
            saved = baseline - r["result"]["per_gpu_memory_with_margin_gb"]
            percent = (saved / baseline) * 100
            print(f"{r['name']:<35} 节省 {saved:>6.2f} GB ({percent:>5.1f}%)")
    print("-" * 100)
    print()
    
    # 推荐配置
    print("💡 推荐配置:")
    print("-" * 100)
    print("1. 如有充足显存 (>80GB): 使用 ZeRO-2 + 梯度检查点")
    print("   - 训练效率高，通信开销小")
    print("   - 适合快速迭代和调试")
    print()
    print("2. 显存受限 (40-80GB): 使用 ZeRO-3 + 梯度检查点")
    print("   - 显著降低单GPU显存占用")
    print("   - 可能增加通信开销")
    print()
    print("3. 显存非常受限 (<40GB): 考虑以下组合")
    print("   - ZeRO-3 + 梯度检查点 + 减小序列长度")
    print("   - 或增加GPU数量")
    print("   - 或使用更小的模型")
    print()
    print("4. 最大化训练速度: 增加GPU数量")
    print("   - 每个GPU显存占用更少")
    print("   - 可以增大batch_size提升效率")
    print("-" * 100)


if __name__ == "__main__":
    compare_optimization_strategies()
