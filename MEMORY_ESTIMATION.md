# 偏好对齐训练显存估算工具

## 简介

本工具用于估算多模态大模型偏好对齐训练所需的GPU显存。在开始训练之前使用此工具，可以：

- ✅ 评估所需的GPU硬件资源
- ✅ 避免因显存不足导致的训练失败
- ✅ 优化训练配置以更好地利用现有硬件
- ✅ 比较不同配置下的显存占用

## 快速开始

```bash
# 使用默认配置（对应finetune_preference.sh）
python estimate_preference_memory.py

# 查看帮助信息
python estimate_preference_memory.py --help

# 对比不同优化策略的显存占用
python compare_memory_strategies.py

# 运行示例脚本（展示多种配置）
bash examples_memory_estimation.sh
```

## 主要功能

### 1. 显存组件估算

该工具会估算以下组件的显存占用：

- **模型参数**: 存储模型权重
- **优化器状态**: AdamW优化器的动量和方差
- **梯度**: 反向传播时的梯度
- **激活值**: 前向传播的中间结果
- **批次数据**: 输入的文本和图像数据

### 2. 优化策略支持

- **DeepSpeed ZeRO**: 支持ZeRO-0到ZeRO-3的优化阶段
- **梯度检查点**: 以计算换显存的优化策略
- **混合精度训练**: 支持fp32、fp16和bf16

### 3. 显存优化建议

工具会根据估算结果推荐合适的GPU型号，如：

- 16GB显存: Tesla V100 16GB
- 24GB显存: RTX 3090, RTX 4090
- 40GB显存: A100 40GB
- 80GB显存: A100 80GB, H100

## 使用示例

### 基础示例

```bash
# 估算8B模型，4个GPU，ZeRO-2配置下的显存
python estimate_preference_memory.py \
    --total_params 8.0 \
    --num_gpus 4 \
    --zero_stage 2
```

### 显存优化示例

如果显存不足，可以尝试：

```bash
# 方法1: 使用ZeRO-3
python estimate_preference_memory.py \
    --total_params 8.0 \
    --num_gpus 4 \
    --zero_stage 3

# 方法2: 减小序列长度
python estimate_preference_memory.py \
    --total_params 8.0 \
    --sequence_length 1024

# 方法3: 增加GPU数量
python estimate_preference_memory.py \
    --total_params 8.0 \
    --num_gpus 8
```

## 输出示例

```
================================================================================
偏好对齐训练显存估算结果 (Preference Training Memory Estimation)
================================================================================

模型配置:
  总参数量:         8,000,000,000 (8.00B)
  可训练参数量:     8,000,000,000 (8.00B)
  GPU数量:          4
  ZeRO优化阶段:     Stage 2
  梯度检查点:       启用

显存占用详情 (每个组件):
  模型参数:            14.90 GB
  优化器状态:          59.60 GB
  梯度:                29.80 GB
  激活值:               2.62 GB
  批次数据:             0.00 GB
  ----------------
  单GPU总计(无分片):   106.94 GB

DeepSpeed ZeRO-2 优化后:
  每GPU显存占用:       39.88 GB
  安全边界 (20%):       7.98 GB
  ================
  推荐显存 (含边界):    47.86 GB

GPU推荐:
  ✓ 可以使用 80GB 显存的GPU (如 A100 80GB, H100)
================================================================================
```

## 参数说明

### 常用参数

| 参数 | 说明 | 默认值 |
|------|------|--------|
| `--total_params` | 模型参数量(B) | 8.0 |
| `--num_gpus` | GPU数量 | 4 |
| `--batch_size` | 每GPU批大小 | 1 |
| `--sequence_length` | 序列长度 | 2048 |
| `--zero_stage` | ZeRO优化阶段(0-3) | 2 |
| `--dtype` | 数据类型 | bf16 |
| `--gradient_checkpointing` | 梯度检查点 | True |

更多参数请使用 `--help` 查看。

## 文档

- **完整使用指南**: [docs/memory_estimation_guide.md](docs/memory_estimation_guide.md)
- **示例脚本**: `examples_memory_estimation.sh`

## 注意事项

1. **估算准确性**: 本工具提供理论估算值，实际显存占用可能因环境而异，建议预留20%安全边界
2. **偏好训练特点**: 偏好训练处理正负样本对，数据显存是常规训练的约2倍
3. **多模态模型**: 本工具主要估算LLM部分显存，视觉编码器显存相对较小
4. **实际验证**: 在小规模数据上先行测试，验证估算准确性

## 相关文件

- `estimate_preference_memory.py` - 显存估算主程序
- `compare_memory_strategies.py` - 优化策略对比工具
- `docs/memory_estimation_guide.md` - 详细文档
- `examples_memory_estimation.sh` - 示例脚本
- `finetune_preference.sh` - 偏好训练脚本
- `mllm/ds_pref_config_zero2.json` - DeepSpeed配置

## 反馈与改进

如果发现估算结果与实际情况有较大偏差，欢迎反馈实际训练中的显存占用数据，以帮助改进工具的准确性。
