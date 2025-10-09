# 偏好对齐训练显存估算指南

## 概述

`estimate_preference_memory.py` 是一个用于估算偏好对齐训练（Preference Training）所需GPU显存的工具。它可以帮助您在开始训练之前，评估所需的硬件资源，避免因显存不足而导致的训练失败。

## 快速开始

### 基本使用

使用默认配置（8B参数模型，4个GPU，ZeRO-2优化）：

```bash
python estimate_preference_memory.py
```

### 自定义配置

根据您的实际训练配置进行显存估算：

```bash
python estimate_preference_memory.py \
    --total_params 8.0 \
    --num_gpus 4 \
    --batch_size 1 \
    --sequence_length 2048 \
    --zero_stage 2 \
    --gradient_checkpointing \
    --dtype bf16
```

## 参数说明

### 模型参数

- `--total_params`: 模型总参数量（单位：十亿，B）
  - 默认值：8.0 (8B参数)
  - 示例：对于MiniCPM-V-2.6模型，使用 `--total_params 8.0`

- `--trainable_ratio`: 可训练参数比例
  - 默认值：1.0 (全部参数可训练)
  - 取值范围：0.0 ~ 1.0
  - 示例：如果只微调部分参数，使用 `--trainable_ratio 0.5`

- `--hidden_size`: 隐藏层维度
  - 默认值：4096
  - 常见值：2048, 4096, 8192

- `--num_layers`: 模型层数
  - 默认值：32
  - 示例：对于不同规模的模型，层数可能在 24~80 之间

- `--num_attention_heads`: 注意力头数
  - 默认值：32
  - 通常与hidden_size成比例

### 训练参数

- `--batch_size`: 每个GPU的批大小
  - 默认值：1
  - 说明：偏好训练中，每个batch包含正负样本对，实际占用显存相当于2倍batch_size

- `--gradient_accumulation_steps`: 梯度累积步数
  - 默认值：1
  - 说明：不影响显存估算，但影响有效batch size

- `--sequence_length`: 序列长度
  - 默认值：2048
  - 常见值：512, 1024, 2048, 4096
  - 说明：序列越长，激活值显存占用越大

### 图像参数

- `--image_count`: 每个样本的图片数量
  - 默认值：1
  - 说明：多图对话时可能需要调整

- `--image_size`: 图片尺寸
  - 默认值：448
  - 说明：图片分辨率，与slice_nums相关

### 优化参数

- `--dtype`: 训练数据类型
  - 默认值：bf16
  - 可选值：fp32, fp16, bf16
  - 说明：bf16在保持训练稳定性的同时减少显存占用

- `--optimizer`: 优化器类型
  - 默认值：adamw
  - 可选值：adamw, sgd
  - 说明：AdamW需要更多显存存储动量和方差状态

- `--gradient_checkpointing`: 是否使用梯度检查点
  - 默认值：启用 (True)
  - 说明：梯度检查点以计算时间换取显存，可大幅降低激活值显存占用
  - 禁用方式：`--no_gradient_checkpointing`

- `--zero_stage`: DeepSpeed ZeRO优化阶段
  - 默认值：2
  - 可选值：0, 1, 2, 3
  - 说明：
    - ZeRO-0: 不使用ZeRO优化
    - ZeRO-1: 分片优化器状态
    - ZeRO-2: 分片优化器状态和梯度
    - ZeRO-3: 分片优化器状态、梯度和模型参数

- `--num_gpus`: GPU数量
  - 默认值：4
  - 说明：使用多GPU训练时，ZeRO会在GPU间分片数据

## 使用示例

### 示例1：使用finetune_preference.sh的默认配置

根据 `finetune_preference.sh` 中的配置：

```bash
python estimate_preference_memory.py \
    --total_params 8.0 \
    --batch_size 1 \
    --sequence_length 2048 \
    --num_gpus 4 \
    --zero_stage 2 \
    --gradient_checkpointing \
    --dtype bf16 \
    --trainable_ratio 1.0
```

### 示例2：资源受限场景（减少显存占用）

如果显存不足，可以尝试：

```bash
python estimate_preference_memory.py \
    --total_params 8.0 \
    --batch_size 1 \
    --sequence_length 1024 \
    --num_gpus 4 \
    --zero_stage 3 \
    --gradient_checkpointing \
    --dtype bf16
```

主要优化点：
- 使用ZeRO-3进一步减少单GPU显存
- 减小sequence_length
- 确保启用gradient_checkpointing

### 示例3：单GPU训练估算

```bash
python estimate_preference_memory.py \
    --total_params 2.0 \
    --batch_size 1 \
    --sequence_length 1024 \
    --num_gpus 1 \
    --zero_stage 0 \
    --gradient_checkpointing \
    --dtype bf16
```

## 输出解读

### 输出示例

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
  模型参数:            16.00 GB
  优化器状态:          64.00 GB
  梯度:                32.00 GB
  激活值:               8.50 GB
  批次数据:             0.50 GB
  ----------------
  单GPU总计(无分片):  121.00 GB

DeepSpeed ZeRO-2 优化后:
  每GPU显存占用:       40.50 GB
  安全边界 (20%):       8.10 GB
  ================
  推荐显存 (含边界):   48.60 GB

GPU推荐:
  ✓ 可以使用 40GB 显存的GPU (如 A100 40GB)
  ✓ 可以使用 80GB 显存的GPU (如 A100 80GB, H100)

================================================================================
```

### 关键指标说明

1. **单GPU总计(无分片)**: 如果不使用ZeRO优化，单个GPU需要的显存
2. **每GPU显存占用**: 使用ZeRO优化后，每个GPU实际需要的显存
3. **推荐显存**: 加上20%安全边界后的显存需求，建议按此配置GPU

### 各组件显存占用

1. **模型参数**: 存储模型权重的显存
   - bf16: 2 bytes/参数
   - fp32: 4 bytes/参数

2. **优化器状态**: AdamW优化器需要存储动量和方差
   - 通常是模型参数的2倍（fp32存储）

3. **梯度**: 存储参数梯度
   - 通常以fp32存储，与参数量相同

4. **激活值**: 前向传播中间结果
   - 使用梯度检查点可大幅减少
   - 与batch_size和sequence_length成正比

5. **批次数据**: 输入数据（文本和图像）
   - 偏好训练中包含正负样本对，是常规训练的2倍

## 显存优化建议

### 如果显存不足，可以尝试以下方法：

1. **启用梯度检查点** (`--gradient_checkpointing`)
   - 可减少约75%的激活值显存
   - 训练时间增加约20%

2. **使用更高级别的ZeRO优化**
   - ZeRO-2 → ZeRO-3: 进一步减少单GPU显存
   - 注意：ZeRO-3可能增加通信开销

3. **减小batch_size**
   - 可以使用gradient_accumulation_steps保持有效batch size

4. **减小sequence_length**
   - 如果数据允许，可以截断较长序列

5. **增加GPU数量**
   - ZeRO会在更多GPU间分片，减少单GPU显存

6. **使用混合精度训练**
   - 使用bf16而非fp32可减少50%显存

## 注意事项

1. **估算的准确性**
   - 本工具提供的是理论估算值
   - 实际显存占用可能因PyTorch版本、CUDA版本、硬件特性等因素有所不同
   - 建议预留20%的安全边界

2. **偏好训练特点**
   - 偏好训练需要处理正负样本对
   - 实际数据显存是常规训练的约2倍
   - 需要额外存储reference model的logp（已预先计算）

3. **多模态模型特点**
   - 除了LLM部分，还包括视觉编码器
   - 图像数据占用的显存与图片数量和分辨率相关
   - 本工具主要估算LLM部分的显存

4. **DeepSpeed配置**
   - 确保训练脚本中的DeepSpeed配置与估算参数一致
   - ZeRO-3可能需要额外的配置（如offload）

## 相关文件

- `finetune_preference.sh`: 偏好训练启动脚本
- `mllm/ds_pref_config_zero2.json`: DeepSpeed ZeRO-2配置文件
- `mllm/train/trainer.py`: 偏好训练器实现

## 参考资料

- [DeepSpeed ZeRO Documentation](https://www.deepspeed.ai/tutorials/zero/)
- [PyTorch Memory Management](https://pytorch.org/docs/stable/notes/cuda.html)
- [Gradient Checkpointing](https://pytorch.org/docs/stable/checkpoint.html)
