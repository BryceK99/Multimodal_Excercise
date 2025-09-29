# THUNLP Multimodal Exercise

## 背景
\n## 核心数据结构解析

多模态大模型的核心数据结构涉及多种模态数据的表示、处理和交互。本节将详细解析MLLM中常用的数据结构及其作用。

### 1. 输入数据结构

#### 文本输入结构
文本数据在MLLM中的处理流程如下：
- **原始文本**: 用户输入的自然语言文本
- **Token序列**: 通过tokenizer转换为token ID列表
- **特殊标记**: 添加BOS、EOS等控制标记
- **注意力遮罩**: 指示有效输入位置的二进制矩阵

示例数据结构：
```python
{
    "input_ids": [101, 2023, 2003, 1037, 2931, 102],  # token IDs
    "attention_mask": [1, 1, 1, 1, 1, 1],  # 有效位置标记
    "image": PIL.Image对象或tensor  # 对应的图像
}
```

#### 图像输入结构
图像数据的处理涉及多个步骤：
- **预处理**: resize到固定尺寸，归一化
- **分块**: 将图像分割成patches (如16x16)
- **嵌入**: 每个patch转换为向量表示
- **位置编码**: 添加二维位置信息

### 2. 嵌入层数据结构

#### 文本嵌入
- **词嵌入矩阵**: vocab_size × embedding_dim
- **位置编码**: 最大长度 × embedding_dim
- **输出**: sequence_length × embedding_dim

#### 视觉嵌入
- **Patch嵌入**: num_patches × embedding_dim
- **分类标记**: [CLS] token用于分类任务
- **位置编码**: 与文本位置编码对齐

### 3. 注意力机制数据流

#### 自注意力 (Self-Attention)
在Transformer层中的数据流：
1. 输入嵌入: batch_size × seq_len × embed_dim
2. 线性变换: 生成Q, K, V矩阵
3. 注意力计算: softmax(QK^T / sqrt(d_k))
4. 加权求和: attention_weights × V
5. 输出投影: 最终的注意力输出

#### 交叉注意力 (Cross-Attention)
用于模态融合：
- Query来自文本
- Key和Value来自图像
- 计算模态间的注意力权重

### 4. 输出数据结构

#### 文本生成输出
- **logits**: batch_size × seq_len × vocab_size
- **概率分布**: softmax后的概率
- **生成的序列**: 通过采样或贪婪解码获得

#### 视觉定位输出
- **坐标框**: [x1, y1, x2, y2] 格式
- **置信度**: 预测的置信分数

### 5. 训练数据批处理

#### DataLoader输出
```python
{
    "input_ids": torch.Tensor(batch_size, max_len),
    "attention_mask": torch.Tensor(batch_size, max_len),
    "labels": torch.Tensor(batch_size, max_len),  # 训练目标
    "images": torch.Tensor(batch_size, 3, 224, 224)  # 图像批次
}
```

#### 数据整理器 (Data Collator)
- 处理变长序列的填充
- 生成注意力遮罩
- 处理特殊标记的遮罩

理解这些核心数据结构对于调试和优化MLLM至关重要，它们直接影响模型的性能和效率。
多模态大模型（Multimodal Large Language Models, MLLM）的构建过程中，模型结构、模型预测、指令微调以及偏好对齐训练是其中重要的组成部分。本次任务中，将提供一个不完整的多模态大模型结构及微调代码，请根据要求，补全过程中的关键步骤，并在提供的数据上实现简单的微调与推理。

## 任务1. 多模态大模型的结构与推理

### 1. Transformer 结构中的多头注意力（Multihead Self-Attention）

Transformer[1] 模型是现代自然语言处理的核心，也是目前主流大模型的结构基础，其核心是自注意力机制（self-attention mechanism）。

多头注意力（Multihead Self-Attention）是Transformer结构中的关键组件，通过引入多个注意力头，它能够捕捉序列中不同位置的特征和依赖关系。每个注意力头在进行线性变换后，独立地计算注意力分数，然后将结果拼接并再次投影。这样可以增强模型的表示能力，允许不同的头关注输入序列的不同部分。
\n多头注意力允许模型从不同的表示子空间中学习信息，每个头可以关注输入序列的不同方面。这种并行处理机制显著提高了模型捕捉复杂依赖关系的能力。

在实现多头注意力时，forward函数的核心任务是计算输入序列的自注意力输出。具体步骤包括：将输入映射至多个子空间（对应多个头），计算注意力权重，通过这些权重加权求和得到注意力输出，最后将所有头的输出拼接并通过线性变换得到最终输出。
\n
**实现要点：**
- 将输入投影到多个头：`q, k, v = self.q_proj(x), self.k_proj(x), self.v_proj(x)`
- 重塑为多头格式：`q = q.view(batch, num_heads, seq_len, head_dim)`
- 计算注意力权重：`attn_weights = torch.matmul(q, k.transpose(-2, -1)) / sqrt(head_dim)`
- 应用遮罩和softmax
- 加权求和并拼接所有头

<div align="center">
    <img src=assets/transformer_multihead_attn.png width=60% />
</div>

**参考论文：** [1] [Attention is All You Need](https://arxiv.org/abs/1706.03762)

**任务描述：** 请根据论文中的 attention 计算公式，完善[mllm/model/llm/llm_architecture.py](mllm/model/llm/llm_architecture.py#L216)，补充 `LLMAttention` 中的 `forward` 计算。

### 2. 多模态特征融合

多模态大模型旨在处理和融合来自不同模态的信号，如文本和图像，以支持更加丰富的应用。多模态特征融合是该过程中至关重要的部分，它通过将来自不同模态的信号表征为嵌入向量（embedding），并进行拼接，从而统一处理来自不同模态的信号。

在实现多模态特征融合的过程中，常见的方式为，首先分别获取图像和文本的嵌入表示。然后，利用一个桥阶层（如 多层全连接层[1]、交叉注意力层[2] 等）将图片表示映射至文本表示空间。最后，将图片映射后的表示与文本表示进行拼接，形成最终的多模态输入信号，传递给语言大模型进行后续处理。
\n
**桥接层设计考虑：**
- **全连接层**: 简单但可能丢失空间信息
- **交叉注意力**: 保留图像的空间结构，计算效率高
- **适配器层**: 参数高效，适合微调场景
- **输出对齐**: 确保视觉和文本嵌入维度一致

<div align="center">
    <img src=assets/llava_architecture.png width=60% />
</div>

**参考论文：** [1] [Visual Instruction Tuning](https://arxiv.org/abs/2304.08485)

[2] [Qwen-VL: A Versatile Vision-Language Model for Understanding, Localization, Text Reading, and Beyond](https://arxiv.org/abs/2308.12966)

**任务描述：** 完善[mllm/model/modeling_mllm.py](mllm/model/modeling_mllm.py#L80)，补充 `get_vllm_embedding()` 中，拼接视觉嵌入与文本嵌入的计算流程。

### 3. 多模态大模型的推理预测

在多模态大模型的推理过程中，模型需要接收和处理文本与图像输入，并采用自回归解码（autoregressive decoding）的形式输出回复。这涉及多个关键步骤，包括输入预处理、模型推理和输出后处理。

输入预处理的过程包括：将对话输入转换为与模型训练阶段一致的模板格式，以确保输入的有效性，同时，将转换后的文本对应到相应的单词ID上；对于图像输入，可能需要对图片进行分块（slice）、调整大小（resize）等操作，以适合视觉编码器的输入要求。在实现中，transformers 库提供了 tokenizer 类，可以便捷地实现输入文本的形式转换，同时利用自定义的 processor 类将图像和文本处理为模型计算时所需的输入格式。

<div align="center">
    <img src=assets/minicpmv_input_process.png width=60% />
</div>

在推理阶段，模型将会根据已有输入和已经产生的输出文本依次循环生成下一个token，并在达到生成结束条件（stopping criteria）时停止输出，通常的结束条件是生成了特定停止符（EOS token），或达到最大输出长度限制。Huggingface transformers 库中集成了这一生成过程，可以使用模型的 `generate()` 方法便捷地控制模型的生成行为。原始的模型输出是一串与文本对应的ID序列，为了得到最终的文本，还需要对生成的输出文本进行解码，并去除相应的特殊符号，如 EOS 标记。

<div align="center">
    <img src=assets/decoding.png width=55% />
</div>

**参考资料：** [Transformers库 generate 方法](https://huggingface.co/docs/transformers/v4.40.2/en/main_classes/text_generation)，[Transformers库 Tokenizer](https://huggingface.co/docs/transformers/v4.40.2/en/main_classes/tokenizer#tokenizer)

**任务描述：**
1. 完善 [eval/model_eval.py](eval/model_eval.py#L120)，补充`MLLMEvalModel`类中的`prepare_chat_inputs()`，对推理时的输入数据进行预处理；
2. 完善 [mllm/model/modeling_mllm.py](mllm/model/modeling_mllm.py#L215)，补充模型推理函数 `generate()`及其相关函数

### 4. 多模态大模型推理效果验证

多模态大模型的实际效果评价需要通过在不同类型的测试集上进行评测来获得，其中一个重要的评测方面是多模态大模型回复中的“幻觉”程度。“幻觉”是指多模态大模型在回答用户输入的问题时产生的与图片不符的回复，模型的幻觉问题也极大地影响了模型的实际应用。

为了评估模型的幻觉水平，一个常用的指标是 $CHAIR$ 指标。这一指标通过计算模型在图片详细描述中，幻觉物体占全部物体提及中的比例，来表征模型的幻觉情况。$CHAIR$ 指标有两个计算方式，其一是回复级别幻觉率 $CHAIR_s$，这一指标计算所有描述条目中，存在幻觉物体的条目所占的比例；其二是物体级别幻觉率 $CHAIR_i$，这一指标计算所有描述条目中，幻觉物体占全部物体提及的比例。

<div align="center">
    <img src=assets/model_hall.png width=50% />
</div>

**参考论文：** [1] [Object Hallucination in Image Captioning](https://arxiv.org/abs/1809.02156)

**任务描述：**
1. 使用编写好的 [eval/model_eval.py](eval/model_eval.py) 代码，在[objhal_bench.jsonl](https://drive.google.com/drive/folders/1j2kw_UZZq1JXfZI644RGNZzbLIB7bTT5) 数据上进行推理，并运行 [eval/eval_chair.py](eval/eval_chair.py) 计算 CHAIR 得分。
2. 控制模型进行随机解码（sampling decoding）、贪婪解码（greedy search）以及束搜索解码（beam search），观察总结不同解码策略下的模型输出特性。

**要求：** 模型的 CHAIR 得分应当接近 **CHAIRs = 32.7, CHAIRi = 8.5, Recall = 61.7, Len = 126**

## 任务2. 多模态大模型的指令微调（Supervised Finetuning， SFT）

多模态大模型的能力需要通过训练获得。目前，主要的训练流程可以划分为三个阶段：1. 预训练（Pretraining）阶段；2. 多任务预训练（Multi-task Pretraining）；2. 监督微调阶段（Supervised Finetuning）。

预训练阶段，使用海量的图像-文本对数据进行模型训练，增强多模态大模型的图像编码能力，并将图像与文本表示统一至相同的空间。

多任务预训练阶段，使用图文交错数据、已有的简单视觉问答数据进行训练，进一步提升模型的知识。

监督微调阶段，使用高质量的图像对话数据进行训练，增强模型在细节描述、指令理解、对话以及复杂推理的能力。

<div align="center">
    <img src=assets/mllm_sft.png width=70% />
</div>

**参考论文：** [1] [Qwen-VL: A Versatile Vision-Language Model for Understanding, Localization, Text Reading, and Beyond](https://arxiv.org/abs/2308.12966)

### 1. 指令微调数据集

为了对模型进行训练，首先需要建立数据集，对原始的输入数据进行处理，以获得模型训练所需要的数据格式。其中涉及的操作与模型推理阶段类似，需要对输入文本格式进行转换，并对图像进行预先处理。

在PyTorch中，Dataset类提供了灵活的接口来定义和处理自定义数据集。通过继承Dataset类，可以实现指令微调所需的数据集，其功能应该包括执行必要的预处理操作，并将数据整理为模型可接受的输入格式。

**参考资料：** [Pytorch Dataset](https://pytorch.org/docs/stable/data.html#torch.utils.data.Dataset)

**任务描述：** 在 [mllm/train/datasets.py](mllm/train/datasets.py#L20) 中，编写 `SupervisedDataset`，读取训练数据并处理为训练所需格式。

### 2. 训练数据预处理

在模型训练中，批处理（batch processing）是提高计算效率的关键。由于输入数据长度各异，需通过填充（padding）技术使每个批次中的数据条目等长。填充时使用特定的填充标记（padding token）将短于最大长度的序列扩展，而对于超过模型最大输入长度的序列，则进行截断。通常，这一批处理过程在`data_collator()`函数中实现。

在实现data_collator()时需要注意，对长度不足的条目进行填充时，填充标记对输入信息的理解没有帮助，因此不应该对其进行注意力的计算。为此，应当为输入和输出数据创建相应的注意力遮罩（attention mask），以忽略填充部分的注意力计算。

**任务描述：** 在 [mllm/train/preprocess.py](mllm/train/preprocess.py#L71)，补充 `data_collator()`。

### 3. 指令微调损失函数

多模态大模型的指令微调与语言模型的指令微调过程类似，采用自回归损失（autoregressive loss）作为模型优化的目标。该优化目标希望模型最大化训练数据中的回复序列的预测概率。由于输入文本和图像不需要进行预测，因此不参与损失计算，损失仅在输出文本上计算。

<div align="center">
    <img src=assets/language_modeling_loss.png width=40% />
</div>

**任务描述：** 在 [mllm/train/trainer.py](mllm/train/trainer.py#L30) 中，补充 `SFTTrainer` 中的 `compute_loss()` 函数。

### 4. 指令微调训练：

在训练模型的过程中，通过使用训练集和测试集，监控训练和验证过程中的损失变化，可以评估模型的学习效果。

对于不同的训练参数设置，模型可能出现欠拟合（训练损失与测试损失下降均不明显）、过拟合（训练损失下降明显，但测试损失反而上升）的现象。优化训练过程中的超参数，如学习率（learning_rate）、批大小（batch_size）、梯度累积步数（gradient_accumulation_step）等，有助于提升模型在测试集上的性能。

**任务描述：**
1. 使用 [data/train.json](data/train.json) 为训练集，[data/test.json](data/test.json) 为测试集，进行模型训练，监控模型训练集、验证集 loss 变化曲线
2. 调整训练过程超参数，如 learning_rate，batch_size，gradient_accumulation_step等，优化模型在测试集上的 loss 表现

**要求：** 模型训练后，在测试集上的损失值需 **低于0.11**，编写代码评测此时模型在测试集上的回答正确率


### 5. 视觉定位能力增强

视觉定位（Visual Grounding, VG）[1] 的目标是给定针对图像中的对象的*自然语言描述*，获得该对象在图像中的*坐标框*。形式化地，给定一个图像$`I`$和一句自然语言问句$`Q`$，其中问句是对N个名字短语$`\mathcal{P}=\{p_i\}_{i=1}^N`$的位置的询问，视觉定位任务的目标是预测N个名词短语在图像上所对应的位置坐标框$`\mathcal{B}=\{b_i\}_{i=1}^N`$，其中$`b_i\in\mathbb{R}^4`$是表示位置框的左上角和右下角（有些做法中也采用中心点坐标和宽、高值）的浮点数像素坐标值。

<div align="center">
    <img src=assets/vg_example.jpg width=60% />
</div>

**任务描述：**
  - 基于 [Shikra](https://github.com/shikras/shikra/tree/main)中提供的 VG 相关数据集（FlickrDataset、RECDataset等），仿照 [data/train.json](data/train.json) 构造用于 VG 功能的指令微调数据，补充完整 [data/prepare_grounding.py](data/prepare_grounding.py)
  - 基于构造的用于 VG 功能的数据微调模型，使模型能够实现 VG 功能，补充完整 [finetune_grounding.sh](finetune_grounding.sh)和[mllm/train/datasets_grounding.py](mllm/train/datasets_grounding.py)
  - 用一个 case 测试模型的 VG 功能，实现输入关于目标对象的询问输出对象对应的坐标框（可以使用上述例子中的图像，也可以自选一张训练集中为出现过的图像）
  - 参照 [Qwen-VL](https://github.com/QwenLM/Qwen-VL/blob/master/eval_mm/evaluate_grounding.py) 的指标计算方式，在 RefCOCO 的三个子集：val、test-A、test-B，RefCOCO+的三个子集：val test-A、test-B，和 RefCOCOg 的两个子集：val-u、test-u 上计算模型的准确率，补充完整 [eval/grounding_eval.py](eval/grounding_eval.py)

**参考论文：** [1] [KOSMOS-2: Grounding Multimodal Large Language Models to the World](https://arxiv.org/pdf/2306.14824), [2] [Qwen-VL: A Versatile Vision-Language Model for Understanding, Localization, Text Reading, and Beyond](https://arxiv.org/abs/2308.12966)


## 任务3. 多模态大模型的偏好对齐训练

大模型的偏好对齐训练通过结合人类的反馈来优化模型的行为，使其生成的内容更加符合人类的期望和需求。在监督微调阶段，模型仅仅对“正例”进行模仿学习，缺少“负例”来抑制有害输出。在偏好对齐训练阶段，通过收集一系列的正负样本对，对模型的输出进行双向的监督，从而更有效的控制模型的输出。

在偏好对齐训练中，最为经典的算法是 RLHF[1] 算法。该方法首先使用正负样本对训练一个打分模型（reward model），再利用打分模型给出的得分，对大模型的回复进行优化，优化目标是提升大模型回复在打分模型评判下的得分，同时不要与原始的模型参数相差太远。但上述方法存在训练不稳定，不易优化的缺点，一个更简便的算法是 DPO[2] 算法，简化了 RLHF 算法中的两阶段训练过程。

在多模态大模型的构建中，可以利用偏好对齐算法来提升模型在回复上的可信度，降低模型幻觉[3]。

<div align="center">
    <img src=assets/rlhf-v.png width=80% />
</div>

**参考论文：** [1] [Training language models to follow instructions with human feedback](https://arxiv.org/abs/2203.02155),

[2] [Direct Preference Optimization: Your Language Model is Secretly a Reward Model](https://arxiv.org/abs/2305.18290),

[3] [RLHF-V: Towards Trustworthy MLLMs via Behavior Alignment from Fine-grained Correctional Human Feedback](https://arxiv.org/abs/2312.00849)

### 1. 偏好数据对数概率（log probability）计算

为了计算偏好对齐训练中的损失值，需要得到模型在正样本与负样本上的输出概率值，形式化地表示为 $\pi(y|x)$，也即模型在每条训练数据输入 $x$ 下，输出文本 $y$ 的概率值。

**任务描述：** 在 [mllm/train/inference_logp.py](mllm/train/inference_logp.py#L130) 中，实现函数 `get_batch_logps()`。

### 2. 偏好优化损失函数

在 DPO 优化算法提出后，产生了不同的优化目标改进方式（如[1]、[2]等），以提升偏好对齐训练的效果。其中一种改进后的损失函数计算公式如下：

<div align="center">

$$
r_\theta(x, y) := \beta log \frac{\pi_{policy}(y|x)}{\pi_{reference}(y|x)}
$$

$$
L_{preference} = − log \sigma(r_\theta(x, y_w)) − \frac{1}{2}\sum_{y\in\\{y_w,y_l\\}}log\sigma(−r_\theta(x, y))
$$

</div>

其中，$`r_\theta(x, y)`$ 表示模型奖励函数；$`\pi_{policy}`$ 表示策略模型，即优化中的多模态大模型；$`\pi_{reference}`$ 表示参考模型，即初始多模态大模型，$`L_{preference}`$ 表示优化损失函数，$`y_w`$ 表示偏好数据对中的正样本，$`y_l`$ 表示偏好数据对中的负样本。

**参考论文：** [1] [Noise Contrastive Alignment of Language Models with
Explicit Rewards](https://arxiv.org/abs/2402.05369)

[2] [KTO: Model Alignment as Prospect Theoretic Optimization](https://arxiv.org/abs/2402.01306)

**任务描述：** 在 [mllm/train/trainer.py](mllm/train/trainer.py#L263) 中，根据上述损失函数实现 `PreferenceTrainer` 中的 `compute_loss()` 及其相关函数。

### 3. 偏好对齐训练

偏好对齐训练中，损失函数与模型输出的奖励值均可以提现模型优化的效果。在优化过程中，我们希望模型输出的损失值减小，正样本上的奖励值增加，负样本上的奖励值减小。

偏好对齐训练的效果同样受到超参数设置的影响。除了通用的超参数，如 学习率、批大小、梯度累计 以外，在偏好对齐训练中还存在一个新的超参数 $\beta$，这个参数控制了模型偏离原始模型参数的程度。

**任务描述：**
1. 使用 [preference_train.json](https://drive.google.com/drive/folders/1j2kw_UZZq1JXfZI644RGNZzbLIB7bTT5?usp=sharing) 为训练集，进行模型训练，监控模型训练的 loss 变化、reward 变化；
2. 使用 [objhal_bench.jsonl](https://drive.google.com/drive/folders/1j2kw_UZZq1JXfZI644RGNZzbLIB7bTT5?usp=sharing) 为测试集，在训练前后的模型上进行推理，并计算 CHAIR 指标；
3. 调整训练过程超参数，如 偏好对齐超参数 beta，以及其他通用超参数，优化模型在测试集上的 CHAIR 指标表现。

**要求：** 训练后模型的 CHAIR 指标应满足 **CHAIRs < 29.5, CHAIRi < 7.8**

## 数据下载与环境配置
### 1. 数据下载

- 下载地址：https://drive.google.com/drive/folders/1j2kw_UZZq1JXfZI644RGNZzbLIB7bTT5?usp=sharing

- 内容：
  - `chair_300.pkl`: CHAIR 评测GT数据
  - `flash_attn-2.3.4+cu118torch2.1cxx11abiFALSE-cp310-cp310-linux_x86_64.whl`: flash-attention 库 wheel 文件
  - `nltk_data.tar.gz`: CHAIR评测中所需 nltk 库数据。下载后解压到 `/home/user` 目录下
  - `objhal_bench.jsonl`: 幻觉评测集
  - `preference_train.json`: 偏好对齐训练数据
  - `sft_images.tar.gz`: `data/train.json` 及 `data/test.json` 中涉及的图片数据。下载后，请将其解压在 `data/`目录下，并将解压后的文件夹重命名为 `images`

### 2. 模型下载
- 下载地址：https://huggingface.co/HaoyeZhang/MLLM_Excercise_Model

### 3. 环境配置
```bash
# 使用 anaconda 新建虚拟环境
conda create -n MiniCPM-V python=3.10 -y
conda activate MiniCPM-V

# 安装环境依赖库
pip install -r requirements.txt

# 安装训练所需的 flash-attention-2 库
pip install flash_attn-2.3.4+cu118torch2.1cxx11abiFALSE-cp310-cp310-linux_x86_64.whl
```
