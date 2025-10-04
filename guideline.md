# THUNLP Multimodal Exercise 指南

## 项目概览
该仓库提供了一个多模态大模型（Multimodal Large Language Model, MLLM）的教学与实验框架，涵盖模型结构补全、推理流程、监督微调（SFT）、视觉定位（VG）以及偏好对齐训练等环节。`README.md` 中的任务描述给出了每个阶段需要完成的代码位置和目标评测指标。

## 环境准备
- 推荐使用 Python 3.10+ 与 CUDA 兼容的 PyTorch 环境。
- 依赖列表位于 `requirements.txt`，可在新环境中执行：
  ```bash
  pip install -r requirements.txt
  ```
- 多卡训练脚本默认基于 DeepSpeed 与 torchrun，确保已正确安装并配置 GPU 与 NCCL 环境。
- 通过 `export PYTHONPATH=$PYTHONPATH:$(realpath .)` 保证自定义模块可被正确导入。

## 目录结构速览
- `README.md`：完整的课程任务说明、参考资料与评测要求。
- `requirements.txt`：项目依赖清单。
- `assets/`：配套图示，包含注意力结构、输入流程、RLHF 等示意图，可用于理解任务背景。
- Shell 脚本
  - `eval.sh`：使用 `eval/model_eval.py` 对 `data/objhal_bench.jsonl` 进行推理评测的示例脚本。
  - `chair_eval.sh`：在 COCO 数据集上计算 CHAIR 指标的示例脚本，依赖 `eval/chair.py`。
  - `finetune_ds_witheval.sh`：监督微调（SFT）示例，演示如何使用多卡 torchrun 进行训练与评估。
  - `finetune_preference.sh`：基于 DeepSpeed 的偏好对齐训练脚本，调用 DPO/Preference 流程。
  - `finetune_grounding.sh`：Visual Grounding 微调脚本的待补充模板。
- `data/`
  - `train.json` / `test.json`：指令微调示例数据。
  - `prepare_grounding.py`：构造视觉定位训练数据的待补充脚本。
  - `test.json`：用于推理或评测的示例数据。
- `eval/`
  - `model_eval.py`：核心推理与输出格式化逻辑，需补全 `prepare_chat_inputs()` 与生成流程。
  - `chair.py`：CHAIR 指标计算与评估逻辑。
  - `grounding_eval.py`：视觉定位指标计算脚本，需结合 RefCOCO 系列数据实现。
- `mllm/`
  - `finetune.py`：统一的训练入口，整合数据模块、Trainer、PEFT/LoRA 配置、DeepSpeed 参数等。
  - `ds_config_zero2.json` 等：DeepSpeed 训练配置模板。
  - `model/`
    - `modeling_mllm.py`：多模态模型主体，需要实现视觉-文本嵌入拼接与生成接口。
    - `modeling_navit_siglip.py`：视觉编码与特征处理实现。
    - `configuration.py`、`processing.py`、`image_processing.py`、`resampler.py`：模型配置、预处理与视觉特征重采样组件。
    - `llm/llm_architecture.py`：语言模型 Attention 等核心模块，需在 `LLMAttention.forward()` 中补完自注意力计算。
  - `train/`
    - `datasets.py`：监督微调数据集与 Preference 数据集定义，`SupervisedDataset` 待完成。
    - `datasets_grounding.py`：视觉定位数据加载逻辑的待完成脚本。
    - `preprocess.py`：数据增广、`data_collator()` 等批处理逻辑，需处理填充与掩码构建。
    - `trainer.py`：训练循环（SFT 与 Preference Trainer），`compute_loss()` 等函数待实现。
    - `inference_logp.py`：生成 log-prob 输出的推理辅助脚本。
- `utils/file_io.py`：文件读取、缓存等通用辅助函数。

## 常见工作流
### 1. 模型推理
1. 配置并下载基础模型权重，将路径写入 `eval.sh` 中的 `model_name_or_path`。
2. 准备输入问题文件（默认为 `data/objhal_bench.jsonl`）。
3. 运行 `eval.sh`，脚本会调用 `eval/model_eval.py` 加载模型、格式化输入并生成回答。
4. 如需评估幻觉指标，可将生成答案交给 `chair_eval.sh` 计算 CHAIR 指标。

### 2. 监督微调（SFT）
1. 准备训练与验证数据（JSON 格式，与 `data/train.json` / `data/test.json` 对齐）。
2. 检查 `mllm/train/datasets.py`、`preprocess.py`、`trainer.py` 中待补充的函数，确保数据集、collator、损失计算已实现。
3. 更新 `finetune_ds_witheval.sh` 中的模型与数据路径，并根据硬件设置 GPU/DeepSpeed 相关参数。
4. 运行脚本，输出目录默认为 `output/mllm_sft_training`，留意日志与 TensorBoard。

### 3. 偏好对齐训练
1. 准备 logp 数据目录与参考模型，填写 `finetune_preference.sh` 中的变量。
2. 确保 `PreferenceTrainDataset`、`PreferenceTrainer` 等实现完整。
3. 运行脚本启动 DeepSpeed 训练，结果默认存放在 `output/mllm_preference_training`。

### 4. 视觉定位（VG）
1. 在 `data/prepare_grounding.py` 中实现数据构建逻辑，生成视觉定位训练集。
2. 实现 `mllm/train/datasets_grounding.py` 并补充 `finetune_grounding.sh`。
3. 训练完成后，使用 `eval/grounding_eval.py` 在 RefCOCO 系列数据集上计算准确率。

## 数据与模型路径建议
- 将真实模型权重与大规模图片数据保存在快速存储中，脚本中通过变量传入路径。
- 由于训练脚本默认启用了 `gradient_checkpointing` 与 DeepSpeed，需要保证磁盘有足够的读写性能与空间。
- 输出目录（如 `output/`）会在运行时自动创建，可根据需要调整。

## 开发提示
- 逐步完成 `README.md` 中列出的 TODO，建议按照“模型结构 → 推理 → SFT → VG → 偏好训练”的顺序推进。
- 补全代码后，编写小规模单元/集成测试（例如构造伪数据运行数据集与 collator）有助于提前发现问题。
- 运行脚本前可使用 `python -m mllm.train.datasets` 等方式进行快速 sanity check。
- 使用 Git 进行版本管理，并在修改深度学习训练脚本时做好日志记录（可选 `tensorboard`、`wandb` 等）。

如需进一步说明，请参考 `README.md` 中的详细任务描述与参考文献。祝调试顺利！
