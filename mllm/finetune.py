import glob
import json
import logging
import os
import pathlib
from dataclasses import dataclass, field
from functools import partial
from typing import Dict, List, Optional, Tuple
from types import MethodType

import torch
import transformers
from accelerate.utils import DistributedType

from model import MLLMModel
from transformers import AutoTokenizer
from transformers.integrations import deepspeed

from mllm.train.datasets import SupervisedDataset, PreferenceTrainDataset
from mllm.train.datasets_grounding import GroundingSupervisedDataset
from mllm.train.trainer import SFTTrainer, PreferenceTrainer
from mllm.train.preprocess import data_collator, build_transform, preference_collator_fn, PreferenceDatasetDataCollator

from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training

@dataclass
class ModelArguments:
    model_name_or_path: Optional[str] = field(default=None)


@dataclass
class DataArguments:
    data_path: str = field(
        default=None, metadata={"help": "Path to the training data."}
    )
    eval_data_path: str = field(
        default=None, metadata={"help": "Path to the evaluation data."}
    )

    data_dir: Optional[str] = field(default=None, metadata={"help": "Directory for the logp file."})
    ref_name: Optional[str] = field(default=None, metadata={"help": "Preference reference model name."})
    image_folder: Optional[str] = field(default="", metadata={"help": "Base directory for the input images."})
    preference_beta: Optional[float] = field(default=0.1)

@dataclass
class TrainingArguments(transformers.TrainingArguments):
    cache_dir: Optional[str] = field(default=None)
    optim: str = field(default="adamw_torch")
    model_max_length: int = field(
        default=2048,
        metadata={
            "help": "Maximum sequence length. Sequences will be right padded (and possibly truncated)."
        },
    )
    tune_vision: Optional[bool] = field(default=True)
    tune_llm: Optional[bool] = field(default=True)
    use_lora: Optional[bool] = field(default=False)
    max_slice_nums: Optional[int] = field(default=9)

    task: str = field(default='LM')

    preference_use_average_logp: Optional[bool] = field(default=False)

@dataclass
class LoraArguments:
    lora_r: int = 64
    lora_alpha: int = 64
    lora_dropout: float = 0.05
    lora_target_modules: str = r"llm\..*layers\.\d+\.self_attn\.(q_proj|k_proj|v_proj)"
    lora_weight_path: str = ""
    lora_bias: str = "none"
    q_lora: bool = False
    lora_modules_to_save: str = ""
    lora_layer_replication: Optional[List[Tuple[int, int]]] = None
    lora_layers_to_transform: Optional[List[int]] = None
    lora_layers_pattern: Optional[str] = None

local_rank = None
def rank0_print(*args):
    if local_rank == 0:
        print(*args)


def safe_save_model_for_hf_trainer(trainer, output_dir: str, bias="none"):
    """Collects the state dict and dump to disk."""
    if trainer.args.should_save and trainer.args.local_rank == 0:
        trainer.save_model(output_dir,)

def make_supervised_data_module(
    tokenizer: transformers.PreTrainedTokenizer,
    data_args,
    transform,
    data_collator=None,
    slice_config=None,
    patch_size=14,
    query_nums=64,
    batch_vision=False,
    max_length=2048,
    dataset_cls = SupervisedDataset
) -> Dict:
    """Make dataset and collator for supervised fine-tuning."""

    rank0_print("Loading data...")

    train_json = json.load(open(data_args.data_path, "r"))
    train_dataset = dataset_cls(
        train_json,
        transform,
        tokenizer,
        slice_config=slice_config,
        patch_size=patch_size,
        query_nums=query_nums,
        batch_vision=batch_vision,
        max_length=max_length,
    )
    print(f'Train data size is {len(train_dataset)}', flush=True)

    if data_args.eval_data_path:
        eval_json = json.load(open(data_args.eval_data_path, "r"))
        eval_dataset = dataset_cls(
            eval_json,
            transform,
            tokenizer,
            slice_config=slice_config,
            patch_size=patch_size,
            query_nums=query_nums,
            batch_vision=batch_vision,
            max_length=max_length,
        )
    else:
        eval_dataset = None

    return dict(
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator= partial(data_collator, max_length=max_length),
    )

def make_preference_data_module(tokenizer, data_args,
                         reference_model, transform,
                         slice_config, batch_vision,
                         max_length=2048,):
    train_dataset = PreferenceTrainDataset(tokenizer=tokenizer,
                               data_dir=data_args.data_dir,
                               filepath=data_args.data_path,
                               ref_name=data_args.ref_name,
                               multimodal_cfg=dict(
                                    is_multimodal=data_args.is_multimodal,
                                    image_token_len=data_args.image_token_len,
                                    image_folder=data_args.image_folder,
                                    image_transform=transform,
                                    patch_size=slice_config["patch_size"],
                                    slice_config=slice_config,
                                    batch_vision=batch_vision,
                                    max_length=max_length,
                                ),
                               reference_model=reference_model)
    print(f'Train data size is {len(train_dataset)}', flush=True)
    data_collator = PreferenceDatasetDataCollator(
        tokenizer=tokenizer, beta=data_args.preference_beta,
        preference_collator_fn=partial(preference_collator_fn, pad_token_id=0, max_length=max_length))

    eval_datasets = None

    return dict(train_dataset=train_dataset,
                eval_dataset=eval_datasets,
                data_collator=data_collator)

def get_parameter_number(model):
    trainable_params, all_param = 0, 0
    for param in model.parameters():
        num_params = param.numel()
        # if using DS Zero 3 and the weights are initialized empty
        if num_params == 0 and hasattr(param, "ds_numel"):
            num_params = param.ds_numel

        all_param += num_params
        if param.requires_grad:
            trainable_params += num_params

    return {'Total': all_param, 'Trainable': trainable_params}


def init_model(model_args, data_args, training_args, lora_args):
    global local_rank

    compute_dtype = (
        torch.float16
        if training_args.fp16
        else (torch.bfloat16 if training_args.bf16 else torch.float32)
    )

    local_rank = training_args.local_rank
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    ddp = world_size != 1
    device_map = None
    if lora_args.q_lora:
        device_map = {"": int(os.environ.get("LOCAL_RANK") or 0)} if ddp else None
        if len(training_args.fsdp) > 0 or deepspeed.is_deepspeed_zero3_enabled():
            logging.warning(
                "FSDP or ZeRO3 are not incompatible with QLoRA."
            )

    model = MLLMModel.from_pretrained(
        model_args.model_name_or_path,
        torch_dtype=compute_dtype,
        device_map=device_map,
    )

    tokenizer = AutoTokenizer.from_pretrained(
        model_args.model_name_or_path, trust_remote_code=True
    )

    if not training_args.tune_vision:
        model.vpm.requires_grad_(False)
    if not training_args.tune_llm:
        model.llm.requires_grad_(False)

    if training_args.use_lora and training_args.task == 'LM':
        if training_args.use_lora and training_args.tune_llm:
            raise ValueError("The model cannot simultaneously adjust LLM parameters and apply LoRA.")

        rank0_print("Currently using LoRA for fine-tuning the MiniCPM-V model.")
        for name, param in model.llm.named_parameters():
            param.requires_grad = False
        modules_to_save = ['embed_tokens','resampler']
        if training_args.tune_vision:
            modules_to_save.append('vpm')
        lora_config = LoraConfig(
            r=lora_args.lora_r,
            lora_alpha=lora_args.lora_alpha,
            target_modules=lora_args.lora_target_modules,
            lora_dropout=lora_args.lora_dropout,
            bias=lora_args.lora_bias,
            layers_to_transform=lora_args.lora_layers_to_transform,
            modules_to_save=modules_to_save,
        )
        if not hasattr(model, 'get_input_embeddings'):
            def get_input_embeddings(self):
                return self.llm.get_input_embeddings()
            model.get_input_embeddings = MethodType(get_input_embeddings, model)
        if lora_args.q_lora:
            model = prepare_model_for_kbit_training(
                model, use_gradient_checkpointing=training_args.gradient_checkpointing
            )
        model = get_peft_model(model, lora_config)
        if training_args.gradient_checkpointing:
            model.enable_input_require_grads()
    elif training_args.use_lora and training_args.task == 'Preference':
        raise NotImplementedError("Lora is not implemented on preference training.")

    rank0_print(get_parameter_number(model))

    params_no_grad = [
        n for n, p in model.named_parameters() if not p.requires_grad]
    # rank0_print(f'No grad params are : {params_no_grad}')

    # Load data
    if hasattr(model.config, "slice_config"):
        model.config.slice_config.max_slice_nums = training_args.max_slice_nums
        slice_config = model.config.slice_config.to_dict()
    else:
        model.config.max_slice_nums = training_args.max_slice_nums
        slice_config = model.config.to_dict()

    if hasattr(model.config, "batch_vision_input"):
        batch_vision = model.config.batch_vision_input
    else:
        batch_vision = False

    transform_func = build_transform()

    if training_args.task in ['LM', 'Grounding']:
        data_module = make_supervised_data_module(
            tokenizer=tokenizer,
            data_args=data_args,
            transform=transform_func,
            data_collator=data_collator,
            slice_config=slice_config,
            patch_size=model.config.patch_size,
            query_nums=model.config.query_num,
            batch_vision=batch_vision,
            max_length=training_args.model_max_length,
            dataset_cls=SupervisedDataset if training_args.task == 'LM' else GroundingSupervisedDataset
        )
    elif training_args.task == 'Preference':
        data_args.image_token_len = model.config.query_num
        data_args.is_multimodal = True
        data_module = make_preference_data_module(
            tokenizer, data_args=data_args,
            reference_model=model_args.model_name_or_path,
            transform=transform_func,
            slice_config=slice_config,
            batch_vision=batch_vision,
            max_length=training_args.model_max_length,
        )
    return model, data_module, tokenizer


local_rank = 0

def train():
    parser = transformers.HfArgumentParser(
        (ModelArguments, DataArguments, TrainingArguments, LoraArguments)
    )

    (
        model_args,
        data_args,
        training_args,
        lora_args,
    ) = parser.parse_args_into_dataclasses()

    if getattr(training_args, "deepspeed", None) :
        training_args.distributed_state.distributed_type = DistributedType.DEEPSPEED

    model, data_module, tokenizer = init_model(model_args, data_args, training_args, lora_args)

    training_args.gradient_checkpointing_kwargs={"use_reentrant":False}

    if training_args.task in ['LM', 'Grounding']:
        trainer = SFTTrainer(
            model=model,
            tokenizer=tokenizer,
            args=training_args,
            **data_module,
        )
    elif training_args.task == 'Preference':
        trainer = PreferenceTrainer(
            model=model,
            tokenizer=tokenizer,
            args=training_args,
            **data_module
        )

    if list(pathlib.Path(training_args.output_dir).glob("checkpoint-*")):
        print(f'Resume from checkpoint.')
        trainer.train(resume_from_checkpoint=True)
    else:
        print(f'Train from start.')
        trainer.train()
    trainer.save_state()

    safe_save_model_for_hf_trainer(
        trainer=trainer,
        output_dir=training_args.output_dir,
        bias=lora_args.lora_bias)


if __name__ == "__main__":
    train()