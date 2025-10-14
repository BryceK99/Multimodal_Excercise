import os, re
import json
import pickle
import random
import torch
import logging
import pandas as pd
import os.path as op
import transformers
from torch.utils.data import Dataset
import math

from PIL import Image
from typing import Dict
from utils.file_io import read_json, bytes_to_PIL_image
from mllm.train.preprocess import preprocess, fill_boxes_in_conversations
from mllm.train.inference_logp import get_dataset_inference_logp
from mllm.train.preprocess import find_best_resize

logger = logging.getLogger(__name__)



class GroundingSupervisedDataset(Dataset):
    """Dataset for supervised fine-tuning."""

    def __init__(
        self,
        raw_data,
        transform,
        tokenizer,
        slice_config,
        patch_size=14,
        query_nums=64,
        batch_vision=False,
        max_length=2048,
    ):
        super(GroundingSupervisedDataset, self).__init__()
        self.raw_data = raw_data
        self.tokenizer = tokenizer
        self.transform = transform
        self.slice_config = slice_config
        self.patch_size = patch_size
        self.query_nums=query_nums
        self.batch_vision = batch_vision
        self.max_length = max_length

    def __len__(self):
        return len(self.raw_data)

    def __getitem__(self, i) -> Dict[str, torch.Tensor]:
        ### ==> TODO: Visual Grounding数据处理流程
        # 基于 datasets.SupervisedDataset 的实现：
        # 1) 读取原始样本，加载图像
        # 2) 使用 preprocess 将图像与对话转换为模型输入张量
        # 3) 构造并返回训练所需字段

        raw = self.raw_data[i]

        # image path -> PIL.Image
        images_dict = {"<image>": Image.open(raw["image"]).convert("RGB")}
        conversations = raw["conversations"]
        # If target boxes present, fill '<boxes>' placeholders before preprocess
        target = raw.get('target', {})
        boxes_seq = None
        # Try to find boxes_seq from either human/assistant turns or top-level
        # but our unified json places boxes_seq in conversations entries sometimes
        # so just pass None here; fill_boxes_in_conversations will insert all boxes if boxes_seq is None
        conversations = fill_boxes_in_conversations(
            conversations=conversations,
            tokenizer=self.tokenizer,
            target=target,
            boxes_seq=boxes_seq,
        )

        preprocessed = preprocess(
            images_dict=images_dict,
            conversations=conversations,
            tokenizer=self.tokenizer,
            transform=self.transform,
            query_nums=self.query_nums,
            slice_config=self.slice_config,
            patch_size=self.patch_size,
            batch_vision=self.batch_vision,
            max_length=self.max_length,
        )

        attention_mask = preprocessed.get("attention_mask", None)
        if attention_mask is None:
            pad_id = 0 if self.tokenizer.pad_token_id is None else self.tokenizer.pad_token_id
            attention_mask = preprocessed["input_ids"].ne(pad_id)
            attention_mask = attention_mask.to(torch.bool)

        ret = dict(
            input_ids=preprocessed["input_ids"],
            position_ids=preprocessed["position_ids"],
            labels=preprocessed["target"],
            attention_mask=attention_mask,
            pixel_values=preprocessed.get("pixel_values", None),
            tgt_sizes=preprocessed.get("tgt_sizes", None),
            image_bound=preprocessed["image_bound"],
        )

        return ret
        ### <===
