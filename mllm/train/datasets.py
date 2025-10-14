import os
import json
import pickle
import random
import torch
import logging
import pandas as pd
import os.path as op
import transformers
from torch.utils.data import Dataset

from PIL import Image
from typing import Dict
from utils.file_io import read_json, bytes_to_PIL_image, b64_to_PIL_image
from mllm.train.preprocess import preprocess, fill_boxes_in_conversations
from mllm.train.inference_logp import get_dataset_inference_logp

logger = logging.getLogger(__name__)

class SupervisedDataset(Dataset):
    """Dataset for supervised fine-tuning."""
    ### ===> TODO: 实现监督微调数据集，能够预处理数据为训练所需格式
    # 图片可以通过 images_dict = { "<image>" : Image.open(self.raw_data[i]["image"]).convert("RGB") } 获取
    # 调用时应该返回一下信息:
    # ret = dict(
    #     input_ids = ,
    #     position_ids = ,
    #     labels = ,
    #     attention_mask = ,
    #     pixel_values = ,
    #     tgt_sizes = ,
    #     image_bound = ,
    # )

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
        super(SupervisedDataset, self).__init__()
        self.raw_data = raw_data
        self.tokenizer = tokenizer
        self.transform = transform
        self.slice_config = slice_config
        self.patch_size = patch_size
        self.query_nums=query_nums
        self.batch_vision = batch_vision
        self.max_length = max_length
        self.length = len(raw_data)
    
    def __len__(self):
        return self.length

    def __getitem__(self, i):
        
        raw = self.raw_data[i]

        image_path = raw["image"]
        image_path = "data/sft" + image_path[image_path.find("/"):]

        images_dict = { "<image>" : Image.open(image_path).convert("RGB") } 
        conversations = raw["conversations"]
        
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


class ModelPreferenceDataset(Dataset):
    def __init__(self,
                 data_dir: str,
                 image_dir: str,
                 filepath: str,
                 reference_name=None,
                 reference_model=None,
                 transform=None,
                 slice_config=None,
                 batch_vision=True,
                 max_length=2048):
        super().__init__()

        filename = filepath.split('/')[-1]

        self.data_path = op.join(data_dir, filename.replace('.json', '') + f'.{reference_name}_logp.pkl')
        self.img_dir = image_dir

        if not op.exists(self.data_path):
            os.makedirs(data_dir, exist_ok=True)
            assert reference_model is not None, "`reference_model` is mandatory when logps do not exist."

            hf_data = read_json(filepath)

            get_dataset_inference_logp(model_path=reference_model,
                                       data_path=hf_data,
                                       img_dir=image_dir,
                                       cache_file=self.data_path,
                                       transform=transform,
                                       slice_config=slice_config,
                                       batch_vision=batch_vision,
                                       max_length=max_length,
                                       )

            torch.distributed.barrier()

            # self.data = pd.read_parquet(self.data_path)
            with open(self.data_path, 'rb') as f:
                self.data = pickle.load(f)
        else:
            if self.data_path.endswith('.parquet'):
                self.data = pd.read_parquet(self.data_path)
            else:
                with open(self.data_path, 'rb') as f:
                    self.data = pickle.load(f)

        self.line_numbers = list(range(len(self.data)))
        random.shuffle(self.line_numbers)
        # random.shuffle(self.data)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        try:
            sample = self.data.iloc[self.line_numbers[index]]
        except:
            sample = self.data[self.line_numbers[index]]

        question = {'role': 'user', 'content': f"<image>\n{sample['question']}"}
        chosen = {'role': 'assistant', 'content': sample['chosen']}
        rejected = {'role': 'assistant', 'content': sample['rejected']}

        if 'image' in sample.keys():
            images_dict = { "<image>": b64_to_PIL_image(sample['image'])}
        elif 'image_path' in sample.keys() and isinstance(sample['image_path'], str):
            images_dict = { "<image>" : Image.open(os.path.join(self.img_dir, sample['image_path'])).convert("RGB") }

        metainfo = sample['metainfos'] if 'metainfos' in sample else ''

        data_dict = {
            'image': images_dict,
            "question": question,
            "chosen": chosen,
            "rejected": rejected,
            "idx": sample['idx'],
            "metainfo": metainfo
        }
        logps=json.loads(sample['logps'])

        if type(logps) == type([]):
            (data_dict['ref_win_logp'], data_dict['ref_win_avg_logp'], data_dict['ref_win_per_token_logp'],
            data_dict['ref_rej_logp'], data_dict['ref_rej_avg_logp'], data_dict['ref_rej_per_token_logp']) = logps
        else:
            (data_dict['ref_win_logp'], data_dict['ref_win_avg_logp'], data_dict['ref_win_per_token_logp'],
            data_dict['ref_rej_logp'], data_dict['ref_rej_avg_logp'], data_dict['ref_rej_per_token_logp']) = logps['logps']

        return data_dict


class PreferenceTrainDataset(Dataset):
    def __init__(self,
                 tokenizer: transformers.PreTrainedTokenizer,
                 data_dir: str,
                 filepath: str,
                 ref_name: str,
                 multimodal_cfg: dict,
                 reference_model = None):
        super(PreferenceTrainDataset, self).__init__()

        self.tokenizer = tokenizer
        self.list_data_dict = ModelPreferenceDataset(
            data_dir=data_dir,
            image_dir=multimodal_cfg['image_folder'],
            filepath=filepath,
            reference_name=ref_name,
            reference_model=reference_model,
            transform=multimodal_cfg['image_transform'],
            slice_config=multimodal_cfg['slice_config'],
            batch_vision=multimodal_cfg['batch_vision'],
            max_length=multimodal_cfg['max_length'],
        )
        self.multimodal_cfg = multimodal_cfg

    def prepare_preference_data_dict(self, image, convs, source, cls):
        multimodal_cfg = self.multimodal_cfg
        ret = preprocess(
            image,
            convs,
            self.tokenizer,
            multimodal_cfg['image_transform'],
            query_nums=multimodal_cfg['image_token_len'],
            slice_config=multimodal_cfg['slice_config'],
            patch_size=multimodal_cfg['patch_size'],
            batch_vision=multimodal_cfg['batch_vision'],
            max_length=multimodal_cfg['max_length']
        )
        ret = dict(
            input_ids=ret["input_ids"],
            position_ids=ret["position_ids"],
            labels=ret["target"],
            attention_mask=torch.ones_like(ret["input_ids"], dtype=torch.bool),
            pixel_values=ret.get("pixel_values", None),
            tgt_sizes=ret.get("tgt_sizes", None),
            image_bound=ret["image_bound"],
        )

        if cls == 'rej':
            ret['ref_rej_logp'] = source['ref_rej_logp']
            ret['ref_rej_avg_logp'] = source['ref_rej_avg_logp']
            ret['ref_rej_per_token_logp'] = source['ref_rej_per_token_logp']
        elif cls == 'win':
            ret['ref_win_logp'] = source['ref_win_logp']
            ret['ref_win_avg_logp'] = source['ref_win_avg_logp']
            ret['ref_win_per_token_logp'] = source['ref_win_per_token_logp']

        # print(f"type ret['ref_{cls}_logp']:", type(ret[f'ref_{cls}_logp']), ret[f'ref_{cls}_logp'])
        # print("dtype ret['ref_rej_logp']:", ret['ref_rej_logp'].dtype)

        return ret

    def __len__(self):
        return len(self.list_data_dict)

    def __getitem__(self, i):
        source: dict = self.list_data_dict[i]
        rej_data_dict = self.prepare_preference_data_dict(
            image = source['image'],
            convs=[source['question'], source['rejected']],
            source=source,
            cls='rej'
        )
        win_data_dict = self.prepare_preference_data_dict(
            image = source['image'],
            convs=[source['question'], source['chosen']],
            source=source,
            cls='win'
        )
        # print("out image:", len(rej_data_dict['image']))
        # print("out image bound:", rej_data_dict['image_bounds'].shape)
        return rej_data_dict, win_data_dict