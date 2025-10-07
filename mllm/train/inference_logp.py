import os
import json
import copy
import torch
import pickle
from copy import deepcopy

import io
import tqdm
import itertools
from PIL import Image

from model import MLLMModel
from transformers import AutoTokenizer
import torch.utils.data as torch_data
from torch.utils.data import Dataset

from utils.file_io import read_json, bytes_to_PIL_image, b64_to_PIL_image
from mllm.train.preprocess import data_collator, build_transform, preprocess


class InferenceSampler(torch.utils.data.sampler.Sampler):

    def __init__(self, size):
        self._size = int(size)
        assert size > 0
        self._rank = torch.distributed.get_rank()
        self._world_size = torch.distributed.get_world_size()
        self._local_indices = self._get_local_indices(size, self._world_size,
                                                      self._rank)

    @staticmethod
    def _get_local_indices(total_size, world_size, rank):
        shard_size = total_size // world_size
        left = total_size % world_size
        shard_sizes = [shard_size + int(r < left) for r in range(world_size)]

        begin = sum(shard_sizes[:rank])
        end = min(sum(shard_sizes[:rank + 1]), total_size)
        return range(begin, end)

    def __iter__(self):
        yield from self._local_indices

    def __len__(self):
        return len(self._local_indices)


class PreferenceInferenceDataset(Dataset):
    def __init__(self,
                 data, img_dir,
                 tokenizer = None,
                 transform=None,
                 slice_config=None,
                 batch_vision=True,
                 max_length=2048,
                 ):

        self.data = data
        self.img_dir = img_dir

        self.tokenizer = tokenizer
        self.transform = transform
        self.slice_config = slice_config
        self.batch_vision = batch_vision
        self.max_length = max_length

    def preprocess_input(self, image_list, msgs):
        model_inputs = preprocess(
            images_dict=image_list,
            conversations=msgs,
            tokenizer=self.tokenizer,
            transform=self.transform,
            slice_config=self.slice_config,
            batch_vision=self.batch_vision,
            max_length=self.max_length
        )

        model_inputs = dict(
            input_ids=model_inputs["input_ids"],
            position_ids=model_inputs["position_ids"],
            labels=model_inputs["target"],
            attention_mask=torch.ones_like(model_inputs["input_ids"], dtype=torch.bool),
            pixel_values=model_inputs.get("pixel_values", None),
            tgt_sizes=model_inputs.get("tgt_sizes", None),
            image_bound=model_inputs["image_bound"],
        )

        return model_inputs

    def prepare_inputs(self, index):
        try:
            sample = self.data[index]
        except:
            sample = self.data.iloc[index]

        question = {'role': 'user', 'content': f"<image>\n{sample['question']}"}
        chosen = {'role': 'assistant', 'content': sample['chosen']}
        rejected = {'role': 'assistant', 'content': sample['rejected']}

        if 'image' in sample.keys():
            images_dict = { "<image>": b64_to_PIL_image(sample['image'])}
        elif 'image_path' in sample.keys() and isinstance(sample['image_path'], str):
            images_dict = { "<image>" : Image.open(os.path.join(self.img_dir, sample['image_path'])).convert("RGB") }

        formated_sample = {
            'image': images_dict,
            "chosen": [question, chosen],
            "rejected": [question, rejected],
            "idx": sample['idx'],
        }

        return formated_sample

    def __getitem__(self, index):
        formated_sample = self.prepare_inputs(index)

        # return formated_sample

        sample = {
            "chosen": self.preprocess_input(formated_sample['image'], formated_sample['chosen']),
            "rejected": self.preprocess_input(formated_sample['image'], formated_sample['rejected'])
        }

        return sample

    def __len__(self):
        return len(self.data)


def get_batch_logps(logits: torch.FloatTensor, labels: torch.LongTensor, tokenizer, return_per_token_logp=False, return_all=False) -> torch.FloatTensor:
    """Compute the log probabilities of the given labels under the given logits.

    Args:
        logits: Logits of the model (unnormalized). Shape: (batch_size, sequence_length, vocab_size)
        labels: Labels for which to compute the log probabilities. Label tokens with a value of -100 are ignored. Shape: (batch_size, sequence_length)
    Returns:
        A tensor of shape (batch_size,) containing the average/sum log probabilities of the given labels under the given logits.
    """
    ### ===> TODO: 实现 logp 计算
    # per_token_logps: 每个位置的logp取值
    # log_prob: 完整回复的 logp 之和
    # average_log_prob: 完整回复中每个词 logp 的平均值
    ## 注意：
    ## 计算时注意logits与label对应关系是否正确，当前位置logits应该以后一个词为目标
    ## 只有输出部分应该被计算再内
    shifted_logits = logits[:, :-1, :].contiguous()
    all_logps = torch.log_softmax(shifted_logits, dim=-1)

    shift_labels = labels[:, 1:].contiguous()
    invalid_mask = shift_labels.eq(-100)
    valid_labels = shift_labels.clone()
    valid_labels[invalid_mask] = 0
    all_logps[invalid_mask] = 0 

    per_token_logps = torch.zeros_like(labels, dtype=logits.dtype, device=logits.device)
    per_token_logps[:, 1:] = torch.gather(all_logps, dim=-1, index=valid_labels.unsqueeze(-1)).squeeze(-1)
    log_prob = torch.sum(per_token_logps, dim=-1)
    average_log_prob = log_prob/torch.sum(~invalid_mask, dim=-1).clamp_min(1)
    ### <===

    assert per_token_logps.shape == labels.shape, f"per_token_logps.shape={per_token_logps.shape}, labels.shape={labels.shape}"

    if return_per_token_logp:
        return per_token_logps

    if return_all:
        return per_token_logps, log_prob, average_log_prob

    return log_prob, average_log_prob



def save_logp_pkl(data, cache_file, logps, overwrite_logps=False):
    out_data = []

    for index in range(len(logps)):
        try:
            line = data[index]
        except:
            line = data.iloc[index]
        logp_data = {}
        logp_data['logps']=logps[index]

        new_line = copy.deepcopy(line)

        if 'logps' in new_line.keys():
            assert overwrite_logps, 'Found existing logp data, pass overwrite_logps=True to force overwritting'
            new_line['logps'] = json.dumps(logp_data)

        else:
            assert (('question' in list(new_line.keys()))
                    and ('chosen' in list(new_line.keys()))
                    and ('rejected' in list(new_line.keys()))), \
                f'Undefined data structure, expecting [Q, Win, Rej] in keys, got {new_line.keys()}'
            new_line['logps'] = json.dumps(logp_data)

        out_data.append(new_line)

    torch.distributed.barrier()

    if torch.distributed.get_rank() == 0:
        with open(cache_file, 'wb') as f:
            pickle.dump(out_data, f)

class PreferenceModel:
    def __init__(self, model_path, max_length=2048) -> None:
        model_name_or_path = model_path
        model = MLLMModel.from_pretrained(
            model_name_or_path,
            torch_dtype=torch.bfloat16,
            # attn_implementation="flash_attention_2",
        )

        self.model = model
        tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, trust_remote_code=True)

        self.model = self.model.to(device='cuda')
        self.tokenizer = tokenizer

        self.model.eval()
        self.config = self.model.config

        self.max_length = max_length

        if hasattr(self.model.config, "slice_config"):
            self.model.config.slice_config.max_slice_nums = 2
            slice_config = self.model.config.slice_config.to_dict()
        else:
            self.model.config.max_slice_nums = 2
            slice_config = self.model.config.to_dict()
        self.slice_config = slice_config
        # print("self.slice_config:", self.slice_config)

    def inference_logp(self, sample, ans_key):
        assert len(sample[ans_key]) == 1, f'len(sample[ans_key]) = {len(sample[ans_key])}'
        model_inputs = sample[ans_key][0]

        for key in model_inputs:
            if isinstance(model_inputs[key], list):
                model_inputs[key] = [model_inputs[key][i].to(self.model.device) for i in range(len(model_inputs[key]))]
            else:
                model_inputs[key] = model_inputs[key].to(self.model.device)

        model_inputs = data_collator([model_inputs], max_length=self.max_length)

        with torch.inference_mode():
            output = self.model(
                model_inputs
            )

            per_token_logp, log_prob, average_log_prob = get_batch_logps(
                output.logits, model_inputs['labels'], self.tokenizer, return_all=True)

            assert per_token_logp.size(1) >= model_inputs['input_ids'].size(1) - 1
            per_token_logp = per_token_logp.tolist()
            log_prob = log_prob.tolist()
            average_log_prob = average_log_prob.tolist()

        return per_token_logp, log_prob, average_log_prob

def get_multimodal_sample_logps(model, dataloader):
    win_logp_list = []
    rej_logp_list = []

    win_avg_logp_list = []
    rej_avg_logp_list = []

    win_per_token_logp_list = []
    rej_per_token_logp_list = []

    with torch.inference_mode():
        idx=0
        for batch in tqdm.tqdm(dataloader):
            for key in ['chosen', 'rejected']:
                per_token_logp, log_prob, average_log_prob = model.inference_logp(
                    sample=batch,
                    ans_key=key
                )

                if key == 'chosen':
                    win_logp_list += log_prob
                    win_avg_logp_list += average_log_prob
                    win_per_token_logp_list += per_token_logp
                else:
                    rej_logp_list += log_prob
                    rej_avg_logp_list += average_log_prob
                    rej_per_token_logp_list += per_token_logp

            idx += 1

    return win_logp_list, win_avg_logp_list, win_per_token_logp_list, rej_logp_list, rej_avg_logp_list, rej_per_token_logp_list

def colloator_fn(data_list):
    data = {}
    for key in data_list[0]:
        data[key] = [x[key] for x in data_list]

    return data

def get_dataset_inference_logp(model_path, data_path, img_dir, cache_file, transform=None, slice_config=None, batch_vision=True, max_length=2048):
    model = PreferenceModel(model_path, max_length=max_length)
    org_data = read_json(data_path) if isinstance(data_path, str) else data_path
    dataset = PreferenceInferenceDataset(data=org_data, img_dir=img_dir,
                                         tokenizer=model.tokenizer,
                                         transform=transform if transform is not None else build_transform(),
                                         slice_config=slice_config if slice_config is not None else model.slice_config,
                                         batch_vision=batch_vision,
                                         max_length=max_length)

    dataloader = torch_data.DataLoader(dataset, batch_size=1, collate_fn=colloator_fn,
                                       num_workers=5, shuffle=False,
                                       sampler=InferenceSampler(len(dataset)))

    outputs = get_multimodal_sample_logps(model, dataloader) # win_logp_list, win_avg_logp_list, win_per_token_logp_list, rej_logp_list, rej_avg_logp_list, rej_per_token_logp_list

    world_size = torch.distributed.get_world_size()
    merged_outputs = [[None for _ in range(world_size)] for i in range(len(outputs))]
    for i in range(len(outputs)):
        torch.distributed.all_gather_object(merged_outputs[i], outputs[i])
        merged_outputs[i] = [_ for _ in itertools.chain.from_iterable(merged_outputs[i])]


    win_logp_list, win_avg_logp_list, win_per_token_logp_list, rej_logp_list, rej_avg_logp_list, rej_per_token_logp_list \
        = merged_outputs

    logps = list(zip(win_logp_list, win_avg_logp_list, win_per_token_logp_list, rej_logp_list, rej_avg_logp_list, rej_per_token_logp_list))

    data_with_logp = save_logp_pkl(dataset.data, cache_file, logps, overwrite_logps=True)

    torch.distributed.barrier()

    del model
