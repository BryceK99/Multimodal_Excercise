import math
from typing import List, Optional
import json
import torch
import torchvision

from threading import Thread
from copy import deepcopy
from PIL import Image
from transformers import AutoProcessor, TextIteratorStreamer

from .configuration import ModelConfig
from .modeling_navit_siglip import SiglipVisionTransformer
from .resampler import Resampler

from .image_processing import ModelImageProcessor
from .processing import ModelProcessor
from .llm.llm_architecture import LLMPreTrainedModel, LLMForCausalLM


class MLLMPreTrainedModel(LLMPreTrainedModel):
    config_class = ModelConfig


class MLLMModel(MLLMPreTrainedModel):

    def __init__(self, config):
        super().__init__(config)
        self.llm = LLMForCausalLM(config)
        self.vpm = self.init_vision_module()
        self.vision_dim = self.vpm.embed_dim
        self.embed_dim = self.llm.config.hidden_size
        self.resampler = self.init_resampler(self.embed_dim, self.vision_dim)
        self.processor = None

        self.terminators = ['<|im_end|>', '<|endoftext|>']

    def init_vision_module(self):
        # same as HuggingFaceM4/siglip-so400m-14-980-flash-attn2-navit add tgt_sizes
        if self.config._attn_implementation == 'flash_attention_2':
            self.config.vision_config._attn_implementation = 'flash_attention_2'
        else:
            # not suport sdpa
            self.config.vision_config._attn_implementation = 'eager'
        model = SiglipVisionTransformer(self.config.vision_config)
        if self.config.drop_vision_last_layer:
            model.encoder.layers = model.encoder.layers[:-1]

        setattr(model, 'embed_dim', model.embeddings.embed_dim)
        setattr(model, 'patch_size', model.embeddings.patch_size)

        return model

    def init_resampler(self, embed_dim, vision_dim):
        return Resampler(num_queries=self.config.query_num,
                         embed_dim=embed_dim,
                         num_heads=embed_dim // 128,
                         kv_dim=vision_dim,
                         adaptive=True)

    def get_input_embeddings(self):
        return self.llm.get_input_embeddings()

    def set_input_embeddings(self, value):
        self.llm.embed_tokens = value

    def get_output_embeddings(self):
        return self.llm.lm_head

    def set_output_embeddings(self, new_embeddings):
        self.llm.lm_head = new_embeddings

    def set_decoder(self, decoder):
        self.llm = decoder

    def get_decoder(self):
        return self.llm

    def get_vllm_embedding(self, data):
        vision_hidden_states = self.get_vision_hidden_states(data)

        if hasattr(self.llm.config, 'scale_emb'):
            vllm_embedding = self.llm.model.embed_tokens(
                data['input_ids']) * self.llm.config.scale_emb
        else:
            vllm_embedding = self.llm.model.embed_tokens(data['input_ids'])

        vision_hidden_states = [
            i.type(vllm_embedding.dtype) if isinstance(i, torch.Tensor) else i
            for i in vision_hidden_states
        ]

        bs = len(data['input_ids'])
        ### ===> 合并 vision_hidden_states 与 vllm_embedding，
        # 其中，vision_hidden_states 为视觉编码，当前 vllm_embedding 仅为语言模型编码
        embed_dtype = vllm_embedding.dtype
        embed_device = vllm_embedding.device
        vllm_embedding_list = []
        for i in range(bs):
            new_embedding_list = []
            last_end = 0
            for id, (start, end) in enumerate(data['image_bound'][i]):
                # print(id, start, end, vision_hidden_states[i][id].shape, vllm_embedding[i].shape)
                new_embedding_list.append(vllm_embedding[i, last_end:start])
                placeholder_len = end - start + 1
                if id < len(vision_hidden_states[i]) and isinstance(
                        vision_hidden_states[i][id], torch.Tensor):
                    vision_feat = vision_hidden_states[i][id]
                    vision_feat = vision_feat.to(dtype=embed_dtype,
                                                 device=embed_device)
                    vision_len = vision_feat.shape[0]
                    if vision_len < placeholder_len:
                        pad = vision_feat.new_zeros((placeholder_len - vision_len,
                                                     vision_feat.shape[1]))
                        vision_feat = torch.cat((vision_feat, pad))
                    elif vision_len > placeholder_len:
                        vision_feat = vision_feat[:placeholder_len]
                    new_embedding_list.append(vision_feat)
                else:
                    # 当不存在有效视觉特征时，回退到原始 token 的嵌入
                    new_embedding_list.append(vllm_embedding[i, start:end + 1])
                # 无论是否替换为视觉特征，都需要推进游标
                last_end = end + 1
            new_embedding_list.append(vllm_embedding[i, last_end:])
            new_embedding = torch.cat(new_embedding_list, dim=0)
            vllm_embedding_list.append(new_embedding)
        vllm_embedding = torch.stack(vllm_embedding_list, dim=0)
        ### <===

        return vllm_embedding, vision_hidden_states

    def get_vision_hidden_states(self, data):
        if 'vision_hidden_states' not in data:
            dtype = self.llm.model.embed_tokens.weight.dtype
            device = self.llm.model.embed_tokens.weight.device
            tgt_sizes = data['tgt_sizes']
            pixel_values_list = data['pixel_values']
            vision_hidden_states = []
            all_pixel_values = []
            img_cnt = []
            for pixel_values in pixel_values_list:
                img_cnt.append(len(pixel_values))
                all_pixel_values.extend(
                    [i.flatten(end_dim=1).permute(1, 0) for i in pixel_values])

            # exist image
            if all_pixel_values:
                tgt_sizes = [
                    tgt_size for tgt_size in tgt_sizes
                    if isinstance(tgt_size, torch.Tensor)
                ]
                tgt_sizes = torch.vstack(tgt_sizes).type(torch.int32)

                max_patches = torch.max(tgt_sizes[:, 0] * tgt_sizes[:, 1])

                all_pixel_values = torch.nn.utils.rnn.pad_sequence(
                    all_pixel_values, batch_first=True, padding_value=0.0)
                B, L, _ = all_pixel_values.shape
                all_pixel_values = all_pixel_values.permute(0, 2, 1).reshape(
                    B, 3, -1, L)

                patch_attn_mask = torch.zeros((B, 1, max_patches),
                                              dtype=torch.bool,
                                              device=device)
                for i in range(B):
                    patch_attn_mask[i, 0, :tgt_sizes[i][0] *
                                    tgt_sizes[i][1]] = True

                vision_batch_size = self.config.vision_batch_size
                all_pixel_values = all_pixel_values.type(dtype)
                if B > vision_batch_size:
                    hs = []
                    for i in range(0, B, vision_batch_size):
                        start_idx = i
                        end_idx = i + vision_batch_size
                        tmp_hs = self.vpm(
                            all_pixel_values[start_idx:end_idx],
                            patch_attention_mask=patch_attn_mask[
                                start_idx:end_idx],
                            tgt_sizes=tgt_sizes[start_idx:end_idx]
                        ).last_hidden_state
                        hs.append(tmp_hs)
                    vision_embedding = torch.cat(hs, dim=0)
                else:
                    vision_embedding = self.vpm(
                        all_pixel_values,
                        patch_attention_mask=patch_attn_mask,
                        tgt_sizes=tgt_sizes).last_hidden_state
                vision_embedding = self.resampler(vision_embedding, tgt_sizes)

                start = 0
                for pixel_values in pixel_values_list:
                    img_cnt = len(pixel_values)
                    if img_cnt > 0:
                        vision_hidden_states.append(
                            vision_embedding[start:start + img_cnt])
                        start += img_cnt
                    else:
                        vision_hidden_states.append([])
            else:  # no image
                if self.training:
                    dummy_image = torch.zeros((1, 3, 224, 224),
                                              device=device,
                                              dtype=dtype)
                    tgt_sizes = torch.Tensor([[
                        (224 // self.config.patch_size),
                        math.ceil(224 / self.config.patch_size)
                    ]]).type(torch.int32)
                    dummy_feature = self.resampler(
                        self.vpm(dummy_image).last_hidden_state, tgt_sizes)
                else:
                    dummy_feature = []
                for _ in range(len(pixel_values_list)):
                    vision_hidden_states.append(dummy_feature)

        else:
            vision_hidden_states = data['vision_hidden_states']

        return vision_hidden_states

    def forward(self, data, **kwargs):
        vllm_embedding, vision_hidden_states = self.get_vllm_embedding(data)
        position_ids = data["position_ids"]
        if position_ids.dtype != torch.int64:
            position_ids = position_ids.long()

        return self.llm(input_ids=None,
                        position_ids=position_ids,
                        inputs_embeds=vllm_embedding,
                        **kwargs)

    def _decode(self,
                inputs_embeds,
                tokenizer,
                attention_mask,
                decode_text=False,
                **kwargs):
        terminators = [
            tokenizer.convert_tokens_to_ids(i) for i in self.terminators
        ]
        ### ===> TODO: 实现语言模型 generate
        output = self.llm.generate(inputs_embeds=inputs_embeds,
                                   pad_token_id=0,
                                   eos_token_id=terminators,
                                   attention_mask=attention_mask,
                                   **kwargs)

        ### <===
        if decode_text:
            prompt_lengths = None
            if attention_mask is not None:
                if isinstance(attention_mask, torch.Tensor):
                    prompt_lengths = attention_mask.sum(dim=-1).tolist()
                else:
                    prompt_lengths = [int(mask.sum()) for mask in attention_mask]
            return self._decode_text(output, tokenizer, prompt_lengths=prompt_lengths)
        return output

    def _decode_stream(self, inputs_embeds, tokenizer, **kwargs):
        terminators = [
            tokenizer.convert_tokens_to_ids(i) for i in self.terminators
        ]
        streamer = TextIteratorStreamer(tokenizer=tokenizer)
        generation_kwargs = {
            'inputs_embeds': inputs_embeds,
            'pad_token_id': 0,
            'eos_token_id': terminators,
            'streamer': streamer
        }
        generation_kwargs.update(kwargs)

        thread = Thread(target=self.llm.generate, kwargs=generation_kwargs)
        thread.start()

        return streamer

    def _decode_text(self, result_ids, tokenizer, prompt_lengths=None):
        terminators = [
            tokenizer.convert_tokens_to_ids(i) for i in self.terminators
        ]
        ### TODO: ===> 编写输出解码过程
        # 其中应该去除tokenizer.bos_id（句子起始特殊符号），以及terminators中的符号
        # 仅解码新生成部分，避免包含提示词
        if isinstance(result_ids, torch.Tensor):
            sequences = result_ids
        else:
            try:
                sequences = torch.as_tensor(result_ids)
            except Exception:
                # 某些情况下 transformers 返回 list[list[int]]
                sequences = torch.tensor(result_ids)

        if sequences.dim() == 1:
            sequences = sequences.unsqueeze(0)

        texts = []
        bsz = sequences.size(0)
        for b in range(bsz):
            seq = sequences[b]
            start = 0
            if prompt_lengths is not None and b < len(prompt_lengths):
                start = int(prompt_lengths[b])
                # 如果返回的序列仅包含新生成 token（长度小于等于提示长度），则不剪切
                if seq.numel() <= start:
                    start = 0
            gen_ids = seq[start:]
            gen_ids = gen_ids[gen_ids != 0]
            gen_ids = [
                int(tid) for tid in gen_ids.tolist()
                if tid != tokenizer.bos_token_id and tid not in terminators
            ]
            texts.append(tokenizer.decode(gen_ids, skip_special_tokens=True))

        # 如果为空，尝试回退一次
        if len(texts) == 1 and (texts[0] is None or texts[0].strip() == ""):
            try:
                # 回退路径：直接整体解码（去除特殊符号）
                ids = sequences[0]
                ids = ids[ids != 0]
                ids = [int(tid) for tid in ids.tolist() if tid != tokenizer.bos_token_id and tid not in terminators]
                fallback_text = tokenizer.decode(ids, skip_special_tokens=True)
                texts[0] = fallback_text
            except Exception:
                pass
        # 仍为空则尽量截取到第一个终止符之前
        if len(texts) == 1 and (texts[0] is None or texts[0].strip() == ""):
            try:
                ids = sequences[0].tolist()
                cut = len(ids)
                for idx, tid in enumerate(ids):
                    if tid in terminators:
                        cut = idx
                        break
                ids = [int(tid) for tid in ids[:cut] if tid != 0 and tid != tokenizer.bos_token_id]
                texts[0] = tokenizer.decode(ids, skip_special_tokens=True)
            except Exception:
                pass

        result_text = texts[0] if len(texts) == 1 else texts
        ### <===
        return result_text

    def generate(self,
                 input_ids=None,
                 pixel_values=None,
                 tgt_sizes=None,
                 image_bound=None,
                 attention_mask=None,
                 tokenizer=None,
                 vision_hidden_states=None,
                 return_vision_hidden_states=False,
                 stream=False,
                 decode_text=False,
                 **kwargs):
        assert input_ids is not None
        assert len(input_ids) == len(pixel_values)

        model_inputs = {
            "input_ids": input_ids,
            "image_bound": image_bound,
        }

        if vision_hidden_states is None:
            model_inputs["pixel_values"] = pixel_values
            model_inputs['tgt_sizes'] = tgt_sizes
        else:
            model_inputs["vision_hidden_states"] = vision_hidden_states

        ### ===> TODO: 实现多模态大模型的 generation，注意不要计算模型参数的梯度。
        # 1. 获取模型视觉信号
        # 2. 实现 self._decode()，返回解码后的文本
        with torch.inference_mode():
            vision_hidden_states = self.get_vision_hidden_states(
                model_inputs)
            vllm_embedding, vision_hidden_states = self.get_vllm_embedding(
                model_inputs)
            if stream:
                result = self._decode_stream(vllm_embedding,
                                             tokenizer,
                                             max_new_tokens=kwargs.get(
                                                 'max_new_tokens', 50),
                                             attention_mask=attention_mask,
                                             **kwargs)
            else:
                result = self._decode(vllm_embedding,
                                      tokenizer,
                                      decode_text=decode_text,
                                      attention_mask=attention_mask,
                                      **kwargs)
        ### <===
        if return_vision_hidden_states:
            return result, vision_hidden_states

        return result
