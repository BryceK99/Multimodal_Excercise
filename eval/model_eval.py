import io
import os
import copy
import json
import base64
import argparse
from PIL import Image
from tqdm import tqdm

import torch
from transformers import AutoTokenizer

from mllm.model import MLLMModel
from mllm.model.processing import ModelProcessor
from mllm.model.image_processing import ModelImageProcessor
from utils.file_io import read_jsonlines, read_json


class MLLMEvalModel(MLLMModel):

    def __init__(self, config):
        super().__init__(config)

    def chat(self,
             image,
             msgs,
             tokenizer,
             processor=None,
             vision_hidden_states=None,
             max_new_tokens=4096*2,
             min_new_tokens=0,
             sampling=True,
             max_inp_length=8192,
             system_prompt='',
             stream=False,
             max_slice_nums=None,
             use_image_id=None,
             **kwargs):
        if isinstance(msgs[0], list):
            batched = True
        else:
            batched = False
        msgs_list = msgs
        images_list = image

        if batched is False:
            images_list, msgs_list = [images_list], [msgs_list]
        else:
            assert images_list is None, "Please integrate image to msgs when using batch inference."
            images_list = [None] * len(msgs_list)
        assert len(images_list) == len(
            msgs_list
        ), "The batch dim of images_list and msgs_list should be the same."

        assert self.config.query_num == processor.image_processor.image_feature_size, "These two values should be the same. Check `config.json` and `preprocessor_config.json`."
        assert self.config.patch_size == processor.image_processor.patch_size, "These two values should be the same. Check `config.json` and `preprocessor_config.json`."
        assert self.config.use_image_id == processor.image_processor.use_image_id, "These two values should be the same. Check `config.json` and `preprocessor_config.json`."
        assert self.config.slice_config.max_slice_nums == processor.image_processor.max_slice_nums, "These two values should be the same. Check `config.json` and `preprocessor_config.json`."
        assert self.config.slice_mode == processor.image_processor.slice_mode, "These two values should be the same. Check `config.json` and `preprocessor_config.json`."

        assert sampling or not stream, "if use stream mode, make sure sampling=True"

        prompts_lists, input_images_lists = self.prepare_chat_inputs(
            tokenizer, system_prompt, msgs_list, images_list)

        inputs = processor(prompts_lists,
                           input_images_lists,
                           max_slice_nums=max_slice_nums,
                           use_image_id=use_image_id,
                           return_tensors="pt",
                           max_length=max_inp_length).to(self.device)

        if sampling:
            generation_config = {
                "top_p": 0.8,
                "top_k": 100,
                "temperature": 0.7,
                "do_sample": True,
                "repetition_penalty": 1.05
            }
        else:
            generation_config = {
                "num_beams": 3,
                "repetition_penalty": 1.2,
            }

        if min_new_tokens > 0:
            generation_config['min_new_tokens'] = min_new_tokens

        generation_config.update(
            (k, kwargs[k]) for k in generation_config.keys() & kwargs.keys())

        inputs.pop("image_sizes")
        with torch.inference_mode():
            res = self.generate(**inputs,
                                tokenizer=tokenizer,
                                max_new_tokens=max_new_tokens,
                                vision_hidden_states=vision_hidden_states,
                                stream=stream,
                                decode_text=True,
                                **generation_config)

        if stream:

            def stream_gen():
                for text in res:
                    for term in self.terminators:
                        text = text.replace(term, '')
                    yield text

            return stream_gen()

        else:
            if batched:
                return res
            # 非 batch：res 可能是字符串或长度为 1 的列表
            if isinstance(res, str):
                return res
            return res[0]

    def prepare_chat_inputs(self, tokenizer, system_prompt, msgs_list,
                            images_list):
        ### ===> TODO:
        # 将输入文本转换为预处理函数所需的格式
        # Rule:
        # 1. 输入图片的位置应该替换为 (<image>./</image>) 字符串
        # 2. 使用 tokenizer 将输入文本转换为模型所需的输入格式，并进行分词（tokenize）
        # 提示：使用 tokenizer.apply_chat_template 进行输入文本格式转换

        prompts_lists = []
        input_images_lists = []

        for msgs, image in zip(msgs_list, images_list):
            conversation = copy.deepcopy(msgs)

            # 确保 image 是一个列表，以便统一处理
            if image and not isinstance(image, list):
                image = [image]

            if image:
                # 根据图片数量插入同样数量的标签（去掉换行，避免对提示分词的微扰）
                image_tags = "".join(["<image>./</image>"] * len(image))
                if len(conversation) > 0 and "content" in conversation[0]:
                    conversation[0]["content"] = image_tags + conversation[0]["content"]

            # 不在此处重复添加特殊符号，避免 tokenizer 发生不必要变化
            prompt = tokenizer.apply_chat_template(conversation,
                                                   tokenize=False,
                                                   add_generation_prompt=True,
                                                   system_prompt=system_prompt)
            prompts_lists.append(prompt)
            # 直接使用 image 列表，不再进行二次包装
            input_images_lists.append(image if image is not None else [])

        ### <===

        return prompts_lists, input_images_lists


def eval_model(args):
    model = MLLMEvalModel.from_pretrained(
        args.model_name_or_path,
        torch_dtype=torch.bfloat16,
    )
    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path,
                                              trust_remote_code=True)

    img_processor_config = read_json(
        'mllm/model/mllm_preprocessor_config.json')
    image_processor = ModelImageProcessor(**img_processor_config)
    processor = ModelProcessor(image_processor, tokenizer)

    model.eval().cuda()

    input_data = read_jsonlines(args.question_file)

    ans_file = open(args.answers_file, 'w')

    # 兼容旧参数：未显式指定 --decoding 时，沿用 --sampling 的语义
    decoding_mode = args.decoding if args.decoding is not None else ("sampling" if args.sampling else "beam")
    if decoding_mode == "sampling":
        use_sampling = True
        gen_kwargs = dict(temperature=args.temperature, top_p=args.top_p, top_k=args.top_k)
    elif decoding_mode == "greedy":
        use_sampling = False
        gen_kwargs = dict(num_beams=1)
    else:  # beam
        use_sampling = False
        gen_kwargs = dict(num_beams=args.num_beams)

    with torch.inference_mode():
        i = 0
        for item in tqdm(input_data):
            image = item['image']
            msgs = [{"role": "user", "content": item['question']}]

            if len(image) > 1000:
                image = Image.open(io.BytesIO(
                    base64.b64decode(image))).convert('RGB')
            else:
                image = Image.open(image).convert('RGB')

            # 传入长度与重复惩罚
            chat_kwargs = dict(max_new_tokens=args.max_new_tokens,
                               repetition_penalty=args.repetition_penalty)
            chat_kwargs.update(gen_kwargs)

            answer = model.chat(image=image,
                                msgs=msgs,
                                tokenizer=tokenizer,
                                sampling=use_sampling,
                                processor=processor,
                                **chat_kwargs)

            answer_dict = {
                "idx": i,
                "question": msgs[0]['content'],
                "answer": answer,
                "model": args.model_name_or_path,
                "metainfos": {
                    key: value
                    for key, value in item.items()
                    if key not in ['image', 'question']
                }
            }

            if 'image_id' in item.keys():
                answer_dict['image_id'] = item['image_id']

            ans_file.write(json.dumps(answer_dict) + '\n')
            ans_file.flush()

            i += 1
            if args.max_samples is not None and i >= args.max_samples:
                break


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-name-or-path", type=str)
    parser.add_argument("--question-file", type=str)
    parser.add_argument("--answers-file", type=str)
    parser.add_argument("--sampling", action='store_true')
    parser.add_argument("--max-samples", type=int, default=None)
    # 新增：解码方式与超参数（优先于 --sampling）
    parser.add_argument("--decoding",
                        choices=["greedy", "beam", "sampling"],
                        default=None,
                        help="选择解码方式；不传则与 --sampling 行为兼容（默认 beam）")
    parser.add_argument("--num-beams", type=int, default=3, dest="num_beams",
                        help="beam search 的束宽，--decoding beam 时生效")
    parser.add_argument("--temperature", type=float, default=0.7,
                        help="采样温度，--decoding sampling 时生效")
    parser.add_argument("--top-p", type=float, default=0.8, dest="top_p",
                        help="nucleus 采样阈值，--decoding sampling 时生效")
    parser.add_argument("--top-k", type=int, default=100, dest="top_k",
                        help="top-k 采样阈值，--decoding sampling 时生效")
    parser.add_argument("--max-new-tokens", type=int, default=128, dest="max_new_tokens",
                        help="最大新生成token数，控制输出长度")
    parser.add_argument("--repetition-penalty", type=float, default=1.05, dest="repetition_penalty",
                        help="重复惩罚，>1会降低重复，过大可能损伤Recall")
    args = parser.parse_args()

    eval_model(args)
