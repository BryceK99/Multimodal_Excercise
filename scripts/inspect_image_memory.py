import json
import os
from PIL import Image
import math

from mllm.train.preprocess import preprocess_image, build_transform


def inspect_sample(json_path, sample_index=0, slice_config=None, query_nums=64):
    if not os.path.exists(json_path):
        print(f"File not found: {json_path}")
        return None
    j = json.load(open(json_path, 'r'))
    if len(j) == 0:
        print(f"Empty json: {json_path}")
        return None
    raw = j[sample_index]
    img_path = raw.get('image')
    if img_path is None:
        print(f"No 'image' field in sample {sample_index} of {json_path}")
        return None
    if not os.path.exists(img_path):
        # try relative to data dir
        candidate = os.path.join(os.path.dirname(json_path), os.path.basename(img_path))
        if os.path.exists(candidate):
            img_path = candidate
        else:
            print(f"Image not found: {img_path}")
            return None

    img = Image.open(img_path).convert('RGB')
    images_dict = {"<image>": img}

    class DummyTok:
        im_start = '<|im_start|>'
        unk_token = '<|unk|>'
        im_end = '<|im_end|>'
        im_id_start = '<|im_id_start|>'
        im_id_end = '<|im_id_end|>'
        slice_start = '<|slice_start|>'
        slice_end = '<|slice_end|>'

    tok = DummyTok()
    transform = build_transform()
    placeholder, images, image_placeholder = preprocess_image(images_dict, tok, transform, query_nums, slice_config, tok.im_start + tok.unk_token * query_nums + tok.im_end, use_image_id=True)

    # images is a list of tensors (transformed). Print shapes and estimate memory
    total_elems = 0
    print(f"Sample from {json_path}, image file: {img_path}")
    for i, t in enumerate(images):
        if hasattr(t, 'shape'):
            # t is a tensor or PIL? preprocess_image returns transform applied tensors
            shape = t.shape
            elems = 1
            for d in shape:
                elems *= d
            total_elems += elems
            print(f"  image[{i}] shape: {shape}, elems: {elems}")
        else:
            print(f"  image[{i}] has no shape, type={type(t)}")

    # assume float32 (4 bytes)
    bytes_total = total_elems * 4
    mb = bytes_total / (1024**2)
    print(f"  total elems: {total_elems}, approx memory (MB, float32): {mb:.2f}\n")
    return dict(total_elems=total_elems, mb=mb, images_count=len(images))


if __name__ == '__main__':
    # set slice_config similar to model defaults
    slice_config = dict(max_slice_nums=9, scale_resolution=448, patch_size=14)

    grounding_json = 'data/vg/all_vg_dataset.json'
    sft_json = 'data/sft/train.json'

    print('Inspect grounding sample:')
    g = inspect_sample(grounding_json, sample_index=0, slice_config=slice_config, query_nums=64)

    print('Inspect sft sample:')
    s = inspect_sample(sft_json, sample_index=0, slice_config=slice_config, query_nums=64)

    print('Summary:')
    print('  grounding:', g)
    print('  sft     :', s)
