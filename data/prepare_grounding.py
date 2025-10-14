import os
import json
import numpy as np
from numpy import random as nprdm
import random
import tqdm
import multiprocessing
import argparse
import threading
from os.path import join as pjoin
from PIL import Image
from tqdm import tqdm

import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from utils.file_io import read_jsonlines

random.seed(71)
nprdm.seed(71)


IMAGE_PLACEHOLDER = '<image>'
BOXES_PLACEHOLDER = '<boxes>'
EXPR_PLACEHOLDER = '<expr>'
OBJS_PLACEHOLDER = '<objs>'
QUESTION_PLACEHOLDER = '<question>'
POINTS_PLACEHOLDER = '<points>'
PHRASE_ST_PLACEHOLDER = '<ph_st>'
PHRASE_ED_PLACEHOLDER = '<ph_ed>'

# A central place to register jsonl and image roots for easy maintenance
# You can modify these paths according to your local layout.
IMAGE_ROOTS = {
    'VG_100K': 'data/vg/VG_100K',
    'VG_100K_2': 'data/vg/VG_100K_2',
    'COCO_TRAIN2014': 'data/vg/coco/train2014',
    'FLICKR30K': 'data/vg/flickr30k-images',
}

TEMPLATE_FILES = {
    'REC': 'data/vg/templete/REC.json',
    'GC': 'data/vg/templete/GC.json',
    'REG': 'data/vg/templete/REG.json',
    'FLICKR': 'data/vg/templete/flickr30k.json',
}

DATA_SOURCES = {
    'REC_VG': {
        'jsonls': [
            'data/vg/REC_ref3_train.jsonl',
        ],
        'image_roots': {
            'images/VG_100K': IMAGE_ROOTS['VG_100K'],
            'images2/VG_100K_2': IMAGE_ROOTS['VG_100K_2'],
            'train2014': IMAGE_ROOTS['COCO_TRAIN2014'],
        }
    },
    'GC_VG': {
        'jsonls': [
            'data/vg/GC_genome196_train.jsonl',
        ],
        'image_roots': {
            'images/VG_100K': IMAGE_ROOTS['VG_100K'],
            'images2/VG_100K_2': IMAGE_ROOTS['VG_100K_2'],
        }
    },
    'FLICKR30K': {
        'jsonls': [
            'data/vg/CWB_flickr30k_train.jsonl',
        ],
        'image_roots': {
            'flickr30k': IMAGE_ROOTS['FLICKR30K'],
        }
    },
    'GPT4GEN': {
        'jsonls': [
            'data/vg/GPT4GEN_BoxCoT_train.jsonl',
            'data/vg/GPT4GEN_RD_BoxCoT_train.jsonl',
        ],
        'image_roots': {
            'vg': IMAGE_ROOTS['FLICKR30K'],
            'coco': IMAGE_ROOTS['COCO_TRAIN2014'],
        }
    },
}


def _get_image_size_cached(img_path: str, _cache: dict) -> tuple:
    """Return (W, H) for an image path, using an in-memory cache.
    If image can't be opened, return (None, None).
    """
    if not isinstance(img_path, str) or not img_path:
        return (None, None)
    if img_path in _cache:
        return _cache[img_path]
    try:
        with Image.open(img_path) as im:
            w, h = im.size
        _cache[img_path] = (w, h)
        return (w, h)
    except Exception:
        _cache[img_path] = (None, None)
        return (None, None)


def _to_xyxy_rel(box, w: int, h: int):
    """Convert a single bbox to normalized xyxy in [0,1].

    Accepts either [x1,y1,x2,y2] or [x,y,w,h] (absolute or normalized).
    Heuristics:
    - If values look already normalized (all in [0,1]), still convert to xyxy normalized.
    - If third value > first and fourth > second and also <= image size (or <=1), treat as xyxy.
    - Else treat as xywh.
    Safely clamps to [0,1]. Returns None if inputs invalid.
    """
    if not isinstance(box, (list, tuple)) or len(box) != 4:
        return None
    try:
        x0, y0, x2_or_w, y2_or_h = [float(v) for v in box]
    except Exception:
        return None

    # If image size unknown, attempt to infer normalized vs absolute by value ranges
    w = float(w) if w not in (None, 0) else None
    h = float(h) if h not in (None, 0) else None

    def clamp01(v):
        return max(0.0, min(1.0, v))

    # If already normalized numbers in [0,1] range (loosely), treat accordingly
    is_all_unit = all(0.0 <= v <= 1.0 for v in [x0, y0, x2_or_w, y2_or_h])

    if is_all_unit:
        # If it looks like xyxy normalized (x2> x1 and y2>y1), keep; else treat as xywh normalized
        if x2_or_w > x0 and y2_or_h > y0:
            x1r, y1r, x2r, y2r = x0, y0, x2_or_w, y2_or_h
        else:
            x1r, y1r = x0, y0
            x2r, y2r = x0 + x2_or_w, y0 + y2_or_h
        return [clamp01(x1r), clamp01(y1r), clamp01(x2r), clamp01(y2r)]

    # If not unit, need image size to normalize
    if w is None or h is None:
        # Can't normalize without size; best-effort: assume typical xyxy and try to scale by max
        # but to avoid incorrect scaling, return None to skip normalization
        return None

    # Heuristic decide format using absolute values vs image size
    # Prefer xyxy when x2>x1 and y2>y1 and x2<=w+eps, y2<=h+eps
    eps = 1e-3
    looks_xyxy = (x2_or_w > x0 and y2_or_h > y0 and
                  x2_or_w <= w + eps and y2_or_h <= h + eps)

    if looks_xyxy:
        x1, y1, x2, y2 = x0, y0, x2_or_w, y2_or_h
    else:
        # Treat as xywh; also ensure w,h non-negative
        ww = max(0.0, x2_or_w)
        hh = max(0.0, y2_or_h)
        x1, y1, x2, y2 = x0, y0, x0 + ww, y0 + hh

    # Normalize
    x1r = x1 / w
    y1r = y1 / h
    x2r = x2 / w
    y2r = y2 / h

    # Clamp and ensure non-decreasing
    x1r, y1r, x2r, y2r = map(clamp01, [x1r, y1r, x2r, y2r])
    if x2r < x1r:
        x1r, x2r = x2r, x1r
    if y2r < y1r:
        y1r, y2r = y2r, y1r
    return [x1r, y1r, x2r, y2r]


def _normalize_boxes_for_image(img_path: str, boxes):
    """Normalize one or many boxes for an image path to xyxy relative [0,1].
    - boxes can be a list of 4 numbers or list of such lists.
    - Returns a list of boxes (even for single input), with best-effort fallback: if no size found
      and cannot infer, returns original boxes unchanged.
    """
    size_cache = getattr(_normalize_boxes_for_image, "_size_cache", {})
    if not isinstance(size_cache, dict):
        size_cache = {}
    setattr(_normalize_boxes_for_image, "_size_cache", size_cache)

    W, H = _get_image_size_cached(img_path, size_cache)

    def norm_one(b):
        nb = _to_xyxy_rel(b, W, H)
        return nb if nb is not None else b

    if isinstance(boxes, (list, tuple)) and len(boxes) == 4 and all(isinstance(x, (int, float)) for x in boxes):
        return [norm_one(list(boxes))]
    elif isinstance(boxes, (list, tuple)) and all(isinstance(bb, (list, tuple)) for bb in boxes):
        return [norm_one(list(bb)) for bb in boxes]
    else:
        return boxes


def resolve_image_path(img_rel: str, roots: dict):
    """Resolve an image relative path against a set of candidate roots.
    Returns the first existing path; otherwise returns the original string.
    """
    # try as-is
    if isinstance(img_rel, str) and os.path.isfile(img_rel):
        return img_rel
    candidates = []
    # basename join to handle cases where json stores only filename
    base = os.path.basename(img_rel) if isinstance(img_rel, str) else None
    for root in (roots or {}).values():
        if root:
            if isinstance(img_rel, str) and ('/' in img_rel or '\\' in img_rel):
                # also try root + basename
                if base:
                    candidates.append(pjoin(root, base))
            # and root + original rel (in case it's just a filename)
            candidates.append(pjoin(root, img_rel))
    # original rel as last resort
    if isinstance(img_rel, str):
        candidates.append(img_rel)
    for c in candidates:
        try:
            if os.path.isfile(c):
                return c
        except Exception:
            continue
    return img_rel



class RECDataset():
    def __init__(self, filename="",
                 template_file="",
                 image_folders={
                    'images/VG_100K': '',
                    'images2/VG_100K_2':''
                 },
                 version = 'vg',
                 total = None,
                 ratio = None,
                 shuffle = False,
                ):

        self.datafile = filename
        self.templates = json.load(open(template_file))
        self.image_dirs = image_folders
        self.version = version

        self.total = total
        self.ratio = ratio
        self.shuffle = shuffle

    def get_template(self,):
        return nprdm.choice(self.templates, 1)[0]

    def build(self, return_dict=None, dict_key="key"):
        # Support files containing {img_path | file_path}, {expression}, {bbox}
        result = []
        rows = []
        if os.path.isfile(self.datafile):
            rows = read_jsonlines(self.datafile)
        else:
            # If not specified, fallback to registered REC jsonls
            for jf in DATA_SOURCES['REC_VG']['jsonls']:
                if os.path.isfile(jf):
                    rows.extend(read_jsonlines(jf))

        if self.shuffle:
            random.shuffle(rows)
        if self.ratio is not None:
            k = max(1, int(len(rows) * float(self.ratio)))
            rows = rows[:k]
        if self.total is not None:
            rows = rows[: self.total]

        for item in tqdm(rows):
            img_rel = item.get('img_path') or item.get('file_path') or item.get('image')
            if img_rel is None:
                continue
            # try resolve against known roots, otherwise use as-is
            img_path = resolve_image_path(img_rel, self.image_dirs)

            # REC should be single bbox; if only `boxes` exists, take the first
            single_box = item.get('bbox')
            if single_box is None:
                bx = item.get('boxes') or []
                single_box = bx[0] if len(bx) > 0 else None
            if single_box is None:
                # REC should have at least one bbox
                continue
            # Normalize bbox to relative xyxy
            norm_boxes = _normalize_boxes_for_image(img_path, single_box)

            # Build ask from template set REC
            tmpl = self.get_template()
            expr = item.get('expression') or EXPR_PLACEHOLDER
            question = (
                tmpl
                .replace('<image>', IMAGE_PLACEHOLDER)
                .replace('<expr>', str(expr))
            )
            caption = f"Answer: {BOXES_PLACEHOLDER} ."

            unified = {
                'image': img_path,
                'target': {'boxes': norm_boxes},
                'conversations': [
                    {'role': 'user', 'content': question},
                    {'role': 'assistant', 'content': caption, 'boxes_seq': [[0]]},
                ]
            }
            result.append(unified)

        return result



class GCDataset():
    def __init__(self, filename="",
                 template_file="",
                 image_folders={
                    'images/VG_100K': '',
                    'images2/VG_100K_2':''
                 },
                 total = None,
                 ratio = None,
                 shuffle = False,
                ):

        self.datafile = filename
        self.templates = json.load(open(template_file))
        self.image_dirs = image_folders

        self.total = total
        self.ratio = ratio
        self.shuffle = shuffle

    def get_template(self,):
        return nprdm.choice(self.templates, 1)[0]

    def build(self, return_dict=None, dict_key="key"):
        # Expect {img_path}, {expression or caption}, {bbox or boxes}
        result = []
        rows = []
        if os.path.isfile(self.datafile):
            rows = read_jsonlines(self.datafile)
        else:
            for jf in DATA_SOURCES['GC_VG']['jsonls']:
                if os.path.isfile(jf):
                    rows.extend(read_jsonlines(jf))

        if self.shuffle:
            random.shuffle(rows)
        if self.ratio is not None:
            k = max(1, int(len(rows) * float(self.ratio)))
            rows = rows[:k]
        if self.total is not None:
            rows = rows[: self.total]

        for item in tqdm(rows):
            img_rel = item.get('img_path') or item.get('file_path')
            if img_rel is None:
                continue
            img_path = resolve_image_path(img_rel, self.image_dirs)

            # Prefer single bbox; fall back to boxes list
            single_box = item.get('bbox')
            if single_box is None:
                bx = item.get('boxes') or []
                single_box = bx[0] if len(bx) > 0 else None
            if single_box is None:
                continue
            # Normalize bbox to relative xyxy
            norm_boxes = _normalize_boxes_for_image(img_path, single_box)
            tmpl = self.get_template()
            question = (
                tmpl
                .replace('<image>', IMAGE_PLACEHOLDER)
                .replace('<objs>', BOXES_PLACEHOLDER)
            )
            caption = item.get('expression') or item.get('caption') or ""

            unified = {
                'image': img_path,
                'target': {'boxes': norm_boxes},
                'conversations': [
                    {'role': 'user', 'content': question, 'boxes_seq': [[0]]},
                    {'role': 'assistant', 'content': caption},
                ]
            }
            result.append(unified)

        return result
    
class REGDataset():
    def __init__(self, filename="",
                 template_file="",
                 image_folders={
                    'train2014': '',
                    'val2014':''
                 },
                 total = None,
                 ratio = None,
                 shuffle = False,
                ):

        self.datafile = filename
        self.templates = json.load(open(template_file))
        self.image_dirs = image_folders

        self.total = total
        self.ratio = ratio
        self.shuffle = shuffle

    def get_template(self,):
        return nprdm.choice(self.templates, 1)[0]

    def build(self, return_dict=None, dict_key="key"):
        # Generation with region hint -> prompt asks to generate expression for boxes
        result = []
        rows = []
        if os.path.isfile(self.datafile):
            rows = read_jsonlines(self.datafile)
        else:
            # For simplicity, reuse REC files but flip the role: boxes -> ask for expression
            for jf in DATA_SOURCES['REC_VG']['jsonls']:
                if os.path.isfile(jf):
                    rows.extend(read_jsonlines(jf))

        if self.shuffle:
            random.shuffle(rows)
        if self.ratio is not None:
            k = max(1, int(len(rows) * float(self.ratio)))
            rows = rows[:k]
        if self.total is not None:
            rows = rows[: self.total]

        for item in tqdm(rows):
            img_rel = item.get('img_path') or item.get('file_path')
            if img_rel is None:
                continue
            img_path = resolve_image_path(img_rel, self.image_dirs)

            single_box = item.get('bbox')
            if single_box is None:
                bx = item.get('boxes') or []
                single_box = bx[0] if len(bx) > 0 else None
            if single_box is None:
                continue
            # Normalize bbox to relative xyxy
            norm_boxes = _normalize_boxes_for_image(img_path, single_box)
            tmpl = self.get_template()
            question = (
                tmpl
                .replace('<image>', IMAGE_PLACEHOLDER)
                .replace('<objs>', BOXES_PLACEHOLDER)
            )
            caption = item.get('expression') or ""

            unified = {
                'image': img_path,
                'target': {'boxes': norm_boxes},
                'conversations': [
                    {'role': 'user', 'content': question, 'boxes_seq': [[0]]},
                    {'role': 'assistant', 'content': caption},
                ]
            }
            result.append(unified)

        return result



class FlickrDataset():
    def __init__(self, filename="",
                 template_file="",
                 image_folders={
                    'flickr30k': '',
                 },
                 total = None,
                 ratio = None,
                 shuffle = False,
                ):

        self.datafile = filename
        self.templates = json.load(open(template_file))
        self.image_dirs = image_folders

        self.total = total
        self.ratio = ratio
        self.shuffle = shuffle

    def get_template(self,):
        return nprdm.choice(self.templates, 1)[0]

    def build(self, return_dict=None, dict_key="key"):
        # Expect {image_id}, {boxes}, {sentence with <ph_st>/<ph_ed>}, {boxes_seq}
        result = []
        rows = []
        if os.path.isfile(self.datafile):
            rows = read_jsonlines(self.datafile)
        else:
            for jf in DATA_SOURCES['FLICKR30K']['jsonls']:
                if os.path.isfile(jf):
                    rows.extend(read_jsonlines(jf))

        if self.shuffle:
            random.shuffle(rows)
        if self.ratio is not None:
            k = max(1, int(len(rows) * float(self.ratio)))
            rows = rows[:k]
        if self.total is not None:
            rows = rows[: self.total]

        for item in tqdm(rows):
            img_id = str(item.get('image_id'))
            # Default Flickr30k filename convention: {image_id}.jpg
            img_rel = pjoin('flickr30k', f"{img_id}.jpg")
            img_path = resolve_image_path(img_rel, self.image_dirs)

            boxes = item.get('boxes') or []
            if boxes:
                boxes = _normalize_boxes_for_image(img_path, boxes)
            boxes_seq = item.get('boxes_seq') or []
            sentence = item.get('sentence') or ""
            tmpl = self.get_template()
            question = tmpl.replace('<image>', IMAGE_PLACEHOLDER)
            caption = sentence

            unified = {
                'image': img_path,
                'target': {'boxes': boxes},
                'conversations': [
                    {'role': 'user', 'content': question},
                    {'role': 'assistant', 'content': caption, 'boxes_seq': boxes_seq},
                ]
            }
            result.append(unified)

        return result


 

class GPT4GenDataset():
    def __init__(self, filename="",
                 template_file="",
                 image_folders={
                    'flickr30k': '',
                 },
                 version='p',
                 total = None,
                 ratio = None,
                 shuffle = False,
                ):

        self.datafile = filename
        self.templates = json.load(open(template_file))
        self.image_dirs = image_folders

        self.version = version
        assert version in ['a', 'c', 'bc']

        self.total = total
        self.ratio = ratio
        self.shuffle = shuffle

    def get_template(self,):
        return nprdm.choice(self.templates, 1)[0]

    def build(self, return_dict=None, dict_key="key"):
        # Expect {img_path}, {question}, {boxes}, and possibly {answer_boxes_seq}/{boxes_seq}
        result = []
        rows = []
        if os.path.isfile(self.datafile):
            rows = read_jsonlines(self.datafile)
        else:
            for jf in DATA_SOURCES['GPT4GEN']['jsonls']:
                if os.path.isfile(jf):
                    rows.extend(read_jsonlines(jf))

        if self.shuffle:
            random.shuffle(rows)
        if self.ratio is not None:
            k = max(1, int(len(rows) * float(self.ratio)))
            rows = rows[:k]
        if self.total is not None:
            rows = rows[: self.total]

        for item in tqdm(rows):
            img_rel = item.get('img_path') or item.get('image') or item.get('file_path')
            if img_rel is None:
                continue
            # For GPT4GEN, image names are typically plain jpg in the same folder as json or a known root
            img_path = resolve_image_path(img_rel, self.image_dirs)

            boxes = item.get('boxes') or []
            if not boxes:
                continue
            boxes = _normalize_boxes_for_image(img_path, boxes)
            answer_boxes_seq = item.get('answer_boxes_seq') or item.get('boxes_seq') or []
            query_boxes_seq = item.get('question_boxes_seq') or item.get('query_boxes_seq') or []
            # question: prefer the dataset question; else fall back to template
            base_q = item.get('question')
            if base_q is None:
                tmpl = self.get_template()
                base_q = tmpl.replace('<image>', IMAGE_PLACEHOLDER).replace('<objs>', BOXES_PLACEHOLDER)
            final_question = base_q
            final_answer = item.get('cot_with_ans') or item.get('answer') or ""

            unified = {
                'image': img_path,
                'target': {'boxes': boxes},
                'conversations': [
                    {'role': 'user', 'content': final_question, 'boxes_seq': query_boxes_seq},
                    {'role': 'assistant', 'content': final_answer, 'boxes_seq': answer_boxes_seq},
                ]
            }
            result.append(unified)

        return result
    


if __name__ == '__main__':
    # Configure default image folders for each dataset class
    rec_img_dirs = {
        'images/VG_100K': IMAGE_ROOTS['VG_100K'],
        'images2/VG_100K_2': IMAGE_ROOTS['VG_100K_2'],
        'train2014': IMAGE_ROOTS['COCO_TRAIN2014'],
    }
    gc_img_dirs = {
        'images/VG_100K': IMAGE_ROOTS['VG_100K'],
        'images2/VG_100K_2': IMAGE_ROOTS['VG_100K_2'],
    }
    flickr_img_dirs = {
        'flickr30k': IMAGE_ROOTS['FLICKR30K']
    }
    gpt_img_dirs = {
        'vg': IMAGE_ROOTS['FLICKR30K'],
        'coco': IMAGE_ROOTS['COCO_TRAIN2014'],
    }

    datasets = [
        RECDataset(filename="", template_file=TEMPLATE_FILES['REC'], version='vg', ratio=None, image_folders=rec_img_dirs),
        GCDataset(filename="", template_file=TEMPLATE_FILES['GC'], ratio=None, image_folders=gc_img_dirs),
        REGDataset(filename="", template_file=TEMPLATE_FILES['REG'], image_folders={'train2014': IMAGE_ROOTS['COCO_TRAIN2014']}),
        FlickrDataset(filename="", template_file=TEMPLATE_FILES['FLICKR'], image_folders=flickr_img_dirs),
        GPT4GenDataset(filename="", template_file=TEMPLATE_FILES['GC'], version='bc', image_folders=gpt_img_dirs),
    ]

    results = []
    for ds in datasets:
        results.extend(ds.build())
    tot = len(results)

    # 'image' already holds a resolvable path string; safe to serialize
    with open("data/vg/all_vg_dataset.json", 'w') as f:
        json.dump(results, f)
    print("Total # exmaples: %d" % tot)