import io
import os
import re
import json
import base64
import argparse
from PIL import Image, ImageDraw, ImageFont
from tqdm import tqdm

import torch
from torchvision.ops.boxes import box_area
from transformers import AutoTokenizer

from eval.model_eval import MLLMEvalModel
from mllm.model.processing import ModelProcessor
from mllm.model.image_processing import ModelImageProcessor

from utils.file_io import read_jsonlines, read_json

def box_iou(boxes1, boxes2):
    area1 = box_area(boxes1)
    area2 = box_area(boxes2)

    lt = torch.max(boxes1[:, None, :2], boxes2[:, :2])  # [N,M,2]
    rb = torch.min(boxes1[:, None, 2:], boxes2[:, 2:])  # [N,M,2]

    wh = (rb - lt).clamp(min=0)  # [N,M,2]
    inter = wh[:, :, 0] * wh[:, :, 1]  # [N,M]

    union = area1[:, None] + area2 - inter

    iou = inter / union
    return iou, union


def _to_abs_from_rel(box, w, h):
    x1, y1, x2, y2 = box
    return [x1 * w, y1 * h, x2 * w, y2 * h]


def vis_boxes(img, boxes, expr, save_name='output.png'):
    """可视化VG结果。
    - img: PIL.Image
    - boxes: dict with keys 'pred' and/or 'gt', values are normalized [x1,y1,x2,y2]
    - expr: expression string
    - save_name: output file path
    """
    if not isinstance(img, Image.Image):
        img = Image.open(img).convert('RGB')
    draw = ImageDraw.Draw(img, 'RGBA')
    w, h = img.size

    def draw_box(b, color, text):
        if not b:
            return
        bx = _to_abs_from_rel(b, w, h)
        x1, y1, x2, y2 = bx
        draw.rectangle([x1, y1, x2, y2], outline=color, width=3)
        # semi-transparent fill
        draw.rectangle([x1, y1, x2, y2], fill=(color[0], color[1], color[2], 40))
        if text:
            draw.text((x1, max(0, y1 - 12)), text, fill=(255,255,255,255))

    pred = boxes.get('pred') if isinstance(boxes, dict) else None
    gt = boxes.get('gt') if isinstance(boxes, dict) else None
    draw_box(gt, (0, 200, 0, 255), f'GT: {expr}')
    draw_box(pred, (220, 0, 0, 255), 'PRED')
    os.makedirs(os.path.dirname(save_name) or '.', exist_ok=True)
    img.save(save_name)

    
def parse_predicted_bbox(answer: str, w: int = None, h: int = None):
    """Parse model text to normalized xyxy [0,1] using evaluate_grounding style.
    Primary format: (x1,y1),(x2,y2) with values in 0..999; normalized by dividing 999.
    If not found, return None (treated as zero IoU).
    """
    PATTERN = re.compile(r"\((.*?)\)\s*,\s*\((.*?)\)")
    m = re.findall(PATTERN, answer)
    if not m:
        return None
    try:
        p1, p2 = m[0]
        if ',' not in p1 or ',' not in p2:
            return None
        x1, y1 = [float(v) for v in p1.split(',')]
        x2, y2 = [float(v) for v in p2.split(',')]
        # normalize by 999 to [0,1]
        x1, y1, x2, y2 = x1/999.0, y1/999.0, x2/999.0, y2/999.0
    except Exception:
        return None

    def clamp01(v):
        return max(0.0, min(1.0, v))
    x1, x2 = sorted([x1, x2])
    y1, y2 = sorted([y1, y2])
    return [clamp01(x1), clamp01(y1), clamp01(x2), clamp01(y2)]


def iou_normalized_xyxy(pred, gt):
    """Compute IoU for two normalized [x1,y1,x2,y2] boxes as floats."""
    if not pred or not gt:
        return 0.0
    px1, py1, px2, py2 = pred
    gx1, gy1, gx2, gy2 = gt
    ix1, iy1 = max(px1, gx1), max(py1, gy1)
    ix2, iy2 = min(px2, gx2), min(py2, gy2)
    iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
    inter = iw * ih
    area_p = max(0.0, px2 - px1) * max(0.0, py2 - py1)
    area_g = max(0.0, gx2 - gx1) * max(0.0, gy2 - gy1)
    union = area_p + area_g - inter
    return (inter / union) if union > 0 else 0.0


    

def evaluate_jsonl(model, tokenizer, processor, jsonl_path, image_root, sampling, vis_nums=0):
    """Evaluate a single JSONL split file and return (acc, total, correct)."""
    data = read_jsonlines(jsonl_path)
    correct = 0
    total_cnt = 0
    remaining_vis = int(vis_nums) if vis_nums else 0
    with torch.no_grad():
        for item in tqdm(data, desc=os.path.basename(jsonl_path)):
            img_rel = item.get('img_path') or item.get('image')
            expr = item.get('expression') or item.get('question') or ''
            bbox = item.get('bbox')

            img_path = img_rel
            if isinstance(img_rel, str) and len(img_rel) < 1000:
                # resolve relative path under the provided image_root
                img_path = os.path.join(image_root, img_rel)

            if isinstance(img_path, str) and len(img_path) > 1000:
                image = Image.open(io.BytesIO(base64.b64decode(img_path))).convert('RGB')
            else:
                image = Image.open(img_path).convert('RGB')

            prompt = f"Where is {expr} in image? answer in [x0,y0,x1,y1] format."
            msgs = [{"role": "user", "content": prompt}]

            answer = model.chat(
                image=image,
                msgs=msgs,
                context=None,
                tokenizer=tokenizer,
                sampling=sampling,
                processor=processor,
            )

            W, H = image.size
            pred_norm = parse_predicted_bbox(answer, W, H)
            # Normalize GT bbox if needed
            if isinstance(bbox, list) and len(bbox) == 4:
                mx = max(bbox)
                if mx <= 1.0:
                    gt_norm = bbox
                else:
                    x1, y1, x2, y2 = bbox
                    gt_norm = [x1 / W, y1 / H, x2 / W, y2 / H]
            else:
                gt_norm = None

            iou = iou_normalized_xyxy(pred_norm, gt_norm)
            total_cnt += 1
            if iou >= 0.5:
                correct += 1

            if remaining_vis > 0:
                vis_boxes(image, {"pred": pred_norm, "gt": gt_norm}, expr, save_name=f"vg_vis_{total_cnt}.png")
                remaining_vis -= 1

    acc = (correct / total_cnt) if total_cnt > 0 else 0.0
    return acc, total_cnt, correct


def eval_model(args):
    # Load model/tokenizer/processor once
    model = MLLMEvalModel.from_pretrained(
        args.model_name_or_path,
        torch_dtype=torch.bfloat16,
    )
    tokenizer = AutoTokenizer.from_pretrained(
        args.model_name_or_path, trust_remote_code=True
    )
    img_processor_config = read_json('mllm/model/mllm_preprocessor_config.json')
    img_processor_config['max_slice_nums'] = 1
    image_processor = ModelImageProcessor(**img_processor_config)
    processor = ModelProcessor(image_processor, tokenizer)
    model.eval().cuda()

    # If a specific file is provided, keep single-file mode for flexibility
    if args.question_file:
        image_root = args.image_dir or os.path.join(args.root_dir, 'coco', 'train2014')
        acc, total, correct = evaluate_jsonl(
            model, tokenizer, processor,
            jsonl_path=args.question_file,
            image_root=image_root,
            sampling=args.sampling,
            vis_nums=args.vis_nums,
        )
        print(f"Evaluating {args.question_file} ...")
        print(f"Precision @ 1: {acc:.4f} ({correct}/{total})\n")
        return

    # Default: run the full RefCOCO / RefCOCO+ / RefCOCOg suite under data/vg
    root = args.root_dir
    coco_root = args.image_dir or os.path.join(root, 'coco', 'train2014')

    # File layout consistent with README and provided data/vg
    splits = {
        'RefCOCO': {
            'val': os.path.join(root, 'REC_refcoco_unc_val.jsonl'),
            'testA': os.path.join(root, 'REC_refcoco_unc_testA.jsonl'),
            'testB': os.path.join(root, 'REC_refcoco_unc_testB.jsonl'),
        },
        'RefCOCO+': {
            'val': os.path.join(root, 'REC_refcoco+_unc_val.jsonl'),
            'testA': os.path.join(root, 'REC_refcoco+_unc_testA.jsonl'),
            'testB': os.path.join(root, 'REC_refcoco+_unc_testB.jsonl'),
        },
        'RefCOCOg': {
            'val-u': os.path.join(root, 'REC_refcocog_umd_val.jsonl'),
            'test-u': os.path.join(root, 'REC_refcocog_umd_test.jsonl'),
        },
    }

    results = {}
    for dataset, parts in splits.items():
        results[dataset] = {}
        for name, path in parts.items():
            if not os.path.isfile(path):
                print(f"[Skip] Missing file: {path}")
                continue
            acc, total, correct = evaluate_jsonl(
                model, tokenizer, processor,
                jsonl_path=path,
                image_root=coco_root,
                sampling=args.sampling,
                vis_nums=0,
            )
            results[dataset][name] = (acc, total, correct)

    # Pretty print summary
    print("\n=== RefCOCO-family Visual Grounding Accuracy (IoU>=0.5) ===")
    for dataset in ['RefCOCO', 'RefCOCO+','RefCOCOg']:
        if dataset not in results:
            continue
        rows = []
        for split_name in results[dataset]:
            acc, total, correct = results[dataset][split_name]
            rows.append(f"{split_name}: {acc:.4f} ({correct}/{total})")
        if rows:
            print(f"{dataset}: " + " | ".join(rows))
    print("")


def interactive_dialogue(args):
    model = MLLMEvalModel.from_pretrained(
        args.model_name_or_path,
        torch_dtype=torch.bfloat16,
    )
    tokenizer = AutoTokenizer.from_pretrained(
        args.model_name_or_path, trust_remote_code=True
    )
    img_processor_config = read_json('mllm/model/mllm_preprocessor_config.json')
    image_processor = ModelImageProcessor(**img_processor_config)
    processor = ModelProcessor(image_processor, tokenizer)
    model.eval().cuda()

    img = Image.open(args.image).convert('RGB')
    print("Enter expressions to ground (empty line to exit).")
    while True:
        expr = args.expr if args.expr else input('Expression: ').strip()
        if not expr:
            break
        # Keep original prompt format unchanged
        prompt = "Where is {} in image? answer in [x0,y0,x1,y1] format.".format(expr)
        msgs = [{"role": "user", "content": prompt}]
        answer = model.chat(
            image=img,
            msgs=msgs,
            context=None,
            tokenizer=tokenizer,
            sampling=args.sampling,
            processor=processor,
        )
        W, H = img.size
        pred_norm = parse_predicted_bbox(answer, W, H)
        print(f"Answer: {answer}")
        out_path = args.out if args.out else 'vg_interactive.png'
        vis_boxes(img.copy(), {"pred": pred_norm}, expr, save_name=out_path)
        print(f"Saved visualization to {out_path}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-name-or-path", default="/root/Multimodal_Excercise/outputs/grounding/checkpoint-400", type=str)
    parser.add_argument("--question-file", type=str, default=None, help="Optional: evaluate a single JSONL file")
    parser.add_argument("--image-dir", type=str, default=None, help="Optional: image root; defaults to data/vg/coco/train2014")
    parser.add_argument("--sampling", action='store_true')
    parser.add_argument("--vis-nums", type=int, default=5)
    parser.add_argument("--root-dir", type=str, default="data/vg", help="Root directory containing RefCOCO JSONLs and COCO images")
    # Interactive mode
    parser.add_argument("--interactive", action='store_true', help="Run interactive terminal dialogue mode")
    parser.add_argument("--image", type=str, default=None, help="Image path for interactive mode")
    parser.add_argument("--expr", type=str, default=None, help="Expression for one-shot interactive run")
    parser.add_argument("--out", type=str, default=None, help="Save path for interactive visualization")
    args = parser.parse_args()

    if args.interactive:
        assert args.image, "--image is required in interactive mode"
        interactive_dialogue(args)
    else:
        eval_model(args)
