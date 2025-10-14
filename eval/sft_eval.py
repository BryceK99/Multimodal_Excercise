"""
SFT evaluation using the same pipeline as eval/model_eval.py

- Reuses MLLMEvalModel and its chat() flow for decoding
- Loads base model and optionally merges a LoRA adapter
- Reads data/sft/test.json, runs inference, and computes yes/no accuracy
"""

import argparse
import json
import os
import sys
from typing import Dict, List, Tuple

import torch
from PIL import Image
from tqdm import tqdm
from transformers import AutoTokenizer


# Ensure repository root is on sys.path
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from eval.model_eval import MLLMEvalModel  # reuse model + chat() implementation
from mllm.model.image_processing import ModelImageProcessor
from mllm.model.processing import ModelProcessor
from utils.file_io import read_json


def _try_peft_load(model: torch.nn.Module, adapter_path: str) -> torch.nn.Module:
    from peft import PeftModel
    peft_model = PeftModel.from_pretrained(model, adapter_path)
    merged = peft_model.merge_and_unload()
    print(f"[sft_eval] Loaded and merged LoRA adapter from: {adapter_path}")
    return merged


def try_load_lora(model: torch.nn.Module, adapter_path: str) -> torch.nn.Module:
    """Try to merge a LoRA adapter into the base model. Returns base if unavailable."""
    if not adapter_path or not os.path.isdir(adapter_path):
        print(f"[sft_eval] Adapter path not found or not a directory: {adapter_path}. Running base model.")
        return model
    try:
        from peft import PeftModel  # noqa: F401
    except Exception as e:
        print(f"[sft_eval] peft not available ({e}). Run without LoRA or install peft to use the adapter.")
        return model

    try:
        return _try_peft_load(model, adapter_path)
    except Exception as e:
        print(f"[sft_eval] Failed to load LoRA from {adapter_path}: {e}.")
        # Fallback candidates named _sanitized
        for cand in [
            os.path.join(os.path.dirname(adapter_path), "_sanitized"),
            os.path.join(os.path.dirname(os.path.dirname(adapter_path)), "_sanitized"),
        ]:
            if os.path.isdir(cand):
                try:
                    print(f"[sft_eval] Trying fallback sanitized adapter: {cand}")
                    return _try_peft_load(model, cand)
                except Exception as e2:
                    print(f"[sft_eval] Fallback failed: {e2}")
        print("[sft_eval] Continue with base model (no LoRA applied).")
        return model


def normalize_yesno(s: str) -> str:
    if s is None:
        return ""
    t = s.strip().lower()
    # Fast path
    if t in ("yes", "no"):
        return t
    # Robust prefix check (e.g., "Yes." / "No,")
    if t.startswith("yes"):
        return "yes"
    if t.startswith("no"):
        return "no"
    # Fallback contains check
    if ("yes" in t) ^ ("no" in t):
        return "yes" if "yes" in t else "no"
    return ""


def load_dataset(path: str) -> List[Dict]:
    with open(path, "r") as f:
        data = json.load(f)
    return data


def fix_image_path(p: str) -> str:
    """Map dataset path to actual workspace path if needed."""
    if isinstance(p, str) and p.startswith("data/images/"):
        return p.replace("data/images/", "data/sft/images/")
    return p


def build_messages(user_text: str) -> List[Dict[str, str]]:
    # Remove dataset's inline <image> tag; image objects are passed separately
    content = (user_text or "").replace("<image>", "").strip()
    return [{"role": "user", "content": content}]


def eval_on_sft(args) -> Tuple[int, int, float]:
    # Load base model and tokenizer
    model = MLLMEvalModel.from_pretrained(args.base_model, torch_dtype=torch.bfloat16)
    tokenizer = AutoTokenizer.from_pretrained(args.base_model, trust_remote_code=True)

    # Processor (image + text), same as model_eval.py
    img_processor_config = read_json("mllm/model/mllm_preprocessor_config.json")
    image_processor = ModelImageProcessor(**img_processor_config)
    processor = ModelProcessor(image_processor, tokenizer)

    # LoRA merge (optional)
    model = try_load_lora(model, args.adapter)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)
    model.eval()

    dataset = load_dataset(args.data)

    os.makedirs(args.save_dir, exist_ok=True)
    pred_path = os.path.join(args.save_dir, f"{args.save_prefix}_predictions.jsonl")
    metrics_path = os.path.join(args.save_dir, f"{args.save_prefix}_metrics.json")
    pred_f = open(pred_path, "w")

    # Decoding config aligned with model_eval.py (greedy by default for yes/no)
    decoding_mode = args.decoding if args.decoding is not None else ("beam")
    if decoding_mode == "sampling":
        use_sampling = True
        gen_kwargs = dict(temperature=args.temperature, top_p=args.top_p, top_k=args.top_k)
    elif decoding_mode == "greedy":
        use_sampling = False
        gen_kwargs = dict(num_beams=1)
    else:  # beam
        use_sampling = False
        gen_kwargs = dict(num_beams=args.num_beams)

    correct = 0
    total = 0

    pbar = tqdm(dataset, desc="SFT Evaluating")
    for idx, item in enumerate(pbar):
        # Extract fields
        img_path = fix_image_path(item.get("image"))
        conv = item.get("conversations", [])
        if not conv:
            continue

        # Ground truth
        gt = ""
        if len(conv) >= 2 and conv[1].get("role") == "assistant":
            gt = conv[1].get("content", "")
        gt_norm = normalize_yesno(gt)

        # User prompt -> msgs
        user_msg = conv[0]
        msgs = build_messages(user_msg.get("content", ""))

        # Load image
        try:
            image = Image.open(img_path).convert("RGB") if img_path else None
        except Exception as e:
            # Skip invalid image
            print(f"[sft_eval] Skip sample id={item.get('id')} due to image error: {e}")
            continue

        # Generation args: keep short output and minimal repetition
        chat_kwargs = dict(max_new_tokens=args.max_new_tokens, repetition_penalty=args.repetition_penalty)
        chat_kwargs.update(gen_kwargs)

        with torch.inference_mode():
            pred = model.chat(
                image=image,
                msgs=msgs,
                tokenizer=tokenizer,
                processor=processor,
                sampling=use_sampling,
                **chat_kwargs,
            )

        pred_norm = normalize_yesno(pred)

        evaluated = False
        is_correct = False
        if gt_norm in ("yes", "no") and pred_norm in ("yes", "no"):
            evaluated = True
            is_correct = (pred_norm == gt_norm)
            correct += int(is_correct)
            total += 1

        # Write prediction line
        out = {
            "id": item.get("id", idx),
            "image": img_path,
            "question": user_msg.get("content", ""),
            "gt": gt,
            "gt_norm": gt_norm,
            "pred": pred,
            "pred_norm": pred_norm,
            "evaluated": evaluated,
            "correct": is_correct,
        }
        pred_f.write(json.dumps(out, ensure_ascii=False) + "\n")
        pred_f.flush()

        if total > 0:
            pbar.set_postfix(acc=f"{correct/total:.4f}")

        if args.max_samples is not None and total >= args.max_samples:
            break

    pred_f.close()
    acc = (correct / total) if total else 0.0

    metrics = {
        "accuracy": acc,
        "correct": correct,
        "total": total,
        "base_model": args.base_model,
        "adapter": args.adapter,
        "data": args.data,
        "decoding": decoding_mode,
        "num_beams": args.num_beams,
        "temperature": args.temperature,
        "top_p": args.top_p,
        "top_k": args.top_k,
        "max_new_tokens": args.max_new_tokens,
        "repetition_penalty": args.repetition_penalty,
    }
    with open(metrics_path, "w") as mf:
        json.dump(metrics, mf, ensure_ascii=False, indent=2)

    print(f"[sft_eval] Accuracy: {acc:.4f} ({correct}/{total})")
    print(f"[sft_eval] Saved predictions to: {pred_path}")
    print(f"[sft_eval] Saved metrics to: {metrics_path}")
    return correct, total, acc


def main():
    parser = argparse.ArgumentParser(description="SFT eval using model_eval pipeline on data/sft/test.json")
    parser.add_argument("--base-model", type=str, default="HaoyeZhang/MLLM_Excercise_Model",
                        help="Base model path or repo id (local folder or HF repo)")
    parser.add_argument("--adapter", type=str, default="outputs/sft/checkpoint-600",
                        help="LoRA adapter directory (optional)")
    parser.add_argument("--data", type=str, default="data/sft/test.json",
                        help="Path to the test.json file")
    parser.add_argument("--max-samples", type=int, default=None,
                        help="Optional limit on number of evaluated samples")
    # Decoding controls (aligned with model_eval)
    parser.add_argument("--decoding", choices=["greedy", "beam", "sampling"], default="greedy",
                        help="Decoding strategy for generation")
    parser.add_argument("--num-beams", type=int, default=1, dest="num_beams",
                        help="Beam width when using beam decoding")
    parser.add_argument("--temperature", type=float, default=0.7,
                        help="Sampling temperature when using sampling decoding")
    parser.add_argument("--top-p", type=float, default=0.8, dest="top_p",
                        help="Top-p for nucleus sampling")
    parser.add_argument("--top-k", type=int, default=100, dest="top_k",
                        help="Top-k sampling cutoff")
    parser.add_argument("--max-new-tokens", type=int, default=32, dest="max_new_tokens",
                        help="Max new tokens for each generation")
    parser.add_argument("--repetition-penalty", type=float, default=1.0, dest="repetition_penalty",
                        help="Repetition penalty for generation")
    # Save
    parser.add_argument("--save-dir", type=str, default="outputs/sft/eval_result",
                        help="Directory to save predictions and metrics")
    parser.add_argument("--save-prefix", type=str, default="sft_test",
                        help="Filename prefix for saved outputs")

    args = parser.parse_args()
    eval_on_sft(args)


if __name__ == "__main__":
    main()

