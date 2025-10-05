import sys
import os
# Ensure project root on sys.path so local modules like mllm.* are importable
_PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

# Try to register local custom MLLM architecture so Auto* can load checkpoints with model_type="mllm"
try:
    from transformers import AutoConfig, AutoModelForCausalLM as _AutoCausal
    from mllm.model.configuration import ModelConfig as _MLLMConfig
    from mllm.model.modeling_mllm import MLLMModel as _MLLMModel
    try:
        AutoConfig.register("mllm", _MLLMConfig)
        _AutoCausal.register(_MLLMConfig, _MLLMModel)
    except Exception:
        # Ignore if already registered or HF version without register API
        pass
except Exception:
    # If these imports fail, we'll fallback to trust_remote_code later in the script
    pass

import argparse
import json
import os
import re
import string
from typing import List, Dict, Any

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import AutoPeftModelForCausalLM, PeftModel


def robust_load_model(adapter_dir: str, base_model: str, device_map: str, torch_dtype):
    """Load base+LoRA with local MLLM registered first, with fallbacks."""
    tok_src = None
    model = None
    # Stage 1: try registered local Auto path
    try:
        if base_model:
            base = AutoModelForCausalLM.from_pretrained(
                base_model, torch_dtype=torch_dtype, device_map=device_map
            )
            model = PeftModel.from_pretrained(base, adapter_dir)
            tok_src = base_model
        else:
            model = AutoPeftModelForCausalLM.from_pretrained(
                adapter_dir, torch_dtype=torch_dtype, device_map=device_map
            )
            tok_src = adapter_dir
        return model, tok_src
    except Exception:
        pass
    # Stage 2: direct local class import
    try:
        from mllm.model.modeling_mllm import MLLMModel
        if not base_model:
            raise RuntimeError("local class path requires base_model")
        base = MLLMModel.from_pretrained(
            base_model, torch_dtype=torch_dtype, device_map=device_map
        )
        model = PeftModel.from_pretrained(base, adapter_dir)
        tok_src = base_model
        return model, tok_src
    except Exception:
        pass
    # Stage 3: trust_remote_code fallback
    if base_model:
        base = AutoModelForCausalLM.from_pretrained(
            base_model, torch_dtype=torch_dtype, device_map=device_map, trust_remote_code=True
        )
        model = PeftModel.from_pretrained(base, adapter_dir, trust_remote_code=True)
        tok_src = base_model
    else:
        model = AutoPeftModelForCausalLM.from_pretrained(
            adapter_dir, torch_dtype=torch_dtype, device_map=device_map, trust_remote_code=True
        )
        tok_src = adapter_dir
    return model, tok_src


def main():
    parser = argparse.ArgumentParser(description="Evaluate exact-match accuracy for LoRA-adapted model.")
    parser.add_argument("--adapter_dir", type=str, default="outputs/lora/checkpoint-400", help="Path to LoRA adapter (output_dir).")
    parser.add_argument("--base_model", type=str, default="HaoyeZhang/MLLM_Excercise_Model", help="Optional base model path/name. If empty, AutoPeft will load base automatically.")
    parser.add_argument("--test_path", type=str, default="data/test.json", help="Path to test set .json or .jsonl")
    parser.add_argument("--input_key", type=str, default="", help="Key for input text (if empty, auto-detect).")
    parser.add_argument("--label_key", type=str, default="", help="Key for label/answer text (if empty, auto-detect).")
    parser.add_argument("--prompt_template", type=str, default="{input}", help="Prompt template; use {input} placeholder.")
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--max_new_tokens", type=int, default=128)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top_p", type=float, default=1.0)
    parser.add_argument("--device_map", type=str, default="auto")
    parser.add_argument("--dtype", type=str, default="bf16", choices=["bf16", "fp16", "fp32"])
    parser.add_argument("--case_sensitive", action="store_true", help="Exact match case sensitive if set.")
    parser.add_argument("--keep_punct", action="store_true", help="Do not strip punctuation/spaces if set.")
    parser.add_argument("--save_pred_path", type=str, default="", help="Optional path to save predictions JSONL.")
    args = parser.parse_args()

    dtype_map = {
        "bf16": torch.bfloat16,
        "fp16": torch.float16,
        "fp32": torch.float32,
    }
    torch_dtype = dtype_map[args.dtype]

    # Load model + tokenizer using robust loader
    model, tok_src = robust_load_model(
        adapter_dir=args.adapter_dir,
        base_model=args.base_model,
        device_map=args.device_map,
        torch_dtype=torch_dtype,
    )

    model.eval()
    tokenizer = AutoTokenizer.from_pretrained(tok_src, use_fast=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"  # better for decoder-only batch generation
    
    # Load data
    data = load_data(args.test_path)
    if not data:
        print("No samples in test set.")
        return

    # Detect keys if not provided
    if not args.input_key:
        args.input_key = guess_field(data[0], ["input", "instruction", "question", "query", "prompt"])
    if not args.label_key:
        args.label_key = guess_field(data[0], ["answer", "output", "label", "target"])

    # Filter out multimodal samples with image fields (best-effort)
    text_only = []
    skipped = 0
    for ex in data:
        has_image = any(k in ex for k in ["image", "images", "image_path", "img", "pic"])
        if has_image and ex.get(args.input_key) and ex.get(args.label_key):
            skipped += 1
            continue
        text_only.append(ex)
    if skipped:
        print(f"Warning: skipped {skipped} multimodal samples (image fields present).")

    # Build prompts
    prompts = build_prompts(text_only, args.input_key, args.prompt_template)
    labels = [str(ex.get(args.label_key, "")) for ex in text_only]

    # Batched generation
    preds: List[str] = []
    saved_f = open(args.save_pred_path, "w", encoding="utf-8") if args.save_pred_path else None

    for i in range(0, len(prompts), args.batch_size):
        batch_prompts = prompts[i : i + args.batch_size]
        enc = tokenizer(batch_prompts, return_tensors="pt", padding=True, truncation=True).to(model.device)
        with torch.no_grad():
            gen = model.generate(
                **enc,
                max_new_tokens=args.max_new_tokens,
                do_sample=(args.temperature > 0),
                temperature=max(args.temperature, 1e-6),
                top_p=args.top_p,
                num_beams=1,
                eos_token_id=tokenizer.eos_token_id,
                pad_token_id=tokenizer.pad_token_id,
            )
        # Extract generated portion per sample
        gen_only_ids = slice_generations(gen, enc["input_ids"])
        batch_texts = [tokenizer.decode(ids, skip_special_tokens=True).strip() for ids in gen_only_ids]
        preds.extend(batch_texts)

        if saved_f:
            for p_text, g_text, raw_prompt in zip(batch_texts, labels[i : i + len(batch_texts)], batch_prompts):
                saved_f.write(json.dumps({"prompt": raw_prompt, "pred": p_text, "gold": g_text}, ensure_ascii=False) + "\n")
                saved_f.flush()

    if saved_f:
        saved_f.close()

    # Compute exact-match accuracy
    lower = not args.case_sensitive
    strip_punct = not args.keep_punct

    corr = 0
    for pred, gold in zip(preds, labels):
        p = normalize_text(pred, lowercase=lower, strip_punct=strip_punct)
        g = normalize_text(gold, lowercase=lower, strip_punct=strip_punct)
        print("pred: ", p, "label:", g)
        if p == g:
            corr += 1

    total = len(labels)
    acc = corr / total if total else 0.0
    print(json.dumps({"total": total, "correct": corr, "accuracy": acc}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()