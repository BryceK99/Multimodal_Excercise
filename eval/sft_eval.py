import argparse
import json
import os
import sys
from typing import Tuple

import torch
from PIL import Image
from tqdm import tqdm

from transformers import AutoTokenizer

# Ensure repository root is on sys.path so we can import local packages when
# running this file directly (python eval/sft_eval.py)
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
	sys.path.insert(0, REPO_ROOT)

from eval.model_eval import MLLMEvalModel
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
	"""
	Try to load and merge a LoRA adapter into the given base model.
	Returns the model with adapters applied (merged) if possible; otherwise returns the base model unchanged.
	"""
	if not adapter_path or not os.path.isdir(adapter_path):
		print(f"[sft_eval] Adapter path not found or not a directory: {adapter_path}. Running base model.")
		return model
	try:
		from peft import PeftModel
	except Exception as e:
		print(f"[sft_eval] peft not available ({e}). Run without LoRA or install peft to use the adapter.")
		return model

	try:
		return _try_peft_load(model, adapter_path)
	except Exception as e:
		print(f"[sft_eval] Failed to load LoRA from {adapter_path}: {e}.")
		# Fallback to a sanitized adapter if available
		candidates = []
		# 1) sibling _sanitized under outputs/lora/
		top_level_sanitized = os.path.join(os.path.dirname(adapter_path), "_sanitized")
		candidates.append(top_level_sanitized)
		# 2) walk up one more level and try _sanitized
		parent_sanitized = os.path.join(os.path.dirname(os.path.dirname(adapter_path)), "_sanitized")
		candidates.append(parent_sanitized)
		for cand in candidates:
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
	# Extract the first token-like word to be robust against prefixes like "Answer: Yes." etc.
	for token in ["yes", "no"]:
		if t == token or t.startswith(token):
			return token
	# Fallback: search
	if "yes" in t and "no" not in t:
		return "yes"
	if "no" in t and "yes" not in t:
		return "no"
	# If ambiguous, default to empty string
	return ""


def eval_on_sft(args) -> Tuple[int, int, float]:
	base_model_path = args.base_model
	adapter_path = args.adapter
	data_path = args.data

	# Load base model and tokenizer
	model = MLLMEvalModel.from_pretrained(base_model_path, torch_dtype=torch.bfloat16)
	tokenizer = AutoTokenizer.from_pretrained(base_model_path, trust_remote_code=True)

	# Processor (image + text)
	img_processor_config = read_json('mllm/model/mllm_preprocessor_config.json')
	image_processor = ModelImageProcessor(**img_processor_config)
	processor = ModelProcessor(image_processor, tokenizer)

	# Apply LoRA if available
	model = try_load_lora(model, adapter_path)

	device = 'cuda' if torch.cuda.is_available() else 'cpu'
	model.to(device)
	model.eval()

	with open(data_path, 'r') as f:
		dataset = json.load(f)

	correct = 0
	total = 0

	# Prepare save directory and files
	os.makedirs(args.save_dir, exist_ok=True)
	pred_path = os.path.join(args.save_dir, f"{args.save_prefix}_predictions.jsonl")
	metrics_path = os.path.join(args.save_dir, f"{args.save_prefix}_metrics.json")
	pred_f = open(pred_path, 'w')

	pbar = tqdm(dataset, desc="Evaluating")
	for item in pbar:
		image_path = item.get('image')
		conv = item.get('conversations', [])
		if not conv:
			continue

		# Ground-truth answer is typically the assistant's content in the second turn
		gt = ""
		if len(conv) >= 2 and conv[1].get('role') == 'assistant':
			gt = conv[1].get('content', '')
		gt_norm = normalize_yesno(gt)

		# Only feed the user message for inference
		user_msg = conv[0]
		msgs = [{"role": "user", "content": user_msg.get('content', '')}]

		# Load image
		try:
			if image_path and len(image_path) > 0:
				image = Image.open(image_path).convert('RGB')
			else:
				image = None
		except Exception:
			# If path missing or invalid, skip this sample
			continue

		# Greedy decoding for deterministic yes/no
		with torch.inference_mode():
			pred = model.chat(
				image=image,
				msgs=msgs,
				tokenizer=tokenizer,
				processor=processor,
				sampling=False,  # greedy
				max_new_tokens=args.max_new_tokens,
				repetition_penalty=1.0,
				num_beams=1,
			)

		pred_norm = normalize_yesno(pred)

		evaluated = False
		is_correct = False
		if gt_norm in ("yes", "no") and pred_norm in ("yes", "no"):
			is_correct = pred_norm == gt_norm
			correct += int(is_correct)
			total += 1
			evaluated = True

		# Write per-sample prediction
		out = {
			"id": item.get("id"),
			"image": image_path,
			"question": user_msg.get('content', ''),
			"gt": gt,
			"gt_norm": gt_norm,
			"pred": pred,
			"pred_norm": pred_norm,
			"evaluated": evaluated,
			"correct": is_correct,
		}
		pred_f.write(json.dumps(out, ensure_ascii=False) + "\n")
		pred_f.flush()

		# Optional progress postfix
		if total > 0:
			pbar.set_postfix(acc=f"{correct/total:.4f}")

		if args.max_samples is not None and total >= args.max_samples:
			break

	pred_f.close()
	acc = (correct / total) if total else 0.0
	# Save metrics summary
	metrics = {
		"accuracy": acc,
		"correct": correct,
		"total": total,
		"base_model": base_model_path,
		"adapter": adapter_path,
		"data": data_path,
		"max_new_tokens": args.max_new_tokens,
	}
	with open(metrics_path, 'w') as mf:
		json.dump(metrics, mf, ensure_ascii=False, indent=2)

	print(f"[sft_eval] Accuracy: {acc:.4f} ({correct}/{total})")
	print(f"[sft_eval] Saved predictions to: {pred_path}")
	print(f"[sft_eval] Saved metrics to: {metrics_path}")
	return correct, total, acc


def main():
	parser = argparse.ArgumentParser(description="SFT eval: run yes/no accuracy on data/test.json")
	parser.add_argument(
		"--base-model",
		type=str,
		default="HaoyeZhang/MLLM_Excercise_Model",
		help="Base model path or repo id (local folder or HF repo)."
	)
	parser.add_argument(
		"--adapter",
		type=str,
		default="outputs/lora/checkpoint-600",
		help="LoRA adapter directory (e.g., outputs/lora/checkpoint-400)."
	)
	parser.add_argument(
		"--data",
		type=str,
		default="data/test.json",
		help="Path to the test.json file."
	)
	parser.add_argument(
		"--max-samples",
		type=int,
		default=None,
		help="Optional limit on number of samples to evaluate."
	)
	parser.add_argument(
		"--max-new-tokens",
		type=int,
		default=8,
		help="Max new tokens to generate for each answer."
	)
	parser.add_argument(
		"--save-dir",
		type=str,
		default="outputs",
		help="Directory to save predictions and metrics."
	)
	parser.add_argument(
		"--save-prefix",
		type=str,
		default="sft_test",
		help="File prefix for saved outputs."
	)

	args = parser.parse_args()
	eval_on_sft(args)


if __name__ == "__main__":
	main()

