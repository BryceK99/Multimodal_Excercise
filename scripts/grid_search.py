import argparse
import json
import os
import shlex
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List

# A lightweight grid search runner for finetune.py
# - Sequential execution of experiment configs
# - Each run has its own output_dir and log file
# - User can select task (LM or Grounding) and pass DS config

THIS_DIR = Path(__file__).resolve().parent
ROOT = THIS_DIR.parent
FINETUNE = ROOT / "mllm" / "finetune.py"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, help="Model name or path for --model_name_or_path")
    parser.add_argument("--data_path", required=True, help="Train json path")
    parser.add_argument("--eval_data_path", default=None, help="Eval json path")
    parser.add_argument("--task", default="LM", choices=["LM", "Grounding", "Preference"], help="Training task")
    parser.add_argument("--image_folder", default="", help="Base directory for images if needed")
    parser.add_argument("--output_base", default=str(ROOT / "runs"), help="Base folder to store runs")
    parser.add_argument("--deepspeed_config", default=None, help="Path to a DeepSpeed config json (optional)")
    parser.add_argument("--config_json", default=None, help="Optional JSON defining grid configs (list or {common, configs})")
    parser.add_argument("--dry_run", action="store_true", help="Print commands only, do not execute")
    parser.add_argument("--run_suffix", default=None, help="Optional suffix for run directory (defaults to timestamp)")
    parser.add_argument("--launcher", default="python", choices=["python", "torchrun"], help="Process launcher for finetune.py")
    parser.add_argument("--nproc_per_node", type=int, default=1, help="Processes per node for torchrun")
    parser.add_argument("--nnodes", type=int, default=1, help="Number of nodes for torchrun")
    parser.add_argument("--node_rank", type=int, default=0, help="Node rank for torchrun")
    parser.add_argument("--master_addr", default="localhost", help="Master address for torchrun")
    parser.add_argument("--master_port", default="6000", help="Master port for torchrun")
    return parser.parse_args()


def build_cmd(common: Dict[str, Any], cfg: Dict[str, Any], launcher_prefix: List[str]) -> List[str]:
    """Build a command list for a single run."""
    cmd = list(launcher_prefix)

    # Required arguments
    cmd += [
        "--model_name_or_path", common["model"],
        "--data_path", common["data_path"],
        "--task", common["task"],
        "--output_dir", cfg["output_dir"],
    ]
    if common.get("eval_data_path"):
        cmd += ["--eval_data_path", common["eval_data_path"]]
    if common.get("image_folder"):
        cmd += ["--image_folder", common["image_folder"]]
    if common.get("deepspeed_config"):
        cmd += ["--deepspeed", common["deepspeed_config"]]

    def add_flag(flag: str, value: Any) -> None:
        if isinstance(value, bool):
            cmd.append(f"--{flag}")
            cmd.append("True" if value else "False")
        else:
            cmd.extend([f"--{flag}", str(value)])

    # Training hyper-parameters
    add_flag("per_device_train_batch_size", cfg.get("per_device_train_batch_size", 1))
    add_flag("gradient_accumulation_steps", cfg.get("gradient_accumulation_steps", 1))
    add_flag("learning_rate", cfg.get("learning_rate", 2e-5))
    add_flag("num_train_epochs", cfg.get("num_train_epochs", 1))
    add_flag("model_max_length", cfg.get("model_max_length", 2048))
    add_flag("fp16", cfg.get("fp16", True))
    add_flag("bf16", cfg.get("bf16", False))
    add_flag("gradient_checkpointing", cfg.get("gradient_checkpointing", True))

    # Task/model specific
    add_flag("tune_vision", cfg.get("tune_vision", True))
    add_flag("tune_llm", cfg.get("tune_llm", True))
    add_flag("use_lora", cfg.get("use_lora", False))
    if cfg.get("use_lora", False):
        add_flag("lora_r", cfg.get("lora_r", 64))
        add_flag("lora_alpha", cfg.get("lora_alpha", 64))
        add_flag("lora_dropout", cfg.get("lora_dropout", 0.05))
        add_flag("q_lora", cfg.get("q_lora", False))

    if "max_slice_nums" in cfg:
        add_flag("max_slice_nums", cfg["max_slice_nums"])

    # Logging / evaluation
    add_flag("evaluation_strategy", cfg.get("evaluation_strategy", "steps"))
    add_flag("eval_steps", cfg.get("eval_steps", 200))
    add_flag("save_steps", cfg.get("save_steps", 200))
    add_flag("logging_steps", cfg.get("logging_steps", 50))
    add_flag("save_total_limit", cfg.get("save_total_limit", 2))

    return cmd


def run_cmd(cmd: List[str], log_path: Path, dry_run: bool = False) -> int:
    os.makedirs(log_path.parent, exist_ok=True)
    cmd_str = " ".join(shlex.quote(x) for x in cmd)
    print(f"[Run] {cmd_str}")
    if dry_run:
        return 0
    with open(log_path, "w", encoding="utf-8") as handle:
        proc = subprocess.Popen(cmd, stdout=handle, stderr=subprocess.STDOUT)
        return proc.wait()


def load_configs(args: argparse.Namespace, common: Dict[str, Any]) -> List[Dict[str, Any]]:
    if args.config_json:
        cfg_obj = json.loads(Path(args.config_json).read_text(encoding="utf-8"))
        if isinstance(cfg_obj, dict):
            json_common = cfg_obj.get("common", {})
            common.update(json_common)
            configs = cfg_obj.get("configs", [])
            if not isinstance(configs, list):
                raise ValueError("config_json: 'configs' must be a list")
            return configs
        if isinstance(cfg_obj, list):
            return cfg_obj
        raise ValueError("config_json must be a JSON list or an object with {common, configs}")

    # Default presets (LoRA only)
    return [
        {
            "name": "lora_bs1_lr2e-5_len2k_r64",
            "use_lora": True,
            "tune_llm": False,
            "tune_vision": False,
            "per_device_train_batch_size": 1,
            "gradient_accumulation_steps": 16,
            "learning_rate": 2e-5,
            "num_train_epochs": 3,
            "model_max_length": 2048,
            "gradient_checkpointing": True,
            "lora_r": 64,
            "lora_alpha": 64,
            "lora_dropout": 0.05,
            "evaluation_strategy": "steps",
            "eval_steps": 500,
        },
        {
            "name": "lora_bs2_lr1e-5_len2k_r128",
            "use_lora": True,
            "tune_llm": False,
            "tune_vision": True,
            "per_device_train_batch_size": 2,
            "gradient_accumulation_steps": 8,
            "learning_rate": 1e-5,
            "num_train_epochs": 3,
            "model_max_length": 2048,
            "gradient_checkpointing": True,
            "lora_r": 128,
            "lora_alpha": 128,
            "lora_dropout": 0.05,
            "evaluation_strategy": "steps",
            "eval_steps": 500,
        },
        {
            "name": "lora_bs1_lr5e-5_len1k_r64",
            "use_lora": True,
            "tune_llm": False,
            "tune_vision": False,
            "per_device_train_batch_size": 1,
            "gradient_accumulation_steps": 32,
            "learning_rate": 5e-5,
            "num_train_epochs": 2,
            "model_max_length": 1024,
            "gradient_checkpointing": True,
            "lora_r": 64,
            "lora_alpha": 32,
            "lora_dropout": 0.1,
            "evaluation_strategy": "steps",
            "eval_steps": 400,
        },
        {
            "name": "qlora_bs1_lr1e-4_len2k_r64",
            "use_lora": True,
            "q_lora": True,
            "tune_llm": False,
            "tune_vision": False,
            "per_device_train_batch_size": 1,
            "gradient_accumulation_steps": 16,
            "learning_rate": 1e-4,
            "num_train_epochs": 3,
            "model_max_length": 2048,
            "gradient_checkpointing": True,
            "lora_r": 64,
            "lora_alpha": 16,
            "lora_dropout": 0.05,
            "evaluation_strategy": "steps",
            "eval_steps": 500,
        },
        {
            "name": "lora_mild_bs1_lr5e-6_len1k_r32",
            "use_lora": True,
            "tune_llm": False,
            "tune_vision": True,
            "per_device_train_batch_size": 1,
            "gradient_accumulation_steps": 8,
            "learning_rate": 5e-6,
            "num_train_epochs": 2,
            "model_max_length": 1024,
            "gradient_checkpointing": True,
            "lora_r": 32,
            "lora_alpha": 16,
            "lora_dropout": 0.05,
            "evaluation_strategy": "steps",
            "eval_steps": 800,
        },
    ]


def main() -> None:
    args = parse_args()

    run_suffix = args.run_suffix or datetime.now().strftime("%Y%m%d_%H%M%S")
    base = Path(args.output_base) / f"grid_{args.task}_{run_suffix}"

    common: Dict[str, Any] = {
        "model": args.model,
        "data_path": args.data_path,
        "eval_data_path": args.eval_data_path,
        "task": args.task,
        "image_folder": args.image_folder,
        "deepspeed_config": args.deepspeed_config,
    }

    configs = load_configs(args, common)

    if args.launcher == "python":
        launcher_prefix = [sys.executable, str(FINETUNE)]
    else:
        launcher_prefix = [
            "torchrun",
            "--nproc_per_node", str(args.nproc_per_node),
            "--nnodes", str(args.nnodes),
            "--node_rank", str(args.node_rank),
            "--master_addr", args.master_addr,
            "--master_port", str(args.master_port),
            str(FINETUNE),
        ]

    exit_codes = []
    base.mkdir(parents=True, exist_ok=True)

    for cfg in configs:
        out_dir = base / cfg["name"]
        cfg["output_dir"] = str(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)

        cmd = build_cmd(common, cfg, launcher_prefix)
        log_path = out_dir / "train.log"
        code = run_cmd(cmd, log_path, dry_run=args.dry_run)
        exit_codes.append((cfg["name"], code))
        if code != 0:
            print(f"[Warn] {cfg['name']} exited with code {code}")

    summary = {name: code for name, code in exit_codes}
    (base / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print("Grid finished. Summary:")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
