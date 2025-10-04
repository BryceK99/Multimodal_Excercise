#!/usr/bin/env bash
# Integrated grid + evaluation launcher for LoRA finetuning
# Combines scripts/grid_search.py with torchrun Deepspeed-style launch and
# aggregates evaluation metrics from trainer_state.json files.

set -euo pipefail

export PYTHONPATH="${PYTHONPATH}:$(realpath .)"
export CUDA_VISIBLE_DEVICES=0,1

# Distributed defaults (can be overridden via CLI)
GPUS_PER_NODE=4
NNODES=1
NODE_RANK=0
MASTER_ADDR=localhost
MASTER_PORT=6001
CUDA_DEVICES="0,1"

# Workload defaults
MODEL="./weights"
DATA_PATH="data/train.json"
EVAL_PATH="data/test.json"
OUTPUT_DIR="outputs/lora/"
TASK="LM"
IMAGE_FOLDER="data/images"

CONFIG_JSON=""
OUTPUT_BASE="runs"
DEEPSPEED_CONFIG=""
RUN_SUFFIX=""
DRY_RUN=0

usage() {
  cat <<'USAGE'
Usage: bash run_grid_lora_eval.sh --model <path> --data_path <train.json> [options]

Required:
  --model PATH              Model name or path passed to finetune.py
  --data_path PATH          Training JSON file

Optional:
  --eval_data_path PATH     Evaluation JSON file
  --task {LM|Grounding|Preference}
  --image_folder PATH       Image root if dataset references images
  --config_json PATH        Grid config file (see scripts/grid_example.json)
  --output_base DIR         Base directory for grid outputs (default: runs)
  --deepspeed PATH          DeepSpeed config JSON passed to finetune.py
  --gpus INT                Processes per node for torchrun (default: 4)
  --nnodes INT              Total nodes for torchrun (default: 1)
  --node_rank INT           Rank of this node (default: 0)
  --master_addr HOST        Master address (default: localhost)
  --master_port PORT        Master port (default: 6001)
  --cuda DEVICES            CUDA_VISIBLE_DEVICES value (e.g. 0,1,2,3)
  --run_suffix STRING       Custom suffix for the grid run directory
  --dry_run                 Print commands and exit without execution

Outputs:
  - Runs grid_search.py with torchrun launcher and DeepSpeed config.
  - Collects eval metrics from each run and prints an accuracy summary.
  - Writes metrics summary to <output_base>/grid_<task>_<suffix>/metrics_summary.json
USAGE
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --model) MODEL="$2"; shift 2;;
    --data_path) DATA_PATH="$2"; shift 2;;
    --eval_data_path) EVAL_PATH="$2"; shift 2;;
    --task) TASK="$2"; shift 2;;
    --image_folder) IMAGE_FOLDER="$2"; shift 2;;
    --config_json) CONFIG_JSON="$2"; shift 2;;
    --output_base) OUTPUT_BASE="$2"; shift 2;;
    --deepspeed) DEEPSPEED_CONFIG="$2"; shift 2;;
    --gpus) GPUS_PER_NODE="$2"; shift 2;;
    --nnodes) NNODES="$2"; shift 2;;
    --node_rank) NODE_RANK="$2"; shift 2;;
    --master_addr) MASTER_ADDR="$2"; shift 2;;
    --master_port) MASTER_PORT="$2"; shift 2;;
    --cuda) CUDA_DEVICES="$2"; shift 2;;
    --run_suffix) RUN_SUFFIX="$2"; shift 2;;
    --dry_run) DRY_RUN=1; shift 1;;
    -h|--help) usage; exit 0;;
    *) echo "[Error] Unknown argument: $1" >&2; usage; exit 1;;
  esac
done

if [[ -z "$MODEL" || -z "$DATA_PATH" ]]; then
  echo "[Error] --model and --data_path are required" >&2
  usage
  exit 1
fi

if [[ -n "$CUDA_DEVICES" ]]; then
  export CUDA_VISIBLE_DEVICES="$CUDA_DEVICES"
fi

[[ -z "$RUN_SUFFIX" ]] && RUN_SUFFIX=$(date +%Y%m%d_%H%M%S)
RUN_DIR="${OUTPUT_BASE}/grid_${TASK}_${RUN_SUFFIX}"

GRID_CMD=( python scripts/grid_search.py \
  --model "$MODEL" \
  --data_path "$DATA_PATH" \
  --task "$TASK" \
  --output_base "$OUTPUT_BASE" \
  --launcher torchrun \
  --nproc_per_node "$GPUS_PER_NODE" \
  --nnodes "$NNODES" \
  --node_rank "$NODE_RANK" \
  --master_addr "$MASTER_ADDR" \
  --master_port "$MASTER_PORT" \
  --run_suffix "$RUN_SUFFIX" )

if [[ -n "$EVAL_PATH" ]]; then GRID_CMD+=( --eval_data_path "$EVAL_PATH" ); fi
if [[ -n "$IMAGE_FOLDER" ]]; then GRID_CMD+=( --image_folder "$IMAGE_FOLDER" ); fi
if [[ -n "$CONFIG_JSON" ]]; then GRID_CMD+=( --config_json "$CONFIG_JSON" ); fi
if [[ -n "$DEEPSPEED_CONFIG" ]]; then GRID_CMD+=( --deepspeed_config "$DEEPSPEED_CONFIG" ); fi
if (( DRY_RUN )); then GRID_CMD+=( --dry_run ); fi

echo "[Grid] Launching: ${GRID_CMD[*]}"
"${GRID_CMD[@]}"

if (( DRY_RUN )); then
  echo "[Grid] Dry run complete (no evaluation parsing)."
  exit 0
fi

if [[ ! -d "$RUN_DIR" ]]; then
  echo "[Error] Expected run directory '$RUN_DIR' not found." >&2
  exit 1
fi

python - <<'PY' "$RUN_DIR"
import json
import sys
from pathlib import Path

run_dir = Path(sys.argv[1])
metrics = []
for subdir in sorted(run_dir.iterdir()):
    if not subdir.is_dir():
        continue
    trainer_state = subdir / "trainer_state.json"
    record = {"name": subdir.name}
    if trainer_state.exists():
        try:
            state = json.loads(trainer_state.read_text(encoding="utf-8"))
            log_history = state.get("log_history", [])
            eval_entries = [entry for entry in log_history if any(key.startswith("eval_") for key in entry)]
            if eval_entries:
                latest = eval_entries[-1]
                for key, value in latest.items():
                    if key.startswith("eval_"):
                        record[key] = value
                if "step" in latest:
                    record["step"] = latest["step"]
        except json.JSONDecodeError:
            record["warning"] = "trainer_state.json parse error"
    else:
        record["warning"] = "missing trainer_state.json"
    metrics.append(record)

out_path = run_dir / "metrics_summary.json"
out_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")

best_acc = None
for item in metrics:
    acc = item.get("eval_accuracy")
    if acc is None:
        continue
    if best_acc is None or acc > best_acc.get("eval_accuracy", float("-inf")):
        best_acc = item

best_loss = None
for item in metrics:
    loss = item.get("eval_loss")
    if loss is None:
        continue
    if best_loss is None or loss < best_loss.get("eval_loss", float("inf")):
        best_loss = item

print("\n[Metrics] Evaluation Summary (saved to", out_path, ")")
for item in metrics:
    parts = [f"name={item['name']}"]
    if "eval_accuracy" in item:
        parts.append(f"acc={item['eval_accuracy']:.4f}")
    if "eval_loss" in item:
        parts.append(f"loss={item['eval_loss']:.4f}")
    if "step" in item:
        parts.append(f"step={item['step']}")
    if "warning" in item:
        parts.append(f"warning={item['warning']}")
    print("  -", ", ".join(parts))

if best_acc:
    print("\n[Best] Highest eval_accuracy:", best_acc['name'], f"(acc={best_acc['eval_accuracy']:.4f})")
if best_loss:
    print("[Best] Lowest eval_loss:", best_loss['name'], f"(loss={best_loss['eval_loss']:.4f})")
PY

echo "[Done] Full artifacts at $RUN_DIR"
