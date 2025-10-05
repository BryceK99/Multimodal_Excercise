#!/bin/bash
# 一键执行：按预设跑 sampling 与 beam，自动生成答案、计算 CHAIR，并输出汇总
#
# 可选环境变量：
#   MODEL_NAME_OR_PATH (默认 ./weights)
#   DATA_PATH          (默认 data/objhal_bench.jsonl)
#   COCO_PATH          (默认 ./coco_annotations)
#   CHAIR_CACHE        (默认 ./chair_300.pkl)
#   OUT_BASE           (默认 ./runs)
#   OUT_DIR            (默认 "$OUT_BASE/$(date +%Y%m%d_%H%M%S)")
#   MODES              (默认 sampling,beam)  也可设为 "sampling" 或 "beam,sampling,greedy"
#   MAX_SAMPLES        (默认空，表示全量；也可设为 100 先小样本试跑)
#   MAX_NEW_TOKENS_SAMPLING (默认 160)
#   MAX_NEW_TOKENS_BEAM     (默认 160)
#   REP_PEN_SAMPLING   (默认 1.08)
#   REP_PEN_BEAM       (默认 1.10)
#
# 用法：
#   bash auto_run_chair.sh
#   MAX_SAMPLES=100 bash auto_run_chair.sh
#   MODES="sampling,beam,greedy" MAX_NEW_TOKENS_SAMPLING=180 bash auto_run_chair.sh

set -euo pipefail

export PYTHONPATH=${PYTHONPATH}:"$(realpath .)"

MODEL_NAME_OR_PATH=${MODEL_NAME_OR_PATH:-"./weights"}
DATA_PATH=${DATA_PATH:-"data/objhal_bench.jsonl"}
COCO_PATH=${COCO_PATH:-"./coco_annotations"}
CHAIR_CACHE=${CHAIR_CACHE:-"./chair_300.pkl"}
OUT_BASE=${OUT_BASE:-"./runs"}

timestamp=$(date +%Y%m%d_%H%M%S)
OUT_DIR=${OUT_DIR:-"${OUT_BASE}/${timestamp}"}
MODES=${MODES:-"sampling,beam"}

MAX_NEW_TOKENS_SAMPLING=${MAX_NEW_TOKENS_SAMPLING:-160}
MAX_NEW_TOKENS_BEAM=${MAX_NEW_TOKENS_BEAM:-160}
REP_PEN_SAMPLING=${REP_PEN_SAMPLING:-1.08}
REP_PEN_BEAM=${REP_PEN_BEAM:-1.10}
MAX_SAMPLES=25

mkdir -p "${OUT_DIR}"

echo "Output dir: ${OUT_DIR}"

# 先做必要文件检查
if [[ ! -d "${MODEL_NAME_OR_PATH}" ]]; then
  echo "[ERR] MODEL_NAME_OR_PATH not found: ${MODEL_NAME_OR_PATH}" >&2
  exit 1
fi
if [[ ! -f "${DATA_PATH}" ]]; then
  echo "[ERR] DATA_PATH not found: ${DATA_PATH}" >&2
  exit 1
fi
if [[ ! -d "${COCO_PATH}" ]]; then
  echo "[ERR] COCO_PATH not found: ${COCO_PATH}" >&2
  exit 1
fi
if [[ ! -f "${CHAIR_CACHE}" ]]; then
  echo "[WARN] CHAIR cache not found: ${CHAIR_CACHE}. 将在首次评测时自动构建缓存（较慢）。"
fi

function run_one() {
  local mode="$1"
  local save_path="${OUT_DIR}/objhal_bench_answer_${mode}.jsonl"
  local gen_log="${OUT_DIR}/gen_${mode}.log"
  local eval_json="${OUT_DIR}/eval-chair-300_${mode}.json"
  local chair_log="${OUT_DIR}/chair_${mode}.log"

  # 将必要参数传给 eval.sh
  export MODEL_NAME_OR_PATH
  export DATA_PATH
  export SAVE_PATH="${save_path}"
  if [[ -n "${MAX_SAMPLES}" ]]; then
    export MAX_SAMPLES
  fi

  if [[ "${mode}" == "sampling" ]]; then
    export MAX_NEW_TOKENS="${MAX_NEW_TOKENS_SAMPLING}"
    export REP_PEN="${REP_PEN_SAMPLING}"
  else
    export MAX_NEW_TOKENS="${MAX_NEW_TOKENS_BEAM}"
    export REP_PEN="${REP_PEN_BEAM}"
  fi

  echo "\n>>> [${mode}] Generating answers -> ${save_path}"
  if ! bash ./eval.sh "${mode}" | tee "${gen_log}"; then
    echo "[ERR] Generation failed for mode=${mode}. 查看日志: ${gen_log}" >&2
    return 1
  fi

  echo ">>> [${mode}] Running CHAIR evaluation"
  if ! python eval/chair.py \
      --coco_path "${COCO_PATH}" \
      --cache "${CHAIR_CACHE}" \
      --cap_file "${save_path}" \
      --save_path "${eval_json}" \
      --caption_key answer | tee "${chair_log}"; then
    echo "[ERR] CHAIR evaluation failed for mode=${mode}. 查看日志: ${chair_log}" >&2
    return 1
  fi

  # 解析指标（chair.py 会打印百分数数值）
  local cs ci rc ln
  cs=$(grep -E '^CHAIRs\s*:' "${chair_log}" | awk -F': ' '{print $2}')
  ci=$(grep -E '^CHAIRi\s*:' "${chair_log}" | awk -F': ' '{print $2}')
  rc=$(grep -E '^Recall\s*:' "${chair_log}" | awk -F': ' '{print $2}')
  ln=$(grep -E '^Len\s*:' "${chair_log}" | awk -F': ' '{print $2}')
  echo "${mode},${cs},${ci},${rc},${ln}" >> "${OUT_DIR}/summary.csv"
}

# 初始化汇总表
echo "mode,CHAIRs(%),CHAIRi(%),Recall(%),Len(x0.01 tokens)" > "${OUT_DIR}/summary.csv"

# 执行预设模式列表
IFS=',' read -ra MODES_ARR <<< "${MODES}"
for m in "${MODES_ARR[@]}"; do
  run_one "${m}"
done

echo
echo "===== Summary ====="
if command -v column >/dev/null 2>&1; then
  column -s, -t "${OUT_DIR}/summary.csv" || cat "${OUT_DIR}/summary.csv"
else
  cat "${OUT_DIR}/summary.csv"
fi
echo "Outputs saved to: ${OUT_DIR}"
