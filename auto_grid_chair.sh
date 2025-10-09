#!/bin/bash
# 批量网格：对 sampling 与 beam 各跑 5 组参数，输出多个 runs 子目录，并生成总汇总
set -euo pipefail

export PYTHONPATH="$(realpath .)"
export CUDA_VISIBLE_DEVICES=1

MODEL_NAME_OR_PATH=${MODEL_NAME_OR_PATH:-"HaoyeZhang/MLLM_Excercise_Model"}
DATA_PATH=${DATA_PATH:-"data/objhal_bench.jsonl"}
COCO_PATH=${COCO_PATH:-"./coco_annotations"}
CHAIR_CACHE=${CHAIR_CACHE:-"./chair_300.pkl"}
OUT_BASE=${OUT_BASE:-"./runs_grid"}
MAX_SAMPLES=${MAX_SAMPLES:-300}

mkdir -p "${OUT_BASE}"
master_summary="${OUT_BASE}/summary.csv"
echo "run,mode,temperature,top_p,top_k,num_beams,max_new_tokens,rep_pen,CHAIRs(%),CHAIRi(%),Recall(%),Len(x0.01 tokens)" > "${master_summary}"

function one_mode() {
  local mode="$1"
  local idx="$2"
  local temp="$3"
  local top_p="$4"
  local top_k="$5"
  local num_beams="$6"   # 对 sampling 可忽略
  local max_new="$7"
  local rep_pen="$8"

  # 独立输出目录
  local out_dir="${OUT_BASE}/run_${mode}_${idx}_$(date +%H%M%S)"
  mkdir -p "${out_dir}"

  export MODEL_NAME_OR_PATH DATA_PATH COCO_PATH CHAIR_CACHE
  export OUT_DIR="${out_dir}"
  export MAX_SAMPLES="${MAX_SAMPLES}"
  export MAX_NEW_TOKENS="${max_new}"
  export REP_PEN="${rep_pen}"
  export TEMP="${temp}"
  export TOP_P="${top_p}"
  export TOP_K="${top_k}"
  export NUM_BEAMS="${num_beams}"

  # 直接用 eval.sh 触发指定模式，避免重复跑另一模式
  export SAVE_PATH="${out_dir}/objhal_bench_answer_${mode}.jsonl"
  if ! bash ./eval.sh "${mode}" | tee "${out_dir}/gen_${mode}.log"; then
    echo "[ERR] generation failed: ${mode}-${idx}" >&2
    return 1
  fi

  # 评测
  if ! python eval/chair.py \
      --coco_path "${COCO_PATH}" \
      --cache "${CHAIR_CACHE}" \
      --cap_file "${SAVE_PATH}" \
      --save_path "${out_dir}/eval-chair-300_${mode}.json" \
      --caption_key answer | tee "${out_dir}/chair_${mode}.log"; then
    echo "[ERR] chair failed: ${mode}-${idx}" >&2
    return 1
  fi

  local cs ci rc ln
  cs=$(grep -E '^CHAIRs\s*:' "${out_dir}/chair_${mode}.log" | awk -F': ' '{print $2}')
  ci=$(grep -E '^CHAIRi\s*:' "${out_dir}/chair_${mode}.log" | awk -F': ' '{print $2}')
  rc=$(grep -E '^Recall\s*:' "${out_dir}/chair_${mode}.log" | awk -F': ' '{print $2}')
  ln=$(grep -E '^Len\s*:' "${out_dir}/chair_${mode}.log" | awk -F': ' '{print $2}')

  echo "${out_dir},${mode},${temp},${top_p},${top_k},${num_beams},${max_new},${rep_pen},${cs},${ci},${rc},${ln}" >> "${master_summary}"
}

echo "=== Sampling 5 组 ==="
# 5组：温度/TopP/TopK/长度/重复惩罚的组合
# one_mode sampling 1 0.70 0.85 80  0  160 1.06
# one_mode sampling 2 0.75 0.88 80  0  160 1.08
one_mode sampling 3 0.80 0.90 60  0  170 1.06
# one_mode sampling 4 0.75 0.92 100 0  180 1.05
# one_mode sampling 5 0.72 0.87 80  0  150 1.10

echo "=== Beam 5 组 ==="
# one_mode beam 1  0    0    0   3  160 1.08
# one_mode beam 2  0    0    0   3  170 1.10
# one_mode beam 3  0    0    0   4  160 1.12
# one_mode beam 4  0    0    0   5  160 1.10
# one_mode beam 5  0    0    0   3  180 1.08

echo
echo "===== Master Summary ====="
if command -v column >/dev/null 2>&1; then
  column -s, -t "${master_summary}" || cat "${master_summary}"
else
  cat "${master_summary}"
fi
