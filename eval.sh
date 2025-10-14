#!/bin/bash
set -euo pipefail

export PYTHONPATH="${PYTHONPATH:+$PYTHONPATH:}$(realpath .)"
export CUDA_VISIBLE_DEVICES=0

# MODEL_NAME_OR_PATH=${MODEL_NAME_OR_PATH:-"HaoyeZhang/MLLM_Excercise_Model"}
MODEL_NAME_OR_PATH=${MODEL_NAME_OR_PATH:-"/root/Multimodal_Excercise/outputs/sft/checkpoint-600"}
DATA_PATH=${DATA_PATH:-"data/objhal_bench.jsonl"}

# 传入解码方式（beam | sampling | greedy），默认 beam
DECODING=sampling
SAVE_PATH=${SAVE_PATH:-"data/tmp.jsonl"}

# 可选环境变量控制
MAX_SAMPLES=${MAX_SAMPLES:-}
MAX_NEW_TOKENS=${MAX_NEW_TOKENS:-128}
REP_PEN=${REP_PEN:-1.05}
TEMP=${TEMP:-0.7}
TOP_P=${TOP_P:-0.8}
TOP_K=${TOP_K:-100}
NUM_BEAMS=${NUM_BEAMS:-3}

EXTRA_ARGS=(--max-new-tokens ${MAX_NEW_TOKENS} --repetition-penalty ${REP_PEN})
if [[ -n "${MAX_SAMPLES}" ]]; then
  EXTRA_ARGS+=(--max-samples ${MAX_SAMPLES})
fi

case "$DECODING" in
  sampling)
    python ./eval/model_eval.py \
      --model-name-or-path "$MODEL_NAME_OR_PATH" \
      --question-file "$DATA_PATH" \
      --answers-file "$SAVE_PATH" \
      --decoding sampling \
      --temperature ${TEMP} --top-p ${TOP_P} --top-k ${TOP_K} \
      "${EXTRA_ARGS[@]}"
    ;;
  greedy)
    python ./eval/model_eval.py \
      --model-name-or-path "$MODEL_NAME_OR_PATH" \
      --question-file "$DATA_PATH" \
      --answers-file "$SAVE_PATH" \
      --decoding greedy \
      "${EXTRA_ARGS[@]}"
    ;;
  beam|*)
    python ./eval/model_eval.py \
      --model-name-or-path "$MODEL_NAME_OR_PATH" \
      --question-file "$DATA_PATH" \
      --answers-file "$SAVE_PATH" \
      --decoding beam \
      --num-beams ${NUM_BEAMS} \
      "${EXTRA_ARGS[@]}"
    ;;
esac
