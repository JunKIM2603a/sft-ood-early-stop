#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT_DIR}"

RUN_NAME="qwen2.5-3b_lr-1e-06_seed-42"
RUN_DIR="checkpoints/fullft_sentinel/${RUN_NAME}"

if [[ ! -f "${RUN_DIR}/checkpoints.jsonl" ]]; then
  echo "ERROR: missing ${RUN_DIR}/checkpoints.jsonl"
  exit 1
fi

mkdir -p artifacts/logs/fullft_eval

CUDA_VISIBLE_DEVICES=0 python scripts/evaluate_stage1.py   --run-dir "${RUN_DIR}"   --steps 0,20,40,60   --eval-batch-size 32   --loss-batch-size 16   --output-root results/fullft_sentinel_gpu0   --raw-root results/raw/fullft_sentinel_gpu0   > artifacts/logs/fullft_eval/gpu0.log 2>&1 &
PID0=$!

CUDA_VISIBLE_DEVICES=1 python scripts/evaluate_stage1.py   --run-dir "${RUN_DIR}"   --steps 10,30,50,64   --eval-batch-size 32   --loss-batch-size 16   --output-root results/fullft_sentinel_gpu1   --raw-root results/raw/fullft_sentinel_gpu1   > artifacts/logs/fullft_eval/gpu1.log 2>&1 &
PID1=$!

echo "GPU0 eval PID: ${PID0}"
echo "GPU1 eval PID: ${PID1}"

set +e
wait "${PID0}"
S0=$?
wait "${PID1}"
S1=$?
set -e

echo "GPU0 exit: ${S0}"
echo "GPU1 exit: ${S1}"

if [[ "${S0}" -ne 0 || "${S1}" -ne 0 ]]; then
  echo "ERROR: full-FT sentinel evaluation failed."
  echo "Inspect artifacts/logs/fullft_eval/gpu0.log and gpu1.log"
  exit 1
fi

python scripts/merge_fullft_sentinel_eval.py

echo "PASS: full-FT sentinel evaluation and merge completed."
