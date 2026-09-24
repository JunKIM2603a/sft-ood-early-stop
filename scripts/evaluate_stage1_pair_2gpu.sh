#!/usr/bin/env bash
set -uo pipefail

LR="${1:-5e-5}"
SEED0="${2:-42}"
SEED1="${3:-43}"
EVAL_BATCH="${4:-32}"
LOSS_BATCH="${5:-16}"

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT_DIR}"

LR_SLUG="$(python - <<PY
print(f"{float('${LR}'):.0e}".replace("+",""))
PY
)"
RUN0="lr-${LR_SLUG}_seed-${SEED0}"
RUN1="lr-${LR_SLUG}_seed-${SEED1}"

LOG_DIR="artifacts/logs/stage1_eval"
mkdir -p "${LOG_DIR}"

echo "Launching full checkpoint evaluation on two GPUs"
echo "GPU 0 -> ${RUN0}"
echo "GPU 1 -> ${RUN1}"

CUDA_VISIBLE_DEVICES=0 python scripts/evaluate_stage1.py   --run-dir "checkpoints/stage1/${RUN0}"   --steps all   --eval-batch-size "${EVAL_BATCH}"   --loss-batch-size "${LOSS_BATCH}"   > "${LOG_DIR}/${RUN0}.log" 2>&1 &
PID0=$!

CUDA_VISIBLE_DEVICES=1 python scripts/evaluate_stage1.py   --run-dir "checkpoints/stage1/${RUN1}"   --steps all   --eval-batch-size "${EVAL_BATCH}"   --loss-batch-size "${LOSS_BATCH}"   > "${LOG_DIR}/${RUN1}.log" 2>&1 &
PID1=$!

echo "PID GPU0: ${PID0}"
echo "PID GPU1: ${PID1}"

set +e
wait "${PID0}"
STATUS0=$?
wait "${PID1}"
STATUS1=$?
set -e

echo "GPU0 eval exit status: ${STATUS0}"
echo "GPU1 eval exit status: ${STATUS1}"

if [[ "${STATUS0}" -ne 0 || "${STATUS1}" -ne 0 ]]; then
  echo "ERROR: at least one evaluation failed. Inspect ${LOG_DIR}."
  exit 1
fi

python scripts/summarize_stage1_run.py   --metrics "results/stage1/${RUN0}/checkpoint_metrics.jsonl"
python scripts/summarize_stage1_run.py   --metrics "results/stage1/${RUN1}/checkpoint_metrics.jsonl"

echo "PASS: both Stage-1 evaluations completed and were summarized."
