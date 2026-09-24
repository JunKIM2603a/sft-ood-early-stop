#!/usr/bin/env bash
set -uo pipefail

if [[ "$#" -lt 4 || "$#" -gt 6 ]]; then
  echo "Usage: $0 <lr_gpu0> <seed_gpu0> <lr_gpu1> <seed_gpu1> [eval_batch] [loss_batch]"
  exit 2
fi

LR0="$1"
SEED0="$2"
LR1="$3"
SEED1="$4"
EVAL_BATCH="${5:-32}"
LOSS_BATCH="${6:-16}"

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT_DIR}"

LOG_DIR="artifacts/logs/stage1_eval"
mkdir -p "${LOG_DIR}"

slug() {
  python - "$1" <<'PY'
import sys
print(f"{float(sys.argv[1]):.0e}".replace("+", ""))
PY
}

RUN0="lr-$(slug "${LR0}")_seed-${SEED0}"
RUN1="lr-$(slug "${LR1}")_seed-${SEED1}"

echo "Launching two FULL Stage-1 evaluations"
echo "GPU 0 -> ${RUN0}"
echo "GPU 1 -> ${RUN1}"

CUDA_VISIBLE_DEVICES=0 python scripts/evaluate_stage1.py   --run-dir "checkpoints/stage1/${RUN0}"   --steps all   --eval-batch-size "${EVAL_BATCH}"   --loss-batch-size "${LOSS_BATCH}"   > "${LOG_DIR}/${RUN0}.log" 2>&1 &
PID0=$!

CUDA_VISIBLE_DEVICES=1 python scripts/evaluate_stage1.py   --run-dir "checkpoints/stage1/${RUN1}"   --steps all   --eval-batch-size "${EVAL_BATCH}"   --loss-batch-size "${LOSS_BATCH}"   > "${LOG_DIR}/${RUN1}.log" 2>&1 &
PID1=$!

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

echo "PASS: both evaluations completed and were summarized."
