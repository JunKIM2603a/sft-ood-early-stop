#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT_DIR}"

RUN_NAME="qwen2.5-7b_fulltrain_lr-1e-06_seed-42"
RUN_DIR="checkpoints/modelscale_p2/${RUN_NAME}"
CONFIG="configs/modelscale_sentinel_7b_fulltrain.yaml"
MANIFEST="data/local/generalpoints_exposure_p1/manifest.json"

if [[ ! -f "${RUN_DIR}/checkpoints.jsonl" ]]; then
  echo "ERROR: missing ${RUN_DIR}/checkpoints.jsonl"
  exit 1
fi

readarray -t SHARDS < <(
python - "${RUN_DIR}/checkpoints.jsonl" <<'PY'
import json
import sys
from pathlib import Path
rows = [
    json.loads(line)
    for line in Path(sys.argv[1]).read_text(encoding="utf-8").splitlines()
    if line.strip()
]
steps = [int(row["optimizer_step"]) for row in rows]
print(",".join(map(str, steps[0::2])))
print(",".join(map(str, steps[1::2])))
PY
)

STEPS0="${SHARDS[0]}"
STEPS1="${SHARDS[1]}"

echo "GPU0 steps: ${STEPS0}"
echo "GPU1 steps: ${STEPS1}"

mkdir -p artifacts/logs/modelscale_p2_eval

CUDA_VISIBLE_DEVICES=0 python scripts/evaluate_stage1.py   --config "${CONFIG}"   --manifest "${MANIFEST}"   --run-dir "${RUN_DIR}"   --steps "${STEPS0}"   --skip-id-validation-loss   --eval-batch-size 16   --loss-batch-size 8   --output-root results/modelscale_p2_gpu0   --raw-root results/raw/modelscale_p2_gpu0   > artifacts/logs/modelscale_p2_eval/gpu0.log 2>&1 &
PID0=$!

CUDA_VISIBLE_DEVICES=1 python scripts/evaluate_stage1.py   --config "${CONFIG}"   --manifest "${MANIFEST}"   --run-dir "${RUN_DIR}"   --steps "${STEPS1}"   --skip-id-validation-loss   --eval-batch-size 16   --loss-batch-size 8   --output-root results/modelscale_p2_gpu1   --raw-root results/raw/modelscale_p2_gpu1   > artifacts/logs/modelscale_p2_eval/gpu1.log 2>&1 &
PID1=$!

set +e
wait "${PID0}"
S0=$?
wait "${PID1}"
S1=$?
set -e

echo "GPU0 exit: ${S0}"
echo "GPU1 exit: ${S1}"

if [[ "${S0}" -ne 0 || "${S1}" -ne 0 ]]; then
  echo "ERROR: P2 evaluation failed."
  exit 1
fi

python scripts/merge_modelscale_p2_eval.py
