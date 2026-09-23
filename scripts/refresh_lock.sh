#!/usr/bin/env bash
set -euo pipefail

if [[ "${CONDA_DEFAULT_ENV:-}" != "sft-ood-es" ]]; then
  echo "ERROR: activate the project environment first:"
  echo "  conda activate sft-ood-es"
  exit 1
fi

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
OUT="${ROOT_DIR}/requirements.freeze.txt"

python -m pip freeze --all | LC_ALL=C sort > "${OUT}"

echo "Wrote exact environment snapshot:"
echo "  ${OUT}"
echo
echo "Review it before committing. requirements.lock.txt remains the curated"
echo "bootstrap lock; requirements.freeze.txt records the exact transitive state."
