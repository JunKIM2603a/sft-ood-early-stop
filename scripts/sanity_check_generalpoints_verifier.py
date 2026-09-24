#!/usr/bin/env python3
"""Sanity-check the GeneralPoints verifier against official ground-truth answers."""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any

from datasets import load_dataset

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from sft_ood_early_stop.generalpoints import score_response  # noqa: E402


DEFAULT_MANIFEST = Path("data/local/generalpoints_stage1/manifest.json")
DEFAULT_OUTPUT = Path(
    "artifacts/audits/generalpoints_verifier_groundtruth.json"
)


def fail(message: str) -> None:
    raise RuntimeError(message)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--subset",
        choices=["id_validation", "sft", "anchor"],
        default="id_validation",
    )
    parser.add_argument(
        "--max-examples",
        type=int,
        default=0,
        help="0 means all examples in the selected frozen subset.",
    )
    parser.add_argument(
        "--required-pass-rate",
        type=float,
        default=1.0,
        help="Ground-truth answers should pass exactly; default is 1.0.",
    )
    return parser.parse_args()


def read_manifest(path: Path) -> dict[str, Any]:
    if not path.exists():
        fail(
            f"missing manifest: {path}; run "
            "python scripts/data/smoke_generalpoints.py first"
        )
    return json.loads(path.read_text(encoding="utf-8"))


def main() -> int:
    args = parse_args()
    manifest = read_manifest(args.manifest)

    key_by_subset = {
        "id_validation": "id_validation_indices",
        "sft": "sft_indices",
        "anchor": "anchor_indices",
    }
    index_key = key_by_subset[args.subset]
    if index_key not in manifest:
        fail(
            f"manifest has no {index_key}; rerun "
            "python scripts/data/smoke_generalpoints.py"
        )

    indices = list(manifest[index_key])
    if args.max_examples > 0:
        indices = indices[: args.max_examples]
    if not indices:
        fail("selected frozen subset is empty")

    dataset = load_dataset(
        manifest["train_repo_id"],
        split=manifest.get("train_split", "train"),
    ).select(indices)

    if "answer" not in dataset.column_names:
        fail("selected dataset does not contain official answer labels")

    correct = 0
    parsed = 0
    valid = 0
    reasons: Counter[str] = Counter()
    failures: list[dict[str, Any]] = []

    for position, row in enumerate(dataset):
        extra = row.get("extra_info")
        if not isinstance(extra, dict):
            fail(f"row {position}: missing extra_info dict")

        display_cards = extra.get("display_cards")
        target = extra.get("target", 24)
        if not isinstance(display_cards, list) or len(display_cards) != 4:
            fail(f"row {position}: invalid display_cards")

        answer = row.get("answer")
        if not isinstance(answer, str):
            fail(f"row {position}: answer is not a string")

        score = score_response(
            answer,
            [int(value) for value in display_cards],
            int(target),
        )
        correct += int(score.correct)
        parsed += int(score.parsed)
        valid += int(score.formula_valid)
        reasons[score.reason] += 1

        if not score.correct and len(failures) < 50:
            failures.append(
                {
                    "subset_position": position,
                    "dataset_index": int(indices[position]),
                    "display_cards": display_cards,
                    "target": target,
                    "reason": score.reason,
                    "extracted_formula": score.formula,
                    "answer": answer,
                }
            )

    n = len(dataset)
    pass_rate = correct / n
    report = {
        "status": (
            "PASS" if pass_rate >= args.required_pass_rate else "FAIL"
        ),
        "subset": args.subset,
        "n": n,
        "correct": correct,
        "pass_rate": pass_rate,
        "parse_rate": parsed / n,
        "valid_formula_rate": valid / n,
        "required_pass_rate": args.required_pass_rate,
        "reason_counts": dict(sorted(reasons.items())),
        "failure_examples": failures,
        "manifest": str(args.manifest),
        "train_repo_id": manifest["train_repo_id"],
        "index_key": index_key,
    }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )

    print("=== GeneralPoints ground-truth verifier sanity ===")
    print("Subset:", args.subset)
    print("Examples:", n)
    print(f"Correct: {correct}/{n} ({pass_rate:.4%})")
    print(f"Parse rate: {parsed / n:.4%}")
    print(f"Valid formula rate: {valid / n:.4%}")
    print("Reasons:", dict(sorted(reasons.items())))
    print("Report:", args.output)

    if pass_rate < args.required_pass_rate:
        fail(
            f"ground-truth verifier pass rate {pass_rate:.4%} is below "
            f"required {args.required_pass_rate:.4%}; inspect failures before "
            "launching additional scientific runs"
        )

    print("PASS: official ground-truth answers are accepted by the verifier.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
