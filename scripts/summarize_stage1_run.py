#!/usr/bin/env python3
"""Summarize one FULL Stage-1 checkpoint trajectory with acquisition sanity."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

EXPECTED_STEPS = [0] + list(range(10, 191, 10)) + [192]


def fail(message: str) -> None:
    raise RuntimeError(message)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--metrics",
        type=Path,
        default=Path(
            "results/stage1/lr-1e-05_seed-42/checkpoint_metrics.jsonl"
        ),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
    )
    parser.add_argument(
        "--min-id-gain",
        type=float,
        default=0.05,
        help=(
            "Post-pilot interpretability guard. This is not part of the "
            "original predeclared forgetting criterion."
        ),
    )
    return parser.parse_args()


def read_rows(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        fail(f"missing metrics: {path}")
    rows = [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    rows.sort(key=lambda row: int(row["optimizer_step"]))
    return rows


def main() -> int:
    args = parse_args()
    rows = read_rows(args.metrics)

    steps = [int(row["optimizer_step"]) for row in rows]
    if steps != EXPECTED_STEPS:
        fail(
            "trajectory is incomplete or unexpected. "
            f"expected={EXPECTED_STEPS}, actual={steps}"
        )

    modes = {str(row.get("evaluation_mode")) for row in rows}
    if modes != {"FULL"}:
        fail(f"only FULL evaluation is allowed; found modes={sorted(modes)}")

    id_counts = {int(row["id_task_n"]) for row in rows}
    ood_counts = {int(row["ood_task_n"]) for row in rows}
    if len(id_counts) != 1 or len(ood_counts) != 1:
        fail(
            f"evaluation population changed: ID={sorted(id_counts)}, "
            f"OOD={sorted(ood_counts)}"
        )

    base_id = float(rows[0]["id_task_accuracy"])
    id_values = [float(row["id_task_accuracy"]) for row in rows]
    id_peak_index = max(range(len(rows)), key=lambda i: id_values[i])
    max_id = id_values[id_peak_index]
    max_id_step = steps[id_peak_index]
    id_gain = max_id - base_id
    acquisition_pass = id_gain >= args.min_id_gain

    ood_values = [float(row["ood_task_accuracy"]) for row in rows]
    peak_index = max(range(len(rows)), key=lambda i: ood_values[i])
    peak_ood = ood_values[peak_index]
    peak_step = steps[peak_index]
    final_ood = ood_values[-1]
    decline = peak_ood - final_ood

    later_below = sum(
        float(row["ood_task_accuracy"]) < peak_ood
        for row in rows[peak_index + 1 :]
    )

    n = int(rows[peak_index]["ood_task_n"])
    se = math.sqrt(max(peak_ood * (1.0 - peak_ood), 0.0) / n)
    decline_threshold = max(0.05, 2.0 * se)

    id_at_ood_peak = float(rows[peak_index]["id_task_accuracy"])
    final_id = float(rows[-1]["id_task_accuracy"])
    synchronized_id_decline = id_at_ood_peak - final_id

    original_clear_peak_decline = (
        peak_index < len(rows) - 1
        and later_below >= 2
        and decline >= decline_threshold
        and synchronized_id_decline < decline_threshold
    )

    acquisition_qualified = (
        acquisition_pass and original_clear_peak_decline
    )

    if not acquisition_pass:
        status = "UNDER_LEARNED_INCONCLUSIVE_FOR_FORGETTING"
    elif original_clear_peak_decline:
        status = "CLEAR_PEAK_TO_DECLINE"
    else:
        status = "NO_CLEAR_PEAK_TO_DECLINE"

    summary = {
        "status": status,
        "post_pilot_acquisition_guard": {
            "added_after_first_run": True,
            "not_part_of_original_predeclared_criterion": True,
            "min_id_gain": args.min_id_gain,
            "base_id_accuracy": base_id,
            "max_id_accuracy": max_id,
            "max_id_step": max_id_step,
            "max_id_gain": id_gain,
            "pass": acquisition_pass,
        },
        "original_predeclared_forgetting_criterion": {
            "ood_peak_step": peak_step,
            "ood_peak_accuracy": peak_ood,
            "final_ood_accuracy": final_ood,
            "peak_to_final_decline": decline,
            "later_checkpoints_below_peak": later_below,
            "ood_n_at_peak": n,
            "binomial_se_at_peak": se,
            "decline_threshold": decline_threshold,
            "id_accuracy_at_ood_peak_step": id_at_ood_peak,
            "final_id_accuracy": final_id,
            "id_decline_from_ood_peak_step": synchronized_id_decline,
            "clear_peak_to_decline": original_clear_peak_decline,
        },
        "acquisition_qualified_clear_peak_to_decline": acquisition_qualified,
        "metrics_path": str(args.metrics),
    }

    output = args.output
    if output is None:
        output = args.metrics.parent / "interpretability_summary.json"

    output.write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )

    print("=== Stage-1 run interpretation ===")
    print("Status:", status)
    print(
        f"ID acquisition: {base_id:.4f} -> {max_id:.4f} "
        f"(gain={id_gain:.4f}, threshold={args.min_id_gain:.4f})"
    )
    print(
        f"OOD peak: step={peak_step}, acc={peak_ood:.4f}; "
        f"final={final_ood:.4f}; decline={decline:.4f}; "
        f"required={decline_threshold:.4f}"
    )
    print(
        "Original clear peak->decline:",
        original_clear_peak_decline,
    )
    print(
        "Acquisition-qualified clear peak->decline:",
        acquisition_qualified,
    )
    print("Wrote:", output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
