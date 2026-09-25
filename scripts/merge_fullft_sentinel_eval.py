#!/usr/bin/env python3
"""Merge and interpret the 8-state full-FT capacity sentinel trajectory."""

from __future__ import annotations

import csv
import json
import math
from pathlib import Path
from typing import Any

RUN_NAME = "qwen2.5-3b_lr-1e-06_seed-42"
EXPECTED_STEPS = [0, 10, 20, 30, 40, 50, 60, 64]
INPUT_ROOTS = [
    Path("results/fullft_sentinel_gpu0"),
    Path("results/fullft_sentinel_gpu1"),
]
OUTPUT_DIR = Path("results/fullft_sentinel") / RUN_NAME


def fail(message: str) -> None:
    raise RuntimeError(message)


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        fail(f"missing file: {path}")
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def main() -> int:
    merged: dict[int, dict[str, Any]] = {}

    for root in INPUT_ROOTS:
        path = root / RUN_NAME / "checkpoint_metrics.jsonl"
        rows = read_jsonl(path)
        print(f"{path}: {len(rows)} rows")
        for row in rows:
            step = int(row["optimizer_step"])
            if row.get("evaluation_mode") != "FULL":
                fail(f"step {step}: expected FULL evaluation")
            if step in merged and merged[step] != row:
                fail(f"conflicting duplicate step {step}")
            merged[step] = row

    actual = sorted(merged)
    if actual != EXPECTED_STEPS:
        fail(f"expected steps {EXPECTED_STEPS}, actual {actual}")

    rows = [merged[step] for step in EXPECTED_STEPS]
    id_counts = {int(row["id_task_n"]) for row in rows}
    ood_counts = {int(row["ood_task_n"]) for row in rows}
    if len(id_counts) != 1 or len(ood_counts) != 1:
        fail(
            f"evaluation population changed: "
            f"ID={sorted(id_counts)}, OOD={sorted(ood_counts)}"
        )

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    jsonl_path = OUTPUT_DIR / "checkpoint_metrics.jsonl"
    with jsonl_path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")

    csv_path = OUTPUT_DIR / "checkpoint_metrics.csv"
    fieldnames = list(rows[0].keys())
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    id_values = [float(row["id_task_accuracy"]) for row in rows]
    base_id = id_values[0]
    id_peak_index = max(range(len(rows)), key=lambda i: id_values[i])
    max_id = id_values[id_peak_index]
    id_gain = max_id - base_id
    acquisition_pass = id_gain >= 0.05

    ood_values = [float(row["ood_task_accuracy"]) for row in rows]
    peak_index = max(range(len(rows)), key=lambda i: ood_values[i])
    peak_ood = ood_values[peak_index]
    peak_step = EXPECTED_STEPS[peak_index]
    final_ood = ood_values[-1]
    decline = peak_ood - final_ood
    later_below = sum(
        float(row["ood_task_accuracy"]) < peak_ood
        for row in rows[peak_index + 1 :]
    )

    n = int(rows[peak_index]["ood_task_n"])
    se = math.sqrt(max(peak_ood * (1.0 - peak_ood), 0.0) / n)
    decline_threshold = max(0.05, 2.0 * se)

    id_at_peak = float(rows[peak_index]["id_task_accuracy"])
    final_id = float(rows[-1]["id_task_accuracy"])
    id_decline = id_at_peak - final_id

    clear_forgetting = (
        peak_index < len(rows) - 1
        and later_below >= 2
        and decline >= decline_threshold
        and id_decline < decline_threshold
    )

    if not acquisition_pass:
        decision = "FULLFT_SENTINEL_UNDER_LEARNED__INCONCLUSIVE"
    elif clear_forgetting:
        decision = "ADAPTATION_CAPACITY_BOUNDARY__CONDITIONAL_GO"
    else:
        decision = "KILL_OR_PIVOT__PHENOMENON_NOT_REPRODUCED"

    summary = {
        "decision": decision,
        "acquisition": {
            "base_id_accuracy": base_id,
            "max_id_accuracy": max_id,
            "max_id_step": EXPECTED_STEPS[id_peak_index],
            "id_gain": id_gain,
            "threshold": 0.05,
            "pass": acquisition_pass,
        },
        "forgetting": {
            "ood_peak_step": peak_step,
            "ood_peak_accuracy": peak_ood,
            "final_ood_accuracy": final_ood,
            "peak_to_final_decline": decline,
            "later_checkpoints_below_peak": later_below,
            "binomial_se_at_peak": se,
            "decline_threshold": decline_threshold,
            "id_accuracy_at_ood_peak_step": id_at_peak,
            "final_id_accuracy": final_id,
            "id_decline_from_ood_peak_step": id_decline,
            "clear_peak_to_decline": clear_forgetting,
        },
        "interpretation": {
            "lora_grid_clear_forgetting_count": 0,
            "if_fullft_clear_and_acquired": (
                "LoRA likely suppresses or shifts the target forgetting "
                "trajectory in this resource-adapted regime."
            ),
            "if_fullft_not_clear_and_acquired": (
                "Target phenomenon was not reproduced under either LoRA or "
                "the predeclared full-FT sentinel; proceed to KILL/PIVOT "
                "analysis rather than drift-selector development."
            ),
        },
    }

    summary_path = OUTPUT_DIR / "capacity_sentinel_summary.json"
    summary_path.write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )

    print("=== Full-FT capacity sentinel summary ===")
    print("Decision:", decision)
    print(
        f"ID acquisition: {base_id:.4f} -> {max_id:.4f} "
        f"(gain={id_gain:.4f})"
    )
    print(
        f"OOD peak: step={peak_step}, acc={peak_ood:.4f}; "
        f"final={final_ood:.4f}; decline={decline:.4f}; "
        f"required={decline_threshold:.4f}"
    )
    print("Clear forgetting:", clear_forgetting)
    print("Summary:", summary_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
