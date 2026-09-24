#!/usr/bin/env python3
"""Merge two parallel Stage-1 evaluation shards and validate completeness."""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Any

EXPECTED_STEPS = [0] + list(range(10, 191, 10)) + [192]
DEFAULT_RUN_NAME = "lr-1e-05_seed-42"


def fail(message: str) -> None:
    raise RuntimeError(message)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input-roots",
        nargs="+",
        type=Path,
        default=[Path("results/stage1_gpu0"), Path("results/stage1_gpu1")],
    )
    parser.add_argument("--run-name", default=DEFAULT_RUN_NAME)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("results/stage1") / DEFAULT_RUN_NAME,
    )
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        fail(f"missing evaluation file: {path}")

    rows: list[dict[str, Any]] = []
    for line_number, line in enumerate(
        path.read_text(encoding="utf-8").splitlines(),
        start=1,
    ):
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError as exc:
            fail(f"{path}:{line_number}: invalid JSON: {exc}")
        rows.append(row)
    return rows


def write_outputs(output_dir: Path, rows: list[dict[str, Any]]) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)

    jsonl_path = output_dir / "checkpoint_metrics.jsonl"
    csv_path = output_dir / "checkpoint_metrics.csv"

    with jsonl_path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")

    fieldnames = [
        "optimizer_step",
        "id_validation_loss",
        "id_task_accuracy",
        "ood_task_accuracy",
        "id_parse_rate",
        "ood_parse_rate",
        "id_valid_formula_rate",
        "ood_valid_formula_rate",
        "id_task_n",
        "ood_task_n",
        "evaluation_mode",
    ]

    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key) for key in fieldnames})


def summarize_curve(rows: list[dict[str, Any]]) -> dict[str, Any]:
    steps = [int(row["optimizer_step"]) for row in rows]
    ood = [float(row["ood_task_accuracy"]) for row in rows]

    peak_index = max(range(len(rows)), key=lambda i: ood[i])
    peak_step = steps[peak_index]
    peak_value = ood[peak_index]

    later_rows = rows[peak_index + 1 :]
    later_below = [
        row
        for row in later_rows
        if float(row["ood_task_accuracy"]) < peak_value
    ]

    final_value = float(rows[-1]["ood_task_accuracy"])
    decline = peak_value - final_value

    n = int(rows[peak_index]["ood_task_n"])
    if n <= 0:
        fail("ood_task_n must be positive")

    se = math.sqrt(max(peak_value * (1.0 - peak_value), 0.0) / n)
    threshold = max(0.05, 2.0 * se)

    id_at_ood_peak = float(rows[peak_index]["id_task_accuracy"])
    final_id = float(rows[-1]["id_task_accuracy"])
    id_decline = id_at_ood_peak - final_id

    clear_peak_decline = (
        peak_index < len(rows) - 1
        and len(later_below) >= 2
        and decline >= threshold
        and id_decline < threshold
    )

    return {
        "peak_step": peak_step,
        "peak_ood_accuracy": peak_value,
        "final_step": steps[-1],
        "final_ood_accuracy": final_value,
        "peak_to_final_decline": decline,
        "ood_n_at_peak": n,
        "binomial_se_at_peak": se,
        "clear_decline_threshold": threshold,
        "later_checkpoints_below_peak": len(later_below),
        "id_accuracy_at_ood_peak_step": id_at_ood_peak,
        "final_id_accuracy": final_id,
        "id_decline_from_ood_peak_step": id_decline,
        "clear_peak_to_decline_by_predeclared_rule": clear_peak_decline,
    }


def main() -> int:
    args = parse_args()

    merged: dict[int, dict[str, Any]] = {}
    source_by_step: dict[int, str] = {}

    print("=== Merge Stage-1 parallel evaluation ===")

    for root in args.input_roots:
        path = root / args.run_name / "checkpoint_metrics.jsonl"
        rows = read_jsonl(path)
        print(f"{path}: {len(rows)} rows")

        for row in rows:
            step = int(row["optimizer_step"])
            mode = str(row.get("evaluation_mode", "UNKNOWN"))

            if mode != "FULL":
                fail(
                    f"step {step} from {path} has evaluation_mode={mode}; "
                    "only FULL evaluation can enter the scientific trajectory"
                )

            if step in merged:
                if merged[step] == row:
                    print(
                        f"  duplicate identical step {step}; keeping first "
                        f"({source_by_step[step]})"
                    )
                    continue
                fail(
                    f"conflicting duplicate step {step}: "
                    f"{source_by_step[step]} vs {path}"
                )

            merged[step] = row
            source_by_step[step] = str(path)

    actual_steps = sorted(merged)
    missing = [step for step in EXPECTED_STEPS if step not in merged]
    unexpected = [step for step in actual_steps if step not in EXPECTED_STEPS]

    print("Expected steps:", EXPECTED_STEPS)
    print("Actual steps:  ", actual_steps)

    if missing:
        fail(f"missing checkpoint evaluation steps: {missing}")
    if unexpected:
        fail(f"unexpected checkpoint steps: {unexpected}")
    if len(actual_steps) != 21:
        fail(f"expected 21 unique checkpoints, found {len(actual_steps)}")

    rows = [merged[step] for step in EXPECTED_STEPS]

    id_counts = {int(row["id_task_n"]) for row in rows}
    ood_counts = {int(row["ood_task_n"]) for row in rows}
    if len(id_counts) != 1 or len(ood_counts) != 1:
        fail(
            "evaluation population changed across checkpoints: "
            f"ID n={sorted(id_counts)}, OOD n={sorted(ood_counts)}"
        )

    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        if not args.overwrite:
            fail(
                f"output directory is non-empty: {args.output_dir}. "
                "Use --overwrite after confirming old contents are disposable."
            )

    write_outputs(args.output_dir, rows)

    summary = summarize_curve(rows)
    summary_path = args.output_dir / "phenomenon_summary.json"
    summary_path.write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )

    print("\nPASS: merged 21 FULL checkpoint evaluations.")
    print("Output:", args.output_dir / "checkpoint_metrics.csv")
    print("Phenomenon summary:", summary_path)
    print(
        "OOD peak: step={peak_step}, acc={peak_ood_accuracy:.4f} | "
        "final acc={final_ood_accuracy:.4f} | "
        "decline={peak_to_final_decline:.4f}".format(**summary)
    )
    print(
        "Clear peak->decline by predeclared rule:",
        summary["clear_peak_to_decline_by_predeclared_rule"],
    )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
