#!/usr/bin/env python3
"""Aggregate the six frozen Stage-1 LoRA conditions."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

LRS = [5e-6, 1e-5, 5e-5]
SEEDS = [42, 43]


def lr_slug(value: float) -> str:
    return f"{value:.0e}".replace("+", "")


def run_name(lr: float, seed: int) -> str:
    return f"lr-{lr_slug(lr)}_seed-{seed}"


def fail(message: str) -> None:
    raise RuntimeError(message)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--results-root",
        type=Path,
        default=Path("results/stage1"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("results/stage1_grid"),
    )
    parser.add_argument(
        "--allow-incomplete",
        action="store_true",
        help="Write partial status instead of failing on missing runs.",
    )
    return parser.parse_args()


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def main() -> int:
    args = parse_args()
    rows: list[dict[str, Any]] = []
    missing: list[str] = []

    for lr in LRS:
        for seed in SEEDS:
            name = run_name(lr, seed)
            summary_path = (
                args.results_root / name / "interpretability_summary.json"
            )
            if not summary_path.exists():
                missing.append(name)
                continue

            summary = read_json(summary_path)
            acquisition = summary["post_pilot_acquisition_guard"]
            forgetting = summary["original_predeclared_forgetting_criterion"]

            rows.append(
                {
                    "run_name": name,
                    "learning_rate": lr,
                    "seed": seed,
                    "status": summary["status"],
                    "id_base_accuracy": acquisition["base_id_accuracy"],
                    "id_max_accuracy": acquisition["max_id_accuracy"],
                    "id_max_step": acquisition["max_id_step"],
                    "id_gain": acquisition["max_id_gain"],
                    "acquisition_pass": acquisition["pass"],
                    "ood_peak_step": forgetting["ood_peak_step"],
                    "ood_peak_accuracy": forgetting["ood_peak_accuracy"],
                    "ood_final_accuracy": forgetting["final_ood_accuracy"],
                    "ood_decline": forgetting["peak_to_final_decline"],
                    "decline_threshold": forgetting["decline_threshold"],
                    "later_below_peak": forgetting[
                        "later_checkpoints_below_peak"
                    ],
                    "original_clear_peak_to_decline": forgetting[
                        "clear_peak_to_decline"
                    ],
                    "acquisition_qualified_clear_peak_to_decline": summary[
                        "acquisition_qualified_clear_peak_to_decline"
                    ],
                }
            )

    if missing and not args.allow_incomplete:
        fail(
            "Stage-1 grid is incomplete. Missing summaries: "
            + ", ".join(missing)
        )

    rows.sort(key=lambda row: (float(row["learning_rate"]), int(row["seed"])))

    original_clear = sum(
        bool(row["original_clear_peak_to_decline"]) for row in rows
    )
    qualified_clear = sum(
        bool(row["acquisition_qualified_clear_peak_to_decline"])
        for row in rows
    )
    acquisition_passes = sum(bool(row["acquisition_pass"]) for row in rows)

    if len(rows) == 6:
        if original_clear >= 2:
            gate = "GO"
        elif original_clear == 1:
            gate = "CONDITIONAL_GO"
        else:
            gate = "NO_LORA_CLEAR_FORGETTING__RUN_CAPACITY_SENTINEL"
    else:
        gate = "INCOMPLETE"

    args.output_dir.mkdir(parents=True, exist_ok=True)

    csv_path = args.output_dir / "stage1_grid_summary.csv"
    if rows:
        with csv_path.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)

    summary = {
        "grid_complete": len(rows) == 6 and not missing,
        "completed_runs": len(rows),
        "missing_runs": missing,
        "original_clear_peak_to_decline_count": original_clear,
        "acquisition_qualified_clear_peak_to_decline_count": qualified_clear,
        "acquisition_pass_count": acquisition_passes,
        "stage1_lora_gate": gate,
        "important_note": (
            "If the completed six-run LoRA grid has zero clear forgetting, "
            "the frozen protocol requires the full-FT capacity sentinel before "
            "a KILL/PIVOT conclusion."
        ),
    }

    json_path = args.output_dir / "stage1_grid_summary.json"
    json_path.write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )

    print("=== Stage-1 six-run grid summary ===")
    print("Completed:", len(rows), "/ 6")
    print("Missing:", missing or "none")
    print("Acquisition passes:", acquisition_passes)
    print("Original clear forgetting:", original_clear)
    print("Acquisition-qualified clear forgetting:", qualified_clear)
    print("Gate:", gate)
    print("CSV:", csv_path)
    print("JSON:", json_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
