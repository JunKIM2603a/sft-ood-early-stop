#!/usr/bin/env python3
"""Merge and interpret the P2 7B/full-train model-scale trajectory."""

from __future__ import annotations

import csv
import json
import math
from pathlib import Path
from typing import Any

RUN_NAME = "qwen2.5-7b_fulltrain_lr-1e-06_seed-42"
RUN_DIR = Path("checkpoints/modelscale_p2") / RUN_NAME
INPUT_ROOTS = [
    Path("results/modelscale_p2_gpu0"),
    Path("results/modelscale_p2_gpu1"),
]
OUTPUT_DIR = Path("results/modelscale_p2") / RUN_NAME


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
    index_rows = read_jsonl(RUN_DIR / "checkpoints.jsonl")
    expected = [int(row["optimizer_step"]) for row in index_rows]
    merged: dict[int, dict[str, Any]] = {}

    for root in INPUT_ROOTS:
        path = root / RUN_NAME / "checkpoint_metrics.jsonl"
        for row in read_jsonl(path):
            step = int(row["optimizer_step"])
            if row.get("evaluation_mode") != "FULL":
                fail(f"step {step}: expected FULL evaluation")
            if step in merged and merged[step] != row:
                fail(f"conflicting duplicate step {step}")
            merged[step] = row

    actual = sorted(merged)
    if actual != expected:
        fail(f"expected steps {expected}, actual {actual}")

    rows = [merged[step] for step in expected]
    if any(row.get("id_validation_loss") is not None for row in rows):
        fail("P2 must not reuse the contaminated ID-loss holdout")

    id_values = [float(row["id_task_accuracy"]) for row in rows]
    ood_values = [float(row["ood_task_accuracy"]) for row in rows]

    base_id = id_values[0]
    id_peak_index = max(range(len(rows)), key=lambda i: id_values[i])
    max_id = id_values[id_peak_index]
    id_gain = max_id - base_id
    acquisition_pass = id_gain >= 0.05

    peak_index = max(range(len(rows)), key=lambda i: ood_values[i])
    peak_ood = ood_values[peak_index]
    peak_step = expected[peak_index]
    final_ood = ood_values[-1]
    decline = peak_ood - final_ood
    later_below = sum(value < peak_ood for value in ood_values[peak_index + 1 :])

    n = int(rows[peak_index]["ood_task_n"])
    se = math.sqrt(max(peak_ood * (1.0 - peak_ood), 0.0) / n)
    threshold = max(0.05, 2.0 * se)

    id_at_peak = id_values[peak_index]
    final_id = id_values[-1]
    id_decline = id_at_peak - final_id

    clear_forgetting = (
        peak_index < len(rows) - 1
        and later_below >= 2
        and decline >= threshold
        and id_decline < threshold
    )

    if clear_forgetting and acquisition_pass:
        decision = "MODEL_SCALE_BOUNDARY_CANDIDATE__REPLICATE_SEED43"
    elif acquisition_pass:
        decision = "NEAR_OFFICIAL_RECIPE_NEGATIVE__PROTOCOL_ALIGNMENT_CONTROL"
    else:
        decision = "7B_UNDER_LEARNED__PROTOCOL_ALIGNMENT_CONTROL"

    protocol_path = Path("artifacts/audits/gp_protocol_alignment.json")
    protocol = (
        json.loads(protocol_path.read_text(encoding="utf-8"))
        if protocol_path.exists()
        else None
    )
    p1_path = Path(
        "results/exposure_p1/"
        "qwen2.5-3b_fulltrain_lr-1e-06_seed-42/"
        "exposure_boundary_summary.json"
    )
    p1 = (
        json.loads(p1_path.read_text(encoding="utf-8"))
        if p1_path.exists()
        else None
    )

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    with (OUTPUT_DIR / "checkpoint_metrics.jsonl").open(
        "w", encoding="utf-8"
    ) as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")

    csv_path = OUTPUT_DIR / "checkpoint_metrics.csv"
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    summary = {
        "stage": "P2_MODEL_SCALE_BOUNDARY",
        "decision": decision,
        "acquisition": {
            "base_id_accuracy": base_id,
            "max_id_accuracy": max_id,
            "max_id_step": expected[id_peak_index],
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
            "decline_threshold": threshold,
            "id_accuracy_at_ood_peak_step": id_at_peak,
            "final_id_accuracy": final_id,
            "id_decline_from_ood_peak_step": id_decline,
            "clear_peak_to_decline": clear_forgetting,
        },
        "comparison_to_p1_3b_fulltrain": p1,
        "protocol_alignment_audit": protocol,
        "interpretation": {
            "if_clear": (
                "Model scale is a candidate boundary condition under the "
                "current 10K protocol. Replicate seed 43 before reopening "
                "drift-selector development."
            ),
            "if_not_clear": (
                "The target trajectory remains absent at 7B under the current "
                "10K protocol. Because the public Jin et al. script points to "
                "a different SFTvsRL data path, perform an original-protocol "
                "alignment control before declaring the published phenomenon "
                "non-reproducible."
            ),
        },
    }

    summary_path = OUTPUT_DIR / "modelscale_boundary_summary.json"
    summary_path.write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )

    print("=== P2 model-scale summary ===")
    print("Decision:", decision)
    print(
        f"ID acquisition: {base_id:.4f} -> {max_id:.4f} "
        f"(gain={id_gain:.4f})"
    )
    print(
        f"OOD peak: step={peak_step}, acc={peak_ood:.4f}; "
        f"final={final_ood:.4f}; decline={decline:.4f}; "
        f"required={threshold:.4f}"
    )
    print("Clear forgetting:", clear_forgetting)
    print("Summary:", summary_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
