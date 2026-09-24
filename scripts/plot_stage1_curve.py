#!/usr/bin/env python3
"""Plot the first Stage-1 ID/OOD checkpoint trajectory."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib.pyplot as plt


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--metrics",
        type=Path,
        default=Path(
            "results/stage1/lr-1e-05_seed-42/checkpoint_metrics.csv"
        ),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(
            "results/stage1/lr-1e-05_seed-42/id_ood_curve.png"
        ),
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    if not args.metrics.exists():
        raise FileNotFoundError(args.metrics)

    rows = []
    with args.metrics.open(encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        rows.extend(reader)

    if not rows:
        raise RuntimeError("metrics CSV is empty")

    steps = [int(row["optimizer_step"]) for row in rows]
    id_acc = [float(row["id_task_accuracy"]) for row in rows]
    ood_acc = [float(row["ood_task_accuracy"]) for row in rows]

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(steps, id_acc, marker="o", label="ID task accuracy")
    ax.plot(steps, ood_acc, marker="o", label="OOD task accuracy")
    ax.set_xlabel("SFT optimizer step")
    ax.set_ylabel("Task accuracy")
    ax.set_ylim(0.0, 1.0)
    ax.grid(True, alpha=0.25)
    ax.legend()
    fig.tight_layout()

    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, dpi=180)
    plt.close(fig)

    print("Wrote:", args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
