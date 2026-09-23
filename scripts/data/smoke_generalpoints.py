#!/usr/bin/env python3
"""Download and validate the GeneralPoints dataset used by Stage 1.

The Hugging Face datasets library downloads data into its cache. This script
also writes a small local manifest containing only deterministic subset indices
and metadata; it does not copy the full dataset into Git.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
from pathlib import Path
from typing import Any

from datasets import Dataset, get_dataset_split_names, load_dataset

DEFAULT_REPO = "Xiaofeng77/gp-l-only-10k"
REQUIRED_SPLITS = ("train", "test_id", "test_face_cards_as_regular")
REQUIRED_COLUMNS = ("data_source", "extra_info", "question")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-id", default=DEFAULT_REPO)
    parser.add_argument("--train-size", type=int, default=4096)
    parser.add_argument("--anchor-size", type=int, default=512)
    parser.add_argument("--seed", type=int, default=20260923)
    parser.add_argument(
        "--manifest-dir",
        type=Path,
        default=Path("data/local/generalpoints_stage1"),
    )
    return parser.parse_args()


def fail(message: str) -> None:
    raise RuntimeError(message)


def validate_columns(split_name: str, dataset: Dataset) -> None:
    missing = [name for name in REQUIRED_COLUMNS if name not in dataset.column_names]
    if missing:
        fail(
            f"{split_name}: missing required columns {missing}; "
            f"found {dataset.column_names}"
        )


def validate_question(example: dict[str, Any], split_name: str) -> None:
    question = example.get("question")
    if not isinstance(question, list) or not question:
        fail(f"{split_name}: expected non-empty chat-message list in 'question'")

    first = question[0]
    if not isinstance(first, dict) or not isinstance(first.get("content"), str):
        fail(f"{split_name}: unexpected question message schema: {first!r}")


def extract_rule_flag(extra_info: Any) -> bool | None:
    if not isinstance(extra_info, dict):
        return None
    value = extra_info.get("treat_face_cards_as_10")
    return value if isinstance(value, bool) else None


def validate_extra_info(example: dict[str, Any], split_name: str) -> None:
    extra = example.get("extra_info")
    if not isinstance(extra, dict):
        fail(f"{split_name}: 'extra_info' is not a dict")

    cards = extra.get("cards")
    if not isinstance(cards, list) or len(cards) != 4:
        fail(f"{split_name}: expected exactly four cards in extra_info.cards")

    target = extra.get("target")
    if target is not None and target != 24:
        fail(f"{split_name}: unexpected target {target!r}; expected 24")


def sample_rule_flags(dataset: Dataset, limit: int = 256) -> set[bool]:
    flags: set[bool] = set()
    for i in range(min(limit, len(dataset))):
        value = extract_rule_flag(dataset[i].get("extra_info"))
        if value is not None:
            flags.add(value)
    return flags


def indices_digest(indices: list[int]) -> str:
    payload = ",".join(str(i) for i in indices).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def main() -> int:
    args = parse_args()

    print(f"Repository: {args.repo_id}")
    split_names = get_dataset_split_names(args.repo_id)
    print("Available splits:", ", ".join(split_names))

    missing_splits = [name for name in REQUIRED_SPLITS if name not in split_names]
    if missing_splits:
        fail(f"missing required Stage-1 splits: {missing_splits}")

    datasets_by_split: dict[str, Dataset] = {}
    for split_name in REQUIRED_SPLITS:
        print(f"Downloading/loading split: {split_name}")
        ds = load_dataset(args.repo_id, split=split_name)
        datasets_by_split[split_name] = ds
        validate_columns(split_name, ds)

        if len(ds) == 0:
            fail(f"{split_name}: split is empty")

        validate_question(ds[0], split_name)
        validate_extra_info(ds[0], split_name)
        print(
            f"  rows={len(ds):,} columns={len(ds.column_names)} "
            f"schema={ds.column_names}"
        )

    train = datasets_by_split["train"]
    required_train_rows = args.train_size + args.anchor_size
    if len(train) < required_train_rows:
        fail(
            f"train split has {len(train)} rows, but Stage 1 requires at least "
            f"{required_train_rows}"
        )

    # Check the key rule-shift metadata on a bounded sample. The official
    # fixed-prompt train/ID data should treat face cards as 10, while the
    # face-cards-as-regular OOD split should not.
    train_flags = sample_rule_flags(train)
    id_flags = sample_rule_flags(datasets_by_split["test_id"])
    ood_flags = sample_rule_flags(datasets_by_split["test_face_cards_as_regular"])

    print("Rule flags:")
    print("  train:", sorted(train_flags))
    print("  test_id:", sorted(id_flags))
    print("  test_face_cards_as_regular:", sorted(ood_flags))

    if train_flags and train_flags != {True}:
        fail(f"train rule metadata is unexpected: {train_flags}")
    if id_flags and id_flags != {True}:
        fail(f"test_id rule metadata is unexpected: {id_flags}")
    if ood_flags and ood_flags != {False}:
        fail(
            "test_face_cards_as_regular rule metadata is unexpected: "
            f"{ood_flags}"
        )

    indices = list(range(len(train)))
    random.Random(args.seed).shuffle(indices)
    sft_indices = indices[: args.train_size]
    anchor_indices = indices[
        args.train_size : args.train_size + args.anchor_size
    ]

    if set(sft_indices) & set(anchor_indices):
        fail("SFT and anchor subsets overlap")

    args.manifest_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = args.manifest_dir / "manifest.json"

    manifest = {
        "repo_id": args.repo_id,
        "available_splits": split_names,
        "required_splits": list(REQUIRED_SPLITS),
        "split_rows": {
            name: len(datasets_by_split[name]) for name in REQUIRED_SPLITS
        },
        "shuffle_seed": args.seed,
        "sft_train_size": args.train_size,
        "anchor_size": args.anchor_size,
        "sft_indices": sft_indices,
        "anchor_indices": anchor_indices,
        "sft_indices_sha256": indices_digest(sft_indices),
        "anchor_indices_sha256": indices_digest(anchor_indices),
        "id_split": "test_id",
        "ood_oracle_split": "test_face_cards_as_regular",
    }

    manifest_path.write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )

    print(f"Manifest written to: {manifest_path}")
    print(
        f"SFT subset: {len(sft_indices)} rows | "
        f"anchor subset: {len(anchor_indices)} rows | overlap=0"
    )
    print("PASS: GeneralPoints Stage-1 data smoke test succeeded.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
