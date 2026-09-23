#!/usr/bin/env python3
"""Download and validate the GeneralPoints data used by Stage 1.

Stage 1 intentionally uses two official Hugging Face repositories:
- SFT train: Xiaofeng77/answer-only-gp-l-only-10k
- Evaluation: Xiaofeng77/gp-l-only-10k

The evaluation repository currently exposes `test` rather than `test_id`.
This script supports both names, but verifies the actual rule semantics instead
of trusting the split name.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
from pathlib import Path
from typing import Any

from datasets import Dataset, get_dataset_split_names, load_dataset

DEFAULT_TRAIN_REPO = "Xiaofeng77/answer-only-gp-l-only-10k"
DEFAULT_EVAL_REPO = "Xiaofeng77/gp-l-only-10k"
OOD_SPLIT = "test_face_cards_as_regular"
ID_SPLIT_CANDIDATES = ("test_id", "test")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-repo-id", default=DEFAULT_TRAIN_REPO)
    parser.add_argument("--eval-repo-id", default=DEFAULT_EVAL_REPO)
    parser.add_argument("--train-size", type=int, default=4096)
    parser.add_argument("--anchor-size", type=int, default=512)
    parser.add_argument("--id-validation-size", type=int, default=512)
    parser.add_argument("--seed", type=int, default=20260923)
    parser.add_argument(
        "--manifest-dir",
        type=Path,
        default=Path("data/local/generalpoints_stage1"),
    )
    return parser.parse_args()


def fail(message: str) -> None:
    raise RuntimeError(message)


def prompt_text(question: Any) -> str:
    if isinstance(question, str):
        return question
    if isinstance(question, list) and question:
        first = question[0]
        if isinstance(first, dict) and isinstance(first.get("content"), str):
            return first["content"]
    return ""


def validate_question(example: dict[str, Any], split_name: str) -> None:
    if not prompt_text(example.get("question")).strip():
        fail(f"{split_name}: unsupported or empty 'question' schema")


def validate_extra_info(example: dict[str, Any], split_name: str) -> None:
    extra = example.get("extra_info")
    if not isinstance(extra, dict):
        fail(f"{split_name}: 'extra_info' is not a dict")

    cards = extra.get("cards")
    if not isinstance(cards, list) or len(cards) != 4:
        fail(f"{split_name}: expected exactly four cards")

    target = extra.get("target")
    if target is not None and target != 24:
        fail(f"{split_name}: unexpected target {target!r}; expected 24")


def rule_semantics(example: dict[str, Any]) -> str:
    """Infer the face-card rule from metadata first, then prompt text."""
    extra = example.get("extra_info")
    if isinstance(extra, dict):
        flag = extra.get("treat_face_cards_as_10")
        if flag is True:
            return "all_10"
        if flag is False:
            return "regular_11_12_13"

        mapping = extra.get("face_card_mapping")
        if mapping == "all_10":
            return "all_10"
        if mapping == "mixed_11_12_13":
            return "regular_11_12_13"

    text = prompt_text(example.get("question")).lower()
    if "count as '10'" in text or 'count as "10"' in text:
        return "all_10"
    if (
        "11" in text
        and "12" in text
        and "13" in text
        and ("respectively" in text or "counts as" in text)
    ):
        return "regular_11_12_13"
    return "unknown"


def sample_rule_semantics(dataset: Dataset, limit: int = 128) -> set[str]:
    values = {rule_semantics(dataset[i]) for i in range(min(limit, len(dataset)))}
    values.discard("unknown")
    return values


def indices_digest(indices: list[int]) -> str:
    payload = ",".join(str(i) for i in indices).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def load_and_basic_check(repo_id: str, split: str) -> Dataset:
    print(f"Downloading/loading: {repo_id} :: {split}")
    ds = load_dataset(repo_id, split=split)
    if len(ds) == 0:
        fail(f"{repo_id}::{split}: split is empty")

    required = {"data_source", "extra_info", "question"}
    missing = required.difference(ds.column_names)
    if missing:
        fail(
            f"{repo_id}::{split}: missing required columns {sorted(missing)}; "
            f"found {ds.column_names}"
        )

    validate_question(ds[0], split)
    validate_extra_info(ds[0], split)
    print(f"  rows={len(ds):,} columns={ds.column_names}")
    return ds


def main() -> int:
    args = parse_args()

    print("=== GeneralPoints Stage-1 data smoke test ===")
    print("SFT train repository:", args.train_repo_id)
    print("Evaluation repository:", args.eval_repo_id)

    train_splits = get_dataset_split_names(args.train_repo_id)
    eval_splits = get_dataset_split_names(args.eval_repo_id)
    print("Train repo splits:", ", ".join(train_splits))
    print("Eval repo splits:", ", ".join(eval_splits))

    if "train" not in train_splits:
        fail("SFT train repository does not contain a train split")
    if OOD_SPLIT not in eval_splits:
        fail(f"evaluation repository is missing OOD split {OOD_SPLIT!r}")

    id_split = next((x for x in ID_SPLIT_CANDIDATES if x in eval_splits), None)
    if id_split is None:
        fail(
            "evaluation repository contains neither 'test_id' nor 'test'; "
            f"available={eval_splits}"
        )

    if id_split == "test":
        print(
            "NOTE: public eval repo has no 'test_id'; using 'test' as the ID "
            "candidate and verifying its actual rule semantics."
        )

    train = load_and_basic_check(args.train_repo_id, "train")
    id_eval = load_and_basic_check(args.eval_repo_id, id_split)
    ood_eval = load_and_basic_check(args.eval_repo_id, OOD_SPLIT)

    if "answer" not in train.column_names:
        fail(
            "SFT train repository has no 'answer' column. "
            "Expected the official answer-only SFT dataset."
        )

    required_train_rows = (
        args.train_size + args.anchor_size + args.id_validation_size
    )
    if len(train) < required_train_rows:
        fail(
            f"train split has {len(train)} rows, but Stage 1 requires at least "
            f"{required_train_rows}"
        )

    train_rules = sample_rule_semantics(train)
    id_rules = sample_rule_semantics(id_eval)
    ood_rules = sample_rule_semantics(ood_eval)

    print("Detected rule semantics:")
    print("  SFT train:", sorted(train_rules) or ["unknown"])
    print(f"  ID eval ({id_split}):", sorted(id_rules) or ["unknown"])
    print(f"  OOD eval ({OOD_SPLIT}):", sorted(ood_rules) or ["unknown"])

    if train_rules and train_rules != {"all_10"}:
        fail(f"SFT train is not consistently J=Q=K=10: {train_rules}")
    if id_rules != {"all_10"}:
        fail(
            f"ID candidate split {id_split!r} did not verify as J=Q=K=10: "
            f"{id_rules or {'unknown'}}"
        )
    if ood_rules != {"regular_11_12_13"}:
        fail(
            f"OOD split {OOD_SPLIT!r} did not verify as J=11,Q=12,K=13: "
            f"{ood_rules or {'unknown'}}"
        )

    indices = list(range(len(train)))
    random.Random(args.seed).shuffle(indices)
    sft_indices = indices[: args.train_size]
    anchor_start = args.train_size
    anchor_end = anchor_start + args.anchor_size
    anchor_indices = indices[anchor_start:anchor_end]

    id_validation_end = anchor_end + args.id_validation_size
    id_validation_indices = indices[anchor_end:id_validation_end]

    sft_set = set(sft_indices)
    anchor_set = set(anchor_indices)
    id_validation_set = set(id_validation_indices)
    if sft_set & anchor_set:
        fail("SFT and anchor subsets overlap")
    if sft_set & id_validation_set:
        fail("SFT and ID-validation subsets overlap")
    if anchor_set & id_validation_set:
        fail("anchor and ID-validation subsets overlap")

    args.manifest_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = args.manifest_dir / "manifest.json"

    if manifest_path.exists():
        previous = json.loads(manifest_path.read_text(encoding="utf-8"))
        if previous.get("sft_indices") not in (None, sft_indices):
            fail("existing manifest has different frozen SFT indices")
        if previous.get("anchor_indices") not in (None, anchor_indices):
            fail("existing manifest has different frozen anchor indices")

    manifest = {
        "train_repo_id": args.train_repo_id,
        "eval_repo_id": args.eval_repo_id,
        "train_split": "train",
        "id_eval_split": id_split,
        "ood_oracle_split": OOD_SPLIT,
        "train_rows": len(train),
        "id_eval_rows": len(id_eval),
        "ood_eval_rows": len(ood_eval),
        "shuffle_seed": args.seed,
        "sft_train_size": args.train_size,
        "anchor_size": args.anchor_size,
        "id_validation_size": args.id_validation_size,
        "sft_indices": sft_indices,
        "anchor_indices": anchor_indices,
        "id_validation_indices": id_validation_indices,
        "sft_indices_sha256": indices_digest(sft_indices),
        "anchor_indices_sha256": indices_digest(anchor_indices),
        "id_validation_indices_sha256": indices_digest(id_validation_indices),
    }
    manifest_path.write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )

    print(f"Manifest written to: {manifest_path}")
    print(
        f"SFT subset={len(sft_indices)} | anchor={len(anchor_indices)} | "
        f"ID validation={len(id_validation_indices)} | pairwise overlap=0"
    )
    print("PASS: GeneralPoints Stage-1 data smoke test succeeded.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
