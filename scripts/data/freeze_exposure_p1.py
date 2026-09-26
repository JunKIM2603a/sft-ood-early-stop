#!/usr/bin/env python3
"""Freeze the full-train exposure sentinel manifest and audit all examples."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

from datasets import load_dataset
from transformers import AutoTokenizer


DEFAULT_STAGE1_MANIFEST = Path(
    "data/local/generalpoints_stage1/manifest.json"
)
DEFAULT_OUTPUT = Path(
    "data/local/generalpoints_exposure_p1/manifest.json"
)
DEFAULT_REPO = "Xiaofeng77/answer-only-gp-l-only-10k"
DEFAULT_MODEL = "Qwen/Qwen2.5-3B-Instruct"


def fail(message: str) -> None:
    raise RuntimeError(message)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--stage1-manifest",
        type=Path,
        default=DEFAULT_STAGE1_MANIFEST,
    )
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--train-repo-id", default=DEFAULT_REPO)
    parser.add_argument("--train-split", default="train")
    parser.add_argument("--model-id", default=DEFAULT_MODEL)
    parser.add_argument("--max-sequence-length", type=int, default=2048)
    parser.add_argument("--min-examples", type=int, default=9000)
    parser.add_argument("--max-examples", type=int, default=11000)
    return parser.parse_args()


def load_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        fail(f"missing required manifest: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def normalize_question(value: Any) -> str:
    if isinstance(value, str) and value.strip():
        return value
    if isinstance(value, list) and value:
        parts = [
            item["content"]
            for item in value
            if isinstance(item, dict)
            and isinstance(item.get("content"), str)
        ]
        if parts:
            return "\n".join(parts).strip()
    fail(f"unsupported question format: {type(value).__name__}")


def normalize_answer(value: Any) -> str:
    if isinstance(value, str) and value.strip():
        return value
    fail("answer is empty or unsupported")


def encoded_length(
    tokenizer: Any,
    question: Any,
    answer: Any,
) -> int:
    user_text = normalize_question(question)
    answer_text = normalize_answer(answer)
    ids = tokenizer.apply_chat_template(
        [
            {"role": "user", "content": user_text},
            {"role": "assistant", "content": answer_text},
        ],
        tokenize=True,
        add_generation_prompt=False,
        return_dict=False,
    )
    if not isinstance(ids, list):
        fail("chat template did not return list[int]")
    return len(ids)


def main() -> int:
    args = parse_args()
    stage1 = load_json(args.stage1_manifest)

    dataset = load_dataset(
        args.train_repo_id,
        split=args.train_split,
    )
    n = len(dataset)
    if not (args.min_examples <= n <= args.max_examples):
        fail(
            f"unexpected full-train size {n}; expected "
            f"{args.min_examples}..{args.max_examples}"
        )

    required = {"question", "answer", "extra_info"}
    missing = required - set(dataset.column_names)
    if missing:
        fail(f"training split is missing columns: {sorted(missing)}")

    tokenizer = AutoTokenizer.from_pretrained(args.model_id, use_fast=True)

    index_hash = hashlib.sha256()
    content_hash = hashlib.sha256()
    max_tokens = 0
    max_token_row = -1
    min_card = None
    max_card = None
    targets: set[int] = set()

    for index, row in enumerate(dataset):
        index_hash.update(f"{index}\n".encode("utf-8"))

        question = row["question"]
        answer = row["answer"]
        extra = row["extra_info"]
        if not isinstance(extra, dict):
            fail(f"row {index}: extra_info is not a dict")

        cards = extra.get("display_cards")
        if not isinstance(cards, list) or len(cards) != 4:
            fail(f"row {index}: expected four display_cards")

        int_cards = [int(value) for value in cards]
        row_min = min(int_cards)
        row_max = max(int_cards)
        min_card = row_min if min_card is None else min(min_card, row_min)
        max_card = row_max if max_card is None else max(max_card, row_max)

        target = int(extra.get("target", 24))
        targets.add(target)

        length = encoded_length(tokenizer, question, answer)
        if length > max_tokens:
            max_tokens = length
            max_token_row = index
        if length > args.max_sequence_length:
            fail(
                f"row {index}: token length {length} exceeds "
                f"max_sequence_length={args.max_sequence_length}"
            )

        stable_record = {
            "question": question,
            "answer": answer,
            "display_cards": int_cards,
            "target": target,
            "face_card_mapping": extra.get("face_card_mapping"),
        }
        content_hash.update(
            json.dumps(
                stable_record,
                ensure_ascii=False,
                sort_keys=True,
                separators=(",", ":"),
            ).encode("utf-8")
        )
        content_hash.update(b"\n")

    if max_card is None or max_card > 10:
        fail(
            f"full SFT train semantic check failed: max display card={max_card}; "
            "expected J=Q=K=10 regime with no values above 10"
        )
    if min_card is None or min_card < 1:
        fail(f"unexpected minimum display card value: {min_card}")

    manifest = {
        "schema_version": 1,
        "purpose": "p1_exposure_boundary_full_train_3b",
        "train_repo_id": args.train_repo_id,
        "train_split": args.train_split,
        "selection": "full_train",
        "train_num_rows": n,
        "dataset_fingerprint": getattr(dataset, "_fingerprint", None),
        "full_train_indices_sha256": index_hash.hexdigest(),
        "train_content_sha256": content_hash.hexdigest(),
        "model_id_for_token_audit": args.model_id,
        "max_sequence_length": args.max_sequence_length,
        "max_observed_tokens": max_tokens,
        "max_observed_tokens_row": max_token_row,
        "semantic_check": {
            "min_display_card": min_card,
            "max_display_card": max_card,
            "targets": sorted(targets),
            "expected_id_rule": "J=Q=K=10",
        },
        "eval_repo_id": stage1["eval_repo_id"],
        "id_eval_split": stage1["id_eval_split"],
        "ood_oracle_split": stage1["ood_oracle_split"],
        "source_stage1_manifest": str(args.stage1_manifest),
        "id_loss_validation_available": False,
        "id_loss_validation_reason": (
            "P1 trains on the complete answer-only train split, so the prior "
            "512-example ID-loss holdout is now part of training."
        ),
        "functional_anchor_available": False,
        "functional_anchor_reason": (
            "P1 intentionally consumes the complete answer-only train split; "
            "drift-selector development remains disabled during this pivot."
        ),
    }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )

    print("=== P1 full-train exposure manifest ===")
    print("Train repo:", args.train_repo_id)
    print("Rows:", n)
    print("Dataset fingerprint:", manifest["dataset_fingerprint"])
    print("Content SHA256:", manifest["train_content_sha256"])
    print(
        f"Display-card range: {min_card}..{max_card}; "
        f"targets={sorted(targets)}"
    )
    print(
        f"Token audit: max={max_tokens} at row={max_token_row}; "
        f"limit={args.max_sequence_length}"
    )
    print("ID-loss validation: DISABLED (overlaps full-train exposure)")
    print("Functional anchor: DISABLED (drift stage remains gated)")
    print("Wrote:", args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
