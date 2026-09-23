#!/usr/bin/env python3
"""Audit token lengths for the exact frozen 4,096-example Stage-1 SFT subset.

This script must be run before the decisive Stage-1 training run. It never
truncates examples. If any selected sample exceeds max_length, the audit fails
and the protocol must be revisited explicitly.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
from datasets import load_dataset
from transformers import AutoTokenizer

DEFAULT_MODEL = "Qwen/Qwen2.5-3B-Instruct"
DEFAULT_MANIFEST = Path("data/local/generalpoints_stage1/manifest.json")
DEFAULT_OUTPUT = Path("artifacts/audits/generalpoints_stage1_tokens.json")
DEFAULT_MAX_LENGTH = 2048


def fail(message: str) -> None:
    raise RuntimeError(message)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-id", default=DEFAULT_MODEL)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--max-length", type=int, default=DEFAULT_MAX_LENGTH)
    parser.add_argument("--progress-every", type=int, default=512)
    return parser.parse_args()


def question_to_text(value: Any) -> str:
    if isinstance(value, str) and value.strip():
        return value
    if isinstance(value, list) and value:
        parts = [
            message["content"]
            for message in value
            if isinstance(message, dict) and isinstance(message.get("content"), str)
        ]
        if parts:
            return "\n".join(parts).strip()
    fail(f"unsupported question format: {type(value).__name__}")


def answer_to_text(value: Any) -> str:
    if isinstance(value, str) and value.strip():
        return value
    fail(f"unsupported/empty answer format: {type(value).__name__}")


def token_lengths(tokenizer: Any, question: Any, answer: Any) -> tuple[int, int, int]:
    user_text = question_to_text(question)
    answer_text = answer_to_text(answer)

    prompt_ids = tokenizer.apply_chat_template(
        [{"role": "user", "content": user_text}],
        tokenize=True,
        add_generation_prompt=True,
        return_dict=False,
    )
    full_ids = tokenizer.apply_chat_template(
        [
            {"role": "user", "content": user_text},
            {"role": "assistant", "content": answer_text},
        ],
        tokenize=True,
        add_generation_prompt=False,
        return_dict=False,
    )

    if not isinstance(prompt_ids, list) or not isinstance(full_ids, list):
        fail(
            "chat template did not return list[int] with return_dict=False; "
            f"prompt={type(prompt_ids).__name__}, full={type(full_ids).__name__}"
        )
    if full_ids[: len(prompt_ids)] != prompt_ids:
        fail("chat-template prefix mismatch; cannot define answer-only labels safely")

    prompt_len = len(prompt_ids)
    full_len = len(full_ids)
    assistant_len = full_len - prompt_len
    if assistant_len <= 0:
        fail("sample has no supervised assistant tokens")
    return prompt_len, assistant_len, full_len


def percentile(values: list[int], q: float) -> float:
    return float(np.percentile(np.asarray(values), q))


def main() -> int:
    args = parse_args()
    if not args.manifest.exists():
        fail(
            f"missing {args.manifest}; run "
            "python scripts/data/smoke_generalpoints.py first"
        )

    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    train_repo = manifest["train_repo_id"]
    train_split = manifest.get("train_split", "train")
    sft_indices = manifest["sft_indices"]
    id_validation_indices = manifest.get("id_validation_indices")
    if id_validation_indices is None:
        fail(
            "manifest has no frozen ID validation subset; rerun "
            "python scripts/data/smoke_generalpoints.py"
        )

    print("=== GeneralPoints Stage-1 token-length audit ===")
    print("Model:", args.model_id)
    print("Data:", f"{train_repo}::{train_split}")
    print("Frozen SFT examples:", len(sft_indices))
    print("Max length:", args.max_length)

    full_dataset = load_dataset(train_repo, split=train_split)
    dataset = full_dataset.select(sft_indices)
    id_validation = full_dataset.select(id_validation_indices)
    if len(dataset) != len(sft_indices):
        fail("selected dataset size does not match manifest SFT index count")
    if "answer" not in dataset.column_names:
        fail("answer-only training dataset is missing the 'answer' column")

    tokenizer = AutoTokenizer.from_pretrained(args.model_id, use_fast=True)

    prompt_lengths: list[int] = []
    assistant_lengths: list[int] = []
    full_lengths: list[int] = []
    over_limit: list[dict[str, int]] = []

    for position, example in enumerate(dataset):
        p_len, a_len, f_len = token_lengths(
            tokenizer,
            example["question"],
            example["answer"],
        )
        prompt_lengths.append(p_len)
        assistant_lengths.append(a_len)
        full_lengths.append(f_len)

        if f_len > args.max_length:
            over_limit.append(
                {
                    "subset_position": position,
                    "dataset_index": int(sft_indices[position]),
                    "tokens": f_len,
                }
            )

        done = position + 1
        if args.progress_every > 0 and (
            done % args.progress_every == 0 or done == len(dataset)
        ):
            print(f"  audited {done:,}/{len(dataset):,}")

    quantiles = {
        "min": int(min(full_lengths)),
        "p50": percentile(full_lengths, 50),
        "p90": percentile(full_lengths, 90),
        "p95": percentile(full_lengths, 95),
        "p99": percentile(full_lengths, 99),
        "max": int(max(full_lengths)),
        "mean": float(np.mean(full_lengths)),
    }

    id_validation_lengths: list[int] = []
    id_validation_over_limit: list[dict[str, int]] = []
    for position, example in enumerate(id_validation):
        _, _, full_len = token_lengths(
            tokenizer,
            example["question"],
            example["answer"],
        )
        id_validation_lengths.append(full_len)
        if full_len > args.max_length:
            id_validation_over_limit.append(
                {
                    "subset_position": position,
                    "dataset_index": int(id_validation_indices[position]),
                    "tokens": full_len,
                }
            )

    clean_pass = not over_limit and not id_validation_over_limit
    report = {
        "status": "PASS" if clean_pass else "FAIL",
        "model_id": args.model_id,
        "manifest": str(args.manifest),
        "train_repo_id": train_repo,
        "train_split": train_split,
        "num_examples": len(dataset),
        "max_length": args.max_length,
        "full_length": quantiles,
        "prompt_length_max": int(max(prompt_lengths)),
        "assistant_length_max": int(max(assistant_lengths)),
        "over_limit_count": len(over_limit),
        "over_limit_examples": over_limit[:100],
        "id_validation_num_examples": len(id_validation),
        "id_validation_max_tokens": int(max(id_validation_lengths)),
        "id_validation_over_limit_count": len(id_validation_over_limit),
        "id_validation_over_limit_examples": id_validation_over_limit[:100],
        "manifest_sft_indices_sha256": manifest.get("sft_indices_sha256"),
        "manifest_id_validation_indices_sha256": manifest.get(
            "id_validation_indices_sha256"
        ),
    }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )

    print("\nToken-length summary:")
    print(
        "  min={min} p50={p50:.1f} p90={p90:.1f} p95={p95:.1f} "
        "p99={p99:.1f} max={max} mean={mean:.1f}".format(**quantiles)
    )
    print("  max prompt tokens:", report["prompt_length_max"])
    print("  max supervised assistant tokens:", report["assistant_length_max"])
    print("  SFT over max_length:", len(over_limit))
    print(
        "  ID validation max tokens:",
        report["id_validation_max_tokens"],
        "| over max_length:",
        len(id_validation_over_limit),
    )
    print("Report:", args.output)

    if over_limit or id_validation_over_limit:
        fail(
            "token audit failed: "
            f"SFT over-limit={len(over_limit)}, "
            f"ID-validation over-limit={len(id_validation_over_limit)}. "
            "Do not silently truncate them."
        )

    print(
        "PASS: frozen SFT and ID-validation examples fit without truncation."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
