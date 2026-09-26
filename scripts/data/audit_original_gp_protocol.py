#!/usr/bin/env python3
"""Audit whether the public original SFTvsRL GP-L data matches our 10K P1 data."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from pathlib import Path
from typing import Any

from datasets import load_dataset
from huggingface_hub import snapshot_download


ORIGINAL_REPO = "tianzhechu/SFTvsRL_Data"
CURRENT_REPO = "Xiaofeng77/answer-only-gp-l-only-10k"
DEFAULT_LOCAL = Path("data/external/sftvsrl_protocol")
DEFAULT_OUTPUT = Path("artifacts/audits/gp_protocol_alignment.json")


def normalize_text(value: str) -> str:
    return re.sub(r"\s+", " ", value.strip())


def current_pair(row: dict[str, Any]) -> tuple[str, str]:
    question = row.get("question")
    answer = row.get("answer")

    if isinstance(question, list):
        pieces = [
            item.get("content", "")
            for item in question
            if isinstance(item, dict)
        ]
        question = "\n".join(pieces)

    if not isinstance(question, str) or not isinstance(answer, str):
        raise RuntimeError("unexpected current dataset question/answer format")

    return normalize_text(question), normalize_text(answer)


def original_pair(row: dict[str, Any]) -> tuple[str, str] | None:
    conversations = row.get("conversations")
    if not isinstance(conversations, list):
        return None

    user = None
    assistant = None
    for item in conversations:
        if not isinstance(item, dict):
            continue
        role = item.get("from")
        value = item.get("value")
        if not isinstance(value, str):
            continue
        if role == "human" and user is None:
            user = value
        elif role == "gpt" and assistant is None:
            assistant = value

    if user is None or assistant is None:
        return None
    return normalize_text(user), normalize_text(assistant)


def digest(pair: tuple[str, str]) -> str:
    payload = json.dumps(pair, ensure_ascii=False, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--local-dir", type=Path, default=DEFAULT_LOCAL)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()

    args.local_dir.mkdir(parents=True, exist_ok=True)
    snapshot_path = Path(
        snapshot_download(
            repo_id=ORIGINAL_REPO,
            repo_type="dataset",
            local_dir=str(args.local_dir),
            allow_patterns=["README.md", "**/gp-l/**", "**/gp_l/**"],
        )
    )

    candidates = sorted(
        path
        for path in snapshot_path.rglob("data.json")
        if "gp-l" in path.as_posix() or "gp_l" in path.as_posix()
    )

    current = load_dataset(CURRENT_REPO, split="train")
    current_pairs = [current_pair(dict(row)) for row in current]
    current_hashes = {digest(pair) for pair in current_pairs}
    current_prompt_hashes = {
        hashlib.sha256(pair[0].encode("utf-8")).hexdigest()
        for pair in current_pairs
    }

    report: dict[str, Any] = {
        "original_repo": ORIGINAL_REPO,
        "current_repo": CURRENT_REPO,
        "current_rows": len(current),
        "original_gp_l_candidates": [str(path) for path in candidates],
    }

    if not candidates:
        report.update(
            {
                "status": "ORIGINAL_GP_L_FILE_NOT_PUBLICLY_RESOLVED",
                "scientific_implication": (
                    "P2 can isolate model scale within the current 10K "
                    "protocol, but cannot be called an exact Jin et al. "
                    "data replication."
                ),
            }
        )
    else:
        original_path = candidates[0]
        raw = json.loads(original_path.read_text(encoding="utf-8"))
        if not isinstance(raw, list):
            raise RuntimeError(f"expected list in {original_path}")

        original_pairs = [
            pair
            for row in raw
            if isinstance(row, dict)
            for pair in [original_pair(row)]
            if pair is not None
        ]
        original_hashes = {digest(pair) for pair in original_pairs}
        original_prompt_hashes = {
            hashlib.sha256(pair[0].encode("utf-8")).hexdigest()
            for pair in original_pairs
        }

        pair_overlap = len(current_hashes & original_hashes)
        prompt_overlap = len(current_prompt_hashes & original_prompt_hashes)
        denominator = max(1, len(original_pairs))

        if pair_overlap == len(original_pairs) and original_pairs:
            status = "ORIGINAL_PAIRS_SUBSET_OF_CURRENT"
        elif pair_overlap > 0 or prompt_overlap > 0:
            status = "PARTIAL_PROTOCOL_OVERLAP"
        else:
            status = "NO_EXACT_TEXT_OVERLAP"

        report.update(
            {
                "status": status,
                "original_file": str(original_path),
                "original_rows": len(raw),
                "original_parseable_pairs": len(original_pairs),
                "exact_pair_overlap": pair_overlap,
                "exact_pair_overlap_rate_vs_original": pair_overlap / denominator,
                "exact_prompt_overlap": prompt_overlap,
                "exact_prompt_overlap_rate_vs_original": (
                    prompt_overlap / denominator
                ),
                "scientific_implication": (
                    "P2 keeps the current 10K dataset fixed to isolate model "
                    "scale. Exact original-protocol replication remains a "
                    "separate control unless overlap is complete."
                ),
            }
        )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )

    print("=== GP protocol alignment audit ===")
    print("Status:", report["status"])
    print("Current rows:", report["current_rows"])
    print("Original candidates:", report["original_gp_l_candidates"])
    if "original_rows" in report:
        print("Original rows:", report["original_rows"])
        print("Exact pair overlap:", report["exact_pair_overlap"])
        print("Exact prompt overlap:", report["exact_prompt_overlap"])
    print("Wrote:", args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
