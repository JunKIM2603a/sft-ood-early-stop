#!/usr/bin/env python3
"""Evaluate Stage-1 checkpoints on ID loss, ID task success, and OOD success."""

from __future__ import annotations

import argparse
import csv
import gc
import json
import sys
from pathlib import Path
from typing import Any

import torch
import yaml
from datasets import Dataset, load_dataset
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from sft_ood_early_stop.generalpoints import score_response  # noqa: E402


DEFAULT_CONFIG = Path("configs/pilot.yaml")
DEFAULT_MANIFEST = Path("data/local/generalpoints_stage1/manifest.json")
DEFAULT_RUN_DIR = Path("checkpoints/stage1/lr-1e-05_seed-42")
DEFAULT_OUTPUT_ROOT = Path("results/stage1")
RAW_ROOT = Path("results/raw/stage1")


def fail(message: str) -> None:
    raise RuntimeError(message)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--run-dir", type=Path, default=DEFAULT_RUN_DIR)
    parser.add_argument(
        "--steps",
        default="all",
        help="all or comma-separated optimizer steps, e.g. 0,10",
    )
    parser.add_argument("--eval-batch-size", type=int, default=32)
    parser.add_argument("--loss-batch-size", type=int, default=16)
    parser.add_argument("--max-new-tokens", type=int, default=128)
    parser.add_argument(
        "--max-examples",
        type=int,
        default=0,
        help="0 means full split; positive values are engineering smoke only.",
    )
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument(
        "--raw-root",
        type=Path,
        default=RAW_ROOT,
    )
    parser.add_argument("--overwrite-step", action="store_true")
    return parser.parse_args()


def read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        fail(f"missing JSON file: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def read_yaml(path: Path) -> dict[str, Any]:
    if not path.exists():
        fail(f"missing YAML file: {path}")
    return yaml.safe_load(path.read_text(encoding="utf-8"))


def normalize_messages(question: Any) -> list[dict[str, str]]:
    if isinstance(question, str) and question.strip():
        return [{"role": "user", "content": question}]
    if isinstance(question, list) and question:
        messages = []
        for item in question:
            if not isinstance(item, dict):
                fail("question message is not a dict")
            role = item.get("role")
            content = item.get("content")
            if not isinstance(role, str) or not isinstance(content, str):
                fail("question message lacks string role/content")
            messages.append({"role": role, "content": content})
        return messages
    fail(f"unsupported question format: {type(question).__name__}")


def answer_text(value: Any) -> str:
    if isinstance(value, str) and value.strip():
        return value
    fail("ID validation answer is empty or unsupported")


def encode_labeled(
    tokenizer: Any,
    question: Any,
    answer: Any,
    max_length: int,
) -> dict[str, torch.Tensor]:
    messages = normalize_messages(question)
    response = answer_text(answer)

    prompt_ids = tokenizer.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=True,
        return_dict=False,
    )
    full_ids = tokenizer.apply_chat_template(
        messages + [{"role": "assistant", "content": response}],
        tokenize=True,
        add_generation_prompt=False,
        return_dict=False,
    )

    if not isinstance(prompt_ids, list) or not isinstance(full_ids, list):
        fail("chat template did not return list[int]")
    if full_ids[: len(prompt_ids)] != prompt_ids:
        fail("chat-template prefix mismatch for ID validation loss")
    if len(full_ids) > max_length:
        fail(
            f"ID-validation example has {len(full_ids)} tokens > {max_length}; "
            "rerun the token audit and resolve explicitly."
        )

    labels = [-100] * len(prompt_ids) + full_ids[len(prompt_ids) :]
    return {
        "input_ids": torch.tensor(full_ids, dtype=torch.long),
        "attention_mask": torch.ones(len(full_ids), dtype=torch.long),
        "labels": torch.tensor(labels, dtype=torch.long),
    }


def collate_labeled(
    examples: list[dict[str, torch.Tensor]],
    pad_token_id: int,
) -> dict[str, torch.Tensor]:
    max_len = max(item["input_ids"].numel() for item in examples)
    batch = len(examples)
    ids = torch.full((batch, max_len), pad_token_id, dtype=torch.long)
    mask = torch.zeros((batch, max_len), dtype=torch.long)
    labels = torch.full((batch, max_len), -100, dtype=torch.long)

    for row, item in enumerate(examples):
        length = item["input_ids"].numel()
        ids[row, :length] = item["input_ids"]
        mask[row, :length] = item["attention_mask"]
        labels[row, :length] = item["labels"]

    return {"input_ids": ids, "attention_mask": mask, "labels": labels}


def discover_checkpoints(run_dir: Path) -> dict[int, Path]:
    index = run_dir / "checkpoints.jsonl"
    if not index.exists():
        fail(f"missing checkpoint index: {index}")

    found: dict[int, Path] = {}
    for line in index.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        step = int(row["optimizer_step"])
        path = Path(row["path"])
        if not path.is_absolute():
            path = ROOT / path
        found[step] = path

    if 0 not in found:
        fail("checkpoint index does not contain step 0")
    return dict(sorted(found.items()))


def select_steps(spec: str, available: dict[int, Path]) -> list[int]:
    if spec.strip().lower() == "all":
        return list(available)
    requested = sorted(
        {int(value.strip()) for value in spec.split(",") if value.strip()}
    )
    missing = [step for step in requested if step not in available]
    if missing:
        fail(f"requested unavailable checkpoint steps: {missing}")
    return requested


def load_model(
    base_model_id: str,
    checkpoint_step: int,
    checkpoint_path: Path,
    device: torch.device,
) -> Any:
    if checkpoint_step == 0:
        model = AutoModelForCausalLM.from_pretrained(
            base_model_id,
            dtype=torch.bfloat16,
            attn_implementation="sdpa",
            low_cpu_mem_usage=True,
        )
        model.config.use_cache = True
        model.to(device)
        model.eval()
        return model

    adapter_path = checkpoint_path / "adapter_model.safetensors"
    full_config = checkpoint_path / "config.json"

    if adapter_path.exists():
        base = AutoModelForCausalLM.from_pretrained(
            base_model_id,
            dtype=torch.bfloat16,
            attn_implementation="sdpa",
            low_cpu_mem_usage=True,
        )
        base.config.use_cache = True
        base.to(device)
        model = PeftModel.from_pretrained(
            base,
            checkpoint_path,
            is_trainable=False,
        )
    elif full_config.exists():
        model = AutoModelForCausalLM.from_pretrained(
            checkpoint_path,
            dtype=torch.bfloat16,
            attn_implementation="sdpa",
            low_cpu_mem_usage=True,
        )
        model.config.use_cache = True
        model.to(device)
    else:
        fail(
            f"checkpoint is neither a PEFT adapter nor a full HF model: "
            f"{checkpoint_path}"
        )

    model.eval()
    return model


@torch.inference_mode()
def compute_id_validation_loss(
    model: Any,
    tokenizer: Any,
    dataset: Dataset,
    batch_size: int,
    max_length: int,
    device: torch.device,
) -> float:
    encoded = [
        encode_labeled(tokenizer, row["question"], row["answer"], max_length)
        for row in dataset
    ]

    total_nll = 0.0
    total_tokens = 0
    for start in range(0, len(encoded), batch_size):
        batch = collate_labeled(
            encoded[start : start + batch_size],
            tokenizer.pad_token_id,
        )
        batch = {key: value.to(device) for key, value in batch.items()}

        with torch.autocast("cuda", dtype=torch.bfloat16):
            output = model(**batch)

        if output.loss is None or not torch.isfinite(output.loss):
            fail("non-finite ID validation loss")

        supervised = int((batch["labels"][:, 1:] != -100).sum().item())
        total_nll += float(output.loss.float().item()) * supervised
        total_tokens += supervised

    if total_tokens == 0:
        fail("ID validation set has zero supervised tokens")
    return total_nll / total_tokens


def prompts_to_batch(
    tokenizer: Any,
    rows: list[dict[str, Any]],
    device: torch.device,
) -> dict[str, torch.Tensor]:
    texts = [
        tokenizer.apply_chat_template(
            normalize_messages(row["question"]),
            tokenize=False,
            add_generation_prompt=True,
        )
        for row in rows
    ]
    encoded = tokenizer(
        texts,
        return_tensors="pt",
        padding=True,
        add_special_tokens=False,
    )
    return {key: value.to(device) for key, value in encoded.items()}


@torch.inference_mode()
def evaluate_task_split(
    model: Any,
    tokenizer: Any,
    dataset: Dataset,
    batch_size: int,
    max_new_tokens: int,
    device: torch.device,
    raw_path: Path,
) -> dict[str, Any]:
    raw_path.parent.mkdir(parents=True, exist_ok=True)

    correct = 0
    parsed = 0
    valid = 0
    total = 0

    original_padding_side = tokenizer.padding_side
    tokenizer.padding_side = "left"

    try:
        with raw_path.open("w", encoding="utf-8") as handle:
            for start in tqdm(
                range(0, len(dataset), batch_size),
                desc=f"generate {raw_path.stem}",
            ):
                end = min(start + batch_size, len(dataset))
                rows = [dataset[i] for i in range(start, end)]
                batch = prompts_to_batch(tokenizer, rows, device)
                input_width = batch["input_ids"].shape[1]

                generated = model.generate(
                    **batch,
                    max_new_tokens=max_new_tokens,
                    do_sample=False,
                    use_cache=True,
                    pad_token_id=tokenizer.pad_token_id,
                    eos_token_id=tokenizer.eos_token_id,
                )
                response_ids = generated[:, input_width:]
                responses = tokenizer.batch_decode(
                    response_ids,
                    skip_special_tokens=True,
                )

                for offset, (row, response) in enumerate(zip(rows, responses)):
                    extra = row.get("extra_info")
                    if not isinstance(extra, dict):
                        fail("evaluation example lacks extra_info dict")

                    display_cards = extra.get("display_cards")
                    target = extra.get("target", 24)
                    if not isinstance(display_cards, list) or len(display_cards) != 4:
                        fail("evaluation example lacks four display_cards")

                    score = score_response(
                        response,
                        [int(value) for value in display_cards],
                        int(target),
                    )
                    total += 1
                    correct += int(score.correct)
                    parsed += int(score.parsed)
                    valid += int(score.formula_valid)

                    record = {
                        "row_index": start + offset,
                        "data_source": row.get("data_source"),
                        "display_cards": display_cards,
                        "target": target,
                        "response": response,
                        **score.to_dict(),
                    }
                    handle.write(json.dumps(record, ensure_ascii=False) + "\n")
    finally:
        tokenizer.padding_side = original_padding_side

    if total == 0:
        fail("evaluation split is empty")

    return {
        "n": total,
        "accuracy": correct / total,
        "parse_rate": parsed / total,
        "valid_formula_rate": valid / total,
    }


def read_existing_metrics(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def write_metrics(
    jsonl_path: Path,
    csv_path: Path,
    rows: list[dict[str, Any]],
) -> None:
    rows = sorted(rows, key=lambda row: int(row["optimizer_step"]))

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


def main() -> int:
    args = parse_args()

    if not torch.cuda.is_available():
        fail("CUDA is unavailable")
    if args.eval_batch_size < 1 or args.loss_batch_size < 1:
        fail("batch sizes must be >= 1")

    config = read_yaml(args.config)
    manifest = read_json(args.manifest)
    if "id_validation_indices" not in manifest:
        fail(
            "manifest predates the frozen ID-loss validation set. Rerun "
            "python scripts/data/smoke_generalpoints.py; the script preserves "
            "the existing SFT/anchor indices."
        )

    if not args.run_dir.exists():
        fail(f"run directory does not exist: {args.run_dir}")

    checkpoints = discover_checkpoints(args.run_dir)
    steps = select_steps(args.steps, checkpoints)
    run_name = args.run_dir.name

    output_dir = args.output_root / run_name
    raw_dir = args.raw_root / run_name
    metrics_jsonl = output_dir / "checkpoint_metrics.jsonl"
    metrics_csv = output_dir / "checkpoint_metrics.csv"
    output_dir.mkdir(parents=True, exist_ok=True)

    existing = read_existing_metrics(metrics_jsonl)
    existing_by_step = {int(row["optimizer_step"]): row for row in existing}

    train = load_dataset(
        manifest["train_repo_id"],
        split=manifest.get("train_split", "train"),
    )
    id_validation = train.select(manifest["id_validation_indices"])
    if "answer" not in id_validation.column_names:
        fail("ID validation data lacks answer labels")

    evaluation = load_dataset(manifest["eval_repo_id"])
    id_split_name = manifest["id_eval_split"]
    ood_split_name = manifest["ood_oracle_split"]
    id_task = evaluation[id_split_name]
    ood_task = evaluation[ood_split_name]

    if args.max_examples > 0:
        id_task = id_task.select(range(min(args.max_examples, len(id_task))))
        ood_task = ood_task.select(range(min(args.max_examples, len(ood_task))))
        evaluation_mode = f"SMOKE_MAX_{args.max_examples}"
    else:
        evaluation_mode = "FULL"

    existing_modes = {
        str(row.get("evaluation_mode", "UNKNOWN")) for row in existing
    }
    if existing_modes and existing_modes != {evaluation_mode} and not args.overwrite_step:
        fail(
            f"existing metrics use mode(s) {sorted(existing_modes)}, but this run "
            f"uses {evaluation_mode}. Re-run with --overwrite-step or use a clean "
            "output directory; never mix smoke and full rows."
        )

    base_model_id = config["model"]["name_or_path"]
    max_length = int(config["training"]["max_sequence_length"])

    tokenizer = AutoTokenizer.from_pretrained(base_model_id, use_fast=True)
    if tokenizer.pad_token_id is None:
        if tokenizer.eos_token_id is None:
            fail("tokenizer has neither pad nor EOS token")
        tokenizer.pad_token = tokenizer.eos_token

    device = torch.device("cuda:0")
    torch.cuda.set_device(device)

    print("=== Stage-1 checkpoint evaluation ===")
    print("Run:", run_name)
    print("GPU:", torch.cuda.get_device_name(device))
    print("Steps:", steps)
    print(
        f"ID-loss validation={len(id_validation)} | "
        f"ID task={len(id_task)} ({id_split_name}) | "
        f"OOD task={len(ood_task)} ({ood_split_name})"
    )
    print("Mode:", evaluation_mode)

    if args.overwrite_step:
        rows = [
            row for row in existing
            if int(row["optimizer_step"]) not in steps
        ]
    else:
        rows = list(existing)

    for step in steps:
        if step in existing_by_step and not args.overwrite_step:
            print(f"Skipping step {step}: metrics already exist")
            continue

        print(f"\n--- Evaluating step {step} ---")
        model = load_model(
            base_model_id,
            step,
            checkpoints[step],
            device,
        )

        id_loss = compute_id_validation_loss(
            model,
            tokenizer,
            id_validation,
            batch_size=args.loss_batch_size,
            max_length=max_length,
            device=device,
        )

        id_raw = raw_dir / f"step-{step:06d}_id.jsonl"
        ood_raw = raw_dir / f"step-{step:06d}_ood.jsonl"

        id_metrics = evaluate_task_split(
            model,
            tokenizer,
            id_task,
            batch_size=args.eval_batch_size,
            max_new_tokens=args.max_new_tokens,
            device=device,
            raw_path=id_raw,
        )
        ood_metrics = evaluate_task_split(
            model,
            tokenizer,
            ood_task,
            batch_size=args.eval_batch_size,
            max_new_tokens=args.max_new_tokens,
            device=device,
            raw_path=ood_raw,
        )

        row = {
            "optimizer_step": step,
            "id_validation_loss": id_loss,
            "id_task_accuracy": id_metrics["accuracy"],
            "ood_task_accuracy": ood_metrics["accuracy"],
            "id_parse_rate": id_metrics["parse_rate"],
            "ood_parse_rate": ood_metrics["parse_rate"],
            "id_valid_formula_rate": id_metrics["valid_formula_rate"],
            "ood_valid_formula_rate": ood_metrics["valid_formula_rate"],
            "id_task_n": id_metrics["n"],
            "ood_task_n": ood_metrics["n"],
            "evaluation_mode": evaluation_mode,
        }

        rows = [r for r in rows if int(r["optimizer_step"]) != step]
        rows.append(row)
        write_metrics(metrics_jsonl, metrics_csv, rows)

        print(
            f"step={step} ID_loss={id_loss:.6f} "
            f"ID_acc={id_metrics['accuracy']:.4f} "
            f"OOD_acc={ood_metrics['accuracy']:.4f}"
        )

        del model
        gc.collect()
        torch.cuda.empty_cache()
        torch.cuda.synchronize()

    print("\nEvaluation results:", metrics_csv)
    print("Raw generations (gitignored):", raw_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
