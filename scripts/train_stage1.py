#!/usr/bin/env python3
"""Train one frozen Stage-1 GeneralPoints LoRA condition.

Default invocation runs the first scientific pilot:
  LR=1e-5, seed=42, 3 epochs, effective batch 64.

This script intentionally implements a small explicit PyTorch loop rather than
delegating checkpoint semantics to Trainer. The goal is to make optimizer-step
counting, gradient accumulation, checkpoint cadence, and saved metadata
auditable.
"""

from __future__ import annotations

import argparse
import gc
import json
import math
import os
import random
import shutil
import subprocess
from pathlib import Path
from typing import Any

import torch
import yaml
from datasets import load_dataset
from peft import LoraConfig, TaskType, get_peft_model
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    get_cosine_schedule_with_warmup,
)

DEFAULT_CONFIG = Path("configs/pilot.yaml")
DEFAULT_MANIFEST = Path("data/local/generalpoints_stage1/manifest.json")
DEFAULT_AUDIT = Path("artifacts/audits/generalpoints_stage1_tokens.json")
DEFAULT_OUTPUT_ROOT = Path("checkpoints/stage1")
DEFAULT_LR = 1.0e-5
DEFAULT_SEED = 42


def fail(message: str) -> None:
    raise RuntimeError(message)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--token-audit", type=Path, default=DEFAULT_AUDIT)
    parser.add_argument("--learning-rate", type=float, default=DEFAULT_LR)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--overwrite-output", action="store_true")
    parser.add_argument(
        "--allow-missing-token-audit",
        action="store_true",
        help="Engineering escape hatch only; decisive runs should not use this.",
    )
    return parser.parse_args()


def read_yaml(path: Path) -> dict[str, Any]:
    if not path.exists():
        fail(f"missing config: {path}")
    return yaml.safe_load(path.read_text(encoding="utf-8"))


def read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        fail(f"missing JSON file: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


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


def encode_example(
    tokenizer: Any,
    question: Any,
    answer: Any,
    max_length: int,
) -> dict[str, torch.Tensor]:
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
        fail("chat template did not return list[int] with return_dict=False")
    if full_ids[: len(prompt_ids)] != prompt_ids:
        fail("chat-template prefix mismatch; cannot build answer-only labels")
    if len(full_ids) > max_length:
        fail(
            f"sample has {len(full_ids)} tokens > max_length={max_length}. "
            "Run the token audit and resolve the protocol; truncation is disabled."
        )

    labels = [-100] * len(prompt_ids) + full_ids[len(prompt_ids) :]
    if not any(token != -100 for token in labels):
        fail("sample has zero supervised assistant tokens")

    return {
        "input_ids": torch.tensor(full_ids, dtype=torch.long),
        "attention_mask": torch.ones(len(full_ids), dtype=torch.long),
        "labels": torch.tensor(labels, dtype=torch.long),
    }


def collate(
    examples: list[dict[str, torch.Tensor]],
    pad_token_id: int,
) -> dict[str, torch.Tensor]:
    max_len = max(item["input_ids"].numel() for item in examples)
    batch_size = len(examples)

    input_ids = torch.full(
        (batch_size, max_len), pad_token_id, dtype=torch.long
    )
    attention_mask = torch.zeros((batch_size, max_len), dtype=torch.long)
    labels = torch.full((batch_size, max_len), -100, dtype=torch.long)

    for row, item in enumerate(examples):
        length = item["input_ids"].numel()
        input_ids[row, :length] = item["input_ids"]
        attention_mask[row, :length] = item["attention_mask"]
        labels[row, :length] = item["labels"]

    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "labels": labels,
    }


def grad_l2_norm(parameters: list[torch.nn.Parameter]) -> float:
    squared = 0.0
    found = False
    for parameter in parameters:
        if parameter.grad is None:
            continue
        found = True
        norm = parameter.grad.detach().float().norm(2).item()
        squared += norm * norm
    return math.sqrt(squared) if found else 0.0


def git_commit() -> str | None:
    try:
        return (
            subprocess.check_output(
                ["git", "rev-parse", "HEAD"],
                text=True,
                stderr=subprocess.DEVNULL,
            )
            .strip()
        )
    except Exception:
        return None


def lr_slug(value: float) -> str:
    return f"{value:.0e}".replace("+", "")


def save_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )


def append_jsonl(path: Path, payload: dict[str, Any]) -> None:
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, ensure_ascii=False) + "\n")


def save_adapter_checkpoint(
    model: Any,
    output_dir: Path,
    step: int,
    metadata: dict[str, Any],
) -> Path:
    target = output_dir / f"checkpoint-{step:06d}"
    if target.exists():
        fail(f"checkpoint already exists: {target}")

    target.mkdir(parents=True)
    model.save_pretrained(target, safe_serialization=True)
    save_json(target / "checkpoint_meta.json", metadata)

    expected = target / "adapter_model.safetensors"
    if not expected.exists():
        fail(f"adapter save failed: {expected} not found")
    return target


def main() -> int:
    args = parse_args()

    if not torch.cuda.is_available():
        fail("CUDA is unavailable")
    if not torch.cuda.is_bf16_supported():
        fail("BF16 is required by the frozen Stage-1 protocol")

    config = read_yaml(args.config)
    manifest = read_json(args.manifest)

    if not args.allow_missing_token_audit:
        audit = read_json(args.token_audit)
        if audit.get("status") != "PASS" or audit.get("over_limit_count") != 0:
            fail(
                "token audit is not a clean PASS; run "
                "python scripts/data/audit_generalpoints_tokens.py"
            )
        if audit.get("manifest_sft_indices_sha256") != manifest.get(
            "sft_indices_sha256"
        ):
            fail("token audit and data manifest refer to different SFT subsets")

    model_cfg = config["model"]
    adapt_cfg = config["adaptation"]
    train_cfg = config["training"]
    ckpt_cfg = config["checkpointing"]

    allowed_lrs = {float(v) for v in train_cfg["learning_rates"]}
    allowed_seeds = {int(v) for v in train_cfg["seeds"]}
    if args.learning_rate not in allowed_lrs:
        fail(
            f"LR {args.learning_rate} is outside frozen grid "
            f"{sorted(allowed_lrs)}"
        )
    if args.seed not in allowed_seeds:
        fail(f"seed {args.seed} is outside frozen grid {sorted(allowed_seeds)}")

    micro_batch = int(train_cfg["micro_batch_size"])
    grad_accum = int(train_cfg["gradient_accumulation_steps"])
    effective_batch = int(train_cfg["effective_batch_size"])
    if micro_batch * grad_accum != effective_batch:
        fail(
            "effective batch invariant failed: "
            f"{micro_batch}*{grad_accum}!={effective_batch}"
        )

    epochs = int(train_cfg["epochs"])
    max_length = int(train_cfg["max_sequence_length"])
    warmup_ratio = float(train_cfg["warmup_ratio"])
    weight_decay = float(train_cfg["weight_decay"])
    save_every = int(ckpt_cfg["save_every_optimizer_steps"])

    sft_indices = manifest["sft_indices"]
    num_examples = len(sft_indices)
    if num_examples % micro_batch != 0:
        fail("frozen SFT subset must be divisible by micro_batch_size")

    micro_batches_per_epoch = num_examples // micro_batch
    if micro_batches_per_epoch % grad_accum != 0:
        fail("micro-batches per epoch must be divisible by gradient accumulation")

    optimizer_steps_per_epoch = micro_batches_per_epoch // grad_accum
    total_optimizer_steps = optimizer_steps_per_epoch * epochs

    expected_steps_per_epoch = int(ckpt_cfg["expected_steps_per_epoch"])
    expected_total_steps = int(ckpt_cfg["expected_total_steps"])
    if optimizer_steps_per_epoch != expected_steps_per_epoch:
        fail(
            f"derived steps/epoch={optimizer_steps_per_epoch}, "
            f"config expects {expected_steps_per_epoch}"
        )
    if total_optimizer_steps != expected_total_steps:
        fail(
            f"derived total steps={total_optimizer_steps}, "
            f"config expects {expected_total_steps}"
        )

    run_name = f"lr-{lr_slug(args.learning_rate)}_seed-{args.seed}"
    output_dir = args.output_root / run_name
    if output_dir.exists():
        if args.overwrite_output:
            shutil.rmtree(output_dir)
        else:
            fail(
                f"output already exists: {output_dir}. "
                "Use --overwrite-output only for an intentionally discarded run."
            )
    output_dir.mkdir(parents=True)

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    device = torch.device("cuda:0")
    torch.cuda.set_device(device)
    torch.cuda.reset_peak_memory_stats()

    print("=== Stage-1 single-condition trainer ===")
    print("Run:", run_name)
    print("GPU:", torch.cuda.get_device_name(device))
    print("Model:", model_cfg["name_or_path"])
    print("Data:", manifest["train_repo_id"])
    print(
        f"examples={num_examples} epochs={epochs} micro_batch={micro_batch} "
        f"grad_accum={grad_accum} effective_batch={effective_batch}"
    )
    print(
        f"optimizer_steps_per_epoch={optimizer_steps_per_epoch} "
        f"total_optimizer_steps={total_optimizer_steps}"
    )
    print(f"LR={args.learning_rate} seed={args.seed} save_every={save_every}")

    dataset = load_dataset(
        manifest["train_repo_id"],
        split=manifest.get("train_split", "train"),
    ).select(sft_indices)
    if "answer" not in dataset.column_names:
        fail("training data does not contain answer labels")

    tokenizer = AutoTokenizer.from_pretrained(
        model_cfg["name_or_path"],
        use_fast=True,
    )
    if tokenizer.pad_token_id is None:
        if tokenizer.eos_token_id is None:
            fail("tokenizer has neither pad nor EOS token")
        tokenizer.pad_token = tokenizer.eos_token

    print("Encoding frozen 4,096-example SFT subset...")
    encoded = [
        encode_example(
            tokenizer,
            row["question"],
            row["answer"],
            max_length=max_length,
        )
        for row in dataset
    ]
    print(
        f"Encoded {len(encoded):,} examples; "
        f"max tokens={max(x['input_ids'].numel() for x in encoded)}"
    )

    print("Loading base model in BF16 with SDPA...")
    model = AutoModelForCausalLM.from_pretrained(
        model_cfg["name_or_path"],
        dtype=torch.bfloat16,
        attn_implementation="sdpa",
        low_cpu_mem_usage=True,
    )
    model.config.use_cache = False
    model.to(device)

    lora = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        inference_mode=False,
        r=int(adapt_cfg["r"]),
        lora_alpha=int(adapt_cfg["alpha"]),
        lora_dropout=float(adapt_cfg["dropout"]),
        target_modules=adapt_cfg["target_modules"],
        bias="none",
    )
    model = get_peft_model(model, lora)

    if bool(adapt_cfg["gradient_checkpointing"]):
        try:
            model.gradient_checkpointing_enable(
                gradient_checkpointing_kwargs={"use_reentrant": False}
            )
        except TypeError:
            model.gradient_checkpointing_enable()
        model.enable_input_require_grads()

    trainable = [p for p in model.parameters() if p.requires_grad]
    trainable_count = sum(p.numel() for p in trainable)
    total_count = sum(p.numel() for p in model.parameters())
    if trainable_count <= 0 or trainable_count >= total_count:
        fail("LoRA trainable parameter invariant failed")

    optimizer = torch.optim.AdamW(
        trainable,
        lr=args.learning_rate,
        weight_decay=weight_decay,
    )
    warmup_steps = math.ceil(total_optimizer_steps * warmup_ratio)
    scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=total_optimizer_steps,
    )

    resolved = {
        "run_name": run_name,
        "git_commit": git_commit(),
        "model_id": model_cfg["name_or_path"],
        "train_repo_id": manifest["train_repo_id"],
        "manifest": str(args.manifest),
        "manifest_sft_indices_sha256": manifest.get("sft_indices_sha256"),
        "token_audit": str(args.token_audit),
        "learning_rate": args.learning_rate,
        "seed": args.seed,
        "epochs": epochs,
        "micro_batch_size": micro_batch,
        "gradient_accumulation_steps": grad_accum,
        "effective_batch_size": effective_batch,
        "max_sequence_length": max_length,
        "optimizer_steps_per_epoch": optimizer_steps_per_epoch,
        "total_optimizer_steps": total_optimizer_steps,
        "warmup_steps": warmup_steps,
        "scheduler": train_cfg["scheduler"],
        "weight_decay": weight_decay,
        "checkpoint_every_optimizer_steps": save_every,
        "lora": {
            "r": int(adapt_cfg["r"]),
            "alpha": int(adapt_cfg["alpha"]),
            "dropout": float(adapt_cfg["dropout"]),
            "target_modules": adapt_cfg["target_modules"],
        },
        "trainable_parameters": trainable_count,
        "total_parameters_with_adapter": total_count,
        "torch_version": torch.__version__,
        "cuda_runtime": torch.version.cuda,
        "gpu": torch.cuda.get_device_name(device),
    }
    save_json(output_dir / "run_config.json", resolved)
    tokenizer.save_pretrained(output_dir / "tokenizer")

    base_dir = output_dir / "checkpoint-000000"
    base_dir.mkdir()
    save_json(
        base_dir / "checkpoint_meta.json",
        {
            "optimizer_step": 0,
            "kind": "base_model",
            "model_id": model_cfg["name_or_path"],
            "run_name": run_name,
        },
    )
    append_jsonl(
        output_dir / "checkpoints.jsonl",
        {
            "optimizer_step": 0,
            "path": str(base_dir),
            "kind": "base_model",
        },
    )

    model.train()
    optimizer_step = 0
    examples_seen = 0
    log_path = output_dir / "train_metrics.jsonl"

    print(
        f"Starting training: warmup_steps={warmup_steps}, "
        f"checkpoint steps=10,20,...,{total_optimizer_steps}"
    )

    for epoch in range(epochs):
        order = list(range(num_examples))
        random.Random(args.seed + epoch).shuffle(order)

        optimizer.zero_grad(set_to_none=True)
        accumulation_losses: list[float] = []

        for micro_index in range(micro_batches_per_epoch):
            begin = micro_index * micro_batch
            batch_positions = order[begin : begin + micro_batch]
            batch_examples = [encoded[position] for position in batch_positions]
            batch = collate(batch_examples, tokenizer.pad_token_id)
            batch = {key: value.to(device) for key, value in batch.items()}

            with torch.autocast("cuda", dtype=torch.bfloat16):
                outputs = model(**batch)
                loss = outputs.loss

            if loss is None or not torch.isfinite(loss):
                fail(
                    f"non-finite loss at epoch={epoch + 1}, "
                    f"micro_batch={micro_index + 1}: {loss}"
                )

            raw_loss = float(loss.detach().float().item())
            accumulation_losses.append(raw_loss)
            (loss / grad_accum).backward()
            examples_seen += micro_batch

            should_step = (micro_index + 1) % grad_accum == 0
            if not should_step:
                continue

            gradient_norm = grad_l2_norm(trainable)
            if not math.isfinite(gradient_norm) or gradient_norm <= 0:
                fail(
                    f"invalid gradient norm before optimizer step: {gradient_norm}"
                )

            optimizer.step()
            scheduler.step()
            optimizer.zero_grad(set_to_none=True)
            optimizer_step += 1
            torch.cuda.synchronize()

            step_loss = float(sum(accumulation_losses) / len(accumulation_losses))
            accumulation_losses.clear()
            current_lr = float(scheduler.get_last_lr()[0])
            peak_gib = torch.cuda.max_memory_allocated() / 1024**3

            metric = {
                "optimizer_step": optimizer_step,
                "epoch": epoch + 1,
                "epoch_fraction": (epoch + 1) - 1 + ((micro_index + 1) / micro_batches_per_epoch),
                "examples_seen": examples_seen,
                "train_loss": step_loss,
                "learning_rate": current_lr,
                "gradient_l2_norm": gradient_norm,
                "cuda_peak_allocated_gib": peak_gib,
            }
            append_jsonl(log_path, metric)

            if optimizer_step == 1 or optimizer_step % 5 == 0:
                print(
                    f"step={optimizer_step:03d}/{total_optimizer_steps} "
                    f"epoch={epoch + 1} loss={step_loss:.6f} "
                    f"lr={current_lr:.3e} grad={gradient_norm:.4f} "
                    f"peak={peak_gib:.2f}GiB"
                )

            checkpoint_due = optimizer_step % save_every == 0
            final_step = optimizer_step == total_optimizer_steps
            if checkpoint_due or final_step:
                checkpoint_meta = {
                    "optimizer_step": optimizer_step,
                    "epoch": epoch + 1,
                    "examples_seen": examples_seen,
                    "train_loss": step_loss,
                    "learning_rate": current_lr,
                    "run_name": run_name,
                    "kind": "lora_adapter",
                }
                target = save_adapter_checkpoint(
                    model,
                    output_dir,
                    optimizer_step,
                    checkpoint_meta,
                )
                append_jsonl(
                    output_dir / "checkpoints.jsonl",
                    {
                        "optimizer_step": optimizer_step,
                        "path": str(target),
                        "kind": "lora_adapter",
                    },
                )
                print(f"  saved {target}")

    if optimizer_step != total_optimizer_steps:
        fail(
            f"training ended at optimizer step {optimizer_step}, "
            f"expected {total_optimizer_steps}"
        )
    if examples_seen != num_examples * epochs:
        fail(
            f"examples_seen={examples_seen}, expected={num_examples * epochs}"
        )

    summary = {
        **resolved,
        "status": "COMPLETED",
        "examples_seen": examples_seen,
        "final_optimizer_step": optimizer_step,
        "final_learning_rate": float(scheduler.get_last_lr()[0]),
        "cuda_peak_allocated_gib": torch.cuda.max_memory_allocated() / 1024**3,
    }
    save_json(output_dir / "run_summary.json", summary)

    del optimizer
    del scheduler
    del model
    gc.collect()
    torch.cuda.empty_cache()

    print("\nCOMPLETED Stage-1 training condition:", run_name)
    print("Output:", output_dir)
    print("Next step: evaluate checkpoint 0/10/.../192 on ID and OOD.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
