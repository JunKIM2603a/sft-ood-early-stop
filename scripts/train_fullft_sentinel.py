#!/usr/bin/env python3
"""Two-GPU FSDP2 full-finetuning capacity sentinel.

Scientific mode reproduces the predeclared sentinel:
- Qwen2.5-3B-Instruct
- full parameter SFT
- 4,096 frozen GeneralPoints answer-only examples
- LR 1e-6
- 1 epoch
- global effective batch 64
- 64 optimizer steps
- checkpoints at 10,20,...,60,64

Run with accelerate launch on exactly two GPUs.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import random
import shutil
from pathlib import Path
from typing import Any

import torch
import yaml
from datasets import load_dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
)


DEFAULT_CONFIG = Path("configs/fullft_sentinel.yaml")
DEFAULT_OUTPUT = Path(
    "checkpoints/fullft_sentinel/qwen2.5-3b_lr-1e-06_seed-42"
)


def fail(message: str) -> None:
    raise RuntimeError(message)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--overwrite-output", action="store_true")
    parser.add_argument("--min-free-disk-gib", type=float, default=50.0)
    return parser.parse_args()


def read_yaml(path: Path) -> dict[str, Any]:
    if not path.exists():
        fail(f"missing config: {path}")
    return yaml.safe_load(path.read_text(encoding="utf-8"))


def read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        fail(f"missing JSON: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def normalize_question(value: Any) -> str:
    if isinstance(value, str) and value.strip():
        return value
    if isinstance(value, list) and value:
        parts = [
            item["content"]
            for item in value
            if isinstance(item, dict) and isinstance(item.get("content"), str)
        ]
        if parts:
            return "\n".join(parts).strip()
    fail(f"unsupported question format: {type(value).__name__}")


def normalize_answer(value: Any) -> str:
    if isinstance(value, str) and value.strip():
        return value
    fail("answer is empty or unsupported")


def encode_example(
    tokenizer: Any,
    question: Any,
    answer: Any,
    max_length: int,
) -> dict[str, list[int]]:
    user_text = normalize_question(question)
    answer_text = normalize_answer(answer)

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
        fail("chat template did not return list[int]")
    if full_ids[: len(prompt_ids)] != prompt_ids:
        fail("chat-template prefix mismatch")
    if len(full_ids) > max_length:
        fail(
            f"sample length {len(full_ids)} exceeds max_length={max_length}; "
            "full-FT sentinel never truncates frozen examples"
        )

    labels = [-100] * len(prompt_ids) + full_ids[len(prompt_ids) :]
    if not any(value != -100 for value in labels):
        fail("zero supervised tokens")

    return {
        "input_ids": full_ids,
        "attention_mask": [1] * len(full_ids),
        "labels": labels,
    }


class EncodedDataset(torch.utils.data.Dataset):
    def __init__(self, rows: list[dict[str, list[int]]]):
        self.rows = rows

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int) -> dict[str, list[int]]:
        return self.rows[index]


class AnswerOnlyCollator:
    def __init__(self, pad_token_id: int):
        self.pad_token_id = pad_token_id

    def __call__(
        self,
        features: list[dict[str, list[int]]],
    ) -> dict[str, torch.Tensor]:
        max_len = max(len(row["input_ids"]) for row in features)
        batch = len(features)

        input_ids = torch.full(
            (batch, max_len),
            self.pad_token_id,
            dtype=torch.long,
        )
        attention_mask = torch.zeros((batch, max_len), dtype=torch.long)
        labels = torch.full((batch, max_len), -100, dtype=torch.long)

        for row_index, row in enumerate(features):
            length = len(row["input_ids"])
            input_ids[row_index, :length] = torch.tensor(
                row["input_ids"],
                dtype=torch.long,
            )
            attention_mask[row_index, :length] = 1
            labels[row_index, :length] = torch.tensor(
                row["labels"],
                dtype=torch.long,
            )

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
        }


def is_main_process() -> bool:
    return int(os.environ.get("RANK", "0")) == 0


def world_size() -> int:
    return int(os.environ.get("WORLD_SIZE", "1"))


def check_disk(output_dir: Path, minimum_gib: float) -> None:
    parent = output_dir.parent
    parent.mkdir(parents=True, exist_ok=True)
    usage = shutil.disk_usage(parent)
    free_gib = usage.free / 1024**3
    print(
        f"[disk] free={free_gib:.1f} GiB under {parent}; "
        f"required>={minimum_gib:.1f} GiB"
    )
    if free_gib < minimum_gib:
        fail(
            f"insufficient free disk for full-state checkpoints: "
            f"{free_gib:.1f} GiB < {minimum_gib:.1f} GiB"
        )


def write_checkpoint_index(
    output_dir: Path,
    expected_final_step: int,
    base_model_id: str,
) -> None:
    rows = [
        {
            "optimizer_step": 0,
            "path": base_model_id,
            "kind": "base_model",
        }
    ]

    checkpoints = []
    for directory in output_dir.glob("checkpoint-*"):
        if not directory.is_dir():
            continue
        try:
            step = int(directory.name.split("-")[-1])
        except ValueError:
            continue
        checkpoints.append((step, directory))

    checkpoints.sort()
    for step, directory in checkpoints:
        rows.append(
            {
                "optimizer_step": step,
                "path": str(directory),
                "kind": "full_model",
            }
        )

    actual = [int(row["optimizer_step"]) for row in rows]
    expected = [0] + list(range(10, expected_final_step, 10))
    if expected_final_step not in expected:
        expected.append(expected_final_step)

    if actual != expected:
        fail(
            f"full-FT checkpoint schedule mismatch: "
            f"expected={expected}, actual={actual}"
        )

    path = output_dir / "checkpoints.jsonl"
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def main() -> int:
    args = parse_args()
    cfg = read_yaml(args.config)

    if world_size() != 2:
        fail(
            f"full-FT sentinel requires exactly 2 distributed processes; "
            f"WORLD_SIZE={world_size()}"
        )

    model_id = cfg["model"]["name_or_path"]
    data_cfg = cfg["data"]
    train_cfg = cfg["training"]
    fsdp_cfg = cfg["fsdp2"]
    ckpt_cfg = cfg["checkpointing"]

    manifest = read_json(Path(data_cfg["manifest"]))
    sft_indices = list(manifest["sft_indices"])

    if len(sft_indices) != int(data_cfg["subset_size"]):
        fail(
            f"manifest SFT size={len(sft_indices)} does not match "
            f"config={data_cfg['subset_size']}"
        )

    if args.smoke:
        selected_indices = sft_indices[:128]
        epochs = 1.0
        max_steps = 2
        save_strategy = "no"
        save_steps = 500
        output_dir = Path("artifacts/smoke/fullft_fsdp2")
        minimum_disk = 2.0
    else:
        selected_indices = sft_indices
        epochs = float(train_cfg["epochs"])
        max_steps = -1
        save_strategy = "steps"
        save_steps = int(ckpt_cfg["save_every_optimizer_steps"])
        output_dir = args.output_dir
        minimum_disk = args.min_free_disk_gib

    if is_main_process():
        if output_dir.exists() and any(output_dir.iterdir()):
            if args.overwrite_output:
                shutil.rmtree(output_dir)
            else:
                fail(
                    f"output directory is non-empty: {output_dir}. "
                    "Use --overwrite-output only for an intentionally "
                    "discarded run."
                )
        check_disk(output_dir, minimum_disk)

    seed = int(train_cfg["seed"])
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    tokenizer = AutoTokenizer.from_pretrained(model_id, use_fast=True)
    if tokenizer.pad_token_id is None:
        if tokenizer.eos_token_id is None:
            fail("tokenizer has neither pad nor EOS token")
        tokenizer.pad_token = tokenizer.eos_token

    dataset = load_dataset(
        manifest["train_repo_id"],
        split=manifest.get("train_split", "train"),
    ).select(selected_indices)

    max_length = int(data_cfg["max_sequence_length"])
    encoded_rows = [
        encode_example(
            tokenizer,
            row["question"],
            row["answer"],
            max_length,
        )
        for row in dataset
    ]

    if is_main_process():
        print("=== Full-FT FSDP2 capacity sentinel ===")
        print("Mode:", "SMOKE" if args.smoke else "SCIENTIFIC")
        print("Model:", model_id)
        print("Examples:", len(encoded_rows))
        print(
            "Max tokens:",
            max(len(row["input_ids"]) for row in encoded_rows),
        )

    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        dtype=torch.bfloat16,
        attn_implementation="sdpa",
        low_cpu_mem_usage=True,
    )
    model.config.use_cache = False

    per_device = int(train_cfg["per_device_train_batch_size"])
    grad_accum = int(train_cfg["gradient_accumulation_steps"])
    expected_effective = int(train_cfg["effective_batch_size"])
    actual_effective = per_device * world_size() * grad_accum

    if actual_effective != expected_effective:
        fail(
            f"effective batch mismatch: {per_device} * {world_size()} * "
            f"{grad_accum} = {actual_effective}, expected {expected_effective}"
        )

    if not args.smoke:
        derived_steps = (
            len(encoded_rows)
            // expected_effective
            * int(train_cfg["epochs"])
        )
        if derived_steps != int(train_cfg["expected_optimizer_steps"]):
            fail(
                f"expected optimizer steps mismatch: "
                f"derived={derived_steps}, "
                f"config={train_cfg['expected_optimizer_steps']}"
            )

    training_args = TrainingArguments(
        output_dir=str(output_dir),
        do_train=True,
        per_device_train_batch_size=per_device,
        gradient_accumulation_steps=grad_accum,
        learning_rate=float(train_cfg["learning_rate"]),
        weight_decay=float(train_cfg["weight_decay"]),
        num_train_epochs=epochs,
        max_steps=max_steps,
        lr_scheduler_type=str(train_cfg["scheduler"]),
        warmup_steps=(
            0 if args.smoke else int(train_cfg["warmup_steps"])
        ),
        bf16=True,
        seed=seed,
        data_seed=seed,
        logging_strategy="steps",
        logging_steps=1,
        logging_first_step=True,
        save_strategy=save_strategy,
        save_steps=save_steps,
        save_total_limit=None,
        save_only_model=True,
        report_to="none",
        remove_unused_columns=False,
        dataloader_drop_last=False,
        dataloader_num_workers=0,
        gradient_checkpointing=False,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=EncodedDataset(encoded_rows),
        data_collator=AnswerOnlyCollator(tokenizer.pad_token_id),
        processing_class=tokenizer,
    )

    distributed_type = str(trainer.accelerator.state.distributed_type)
    if "FSDP" not in distributed_type.upper():
        fail(
            "Accelerate did not initialize FSDP. "
            f"distributed_type={distributed_type}. "
            "Launch with configs/accelerate_fullft_fsdp2.yaml."
        )

    if is_main_process():
        print("Accelerate distributed type:", distributed_type)
        print(
            "FSDP2 execution config: version={version}, "
            "cpu_offload={cpu_offload}, activation_checkpointing={activation}".format(
                version=fsdp_cfg["version"],
                cpu_offload=fsdp_cfg["cpu_offload"],
                activation=fsdp_cfg["activation_checkpointing"],
            )
        )

    result = trainer.train()

    if not args.smoke:
        final_step = int(trainer.state.global_step)
        expected_final = int(train_cfg["expected_optimizer_steps"])
        if final_step != expected_final:
            fail(
                f"full-FT ended at step {final_step}; "
                f"expected {expected_final}"
            )

        final_dir = output_dir / f"checkpoint-{final_step}"
        if not final_dir.exists():
            trainer.save_model(str(final_dir))

        torch.distributed.barrier()

        if is_main_process():
            write_checkpoint_index(
                output_dir,
                expected_final,
                model_id,
            )

    local_peak = torch.cuda.max_memory_allocated() / 1024**3
    if is_main_process():
        summary = {
            "status": "PASS" if args.smoke else "COMPLETED",
            "mode": "SMOKE" if args.smoke else "SCIENTIFIC",
            "model_id": model_id,
            "learning_rate": float(train_cfg["learning_rate"]),
            "seed": seed,
            "examples": len(encoded_rows),
            "global_step": int(trainer.state.global_step),
            "train_loss": float(result.training_loss),
            "local_rank0_peak_allocated_gib": local_peak,
            "world_size": world_size(),
            "per_device_train_batch_size": per_device,
            "gradient_accumulation_steps": grad_accum,
            "effective_batch_size": actual_effective,
            "output_dir": str(output_dir),
        }
        output_dir.mkdir(parents=True, exist_ok=True)
        (output_dir / "run_summary.json").write_text(
            json.dumps(summary, indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
        tokenizer.save_pretrained(output_dir / "tokenizer")
        print(json.dumps(summary, indent=2))
        print(
            "PASS: full-FT FSDP2 "
            + ("smoke" if args.smoke else "capacity sentinel")
            + " completed."
        )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
