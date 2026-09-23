#!/usr/bin/env python3
"""GPU training smoke test for Qwen2.5-3B-Instruct + Stage-1 LoRA.

This is intentionally NOT a scientific run. It checks that the frozen Stage-1
model/data/LoRA stack can complete:
load -> tokenize -> forward -> backward -> optimizer step -> adapter save/reload.

Run from the repository root, preferably after the data smoke test.
"""

from __future__ import annotations

import argparse
import gc
import json
import math
import os
import random
import shutil
import sys
from pathlib import Path
from typing import Any

import torch
from datasets import load_dataset
from peft import LoraConfig, PeftModel, TaskType, get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL_ID = "Qwen/Qwen2.5-3B-Instruct"
DEFAULT_MANIFEST = Path("data/local/generalpoints_stage1/manifest.json")
DEFAULT_OUTPUT = Path("artifacts/smoke/qwen2.5-3b-lora-r32")
MAX_LENGTH = 2048
LR = 1.0e-5
LORA_R = 32
LORA_ALPHA = 16
LORA_DROPOUT = 0.0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-id", default=MODEL_ID)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--samples", type=int, default=4)
    parser.add_argument("--steps", type=int, default=2)
    parser.add_argument("--micro-batch-size", type=int, default=1)
    parser.add_argument("--max-length", type=int, default=MAX_LENGTH)
    parser.add_argument("--learning-rate", type=float, default=LR)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--overwrite-output", action="store_true")
    parser.add_argument("--skip-reload-check", action="store_true")
    return parser.parse_args()


def fail(message: str) -> None:
    raise RuntimeError(message)


def question_to_text(value: Any) -> str:
    if isinstance(value, str) and value.strip():
        return value
    if isinstance(value, list) and value:
        parts: list[str] = []
        for message in value:
            if isinstance(message, dict) and isinstance(message.get("content"), str):
                parts.append(message["content"])
        text = "\n".join(parts).strip()
        if text:
            return text
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
    """Encode one answer-only chat SFT example and mask the user prompt loss."""
    user_text = question_to_text(question)
    answer_text = answer_to_text(answer)

    user_messages = [{"role": "user", "content": user_text}]
    full_messages = [
        {"role": "user", "content": user_text},
        {"role": "assistant", "content": answer_text},
    ]

    # Transformers 5.x defaults apply_chat_template(return_dict=True), which
    # returns a BatchEncoding/dict-like object rather than list[int].
    # Explicitly request the token-id list so masking logic stays unambiguous.
    prompt_ids = tokenizer.apply_chat_template(
        user_messages,
        tokenize=True,
        add_generation_prompt=True,
        return_dict=False,
    )
    full_ids = tokenizer.apply_chat_template(
        full_messages,
        tokenize=True,
        add_generation_prompt=False,
        return_dict=False,
    )

    if not isinstance(prompt_ids, list) or not isinstance(full_ids, list):
        fail(
            "tokenizer chat template did not return token-id lists even with "
            "return_dict=False; "
            f"got prompt={type(prompt_ids).__name__}, "
            f"full={type(full_ids).__name__}"
        )

    if full_ids[: len(prompt_ids)] != prompt_ids:
        fail(
            "Qwen chat-template prefix mismatch; refusing to guess the loss mask. "
            "Inspect the tokenizer/template before continuing."
        )

    if len(full_ids) > max_length:
        fail(
            f"sample length {len(full_ids)} exceeds max_length={max_length}; "
            "the scientific pipeline must define truncation/filtering explicitly."
        )

    labels = [-100] * len(prompt_ids) + full_ids[len(prompt_ids) :]
    supervised_tokens = sum(token != -100 for token in labels)
    if supervised_tokens == 0:
        fail("example contains no supervised assistant tokens")

    return {
        "input_ids": torch.tensor(full_ids, dtype=torch.long),
        "attention_mask": torch.ones(len(full_ids), dtype=torch.long),
        "labels": torch.tensor(labels, dtype=torch.long),
    }


def collate(
    examples: list[dict[str, torch.Tensor]],
    pad_token_id: int,
) -> dict[str, torch.Tensor]:
    if not examples:
        fail("cannot collate an empty batch")

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


def cuda_memory() -> dict[str, float]:
    free_bytes, total_bytes = torch.cuda.mem_get_info()
    return {
        "allocated_gib": torch.cuda.memory_allocated() / 1024**3,
        "reserved_gib": torch.cuda.memory_reserved() / 1024**3,
        "peak_allocated_gib": torch.cuda.max_memory_allocated() / 1024**3,
        "free_gib": free_bytes / 1024**3,
        "total_gib": total_bytes / 1024**3,
    }


def print_memory(label: str) -> None:
    mem = cuda_memory()
    print(
        f"[memory] {label}: allocated={mem['allocated_gib']:.2f} GiB | "
        f"reserved={mem['reserved_gib']:.2f} GiB | "
        f"peak={mem['peak_allocated_gib']:.2f} GiB | "
        f"free={mem['free_gib']:.2f}/{mem['total_gib']:.2f} GiB"
    )


def load_base_model(model_id: str, device: torch.device) -> Any:
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        dtype=torch.bfloat16,
        attn_implementation="sdpa",
        low_cpu_mem_usage=True,
    )
    model.config.use_cache = False
    model.to(device)
    return model


def grad_l2_norm(parameters: list[torch.nn.Parameter]) -> float:
    total = 0.0
    found = False
    for parameter in parameters:
        if parameter.grad is None:
            continue
        found = True
        norm = parameter.grad.detach().float().norm(2).item()
        total += norm * norm
    if not found:
        return 0.0
    return math.sqrt(total)


def main() -> int:
    args = parse_args()

    if not torch.cuda.is_available():
        fail("CUDA is unavailable; this smoke test requires an NVIDIA GPU")
    if not torch.cuda.is_bf16_supported():
        fail("BF16 is unsupported, but the frozen Stage-1 protocol requires BF16")
    if args.steps < 1 or args.samples < 1 or args.micro_batch_size < 1:
        fail("steps, samples, and micro-batch-size must all be >= 1")
    if not args.manifest.exists():
        fail(
            f"missing {args.manifest}; first run: "
            "python scripts/data/smoke_generalpoints.py"
        )

    if args.output_dir.exists():
        if args.overwrite_output:
            shutil.rmtree(args.output_dir)
        elif any(args.output_dir.iterdir()):
            fail(
                f"output directory already exists and is non-empty: "
                f"{args.output_dir}. Use --overwrite-output to replace it."
            )

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    device = torch.device("cuda:0")
    torch.cuda.set_device(device)
    torch.cuda.reset_peak_memory_stats()

    print("=== Stage-1 GPU training smoke test ===")
    print("Visible CUDA devices:", torch.cuda.device_count())
    print("Using:", torch.cuda.get_device_name(device))
    print("Model:", args.model_id)
    print(
        f"LoRA: r={LORA_R}, alpha={LORA_ALPHA}, dropout={LORA_DROPOUT}, "
        "target_modules=all-linear"
    )
    print(
        f"Smoke: samples={args.samples}, steps={args.steps}, "
        f"micro_batch={args.micro_batch_size}, max_length={args.max_length}"
    )
    print_memory("startup")

    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    train_repo = manifest["train_repo_id"]
    train_split = manifest.get("train_split", "train")
    sft_indices = manifest["sft_indices"]
    if len(sft_indices) < args.samples:
        fail(
            f"manifest contains only {len(sft_indices)} SFT indices; "
            f"requested {args.samples}"
        )

    print(f"Loading SFT data: {train_repo}::{train_split}")
    dataset = load_dataset(train_repo, split=train_split)
    selected_indices = sft_indices[: args.samples]
    dataset = dataset.select(selected_indices)

    if "answer" not in dataset.column_names:
        fail("answer-only SFT dataset does not contain an 'answer' column")

    print("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(args.model_id, use_fast=True)
    if tokenizer.pad_token_id is None:
        if tokenizer.eos_token_id is None:
            fail("tokenizer has neither pad_token_id nor eos_token_id")
        tokenizer.pad_token = tokenizer.eos_token

    encoded = [
        encode_example(
            tokenizer,
            example["question"],
            example["answer"],
            args.max_length,
        )
        for example in dataset
    ]
    lengths = [item["input_ids"].numel() for item in encoded]
    supervised = [
        int((item["labels"] != -100).sum().item()) for item in encoded
    ]
    print("Token lengths:", lengths)
    print("Supervised assistant tokens:", supervised)

    print("Loading BF16 base model with PyTorch SDPA...")
    model = load_base_model(args.model_id, device)
    print_memory("base model loaded")

    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        inference_mode=False,
        r=LORA_R,
        lora_alpha=LORA_ALPHA,
        lora_dropout=LORA_DROPOUT,
        target_modules="all-linear",
        bias="none",
    )
    model = get_peft_model(model, lora_config)

    try:
        model.gradient_checkpointing_enable(
            gradient_checkpointing_kwargs={"use_reentrant": False}
        )
    except TypeError:
        model.gradient_checkpointing_enable()
    model.enable_input_require_grads()
    model.config.use_cache = False

    trainable = [p for p in model.parameters() if p.requires_grad]
    trainable_count = sum(p.numel() for p in trainable)
    total_count = sum(p.numel() for p in model.parameters())
    ratio = 100.0 * trainable_count / total_count
    print(
        f"Trainable parameters: {trainable_count:,}/{total_count:,} "
        f"({ratio:.3f}%)"
    )
    if trainable_count == 0 or trainable_count == total_count:
        fail("unexpected trainable-parameter count; LoRA freeze is not correct")

    tracked_name = None
    tracked_parameter = None
    for name, parameter in model.named_parameters():
        if parameter.requires_grad and "lora_B" in name:
            tracked_name = name
            tracked_parameter = parameter
            break
    if tracked_parameter is None:
        fail("could not find a trainable LoRA-B parameter")

    tracked_before = tracked_parameter.detach().float().cpu().clone()

    optimizer = torch.optim.AdamW(
        trainable,
        lr=args.learning_rate,
        weight_decay=0.0,
    )

    model.train()
    losses: list[float] = []
    grad_norms: list[float] = []

    for step in range(args.steps):
        start = (step * args.micro_batch_size) % len(encoded)
        batch_examples = [
            encoded[(start + offset) % len(encoded)]
            for offset in range(args.micro_batch_size)
        ]
        batch = collate(batch_examples, tokenizer.pad_token_id)
        batch = {key: value.to(device) for key, value in batch.items()}

        optimizer.zero_grad(set_to_none=True)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            outputs = model(**batch)
            loss = outputs.loss

        if loss is None or not torch.isfinite(loss):
            fail(f"step {step + 1}: non-finite loss: {loss}")

        loss.backward()
        gradient_norm = grad_l2_norm(trainable)
        if not math.isfinite(gradient_norm) or gradient_norm <= 0.0:
            fail(
                f"step {step + 1}: invalid LoRA gradient norm "
                f"{gradient_norm}"
            )

        optimizer.step()
        torch.cuda.synchronize()

        loss_value = float(loss.detach().float().item())
        losses.append(loss_value)
        grad_norms.append(gradient_norm)
        print(
            f"step={step + 1}/{args.steps} loss={loss_value:.6f} "
            f"lora_grad_l2={gradient_norm:.6f}"
        )
        print_memory(f"after optimizer step {step + 1}")

    tracked_after = tracked_parameter.detach().float().cpu()
    max_abs_update = float((tracked_after - tracked_before).abs().max().item())
    print(f"Tracked parameter: {tracked_name}")
    print(f"Tracked LoRA-B max |delta|: {max_abs_update:.8e}")
    if not math.isfinite(max_abs_update) or max_abs_update <= 0.0:
        fail("LoRA parameter did not change after optimizer steps")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    print(f"Saving adapter: {args.output_dir}")
    model.save_pretrained(args.output_dir, safe_serialization=True)
    tokenizer.save_pretrained(args.output_dir)

    adapter_config = args.output_dir / "adapter_config.json"
    adapter_weights = args.output_dir / "adapter_model.safetensors"
    if not adapter_config.exists() or not adapter_weights.exists():
        fail("PEFT adapter save did not produce expected files")

    result = {
        "status": "PASS",
        "model_id": args.model_id,
        "device": torch.cuda.get_device_name(device),
        "torch_version": torch.__version__,
        "cuda_runtime": torch.version.cuda,
        "bf16_supported": torch.cuda.is_bf16_supported(),
        "train_repo_id": train_repo,
        "selected_dataset_indices": selected_indices,
        "steps": args.steps,
        "samples": args.samples,
        "micro_batch_size": args.micro_batch_size,
        "max_length": args.max_length,
        "learning_rate": args.learning_rate,
        "lora": {
            "r": LORA_R,
            "alpha": LORA_ALPHA,
            "dropout": LORA_DROPOUT,
            "target_modules": "all-linear",
        },
        "trainable_parameters": trainable_count,
        "total_parameters_with_adapter": total_count,
        "trainable_percent": ratio,
        "losses": losses,
        "gradient_l2_norms": grad_norms,
        "tracked_parameter": tracked_name,
        "tracked_max_abs_update": max_abs_update,
        "cuda_memory": cuda_memory(),
        "adapter_dir": str(args.output_dir),
    }

    # Free the training model before a save/reload check.
    del optimizer
    del model
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.synchronize()
    print_memory("after freeing training model")

    if not args.skip_reload_check:
        print("Reloading saved adapter for a final forward check...")
        base_model = load_base_model(args.model_id, device)
        reload_model = PeftModel.from_pretrained(
            base_model,
            args.output_dir,
            is_trainable=False,
        )
        reload_model.eval()

        batch = collate([encoded[0]], tokenizer.pad_token_id)
        batch = {key: value.to(device) for key, value in batch.items()}
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
            reload_loss = reload_model(**batch).loss

        if reload_loss is None or not torch.isfinite(reload_loss):
            fail(f"reloaded adapter produced invalid loss: {reload_loss}")
        result["reload_loss"] = float(reload_loss.float().item())
        print(f"Reload forward loss: {result['reload_loss']:.6f}")

        del reload_model
        del base_model
        gc.collect()
        torch.cuda.empty_cache()

    result_path = args.output_dir / "smoke_result.json"
    result_path.write_text(
        json.dumps(result, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )

    print("\nPASS: Qwen2.5-3B + LoRA GPU training smoke test succeeded.")
    print("Result:", result_path)
    print(
        "This proves the local stack can train/save/reload the Stage-1 LoRA "
        "configuration. It is NOT a scientific Stage-1 result."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
