#!/usr/bin/env python3
"""P2 preflight for Qwen2.5-7B full-train ZeRO-Infinity execution."""

from __future__ import annotations

import argparse
import json
import os
import shlex
import shutil
from pathlib import Path
from typing import Any

from datasets import load_dataset
from transformers import AutoTokenizer


MODEL_ID = "Qwen/Qwen2.5-7B-Instruct"
MANIFEST = Path("data/local/generalpoints_exposure_p1/manifest.json")
AUDIT = Path("artifacts/audits/gp_protocol_alignment.json")
NVME_DIR = Path("artifacts/deepspeed_nvme/p2_7b")
AIO_ENV = Path("artifacts/env/deepspeed_aio.env")
OUTPUT = Path("artifacts/audits/p2_7b_preflight.json")


def fail(message: str) -> None:
    raise RuntimeError(message)


def read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        fail(f"missing required file: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def apply_env_file(path: Path) -> None:
    if not path.exists():
        return

    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        if not line.startswith("export ") or "=" not in line:
            fail(f"unsupported environment line in {path}: {raw_line!r}")
        key, value = line[len("export "):].split("=", 1)
        parsed = shlex.split(value)
        if len(parsed) != 1:
            fail(f"cannot parse environment value for {key} in {path}")
        os.environ[key] = parsed[0]


def meminfo_gib() -> tuple[float, float]:
    values: dict[str, int] = {}
    for line in Path("/proc/meminfo").read_text().splitlines():
        key, rest = line.split(":", 1)
        values[key] = int(rest.strip().split()[0])
    total = values["MemTotal"] / 1024**2
    available = values["MemAvailable"] / 1024**2
    return total, available


def normalize_question(value: Any) -> str:
    if isinstance(value, str):
        return value
    if isinstance(value, list):
        return "\n".join(
            item.get("content", "")
            for item in value
            if isinstance(item, dict)
        )
    fail(f"unsupported question type: {type(value).__name__}")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--min-disk-gib", type=float, default=400.0)
    parser.add_argument("--min-total-ram-gib", type=float, default=96.0)
    args = parser.parse_args()

    manifest = read_json(MANIFEST)
    protocol = read_json(AUDIT)

    NVME_DIR.mkdir(parents=True, exist_ok=True)
    usage = shutil.disk_usage(NVME_DIR)
    disk_free_gib = usage.free / 1024**3
    ram_total_gib, ram_available_gib = meminfo_gib()

    if disk_free_gib < args.min_disk_gib:
        fail(
            f"insufficient free disk for 7B checkpoints + NVMe offload: "
            f"{disk_free_gib:.1f} GiB < {args.min_disk_gib:.1f} GiB"
        )
    if ram_total_gib < args.min_total_ram_gib:
        fail(
            f"host RAM is below P2 minimum: "
            f"{ram_total_gib:.1f} GiB < {args.min_total_ram_gib:.1f} GiB"
        )

    apply_env_file(AIO_ENV)

    try:
        from deepspeed.ops.op_builder import AsyncIOBuilder
        aio_compatible = bool(AsyncIOBuilder().is_compatible())
    except Exception as exc:
        fail(f"cannot validate DeepSpeed async_io support: {exc}")

    if not aio_compatible:
        fail(
            "DeepSpeed async_io is not compatible on this host. "
            "NVMe offload requires libaio headers/libraries. Run "
            "'bash scripts/setup_deepspeed_aio.sh', then rerun this preflight. "
            "For a system-wide Ubuntu install, DeepSpeed documents "
            "'sudo apt-get install -y libaio-dev'."
        )

    dataset = load_dataset(
        manifest["train_repo_id"],
        split=manifest.get("train_split", "train"),
    )
    if len(dataset) != int(manifest["train_num_rows"]):
        fail("P1 frozen dataset row count changed")

    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, use_fast=True)
    max_tokens = 0
    max_row = -1

    for index, row in enumerate(dataset):
        question = normalize_question(row["question"])
        answer = row["answer"]
        if not isinstance(answer, str):
            fail(f"row {index}: answer is not a string")

        ids = tokenizer.apply_chat_template(
            [
                {"role": "user", "content": question},
                {"role": "assistant", "content": answer},
            ],
            tokenize=True,
            add_generation_prompt=False,
            return_dict=False,
        )
        length = len(ids)
        if length > max_tokens:
            max_tokens = length
            max_row = index
        if length > int(manifest["max_sequence_length"]):
            fail(
                f"row {index}: 7B tokenizer length={length} exceeds "
                f"limit={manifest['max_sequence_length']}"
            )

    probe = NVME_DIR / ".write_probe"
    probe.write_bytes(b"p2-nvme-probe")
    probe.unlink()

    report = {
        "status": "PASS",
        "model_id": MODEL_ID,
        "model_parameters_billion": 7.6156,
        "train_rows": len(dataset),
        "dataset_fingerprint": manifest.get("dataset_fingerprint"),
        "train_content_sha256": manifest.get("train_content_sha256"),
        "max_tokens": max_tokens,
        "max_tokens_row": max_row,
        "disk_free_gib": disk_free_gib,
        "host_ram_total_gib": ram_total_gib,
        "host_ram_available_gib": ram_available_gib,
        "deepspeed_async_io_compatible": aio_compatible,
        "aio_env_file": str(AIO_ENV) if AIO_ENV.exists() else None,
        "nvme_path": str(NVME_DIR.resolve()),
        "protocol_alignment_status": protocol.get("status"),
        "protocol_alignment_note": protocol.get("scientific_implication"),
    }

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(
        json.dumps(report, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )

    print("=== P2 7B preflight ===")
    print("Status: PASS")
    print("Train rows:", len(dataset))
    print("7B tokenizer max tokens:", max_tokens)
    print(
        f"Host RAM: total={ram_total_gib:.1f} GiB "
        f"available={ram_available_gib:.1f} GiB"
    )
    print(f"NVMe/filesystem free: {disk_free_gib:.1f} GiB")
    print("DeepSpeed async_io compatible:", aio_compatible)
    print("AIO env file:", report["aio_env_file"])
    print("Protocol alignment:", protocol.get("status"))
    print("Wrote:", OUTPUT)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
