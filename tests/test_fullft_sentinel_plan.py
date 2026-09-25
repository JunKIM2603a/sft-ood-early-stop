from pathlib import Path

import yaml


def test_fullft_sentinel_batch_and_steps():
    cfg = yaml.safe_load(
        Path("configs/fullft_sentinel.yaml").read_text(encoding="utf-8")
    )
    train = cfg["training"]

    assert train["world_size"] == 2
    assert train["per_device_train_batch_size"] == 1
    assert train["gradient_accumulation_steps"] == 32
    assert (
        train["per_device_train_batch_size"]
        * train["world_size"]
        * train["gradient_accumulation_steps"]
        == train["effective_batch_size"]
        == 64
    )
    assert (
        cfg["data"]["subset_size"] // train["effective_batch_size"]
        == train["expected_optimizer_steps"]
        == 64
    )


def test_fullft_uses_explicit_fsdp2_cpu_offload():
    cfg = yaml.safe_load(
        Path("configs/fullft_sentinel.yaml").read_text(encoding="utf-8")
    )
    fsdp = cfg["fsdp2"]

    assert fsdp["version"] == 2
    assert fsdp["reshard_after_forward"] is True
    assert fsdp["activation_checkpointing"] is True
    assert fsdp["cpu_offload"] is True
    assert fsdp["transformer_layer_cls_to_wrap"] == "Qwen2DecoderLayer"

    accel = yaml.safe_load(
        Path("configs/accelerate_fullft_fsdp2.yaml").read_text(
            encoding="utf-8"
        )
    )
    assert accel["distributed_type"] == "FSDP"
    assert accel["fsdp_config"]["fsdp_version"] == 2
    assert accel["fsdp_config"]["fsdp_cpu_offload"] is True
