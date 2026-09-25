from pathlib import Path
import json
import yaml


def test_zero3_cpu_offload_plan_preserves_scientific_batch():
    cfg = yaml.safe_load(
        Path("configs/fullft_sentinel.yaml").read_text(encoding="utf-8")
    )
    train = cfg["training"]

    assert train["per_device_train_batch_size"] == 1
    assert train["world_size"] == 2
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


def test_zero3_explicitly_offloads_optimizer_and_params():
    cfg = json.loads(
        Path("configs/deepspeed_zero3_cpu.json").read_text(
            encoding="utf-8"
        )
    )
    zero = cfg["zero_optimization"]

    assert zero["stage"] == 3
    assert zero["offload_optimizer"]["device"] == "cpu"
    assert zero["offload_param"]["device"] == "cpu"
    assert zero["stage3_gather_16bit_weights_on_model_save"] is True
