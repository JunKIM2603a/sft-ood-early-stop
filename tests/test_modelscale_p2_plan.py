from pathlib import Path
import json
import yaml


def test_p2_changes_model_scale_only_scientifically():
    p1 = yaml.safe_load(
        Path("configs/exposure_sentinel_3b_fulltrain.yaml").read_text(
            encoding="utf-8"
        )
    )
    p2 = yaml.safe_load(
        Path("configs/modelscale_sentinel_7b_fulltrain.yaml").read_text(
            encoding="utf-8"
        )
    )

    assert p1["model"]["name_or_path"] == "Qwen/Qwen2.5-3B-Instruct"
    assert p2["model"]["name_or_path"] == "Qwen/Qwen2.5-7B-Instruct"
    assert p2["data"]["manifest"] == p1["data"]["manifest"]
    assert p2["data"]["selection"] == p1["data"]["selection"] == "full_train"

    for key in [
        "learning_rate",
        "seed",
        "epochs",
        "effective_batch_size",
        "scheduler",
        "warmup_ratio",
        "weight_decay",
    ]:
        assert p2["training"][key] == p1["training"][key]


def test_p2_uses_zero3_nvme_offload():
    cfg = json.loads(
        Path("configs/deepspeed_zero3_nvme.json").read_text(
            encoding="utf-8"
        )
    )
    zero = cfg["zero_optimization"]

    assert zero["stage"] == 3
    assert zero["offload_optimizer"]["device"] == "nvme"
    assert zero["offload_param"]["device"] == "nvme"
    assert zero["stage3_gather_16bit_weights_on_model_save"] is True


def test_p2_keeps_drift_gate_closed():
    cfg = yaml.safe_load(
        Path("configs/modelscale_sentinel_7b_fulltrain.yaml").read_text(
            encoding="utf-8"
        )
    )
    assert cfg["evaluation"]["id_validation_loss"] == (
        "disabled_due_to_training_overlap"
    )
    assert cfg["evaluation"]["functional_drift"] == (
        "disabled_until_reproduction_gate_reopens"
    )
