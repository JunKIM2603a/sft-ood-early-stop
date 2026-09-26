from pathlib import Path

import yaml


def test_p1_changes_exposure_not_model_or_optimization_recipe():
    cfg = yaml.safe_load(
        Path("configs/exposure_sentinel_3b_fulltrain.yaml").read_text(
            encoding="utf-8"
        )
    )
    train = cfg["training"]

    assert cfg["model"]["name_or_path"] == "Qwen/Qwen2.5-3B-Instruct"
    assert cfg["data"]["selection"] == "full_train"
    assert train["method"] == "full_finetuning"
    assert train["learning_rate"] == 1e-6
    assert train["epochs"] == 1
    assert train["seed"] == 42
    assert (
        train["per_device_train_batch_size"]
        * train["world_size"]
        * train["gradient_accumulation_steps"]
        == train["effective_batch_size"]
        == 64
    )


def test_p1_does_not_reuse_contaminated_id_loss_or_anchor():
    cfg = yaml.safe_load(
        Path("configs/exposure_sentinel_3b_fulltrain.yaml").read_text(
            encoding="utf-8"
        )
    )

    assert cfg["evaluation"]["id_validation_loss"] == (
        "disabled_due_to_training_overlap"
    )
    assert cfg["evaluation"]["functional_drift"] == (
        "disabled_until_reproduction_gate_reopens"
    )
    assert cfg["checkpointing"]["save_every_optimizer_steps"] == 10


def test_p1_training_and_evaluation_code_supports_full_train_mode():
    trainer = Path("scripts/train_fullft_sentinel_zero3.py").read_text(
        encoding="utf-8"
    )
    evaluator = Path("scripts/evaluate_stage1.py").read_text(
        encoding="utf-8"
    )

    assert 'selection == "full_train"' in trainer
    assert 'math.ceil(len(encoded_rows) / expected_effective)' in trainer
    assert "--skip-id-validation-loss" in evaluator
    assert "id_loss = None" in evaluator
