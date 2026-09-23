"""Pure planning invariants for the frozen Stage-1 schedule."""

from pathlib import Path

import yaml


def load_config():
    return yaml.safe_load(Path("configs/pilot.yaml").read_text(encoding="utf-8"))


def test_effective_batch_and_step_count():
    cfg = load_config()
    train = cfg["training"]
    ckpt = cfg["checkpointing"]
    n = cfg["data"]["train"]["subset_size"]

    micro = train["micro_batch_size"]
    accum = train["gradient_accumulation_steps"]
    effective = train["effective_batch_size"]
    epochs = train["epochs"]

    assert micro * accum == effective == 64
    assert n % effective == 0
    assert n // effective == ckpt["expected_steps_per_epoch"] == 64
    assert (n // effective) * epochs == ckpt["expected_total_steps"] == 192


def test_checkpoint_schedule_contains_final_step():
    cfg = load_config()
    every = cfg["checkpointing"]["save_every_optimizer_steps"]
    total = cfg["checkpointing"]["expected_total_steps"]

    steps = [0] + list(range(every, total + 1, every))
    if steps[-1] != total:
        steps.append(total)

    assert steps[0] == 0
    assert 10 in steps
    assert 190 in steps
    assert steps[-1] == 192
    assert len(steps) == 21
