# RTX 4000-series torchrun communication guard

The first DeepSpeed ZeRO-3 smoke stopped before DeepSpeed initialization
because Accelerate rejects RTX 4000-series multi-GPU `torchrun` startup
unless NCCL P2P and IB are explicitly disabled.

The launcher now always exports:

```bash
NCCL_P2P_DISABLE=1
NCCL_IB_DISABLE=1
```

This is a launch/communication compatibility fix only. It does not change the
scientific training condition, model, data, LR, seed, epoch count, batch size,
or ZeRO-3 offload policy.

Retry:

```bash
git pull
pytest -q tests/test_deepspeed_sentinel_plan.py
bash scripts/run_fullft_sentinel_2gpu.sh smoke
```

A correct startup prints both environment variables as 1 before `torchrun`.
