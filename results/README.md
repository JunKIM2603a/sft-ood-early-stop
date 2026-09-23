# Results

Keep only compact, reproducible result artifacts under version control.

Recommended layout per run:

```text
results/
└── <experiment_id>/
    ├── config_resolved.yaml
    ├── metadata.json
    ├── checkpoint_metrics.csv
    ├── selector_summary.csv
    └── figures/
```

Raw model outputs and large logs belong under `results/raw/` or external storage and are gitignored.

## Minimum checkpoint table schema

- experiment_id
- seed
- learning_rate
- checkpoint_id
- optimizer_step
- train_loss
- id_validation_loss
- id_score
- ood_score
- parameter_l2
- functional_kl
- spectral_rotation
- principal_angle

## Minimum selector table schema

- experiment_id
- selector
- selected_checkpoint_id
- selected_step
- selected_ood_score
- oracle_checkpoint_id
- oracle_ood_score
- regret

Never overwrite failed runs. Record failure reason in metadata so compute and instability are visible rather than silently dropped.
