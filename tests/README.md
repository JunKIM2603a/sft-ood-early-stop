# Tests

Before long GPU runs, maintain smoke tests for:

- dataset formatting and split identity
- deterministic checkpoint discovery/order
- metric computation on a tiny fixture
- base-vs-base drift equals approximately zero
- functional KL finite/non-negative within numerical tolerance
- principal-angle bounds and shape handling
- selector never reads OOD labels except the oracle evaluator
- result rows map one-to-one to checkpoint IDs

GPU smoke tests should use the smallest viable model/input and complete before launching the Stage-1 grid.
