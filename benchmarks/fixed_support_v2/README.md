# Fixed-support v2 production and validation archive

This directory contains the production-profile gate and the immutable evidence
retained from the pre-promotion v1/v2 comparison.

The mutable v1 solver and its replay runners were retired on 2026-07-27. They
were not a reproducible correctness oracle: current v1 no longer reproduced
all historical successes. The executable comparison remains available in Git
history before the retirement commit, while the evidence used for the
promotion decision remains in the repository.

The following files are intentionally retained:

- `fixed_support_v2_gpu_matrix.json`, which declares the historical cases and
  frozen artifact hashes;
- `data/frozen_v1_baseline/selected_case_summary.json`;
- `data/frozen_v1_baseline/selected_case_summary.md`.

Unit tests verify the declared SHA-256 hashes without importing or executing a
v1 runtime module. The original validation decision, GPU artifact hashes, and
limitations remain recorded in
`documents/fixed_support_solver_v2_validation.md`.

## Production-profile gate

The public default `head_v2` route has a separate production-profile runner:

```console
csh benchmarks/fixed_support_v2/run_fixed_support_v2_production_profile_gpu.csh
```

It runs the five curated product-route families twice in one GPU process.
Before each family it clears JAX's in-memory caches, records one cold run, and
then immediately records one warm run.  It uses the public default
``return_diagnostics=False`` path, so terminal diagnostic kernels are not
included in the production budget.  A run without arguments enforces the
approved `a100_40gb_2026_07` runtime budget.  The runner also samples active
GPU compute processes around every measured call and rejects a runtime
measurement that overlaps another process.

To run a diagnostic with alternate maximum per-family limits:

```console
csh benchmarks/fixed_support_v2/run_fixed_support_v2_production_profile_gpu.csh \
  MAX_COLD_COMPILE MAX_COLD_WALL MAX_WARM_EXECUTE MAX_WARM_WALL
```

This runner calls the public API directly. It does not change the public
default or retry with a retired solver.
