# Fixed-support v2 validation

This directory archives the reproducible inputs and GPU runners for the
fixed-support v2 experimental solver validation.  It does not select v2 from a
production preset.

Run every command from any working directory; each C shell runner enters the
repository root relative to its own location.

The final validation sequence is:

```console
csh benchmarks/fixed_support_v2/run_fixed_support_v2_corrected_gpu.csh
csh benchmarks/fixed_support_v2/run_fixed_support_v2_water128_gpu100.csh
csh benchmarks/fixed_support_v2/run_fixed_support_v2_final_solver_matrix_gpu.csh
```

The first command regenerates the corrected support-lifecycle prerequisite.
The second regenerates the focused water-128 prerequisite.  The final command
verifies both prerequisite artifacts before running the ten-case exact-state
solver matrix.

Generated JSON and Markdown outputs are written below `results/` and are not
repository sources.  The small files below `data/frozen_v1_baseline/` are the
hash-pinned historical baseline inputs required by the matrix preflight.

For a CPU-only input and integrity check:

```console
JAX_PLATFORMS=cpu python \
  benchmarks/fixed_support_v2/fixed_support_v2_unbiased_gpu_experiment.py \
  --preflight-only \
  --cases all \
  --lifecycle-families manifest \
  --output-dir results/fixed_support_v2_preflight
```

The validation decision, original GPU hashes, and limitations are recorded in
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

This runner calls the public API directly.  It does not change the public
default, retry with v1, or replace the archived experimental solver matrix.
