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
