# ExoGibbs Benchmark Freeze

## Current Scope

The current benchmark baseline is intentionally small and fixed before further solver optimization work.

Implemented benchmark runners:

- single-layer benchmark: `benchmarks/run_single_layer.py`
- atmospheric-profile benchmark: `benchmarks/run_profile.py`

Current non-goals:

- FastChem comparison runner
- markdown report generation
- plotting
- CI gating
- solver modifications or solver-side instrumentation

## Anchor Cases

Single-layer anchor:

- case id: `fastchem_extended_single_T3000_P1e-8_reference`
- setup: FastChem extended preset via `exogibbs.presets.fastchem.chemsetup`
- state: `T = 3000 K`, `P = 1e-8 bar`
- abundance policy: normalized linear abundances loaded from the extended FastChem element file

Profile anchor:

- case id: `fastchem_extended_profile_64layer_monotonic_reference`
- setup: same FastChem extended preset and abundance policy
- layers: `64`
- pressure range: `1e-8` to `1e2 bar`
- temperature range: `900` to `3000 K`
- ordering: top-to-bottom, increasing pressure
- supported methods:
  - `vmap_cold`
  - `scan_hot_from_top`
  - `scan_hot_from_bottom`

## Timing Policy

Both runners use the same timing semantics:

- one measured first call
- untimed warmup calls
- repeated measured warm calls
- `block_until_ready()` before stopping timers

Primary optimization metric:

- `warm_call_median_s`

Supporting timing metrics:

- `first_call_s`
- `warm_call_mean_s`
- `warm_call_p95_s`
- `warm_call_min_s`
- raw warm timings

## Diagnostics Policy

Diagnostics are requested only through the public API.

Current policy:

- diagnostics may be collected in a separate jitted run
- missing diagnostics fields must be recorded as `null`
- benchmark status must be conservative
- success must not be inferred only from a returned solution object

Current diagnostic focus:

- convergence flags
- final residuals
- iteration counts
- max-iteration hits
- NaN detection
- per-layer diagnostics for profile runs when available

## Status Semantics

Current benchmark status values:

- `pass`
- `fail_runtime`
- `error`

Interpretation:

- `pass`: diagnostics explicitly indicate acceptable convergence quality
- `fail_runtime`: NaN or explicit convergence failure
- `error`: diagnostics are missing or insufficient to justify success

## JSON Baseline

Current benchmark JSON is the source of truth. The schema is intentionally simple and additive.

Stable metadata currently recorded:

- `benchmark_version`
- `timestamp_utc`
- case id and category
- setup metadata
- axes
- solver options
- execution config
- metrics
- status

This benchmark surface is the baseline to preserve while solver optimization work proceeds.
