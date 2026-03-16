# ExoGibbs Benchmark System Design

## Scope

This document defines a benchmark system for ExoGibbs equilibrium calculations. The benchmark system is intended to establish a stable baseline before solver optimization work. It must measure runtime and convergence quality together, and it must support direct comparison against FastChem-compatible cases.

Non-goal:

- changing solver algorithms or solver defaults in this phase

## 1. Benchmark Goals

The benchmark system should answer four questions:

1. How fast is ExoGibbs on representative single-layer and atmospheric-profile workloads after compilation/warmup?
2. How stable is convergence across temperature, pressure, and abundance regimes relevant to atmospheric chemistry?
3. When run on FastChem-compatible inputs, how closely do ExoGibbs solutions match FastChem outputs?
4. Which benchmark cases are safe to use as optimization gates later, without mixing performance regressions with convergence failures?

The primary timing target is repeated execution speed for HMC-like workloads. First-call compilation cost must still be recorded, but optimization decisions should be based on second-and-later calls.

## 2. Benchmark Categories

### 2.1 Single-Layer Solver Benchmark

Purpose:

- isolate one equilibrium solve at fixed `(T, P, b)`
- measure warm and cold runtime separately
- record solver diagnostics per solve
- provide the lowest-noise optimization anchor

Inputs:

- `exogibbs.api.equilibrium.equilibrium`
- one `ChemicalSetup`
- one `(T, P, b)` case
- one `EquilibriumOptions`

Recommended initial anchor case:

- FastChem extended preset
- `T = 3000 K`
- `P = 1e-8 bar`
- default FastChem-compatible abundances used in existing comparison scripts

Output granularity:

- one record per benchmark case
- repeated-call runtime summary over `N` hot runs
- one optional diagnostics snapshot from a representative run

### 2.2 Atmospheric Profile Benchmark

Purpose:

- benchmark realistic layered workloads
- measure profile method behavior (`vmap_cold`, `scan_hot_from_top`, `scan_hot_from_bottom`)
- evaluate hot-start carryover effects and profile-wide convergence quality

Inputs:

- `exogibbs.api.equilibrium.equilibrium_profile`
- temperature array
- pressure array
- one shared elemental abundance vector
- one profile solve method

Output granularity:

- one record per profile case
- per-layer convergence diagnostics
- aggregate profile runtime and quality metrics

### 2.3 FastChem Comparison Benchmark

Purpose:

- validate that timing is only reported as meaningful when ExoGibbs converges to a FastChem-comparable solution
- detect chemistry regimes where performance is good but agreement is not acceptable

Inputs:

- FastChem-compatible species/setup
- matched temperature-pressure structure
- matched elemental abundance definition
- ExoGibbs result and FastChem result converted to the same comparison basis

Comparison basis:

- species mixing ratios for species common to both systems
- profile-level agreement summary
- case-level pass/fail decision

Output granularity:

- one record per comparison case
- species-level agreement metrics
- profile-level agreement summary

## 3. Required Metrics

Each benchmark record should store both raw measurements and derived pass/fail fields.

### 3.1 Convergence and Numerical Health

Required per solve:

- `converged`: boolean derived from final residual and `max_iter`
- `final_residual`
- `iteration_count`
- `hit_max_iter`: boolean
- `has_nan`: boolean over result arrays and diagnostics

Required per profile:

- `layer_converged_fraction`
- `layer_max_residual`
- `layer_median_residual`
- `layer_max_iteration_count`
- `layer_max_iter_hit_count`
- `layer_nan_count`

Implementation note:

- Use `return_diagnostics=True` from the public API where available.
- If diagnostics content is missing a field, record the missing field explicitly as `null` and treat the case as incomplete rather than silently assuming success.

### 3.2 Runtime

Required timing groups:

- `first_call_wall_s`: first invocation including JIT compilation/tracing
- `warm_call_wall_s`: array of repeated-call timings after warmup
- `warm_call_mean_s`
- `warm_call_median_s`
- `warm_call_p95_s`
- `warm_call_min_s`

Recommended timing protocol:

- run one untimed setup phase
- run exactly one measured first call
- run `W` warmup calls not included in reported warm statistics
- run `N` measured warm calls
- call `block_until_ready()` before stopping each timer

Primary decision metric:

- `warm_call_median_s`

Secondary metrics:

- `first_call_wall_s`
- `warm_call_p95_s`

### 3.3 Agreement with FastChem

Required when a FastChem reference is available:

- `common_species_count`
- `species_max_abs_log10_error`
- `species_median_abs_log10_error`
- `species_max_abs_frac_error`
- `species_count_above_threshold`
- `agreement_pass`: boolean

Recommended comparison rules:

- compare only species present in both ExoGibbs and FastChem
- compare on mixing ratio basis
- apply a floor such as `1e-30` before taking logs
- compute both absolute log-scale error and fractional error
- report the worst species and worst layer for debugging

## 4. Benchmark Case Axes

The system should define cases by explicit axis values, not by ad hoc scripts.

### 4.1 Temperature Range

Minimum axis coverage:

- low: `300-700 K`
- mid: `700-1800 K`
- high: `1800-3500 K`

Case shapes:

- single-layer fixed temperature points
- isothermal profiles
- non-isothermal profiles with monotonic or mild curvature

Recommended first temperature points:

- `300 K`, `700 K`, `1200 K`, `2000 K`, `3000 K`

### 4.2 Pressure Profile

Minimum axis coverage:

- single-layer fixed pressures
- standard atmospheric profile: `1e2` to `1e-8 bar`
- optional narrower subranges for stress tests

Recommended first profile:

- `Nlayer = 100`
- `P = logspace(2, -8, 100)` in bar

### 4.3 Element Range

Benchmark cases should vary chemistry size as well as thermodynamic conditions.

Minimum axis coverage:

- default FastChem preset
- extended FastChem preset
- one reduced subset case for faster iteration during benchmark development

Case metadata should include:

- `n_elements`
- `n_species`
- preset or source files used

### 4.4 Abundance Pattern

Minimum axis coverage:

- solar-like baseline
- metal-rich scaling
- metal-poor scaling
- one perturbed abundance vector with selected element offsets

Recommended first set:

- baseline solar-like vector
- uniform metallicity scaling at `0.1x`, `1x`, `10x`
- one C/O-perturbed case if supported by the preset data

## 5. Pass/Fail Criteria

Benchmark output should distinguish benchmark execution success from scientific acceptance.

### 5.1 Runtime Benchmark Validity

A timing result is valid only if:

- no NaNs are present
- `hit_max_iter == false` for the measured solve, or below an allowed threshold for profiles
- final residual is below the configured residual threshold

Suggested default thresholds:

- single-layer: `final_residual <= max(10 * epsilon_crit, 1e-9)`
- profile: at least `99%` of layers valid, with no catastrophic NaN layer

### 5.2 FastChem Agreement Validity

A FastChem comparison case passes only if:

- ExoGibbs timing validity passes
- FastChem run completes successfully
- enough common species exist for comparison
- agreement metrics stay within configured tolerances

Suggested initial tolerances:

- `species_median_abs_log10_error <= 1e-3`
- `species_max_abs_log10_error <= 1e-1`
- no scientifically important species exceed a stricter case-specific threshold if designated

These thresholds should be stored in case definitions so they can be tightened later without changing the runner.

### 5.3 Overall Case Status

Each case should emit:

- `status = "pass"`
- `status = "fail_runtime"`
- `status = "fail_agreement"`
- `status = "error"`

`error` is reserved for runner failures such as missing reference data, missing FastChem binding, or malformed case configuration.

## 6. Output and Report Format

Use machine-readable JSON as the source of truth, plus a short markdown summary for human review.

### 6.1 JSON Result Schema

Suggested top-level layout:

```json
{
  "benchmark_version": "0.1",
  "git_commit": "abc123",
  "platform": "cpu",
  "jax_backend": "cpu",
  "timestamp_utc": "2026-03-16T00:00:00Z",
  "cases": [
    {
      "case_id": "fastchem_extended_single_T3000_P1e-8_solar",
      "category": "single_layer",
      "status": "pass",
      "setup": {
        "preset": "fastchem_extended",
        "n_elements": 80,
        "n_species": 752
      },
      "axes": {
        "temperature_K": 3000.0,
        "pressure_bar": 1e-8,
        "abundance_pattern": "solar",
        "profile_kind": null
      },
      "solver_options": {
        "epsilon_crit": 1e-10,
        "max_iter": 1000,
        "method": null
      },
      "metrics": {
        "converged": true,
        "final_residual": 2.1e-11,
        "iteration_count": 250,
        "hit_max_iter": false,
        "has_nan": false,
        "first_call_wall_s": 1.23,
        "warm_call_wall_s": [0.061, 0.060, 0.060],
        "warm_call_median_s": 0.060,
        "agreement": null
      }
    }
  ]
}
```

### 6.2 Markdown Summary

The markdown report should be concise and CI-friendly.

Recommended sections:

- environment summary
- benchmark matrix summary
- pass/fail table
- warm-runtime table sorted by category and case
- FastChem agreement table
- notable failures with short diagnostics

Recommended report style:

- one row per case
- show median warm runtime, first-call runtime, final residual, iteration count, status
- include links or file paths to the raw JSON artifact

## 7. Staged Implementation Plan

Implement in the smallest-risk order, keeping solver code unchanged.

### Stage 1: Case Definition and Result Schema

Deliverables:

- `benchmarks/cases.py` or similar static case registry
- JSON schema definition in code or documented dataclasses
- one CLI entry point that loads cases and writes JSON

Why first:

- locks down benchmark vocabulary before timing logic spreads across scripts

### Stage 2: Single-Layer ExoGibbs Runner

Deliverables:

- runner for one `equilibrium(...)` case
- first-call and warm-call timing support
- convergence and NaN checks
- JSON output for ExoGibbs-only cases

Why second:

- lowest complexity
- directly reuses patterns from `examples/benchmark/single_solve_benchmark.py`
- establishes the hot-call timing protocol needed for HMC-like workloads

### Stage 3: Profile Runner

Deliverables:

- runner for `equilibrium_profile(...)`
- support for `vmap_cold`, `scan_hot_from_top`, `scan_hot_from_bottom`
- per-layer diagnostics aggregation

Why third:

- adds realistic atmospheric workloads after single-solve infrastructure is stable

### Stage 4: FastChem Adapter and Agreement Metrics

Deliverables:

- adapter that runs FastChem on matching case definitions when available
- species matching and normalization utilities
- agreement metric computation

Why fourth:

- isolates external dependency handling from the core timing harness
- avoids blocking ExoGibbs-only benchmarking if FastChem is unavailable

### Stage 5: Markdown Report Generation

Deliverables:

- markdown summary generator from JSON artifacts
- sorted summary tables and failure highlights

Why fifth:

- reporting should consume stable JSON, not drive benchmark execution design

### Stage 6: CI/Regression Gating

Deliverables:

- minimal deterministic benchmark subset for CI
- optional larger local benchmark matrix
- threshold checks that fail only on explicit runtime/agreement regressions

Why last:

- gating should begin only after the benchmark outputs and tolerances are stable

## Recommended Initial Deliverable Set

For the first implementation pass, keep the scope narrow:

- 1 single-layer FastChem extended anchor case
- 1 atmospheric profile case on the same chemistry setup
- 1 FastChem comparison case on the same profile
- JSON artifact plus markdown summary
- no solver changes

This is sufficient to create an optimization baseline while keeping benchmark bring-up risk low.
