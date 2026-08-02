# Benchmarks

Current benchmark runners:

- `python -m benchmarks.run_single_layer`
- `python -m benchmarks.run_profile`
- `python -m benchmarks.fastchem4.run_production_comparison` (requires an
  external FastChem v4.0.3 standalone executable)

The fixed-support v2 production-profile gate and frozen pre-promotion evidence
are documented in `benchmarks/fixed_support_v2/README.md`.
The independent production comparison with FastChem 4 is documented in
`benchmarks/fastchem4/README.md`.

Run from the repository root with `PYTHONPATH=src`.

## Example Commands

Single-layer benchmark:

```bash
PYTHONPATH=src python -m benchmarks.run_single_layer --output results/benchmarks/single_layer.json
```

Profile benchmark:

for bash

```bash
PYTHONPATH=src python -m benchmarks.run_profile --method vmap_cold --output results/benchmarks/profile_vmap_cold.json
PYTHONPATH=src python -m benchmarks.run_profile --method scan_hot_from_top --output results/benchmarks/profile_scan_hot_from_top.json
PYTHONPATH=src python -m benchmarks.run_profile --method scan_hot_from_bottom --output results/benchmarks/profile_scan_hot_from_bottom.json
```

for csh

```sh
(setenv PYTHONPATH src; python -m benchmarks.run_profile --method scan_hot_from_top --warmup 3 --repeat 20 --output /home/kawahara/tmp/profile_benchmark_0.3.6_.json)
```

Recommended bring-up run:

- use `--warmup 0 --repeat 1`
- this is the fastest way to confirm the benchmark case, JAX backend, and JSON output path work

Recommended benchmark run:

- use `--warmup 3 --repeat 20`
- treat `warm_call_median_s` as the main optimization metric

Examples:

```bash
PYTHONPATH=src python -m benchmarks.run_single_layer --warmup 0 --repeat 1 --output results/benchmarks/single_layer_bringup.json
PYTHONPATH=src python -m benchmarks.run_single_layer --warmup 3 --repeat 20 --output results/benchmarks/single_layer_benchmark.json

PYTHONPATH=src python -m benchmarks.run_profile --method scan_hot_from_top --warmup 0 --repeat 1 --output results/benchmarks/profile_bringup.json
PYTHONPATH=src python -m benchmarks.run_profile --method scan_hot_from_top --warmup 3 --repeat 20 --output results/benchmarks/profile_benchmark.json
```

## Output

The timing runners write JSON to the path given by `--output`. The FastChem 4
comparison additionally writes `<output-stem>.preflight.json` and a Markdown
report beside its JSON result.

Recommended location:

- `results/benchmarks/`
- FastChem 4 comparison: `results/fastchem4_production_comparison/`

## Interpreting Results

The status semantics below apply to the timing runners. The FastChem 4
comparison uses `complete` or `incomplete` for execution completeness and does
not apply a scientific agreement threshold; see its dedicated README.

- `first_call_s`: first measured call after JIT setup; includes first-execution overhead and is not the main optimization target
- `warm_call_median_s`: main steady-state timing metric for repeated-call workloads
- `status`: conservative benchmark outcome
- convergence-related metrics: use these to decide whether timing is meaningful

Typical status meaning:

- `pass`: diagnostics indicate acceptable convergence
- `fail_runtime`: NaN or explicit convergence failure
- `error`: diagnostics were missing or incomplete, so success was not assumed
