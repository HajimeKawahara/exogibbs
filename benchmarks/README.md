# Benchmarks

Current benchmark runners:

- `python -m benchmarks.run_single_layer`
- `python -m benchmarks.run_profile`

Run from the repository root with `PYTHONPATH=src`.

## Example Commands

Single-layer benchmark:

```bash
PYTHONPATH=src python -m benchmarks.run_single_layer --output results/benchmarks/single_layer.json
```

Profile benchmark:

```bash
PYTHONPATH=src python -m benchmarks.run_profile --method vmap_cold --output results/benchmarks/profile_vmap_cold.json
PYTHONPATH=src python -m benchmarks.run_profile --method scan_hot_from_top --output results/benchmarks/profile_scan_hot_from_top.json
PYTHONPATH=src python -m benchmarks.run_profile --method scan_hot_from_bottom --output results/benchmarks/profile_scan_hot_from_bottom.json
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

JSON output is written to the path given by `--output`.

Recommended location:

- `results/benchmarks/`

## Interpreting Results

- `first_call_s`: first measured call after JIT setup; includes first-execution overhead and is not the main optimization target
- `warm_call_median_s`: main steady-state timing metric for repeated-call workloads
- `status`: conservative benchmark outcome
- convergence-related metrics: use these to decide whether timing is meaningful

Typical status meaning:

- `pass`: diagnostics indicate acceptable convergence
- `fail_runtime`: NaN or explicit convergence failure
- `error`: diagnostics were missing or incomplete, so success was not assumed
