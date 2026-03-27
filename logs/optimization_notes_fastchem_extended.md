
# ExoGibbs Optimization Notes: FastChem Extended Case

This note records the performance work carried out on the reduced Newton system for a single representative ExoGibbs solve. It is intended as a short technical reference for future optimization work.

## Representative Benchmark Case

All comparisons in this note refer to the same benchmark case and should remain anchored to it unless an explicit benchmark change is made and documented.

- Preset: FastChem extended
- `logK` file: `fastchem/logK/logK_extended.dat`
- Element abundance file: `fastchem/element_abundances/asplund_2020_extended.dat`
- Elements: 80
- Species: 752
- Temperature: 3000 K
- Pressure: `1e-8` bar
- `epsilon_crit = 1e-10`
- `max_iter = 1000`

Important:

- Future optimization comparisons should not silently change this benchmark case.
- The production `B = A diag(n) A^T` assembly remains the `einsum`-based implementation.

## Accepted Changes

### 1. Stacked-RHS Linear Solve

Accepted production change:

- The reduced Schur-complement solve now performs one Cholesky factorization and one `cho_solve` on stacked right-hand sides `(rhs, An)`.

Reason:

- This is algebraically identical to solving the two right-hand sides separately.
- It reduced end-to-end solve time on the representative FastChem extended case without changing iteration count or convergence behavior.

Observed CPU result during this work:

- `split_rhs` baseline: about `61.89 ms` total solve, about `0.2476 ms/iteration`
- `stacked_rhs` accepted path: about `59.91 ms` total solve, about `0.2396 ms/iteration`
- Iteration count stayed at `250`
- Convergence stayed `True`
- Final residual stayed about `2.10e-11`

Observed practical result during this work:

- `stacked_rhs` improved the representative solve on both CPU and GPU.

### 2. Reused-Terms Residual Evaluation

Accepted production change:

- After the damped Newton step, residual evaluation reuses `A^T pi` and updates `gk` algebraically instead of recomputing both from scratch.

Reason:

- This is algebraically equivalent to the original residual-side computation.
- It reduced the profiled residual-evaluation bucket and gave a small end-to-end win on the representative FastChem extended case.

Observed CPU result during this work:

- Residual baseline: about `61.15 ms` total solve, about `0.2446 ms/iteration`, residual-evaluation bucket about `0.1928 ms`
- Reused-terms path: about `60.67 ms` total solve, about `0.2427 ms/iteration`, residual-evaluation bucket about `0.0797 ms`
- Iteration count stayed at `250`
- Convergence stayed `True`
- Final residual stayed about `2.10e-11`

## Rejected or Not-Adopted Experiments

### 1. B-Assembly `matmul` Variant

Tested idea:

- Replace the production `einsum` assembly of `B = A diag(n) A^T` with a `matmul` form.

Outcome on the representative FastChem extended case:

- Did not improve the real representative solve enough to justify replacing the baseline.
- Not adopted.

Observed CPU result during this work:

- `einsum`: about `60.92 ms` total solve, about `0.2437 ms/iteration`
- `matmul`: about `62.26 ms` total solve, about `0.2490 ms/iteration`

Observed practical result during this work:

- On GPU, the `matmul` variant was approximately neutral relative to `einsum`, so it still did not justify replacing the production baseline.

### 2. B-Assembly `outer_products` Variant

Tested idea:

- Precompute constant per-species outer products and assemble `B = sum_k n_k * outer_k`.

Outcome on the representative FastChem extended case:

- Severe end-to-end regression.
- Not adopted.

Observed CPU result during this work:

- `outer_products`: about `803.29 ms` total solve, about `3.2131 ms/iteration`

### 3. Older `split_rhs` Solve

Status:

- Retained only as a historical baseline during the investigation.
- Not kept as production code.

Reason:

- The stacked-RHS solve provided the same numerical behavior with better performance.

## Observed Outcomes

High-level conclusions from this work:

- The accepted production improvements are implementation-level optimizations only.
- The reduced Schur-complement formulation, Newton method, damping, convergence checks, and stopping criterion were not changed.
- The most reliable practical improvements observed on the representative FastChem extended case were:
  - stacked-RHS reduced solve
  - reused-terms residual evaluation
- The `einsum`-based `B` assembly remains the best production choice among the tested low-risk variants.
- In this work, end-to-end solve timing on the representative FastChem extended case was treated as the main decision criterion; profiled subcomponent timings were used only as supporting diagnostics.

## Guidance for Future Optimization

Most promising next directions:

- Prioritize changes that improve the real FastChem extended representative solve, not only a profiled subcomponent.
- Continue low-risk implementation-level improvements only if they produce measurable end-to-end gains on the representative case.
- Use the current benchmark case as the primary optimization anchor before considering smaller surrogate cases.

Not promising based on current results:

- Replacing the current `einsum` `B` assembly with the tested `matmul` variant.
- Precomputed per-species `outer_products` assembly for this representative case.

## Current Benchmark Commands

Single-solve timing:

```bash
NUMBA_CACHE_DIR=/tmp/numba_cache PYTHONPATH=src python examples/benchmark/single_solve_benchmark.py --platform cpu --repeat 20 --warmup 3 --print-diagnostics
```

Newton-profile timing:

```bash
NUMBA_CACHE_DIR=/tmp/numba_cache PYTHONPATH=src python examples/benchmark/newton_iteration_profile.py --platform cpu --solve-repeat 10 --profile-repeat 3
```
