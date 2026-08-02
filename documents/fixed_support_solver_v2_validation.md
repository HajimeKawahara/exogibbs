# Fixed-support solver v2 experimental validation closure

> **Historical-record note (2026-07-27):** this document records the
> pre-promotion validation closure.  The later production migration promoted
> v2, and the cleanup then removed the executable `head_v1` route and mutable
> v1 replay tools.  Frozen summaries and recorded hashes below remain
> unchanged as historical evidence.

## Status

The fixed-support PD-IPM/R-GIE solver v2 experimental solver validation was
closed on 2026-07-23.

This closure covers the M0--M6 experimental implementation, mathematical
contracts, fixed-support GPU solver matrix, and the external support-lifecycle
validation.  It does not promote v2 to a production preset or replace the
current public condensate-equilibrium route.

Further ad hoc globalization changes are outside this closed validation
series.  Any production migration is a separate task with its own API,
performance, regression, and release gates.

## Validated implementation

The validated v2 path contains:

- the canonical fixed-support residual and objective contract;
- the reduced R-GIE normal direction and persistent original filter;
- a persistent physical-amount elastic restoration solver;
- typed NORMAL, RESTORATION, CONVERGED, and FAILED controller ownership;
- the restoration return map with the linearized bound-multiplier update;
- complete fixed-epsilon solves inside barrier continuation;
- exact method-0 SOC as a separate ordered operation;
- prepared-profile support bucketing and external support closure;
- terminal normal/restoration replay diagnostics;
- per-stage restoration-call, accepted-iteration, and return diagnostics.

The v1 solver remains an experimental baseline.  The validation did not add a
new v1 globalization behavior.

## Final GPU solver matrix

Artifact:

```text
results/fixed_support_v2_final_solver_matrix_gpu/
```

SHA-256:

```text
preflight.json  e823c7d321d65a26e0bf5d1ba2873e70b4fdf57906da6ea805bd3f4456fafb34
summary.json    5b8480fe616415b5dbbc7900be053c4f8bcfc14f11e831d18f0b774334eb9281
summary.md      3536a3f7fce7705b1cce5291d911f49fe9248da3d3b6889e28398ec789d73391
```

The run used an NVIDIA A100-PCIE-40GB, x64 JAX, the epsilon schedule
`(-11, -13, -15, -17)`, and a global maximum of 100 accepted restoration
iterations per call.

Results:

- all 10 exact-state v2 cases converged through all four epsilon stages;
- both small-support controls converged;
- all eight large-support stress cases converged, including supports 64, 96,
  and 128;
- all 10 final states were finite;
- all 10 independent KKT gates passed;
- the nine cases that had already converged before the water-128 safeguard
  change retained identical normal-iteration counts and final KKT values.

Maximum final KKT components across the 10 cases:

```text
gas stationarity          9.094947017729282e-13
condensate stationarity   6.9907257760348784e-09
scaled budget             9.808603450597192e-13
complementarity           3.552713678800501e-15
scaled total density      3.811460003439431e-16
```

Measured v2 totals across the 10 single-layer cases:

```text
compilation               218.81969300005585 s
warm execution              3.364669025875628 s
first diagnostic compile   43.891206751577556 s
warm diagnostics            0.06486777681857347 s
```

Correctness is accepted for the experimental validation.  No production
runtime budget has been approved.

## Support lifecycle

The final solver-matrix artifact links the previously completed corrected
support-lifecycle artifact:

```text
results/fixed_support_v2_corrected_gpu/summary.json
sha256 afc07204a81203d6ab9a4473835885a408a5bf28500e3dc3b6bc05ccfee1146d
```

That run recorded:

- 191 fixed-support solves and 191 converged outcomes;
- 37 condensating profile layers closed under the external support lifecycle;
- 17 gas-only layers with no candidate;
- all five curated lifecycle families closed;
- final closed supports from 3 through 57;
- no material representational-floor inventory.

Fixed-support `support_closed=False` values in the solver-only stress matrix are
not solver failures.  Support closure is intentionally evaluated by the
separate outer lifecycle.

## Water-128 closure

The focused safeguard artifact is:

```text
results/fixed_support_v2_water128_gpu100/summary.json
sha256 b5b08e22a0f6f0101d1926df677f2da14b54082e4c39e93392bc43470aa47f77
```

The earlier 50-iteration run stopped with `RESTORATION_MAX_ITER`.  CPU runs
with limits 100 and 200 converged to the same four-stage result.  The final
A100 run required 61 accepted restoration iterations, leaving 39 iterations
of global safeguard margin.

The stage-0 restoration return audit recorded:

```text
accepted restoration iterations    61
dual fraction                       0.821551388451323
bound multiplier reset              true
equality multiplier reset           true
representation floor applied        false
scaled budget floor injection       0
scaled total-density injection      0
```

The return point was primal-feasible but not stationary, as required by the
return-map contract.  The resumed normal solver reduced the large post-return
stationarity and complementarity residuals to the final KKT tolerances.

The water-128 failure was therefore a premature global safeguard, not a
nonfinite direction, restoration line-search failure, or layer-specific
algorithm defect.

## Integrity and tests

The final run recorded:

- 10 of 10 solver preflight cases passing;
- five of five lifecycle-family initialization contracts passing;
- exact provided-state adapter differences equal to zero;
- frozen historical artifact integrity passing;
- 23 relevant runner, v1, and v2 source files with matching recorded hashes;
- both predecessor GPU artifact hashes present and matching;
- 69 focused M0--M6 unit and integration tests passing before the run.

The artifact records Git commit
`202ecbea71f7bec7b04340dd6529bf163eab0afc` and
`worktree_dirty=true`.  The per-file hashes define the exact validated source
state, but a clean archival commit is still required before release work.

The clean archival preparation promotes the validation runner and frozen
baseline inputs from local `volatiles_*` scratch directories to
`benchmarks/fixed_support_v2/`, and replaces workstation-specific paths in the
C shell entry points with repository-relative paths.  The original artifact's
embedded source hashes intentionally continue to identify the pre-archive
files that produced it.  The relocation does not change the v1 or v2 numerical
solver implementation; subsequent runs record hashes for the archived paths.

## Explicit limitations

1. `production_preset_promoted` and `promotion_ready` remain false.
2. The historical v1 convergence labels are not reproduced for
   `small_highT_control` and `large_water_activity64`.  Current v1 source and
   results are hashed, and v2 converges both historical-success cases, but
   claims about exact reproduction of the old v1 trajectory must retain this
   caveat.
3. The final solver matrix is not an outer support-selection benchmark.
4. Compilation dominates execution for many distinct support shapes.
5. No clean release commit, public-route selection, production runtime budget,
   or full production regression has been approved.

## Closure decision

The v2 experimental solver-validation questions are considered answered:

- the solver handles both small and genuinely large fixed supports;
- restoration has persistent ownership and an auditable return;
- continuation does not advance failed stages;
- the external lifecycle closes all curated families;
- the former water-128 failure converges under the global safeguard of 100;
- no layer-, species-, or benchmark-specific branch is required.

The deferred hybrid log-gas/amount-condensate formulation is not triggered by
this matrix because no systematic inactive-bound stall remains.

The next authorized work, if requested, is a separate production migration:

1. create a clean archival commit for the validated source and documents;
2. reconcile or formally supersede the two historical v1 status labels;
3. define the public API/preset opt-in and rollback boundary;
4. agree on compilation and execution budgets;
5. run the full repository and production-profile regression gates;
6. promote only after those migration gates pass.

## Production migration follow-up

Production migration subsequently selected a v2-only product direction.  The
fixed-support v1 implementation remains archival evidence rather than a
production fallback.  The two historical v1 status mismatches are formally
superseded for production gating, while their frozen evidence and mismatch
records remain unchanged.

The production API, immutable validation preset, release rollback boundary,
approved runtime budget, final exclusive GPU gate, and default promotion are
recorded in
`documents/fixed_support_solver_v2_production_migration.md`.
