# Fixed-support solver v2 production migration

## Status

The fixed-support v2 route is promoted as the production default.

The fixed-support v2 solver is the only solver targeted for the new product
route.  The frozen fixed-support v1 implementation remains archival benchmark
evidence and is not a production fallback.

**Runtime retirement follow-up (2026-07-27):** the explicit `head_v1` route,
its mutable replay runners, and its implementation were removed after the
promotion.  Frozen v1 summaries, recorded hashes, and this migration record
remain immutable historical evidence.  Rollback continues to use a preceding
release artifact; it does not dispatch to an older solver inside the current
release.

The production default is:

```python
from exogibbs.api import CondensateEquilibriumOptions

options = CondensateEquilibriumOptions(
    fixed_support_v2_preset="validated_2026_07",
)
```

`condensate_equilibrium` and `condensate_equilibrium_profile` both use the
same v2 outer lifecycle.  Support selection, inactive-driving closure, and
support expansion remain outside the fixed-support solver.

`head_v2` does not silently retry a failed layer with v1.

## Production preset

The named `validated_2026_07` preset fixes the configuration used by the
closed GPU validation:

```text
epsilon schedule                         -11, -13, -15, -17
maximum normal iterations                1000
maximum normal line-search trials        20
maximum restoration calls                2
maximum restoration iterations           100
maximum restoration line-search trials   20
initial support top-k                     8
initial support limit                     16
support additions per round               8
total support limit                       128
maximum lifecycle rounds                  15
budget relative floor                     1e-6
support closure tolerance                 1e-8
```

Arbitrary `FixedSupportV2Config` values remain available only through the
experimental prepared-plan API.  The production route does not translate
environment variables into kernel configuration.

## Historical v1 status supersession

The historical convergence observations for these cases are formally
superseded as production gates:

```text
small_highT_control
large_water_activity64
```

The frozen artifacts remain immutable evidence that the historical run
reported convergence.  They are not a reproducibility contract for the
current v1 trajectory because the historical artifact hashes freeze the
reports, not the then-mutable v1 source.

The final exact-state artifact records:

| Case | Historical v1 | Current v1 at validation | v2 | Independent KKT |
| --- | --- | --- | --- | --- |
| `small_highT_control` | converged | not converged | converged | passed |
| `large_water_activity64` | converged | not converged | converged | passed |

The migration therefore applies these rules:

1. Do not tune or extend v1 to reproduce the historical trajectories.
2. Preserve the frozen artifact hashes and the recorded mismatch.
3. Require v2 to converge both historical-success coverage cases.
4. Do not require current v1 status reproduction for promotion.

## Rollback boundary

Rollback is release-based, not solver-based.

- Before promotion, the current release remains the rollback target.
- After promotion, deploy the preceding wheel, container, or commit.
- Do not add a public v1 production preset.
- Do not add an in-solver v2-to-v1 fallback.

This keeps each result attributable to one solver architecture and avoids
shipping a permanent dual-solver product contract.

## Completed promotion gates

Promotion completed after all of the following:

1. full CPU repository regression;
2. focused v2 and archive-contract regression;
3. CPU archive preflight;
4. the ten-case exact-state GPU matrix;
5. all five production-route lifecycle families;
6. the focused water-128 restoration gate when restoration ownership or
   limits change;
7. approved cold-compilation and warm-execution budgets;
8. an explicit promotion decision.

`CondensateEquilibriumOptions()` now selects `head_v2`.

## Production-profile regression and runtime measurement

The public-route gate is:

```console
csh benchmarks/fixed_support_v2/run_fixed_support_v2_production_profile_gpu.csh
```

It calls `condensate_equilibrium_profile` through the default route for these
families:

```text
solar_highT_no_condensate_gas_regression
solar_silicate_first_condensation
solar_water_condensation
lowT_strong_condensation_budget_stress
near_phase_boundary_support_sensitivity
```

Every layer must report the v2 route identity, public convergence, fixed
support convergence, an independently re-evaluated final KKT pass, finite
final state, and lifecycle closure.  A layer with no thermodynamically
eligible condensate may instead close through the explicit gas-only outcome.
For each family the runner clears JAX's in-memory caches, performs one cold
call, and immediately repeats the same call warm.  It records:

- per-family and total solver compilation time;
- per-family and total solver execution time;
- full public-call wall time;
- device, JAX x64 state, commit, dirty-worktree state, and source hashes.

The runner samples active GPU compute processes before and after every cold
and warm call.  Any process other than the runner invalidates the runtime
measurement and prevents the production-profile gate from passing.

The runtime gate uses the public default `return_diagnostics=false` path.
Terminal replay kernels are therefore excluded from the production budget;
their time is still represented as zero in the artifact schema.

The approved `a100_40gb_2026_07` runtime budgets are maximum per-family cold
compilation, cold wall, warm execution, and warm wall seconds:

```text
maximum cold compilation                 900 s
maximum cold wall                        960 s
maximum warm execution                    20 s
maximum warm wall                         25 s
```

The default runner invocation enforces all four limits:

```console
csh benchmarks/fixed_support_v2/run_fixed_support_v2_production_profile_gpu.csh
```

Alternate limits may be supplied for deliberate diagnostic runs:

```console
csh benchmarks/fixed_support_v2/run_fixed_support_v2_production_profile_gpu.csh \
  MAX_COLD_COMPILE MAX_COLD_WALL MAX_WARM_EXECUTE MAX_WARM_WALL
```

The ten-case exact-state solver matrix remains separate correctness evidence.
The water-128 restoration run remains a focused gate when restoration
ownership or limits change.

## Current migration verification

The promoted-default integration change has passed the following CPU checks:

```text
full repository unit suite              508 passed, 22 warnings
focused API/v2/archive suite             139 passed, 18 warnings
archived solver input preflight           10/10
archived lifecycle family preflight         5/5
frozen v1 baseline integrity              passed
production-profile input preflight         passed
production-profile GPU families              5/5
```

The warning count is unchanged from the pre-migration full-suite baseline;
the test count increases by ten targeted migration tests.

### A100 production-profile measurement

The public production-profile runner completed on 2026-07-24 with JAX 0.4.30,
x64 enabled, and one NVIDIA A100-PCIE-40GB:

```text
cold families passed                       5/5
warm families passed                       5/5
cold layers passed                       54/54
warm layers passed                       54/54
active lifecycle closures                37/37 per phase
gas-only lifecycle closures              17/17 per phase
independent final KKT passes              37/37 per phase
maximum final support count                  57

maximum family cold compilation       742.908809 s
maximum family cold wall              788.475670 s
maximum family warm execution          13.851622 s
maximum family warm wall               17.493919 s

total cold compilation                2094.745725 s
total cold wall                       2235.905232 s
total warm execution                    32.409345 s
total warm wall                         43.154791 s
```

The maximum values all came from `solar_water_condensation`.  This first run
established the measurements before runtime-budget approval and therefore
recorded:

```text
runtime_budget.approved_limits_supplied    false
runtime_budget.passed                      false
production_profile_gate_passed             false
promotion_authorized                       false
```

The ignored raw artifacts are pinned by these SHA-256 values:

```text
production_preflight.json
3b765b1e2159f3878872c0d14c3383dee6db1789b500f53e7057f977246e938f

summary.json
c53d12c34bcbf7923bc33469e0bdc30982c58e4a80c7b884ca204687381e0d6b

summary.md
7899643888f888d2cbc7d41688f34a4bd8be60eeab00cdc24793e84740581176
```

The measured values were accepted with the approved limits above.

### Promoted-default verification and GPU exclusivity

A promoted-default verification started at 11:10:45 JST on 2026-07-24.  All
five cold and five warm families, including every applicable layer KKT and
lifecycle gate, passed through `CondensateEquilibriumOptions()` without an
explicit route.  A different user's GPU compute process started at 11:19:26 and
overlapped the remainder of the run, however, so its runtime values are not
promotion evidence.  In particular, the contended water-family warm values
of 22.844174 seconds execution and 34.956163 seconds wall time are treated as
an invalid measurement rather than a reason to change the approved limits.

The runner now samples GPU compute processes around every measured call,
records `environment.exclusive_gpu_measurement`, and refuses promotion-gate
success if another process is observed.  Subsequent production-profile
regressions must pass both this exclusivity check and the four limits above.

### Final exclusive promoted-default gate

The final promoted-default run completed on 2026-07-24 with no external GPU
compute process in any of the 22 samples taken around the measured calls.  It
recorded:

```text
cold families passed                       5/5
warm families passed                       5/5
cold layers passed                       54/54
warm layers passed                       54/54
active lifecycle closures                37/37 per phase
gas-only lifecycle closures              17/17 per phase
independent final KKT passes              37/37 per phase
maximum final support count                  57
maximum lifecycle rounds                      8

maximum family cold compilation       758.593398 s
maximum family cold wall              802.314860 s
maximum family warm execution          13.886243 s
maximum family warm wall               17.839362 s

total cold compilation                2131.842243 s
total cold wall                       2265.649461 s
total warm execution                    32.748221 s
total warm wall                         44.554101 s
```

All four `a100_40gb_2026_07` runtime checks passed.  Cold and warm runs
selected identical status, lifecycle outcome, support, and terminal status
for all 54 profile layers.  The maximum final independent KKT components over
the active layers were:

```text
gas stationarity          9.094947017729282e-13
condensate stationarity   6.0632787679537614e-09
scaled budget             2.0015079372503833e-12
complementarity           3.552713678800501e-15
scaled total density      1.9092789328702238e-16
```

The gate recorded:

```text
environment.exclusive_gpu_measurement       true
runtime_budget.passed                       true
correctness_passed                          true
production_profile_gate_passed              true
promotion_authorized                        true
```

The ignored final artifacts are pinned by these SHA-256 values:

```text
production_preflight.json
1669ba62480a1a8a3ea11ce4f1b8b4355ef61ce42b2a5d9fd92c44e951588378

summary.json
61f40fe7750be2f77727780985175ecd215ae0770a4bb731d41d331000a53d24

summary.md
f01f678f383096eb4c141bdad04cb3698e9d57c275b27edaa71b3c20feb7b98e
```

The five production-route source hashes in the final artifact match the
current files exactly.  This closes the correctness, exclusivity, and runtime
budget requirements for the promoted default.

## Prohibited migration changes

- reopening M0--M6 solver validation;
- adding v1 globalization behavior;
- adding layer-, species-, or benchmark-specific branches;
- weakening independent KKT tolerances;
- moving support lifecycle into the fixed-support solver;
- adding automatic v1 fallback;
- changing the validated numerical preset to recover runtime;
