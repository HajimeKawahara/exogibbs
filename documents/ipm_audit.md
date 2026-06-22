# IPM Audit Summary

## Restart Guidance: Avoid More Ad Hoc Route Fixes

Future condensate solver sessions should not continue by adding another
support retry, acceptance fallback, or row-specific gate before the PD-IPM/RGIE
contract is audited.  The HEAD route has already accumulated practical repair
layers for public convergence, full-budget consistency, and support closure.
Those layers are useful guards, but they should not substitute for a coherent
solver core.

Before implementing another route-level fix, audit the active-support
PD-IPM/RGIE path as a numerical method:

- define one residual vector for the active-support fixed problem;
- define whether the dual variable is physical-space `z` or log-space
  `rho = log(z)`, and use that convention consistently;
- verify the analytic Jacobian or reduced system against finite differences;
- verify that the Newton direction satisfies the linearized KKT residual before
  clipping or line search;
- replace component-wise direction clipping with regularization, damping,
  trust-region logic, or step-length control that preserves the KKT coupling;
- use inactive condensate driving as an active-set dual-feasibility violation,
  not as another independent replay target.

Concrete first task:

1. Add a small synthetic active-support PD-IPM audit that compares the
   implemented residual/Jacobian/reduced direction against finite differences.
2. Separate or rename the current `rho` conventions so the old physical-dual
   diagnostic path cannot be confused with the algorithm-v1.1 log-dual path.
3. Check the `delta_qtot` row and log-space fraction-to-boundary logic before
   changing support selection policy again.

FastChem4 and fastchem3 outputs remain comparison targets only.  Do not use
their public/runtime/trace values as constructor inputs, and do not treat exact
FastChem replay as the solver objective.

## 2026-06-19 PD-IPM/R-GIE Contract Checkpoint

The first PD-IPM cleanup pass completed the contract checks that should precede
any further HEAD-route retry policy changes:

- `rho` is now treated as the algorithm-v1.1 log dual, `rho = log(eta)`, in the
  PD-IPM/R-GIE state builder.  The builder can infer either `rho` or `eta` from
  the other one, and rejects inconsistent or non-positive dual input.
- An active-support PD-IPM GIE residual/Jacobian audit was added for the fixed
  support problem, using variable order `(q, r, lambda, rho, qtot)`.
- The reduced R-GIE direction is audited against that full active-support GIE
  linearization by checking `J delta + F`.
- Component-wise clipping of `delta_r` and `delta_rho` is now documented and
  tested as a globalization/safety alteration, not as the PD-IPM Newton
  direction itself.  It can break the coupled GIE linearization even when it
  protects individual log variables.
- An opt-in `step_control_policy="scalar_fraction_to_boundary"` path was added
  through the reduced step, thermo-valid callsites, and continuation records.
  This keeps the coupled Newton direction intact and applies a scalar
  fraction-to-boundary step length instead.  HEAD route v1.10 integrates this
  policy as a support-free fallback-only retry family.  HEAD route v1.11 then
  promotes the same policy to the public primary default so the PD-IPM/R-GIE
  path, not the component-clipped globalization path, is the main line.

Fresh explicit-support curated lifecycle comparison was run from public API
restricted payloads only; FastChem values were not used as constructor inputs.
The volatile artifacts are:

- `volatiles_artifacts/pdipm_step_control_curated_comparison.json`
- `volatiles_artifacts/pdipm_step_control_curated_comparison.md`

Summary:

| variant | route converged | reached final barrier | main stop reasons | median final residual |
|---|---:|---:|---|---:|
| `component_clip_default` | 0 / 14 | 0 / 14 | `no_p_armijo_trial`: 13, `current_barrier_not_centered`: 1 | `3.725e+01` |
| `scalar_fraction_to_boundary` | 12 / 14 | 12 / 14 | `final_barrier_centered`: 12, `current_barrier_not_centered`: 2 | `4.386e-05` |

Interpretation:

- The scalar fraction-to-boundary path is a much better PD-IPM candidate on
  these explicit-support lifecycle reruns.
- As a production-safe public surface, it was initially kept behind the v1.10
  retry path.  The v1.11 policy decision intentionally promotes it anyway:
  public metric regressions are now tracked as solver blockers instead of being
  hidden by reverting to the component-clipped primary.
- The two remaining `current_barrier_not_centered` rows were decomposed before
  route-level adoption:
  `carbon_rich_graphite_window__T1300_P1_corrected` and
  `complex_heavy_element_or_boron_titanium_zirconium_case__T1100_P1_corrected`.
  Relaxing the center gate alone did not repair either row.  The IPOPT
  persistent-filter style variant repaired the graphite row, while the
  heavy/Ti/Zr row moved to `no_p_armijo_trial`.
- The line-search/filter interaction still needs auditing.  The scalar policy
  avoids the dominant catastrophic Armijo failures, but many successful rows
  still end inner loops with no selected trial after centering.

HEAD route v1.10 wiring result:

- The support-free retry candidate pool now compares support-cap retry, staged
  support-growth retry, and scalar step-control retry with the same
  `(positive_inactive_count, max_positive_inactive_driving, support_count)`
  closure score.
- Curated end-to-end tests pass (`8 passed, 26 warnings`).  The water
  low-temperature support-growth regression now selects scalar step-control
  retry with 143 active support species and temperature-valid inactive closure
  `0 / 0`, instead of the v1.9 staged candidate with 162 active support species.
- The integration is computationally more expensive because fallback rows now
  evaluate one additional fresh API retry family.  That cost should be measured
  before broadening the retry matrix further.

Scalar primary-default validation was attempted after v1.10.  As a
production-safe metric change it was not ready: the original curated contracts
regressed to `6 failed, 2 passed` when `scalar_fraction_to_boundary` replaced
`component_clip` as the public primary default, including support-free water
rows returning to `not_converged`.  The v1.11 policy decision accepts that
regression and redefines those rows as expected PD-IPM-first blockers.  After
the curated contracts were updated to that blocker surface,
`pytest -q tests/endtoend/curated_cases` passed as `8 passed, 21 warnings`.

HEAD route v1.11 wiring result:

- The public condensate default is now `head_route_v1_11_pdipm_scalar_primary`
  with `head_route_primary_step_control_policy="scalar_fraction_to_boundary"`.
- HEAD route v1.10 remains the production-safe baseline for comparison, not the
  current main line.
- The 99-row FastChem4 comparison under v1.11 has 95 `converged`, 1
  `converged_with_caveat`, and 3 `not_converged` rows.  The largest regression
  is `solar_water_condensation` layer 0 with
  `dG/RT Exo-FC = -3297.0902626575994` and a very large relative budget
  residual.

Next solver work should focus on fixing the v1.11 blocker surface under scalar
step control, especially filter/merit trial acceptance and the water
support-free budget regressions.  Reverting the public default to component
clipping would recover v1.10 metrics, but it would also move the main route away
from the PD-IPM/R-GIE contract being audited here.

## 2026-06-20 Ipopt-Oriented Blocker Audit

The Ipopt reference implementation reinforces the next design rule: do not
repair a PD-IPM Newton step by clipping individual components after the coupled
direction has been computed.  Ipopt instead invests in the surrounding
algorithmic contract:

- bound-pushed primal initialization and multiplier initialization before the
  first nonlinear step;
- separate primal/dual fraction-to-boundary step lengths;
- filter-based line search with h-type feasibility acceptance, tiny-step
  detection, watchdog, second-order correction, and restoration phases;
- warm-start pushes that keep both primal variables and multipliers away from
  pathological boundaries.

The ExoGibbs v1.11 line now follows this direction by keeping
`scalar_fraction_to_boundary` as the public primary step-control policy and
adding blocker diagnostics instead of reverting to component clipping.

Implementation checkpoint:

- scalar fraction-to-boundary reports now identify the limiting active
  condensate variable group (`r` or `rho`), local support index, species name,
  raw alpha, safety alpha, and top limiting species;
- support species names are propagated from the public condensate API through
  lifecycle continuation into the reduced PD-IPM/R-GIE reports;
- the diagnostics are produced from fresh ExoGibbs API inputs and do not use
  FastChem4 public/runtime/trace values as constructor inputs.

Solar-water layer 0, the largest v1.11 blocker, was rerun through the public
support-free path with scalar primary control and support cap 128.  It still
stops at `no_p_armijo_trial`; the scalar step length is
`2.2303859960980942e-08`.  The fraction-to-boundary limiter is an `r` update for
`FeS2(s)`:

| rank | species | group | current log value | direction | safety alpha |
|---:|---|---|---:|---:|---:|
| 1 | `FeS2(s)` | `r` | -12.165855883311064 | -44611112.235311896 | 2.2303859960980942e-08 |
| 2 | `FeS(s,l)` | `rho` | -16.314562413177427 | -39866972.657306716 | 2.4958002418516698e-08 |
| 3 | `MgS(s)` | `r` | -14.662960164420406 | -23917560.19974084 | 4.16012332232274e-08 |
| 4 | `MgO(s,l)` | `r` | -14.462459874411272 | -19682405.697240468 | 5.055276348355638e-08 |
| 5 | `Fe2O3(s)` | `r` | -13.665460035847312 | -12417390.506961612 | 8.012955696626994e-08 |

Interpretation:

- the blocker is not a missing support-cap retry or a FastChem replay issue;
- the active-support state contains many species whose primal/dual interior
  positions are poor enough that a coupled Newton direction points almost
  directly into the log boundary;
- the next repair should be an Ipopt-like initial-point and dual
  initialization pass for the active support, followed by filter/restoration
  work if the step remains unacceptable.

## 2026-06-20 Dual Initialization Repair

The first repair following the blocker audit is HEAD route v1.12.  It keeps
the scalar fraction-to-boundary PD-IPM primary route and changes the
active-support dual initialization:

- when `rho` is inferred from `epsilon`, compute `eta = exp(rho)` as before;
- apply `eta = max(eta, 0.1)` before forming the reduced R-GIE Newton system;
- store `dual_initialization_policy`, `dual_push_floor`,
  `dual_push_applied_count`, and `dual_push_max_log_delta` in continuation
  input diagnostics;
- do not use FastChem4 public/runtime/trace values as constructor inputs.

This is intentionally a pre-Newton initialization repair, not a component-wise
alteration of the Newton direction.  It matches the Ipopt pattern of pushing
multipliers away from pathological boundaries before asking the line search to
globalize the step.

Observed effects:

| check | v1.11 scalar primary | v1.12 dual push |
|---|---:|---:|
| support-free water layer 0 | not converged | converged, tier 1, inactive `0 / 0` |
| support-free water layer 5 | not converged under `0.01` floor | converged, tier 1, inactive `0 / 0` under `0.1` floor |
| full-profile public status | 95 converged, 1 caveat, 3 not converged | 99 converged |
| max `|dG/RT|` vs FastChem4-scaled | 3297.0902626575994 | 0.09632193272911493 |

The remaining high-priority rows after this repair are no longer the original
water layer 0 alpha blocker.  They are `carbon_rich_graphite_window` layer 7
for the largest FastChem4-scaled Gibbs gap and
`solar_metal_sulfide_or_Fe_Ni_S_region` layer 7 for the largest
temperature-valid inactive driving.

## Scope

This note summarizes the diagnostic work on the condensate equilibrium IPM path in:

- [src/exogibbs/optimize/pipm_rgie_cond.py](/home/kawahara/exogibbs/src/exogibbs/optimize/pipm_rgie_cond.py)
- [src/exogibbs/optimize/minimize_cond.py](/home/kawahara/exogibbs/src/exogibbs/optimize/minimize_cond.py)

The goal was to identify whether the current failure mode was caused by:

- stale residual evaluation,
- bad acceptance policy,
- gas-step limiter domination by trace species,
- reduced-system conditioning,
- reduced-vs-full PIPM algebra,
- or the IPM direction itself.

All work below was diagnostic-only. The active production solver path was not replaced.

## What Was Tested

### 1. Fresh post-update residual evaluation

The post-step residual evaluation was corrected to recompute a fresh residual on the updated state.

Finding:

- This was a real bug and needed fixing.
- It did not remove the core crawling behavior at feasible `epsilon`.

### 2. Adaptive sk-guarded epsilon scheduling

Stage-start `sk`-infeasible jumps were removed by using an adaptive guard on the barrier schedule.

Finding:

- This removed a real pathology in the outer continuation schedule.
- After that fix, the remaining dominant issue was still gas-step-limited crawling.

### 3. Acceptance-policy audit via lambda scan

For a fixed current direction, multiple trial `lambda` values were evaluated with fresh post-update residuals.

Finding:

- Larger feasible `lambda` values usually made the fresh residual much worse.
- Smaller `lambda` values were at best only marginally better.
- This ruled out acceptance policy as the main remaining problem.

Conclusion:

- Backtracking line search was not the next primary fix.

### 4. Gas-step limiter decomposition

The gas limiter was decomposed into per-species contributions.

Finding:

- `lam1_gas` was usually set by the shared global gas cap.
- It was not usually dominated by a few trace-species guard hits.

Conclusion:

- Trace-aware gas limiting or special trace clipping was not the main issue.

### 5. Frozen-condensate gas-only reference direction

A gas-only reference direction was computed with condensates frozen into an effective element budget.

Finding:

- The condensate-coupled gas direction differed strongly from this reference.
- But the gas-only reference was often even more pathological.

Conclusion:

- The issue was not that condensate coupling uniquely corrupted an otherwise good gas direction.

### 6. Reduced-system conditioning experiments

Optional reduced-system backends were tested, including:

- `augmented_lu_row_scaled` (default),
- `augmented_lu_rowcol_scaled`,
- `schur_cholesky_reg` with small diagonal regularization.

Finding:

- Row/column scaling made no material difference.
- Schur/Cholesky with small regularization only perturbed the direction slightly.
- None of these experiments materially improved `lam1_gas` or fresh post-step residual.

Conclusion:

- Simple conditioning/scaling changes were not the next primary fix.

### 7. Full-vs-reduced PIPM GIE comparison

The full PIPM GIE linearization was solved directly and compared against the current reduced-GIE path at the same state.

Finding:

- Full and reduced PIPM directions matched to roundoff on representative cases.
- After applying the same condensate clipping and evaluating the same post-step residual, they were effectively identical.

Conclusion:

- The reduced elimination / reconstruction path was not the cause.
- The problem was the PIPM direction itself, not the reduced implementation.

### 8. One-step full PDIPM comparison

An experimental full PDIPM GIE direction was compared against PIPM at one state.

Important detail:

- The first PDIPM diagnostic used `rho = epsilon - ln_mk`.
- That implies `Fc = ln_mk + rho - epsilon = 0` at the initial state.

Finding:

- Under that on-center initialization, PDIPM and PIPM produced the same primal direction to displayed precision.
- One-step fresh residuals were the same.

Conclusion:

- The previous one-step PDIPM equivalence was partly a center-path degeneracy.

### 9. Multi-step full PDIPM trajectory with independent rho

A fixed-`epsilon` full-PDIPM trajectory runner was added that carries:

- `ln_nk`,
- `ln_mk`,
- `rho`,
- `ln_ntot`

across multiple iterations, using a simple residual-based backtracking rule on the full PDIPM residual norm.

Tested `rho` initializations:

- on-center: `rho0 = epsilon - ln_mk`
- off-center positive: `rho0 = epsilon - ln_mk + 1`
- off-center negative: `rho0 = epsilon - ln_mk - 1`

Finding:

- Off-center PDIPM could produce a different trajectory.
- But that only helped in a small number of `epsilon = 0` cases.
- In the feasible crawling regime, `epsilon = -5`, `-10`, and plateau-adjacent probes, PDIPM usually accepted no step at all under the diagnostic backtracking rule.
- The off-center improvements did not appear in the regime that actually matters for the current solver failure.

Representative outcome:

- `layer 45, epsilon = 0`: `rho_offset = +1` beat PIPM.
- `layer 90, epsilon = 0`: `rho_offset = +1` beat PIPM strongly.
- `layer 0, epsilon = 0`: off-center PDIPM did not help.
- For `epsilon = -5`, `-10`, and plateau-adjacent cases across representative layers, PDIPM generally did not beat PIPM.

Conclusion:

- Off-center PDIPM does break the center-path degeneracy.
- But it does not reliably improve the gas-limited crawling regime.

## Overall Diagnosis

The audit ruled out the following as the main remaining issue:

- stale post-update residual evaluation,
- stage-start `sk` infeasible continuation jumps,
- acceptance policy,
- trace-species-dominated gas limiting,
- simple reduced-system scaling / regularization,
- reduced-vs-full PIPM algebra.

The remaining issue is the IPM direction itself.

More specifically:

- PIPM full and reduced directions are the same.
- PDIPM can differ only when `rho` is allowed off-center.
- Even then, it does not materially improve the feasible-epsilon crawling regime.

## Final Recommendation

At this point the evidence does **not** justify more investment in this IPM family as the next primary path.

Recommended interpretation:

- The previous one-step PDIPM/PIPM equivalence was indeed partly due to starting on the center path.
- But off-center multi-step PDIPM still does not improve the regime that matters enough to justify a real PDIPM backend next.
- It is reasonable to stop investing heavily in PIPM/PDIPM formulation variants and pivot to a different solver family or direction-generation strategy.

## Practical Bottom Line

If future work continues from this audit, the next experiment should probably not be:

- another reduced-vs-full IPM comparison,
- another line-search variant on the same IPM direction,
- or another small conditioning tweak inside the current IPM family.

It should instead target a genuinely different solver strategy.

## 2026-06-21 Severe Support-Closure Repair

The next repair kept the PD-IPM scalar primary route instead of reverting to
component-wise clipping or exact FastChem replay.

What was tested:

- promoting persistent h-type filter/restoration directly into the primary
  default;
- leaving the scalar primary default in place and using the final
  temperature-valid inactive-driving diagnostic to decide whether support
  expansion is required.

Finding:

- Direct h-type/restoration promotion regressed explicit-support and water
  curated rows, so it was not kept as the public default.
- The severe support-closure row
  `solar_metal_sulfide_or_Fe_Ni_S_region` layer 7 had a well-centered active
  support solve but poor inactive closure: `96 / 1012.945`.
- Triggering staged support-growth retry only for severe temperature-valid
  closure failures repaired that row to support 67 and inactive closure `0 / 0`.

Validation:

- `pytest -q tests/endtoend/curated_cases`: `8 passed`.
- Target FastChem4 comparison for the repaired row completed with
  `dG/RT Exo-FC = 2.2108387938146734e-06`.
- Full-profile FastChem4 comparison was attempted but stopped because the fresh
  campaign runtime became too long for this session.

Conclusion:

- The current practical PD-IPM route is: keep scalar fraction-to-boundary as the
  primary step control, keep Ipopt-style dual push initialization, and use
  severe inactive support-closure diagnostics to decide when to expand support.

## 2026-06-21 PD-IPM Core Mainline

HEAD route v1.14 moves the public primary continuation closer to an Ipopt-style
PD-IPM contract instead of adding another route-level retry.

Implemented design:

- add `continuation_mode="pdipm_core"` to the algorithm-v1.1 continuation
  driver;
- in core mode, force `scalar_fraction_to_boundary` and
  `direction_policy="algorithm_v11_reduced"` so alternate restoration or budget
  directions do not replace the Newton direction as the main line;
- split fraction-to-boundary diagnostics into primal and dual step-length
  limits while still reporting the combined limiter;
- make the core acceptance sequence use persistent h-type filter acceptance,
  soft restoration, and dedicated restoration rather than component-wise
  clipping;
- report `filter_accept_count`, `restoration_count`, `barrier_update_count`,
  and `tiny_step_count` from the continuation report;
- keep HEAD route responsible for active-set orchestration, support expansion,
  and final gates.

Observed effects:

- Explicit-support curated rows now define a v1.14 PD-IPM-core surface. Several
  rows intentionally move from `converged` to `converged_with_caveat` or
  `not_converged`, while the graphite explicit-support row improves to
  `converged`.
- The previous target row
  `solar_metal_sulfide_or_Fe_Ni_S_region` layer 7 remains converged against the
  FastChem4-scaled comparison, with `dG/RT Exo-FC =
  2.1952751936282766e-06`, but temperature-valid inactive closure is no longer
  perfect: `4 / 23.120042446981557`.
- Runtime is now a major blocker.  The same target solve takes
  `172.27274646499973 s`, and support-free midlayer validation was interrupted
  after `330.18 s`.

Conclusion:

- v1.14 is a solver-development baseline, not a production metric improvement.
  It makes PD-IPM core behavior explicit and measurable.  The next work should
  reduce runtime and repair the new v1.14 blocker surface without returning the
  main line to component-wise clipping or adding another HEAD-route fallback.

## 2026-06-21 PD-IPM Tiny-Step Restoration

Ipopt's line-search implementation treats very small primal steps as a signal
that the current globalization path is failing.  The important design point for
ExoGibbs is not to clip individual Newton components after the reduced PD-IPM
direction is computed.  The surrounding line-search/restoration machinery should
decide whether the coupled step is usable.

HEAD route v1.15 implements that rule in `pdipm_core`:

- keep the primary direction as `algorithm_v11_reduced`;
- keep scalar fraction-to-boundary and separate primal/dual step lengths;
- detect tiny primary steps with `alpha_primal <= 1.0e-8`;
- route a tiny primary step to restoration instead of accepting a nearly zero
  primary update;
- record tiny-step diagnostics in the continuation report and score.

This is a structural PD-IPM-mainline repair, not a metric win.  The explicit
support blocker surface stayed at 8 converged, 4 caveat, and 2 not converged
rows.  The target `solar_metal_sulfide_or_Fe_Ni_S_region` layer 7 stayed
converged with support 52 and temperature-valid inactive closure `4 / 23.120`,
but solve time increased from 172.27 s to 186.35 s.  The support-free midlayer
test still timed out at 240 s.

Next work should target the active-support solve cost and restoration
effectiveness.  Do not treat the v1.15 runtime blocker as a reason to restore
component-wise clipping or to add another route-level retry before auditing the
PD-IPM core step construction and restoration phase.

## 2026-06-21 PD-IPM Fast Diagnostics

HEAD route v1.16 keeps the v1.15 PD-IPM core policy and repairs a different
problem: large reports were being deep-copied on the hot path.

Observed profiling result:

- `solar_metal_sulfide_or_Fe_Ni_S_region` midlayer spent a large fraction of
  wall time in `dataclasses.asdict()` and diagnostic JAX gathers, not in a new
  Newton solve variant.
- Replacing deep report copies with explicit shallow dictionaries, and copying
  diagnostic arrays once before NumPy summaries, reduced the target layer solve
  from 186.35 s to 76.61 s.
- The support-free midlayer curated regression now completes in 221.74 s
  instead of timing out at 240 s.
- The full 99-layer FastChem4 comparison now completes and reports
  `91 converged / 8 not_converged`, with route counts
  `79 primary / 17 gas-only / 3 native fallback`.

This is not a chemical metric improvement.  It is a validation-enabling runtime
repair.  The PD-IPM mainline remains:

- `continuation_mode="pdipm_core"`;
- reduced PD-IPM direction;
- scalar fraction-to-boundary step control;
- dual push-floor initialization;
- filter/restoration globalization;
- tiny primary steps routed to restoration.

Do not use v1.16 as an excuse to revert to component-wise clipping.  The
remaining work is now visible again: repair the remaining blocker rows inside
the PD-IPM/filter/restoration contract, especially the
`complex_heavy_element_or_boron_titanium_zirconium_case` support-free midlayer,
support-free water layers 7/8, graphite layers 0/1, and the explicit-support
caveat/not-converged surface.

## 2026-06-22 PD-IPM Mainline Fixed Point

HEAD route v1.17 and v1.18 keep the PD-IPM mainline and repair the remaining
curated blockers around it.  The important point is that these repairs do not
restore component-wise clipping and do not use FastChem4 trace values as
constructor inputs.

Implemented design:

- keep `continuation_mode="pdipm_core"` as the public primary default;
- keep scalar fraction-to-boundary and separate primal/dual step diagnostics;
- keep Ipopt-style dual push initialization and tiny-step restoration;
- add final full-budget feasibility restoration for accepted active-support
  states by moving gas log amounts and active condensate amounts within the
  ExoGibbs budget frame;
- when support-free outer-loop growth reaches its iteration budget, grow the
  next support from the PD-IPM lifecycle final state rather than from a raw
  fallback seed;
- when an explicit support payload is too narrow, run one guarded
  `explicit_support_closure_retry` from ExoGibbs-native inactive driving;
- when an explicit empty support payload is gas-only, use the same
  `empty_support_strict_gas_retry` that the support-free empty-support path
  already used.

Observed fixed-point score:

- curated fresh API score: `99 / 99 converged`;
- status counts: `89 converged / 10 converged_with_caveat / 0 not_converged`;
- route counts: `71 primary / 18 gas-only / 10 native fallback`;
- highT gas-only family: `18 / 18 converged`, no temperature-valid inactive
  condensates;
- complex heavy family: `9 / 9 converged`;
- full-profile FastChem4 comparison: 99 finite rows, 40 ExoGibbs-lower
  `G/RT` rows, 59 FastChem4-scaled-lower `G/RT` rows, max `abs(dG/RT)` =
  `1.730409e-1`.

Interpretation:

- v1.18 is a practical fixed point for the curated suite, not proof that the
  PD-IPM/filter/restoration implementation is complete.
- The remaining 10 `converged_with_caveat` rows are the next PD-IPM-focused
  quality target.
- Future work should reduce caveats and broad-grid risk inside the same
  PD-IPM/filter/restoration contract.  Do not add ad hoc replay targets or
  route-specific shortcuts before checking whether the active-support PD-IPM
  residual, line search, restoration, or support-closure gate is the real
  source.
