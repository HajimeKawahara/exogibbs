# Fixed-Support PD-IPM Math Contract Audit

This document defines the mathematical contract for the fixed-support
PD-IPM/R-GIE batch path before further globalization or continuation work.
The purpose is to verify that the solver is actually solving the intended
log-barrier KKT system.

## Variables

- `q`: gas log amounts, `q_k = log(n_k)`.
- `r`: active condensate log amounts, `r_j = log(m_j)`.
- `qtot`: gas total log amount, `qtot = log(n_tot)`.
- `lambda`: element potentials, one value per element row.
- `rho`: condensate log activity-correction dual, `rho_j = log(eta_j)`.
- `epsilon`: log barrier parameter, so `mu = exp(epsilon)`.

## Fixed-Support KKT Residual Contract

For a fixed active condensate support, every active condensate must participate
in the strict residual.  No activity proxy should mask active components.

Given:

- gas stoichiometry `A_g`
- active condensate stoichiometry `A_c`
- element inventory `b`
- gas stationarity source `s_g`
- condensate standard source `h_c`

the strict residual components are:

1. Gas stationarity:

   ```text
   q + s_g - A_g.T @ lambda = 0
   ```

2. Active condensate stationarity:

   ```text
   h_c - A_c.T @ lambda - eta = 0
   eta = exp(rho)
   ```

3. Element budget:

   ```text
   A_g @ exp(q) + A_c @ exp(r) - b = 0
   ```

4. Log complementarity:

   ```text
   r + rho - epsilon = 0
   ```

   Equivalently:

   ```text
   exp(r) * exp(rho) = exp(epsilon)
   m * eta = mu
   ```

5. Gas total density:

   ```text
   sum(exp(q)) - exp(qtot) = 0
   ```

The reported `final_residual` must be the norm of exactly these reported
components.

## HEAD Route Continuation Contract

The HEAD route lifecycle also uses the same log-complementarity convention:

```text
r + rho - epsilon = 0
```

For any HEAD route continuation run, the state passed into
`run_algorithm_v11_pdipm_continuation` must be centered at that continuation
policy's `initial_epsilon`.  Therefore, when `rho` is inferred from epsilon:

```text
rho_init = initial_epsilon - ln_mk
ln_mk + rho_init = initial_epsilon
```

If a continuation candidate uses a different `initial_epsilon` from the
lifecycle input, the log dual must be shifted by the epsilon difference:

```text
rho_candidate = rho_input + initial_epsilon_candidate - epsilon_input
```

This preserves any dual push-floor offset while changing the barrier center.

## Current Code Mapping

- Main batch core:
  `src/exogibbs/optimize/minimize_cond.py::_pdipm_activity_fixed_support_batch_core`
- Step-local residual:
  `residual_components(qi, ri, lami, rhoi, qtoti)`
- Final reported components:
  `gas_residual_norm`, `condensate_stationarity_residual_norm`,
  `budget_residual_norm`, `complementarity_residual_norm`,
  `total_density_residual_norm`
- Profile/many-plan propagation:
  `src/exogibbs/api/condensate_equilibrium.py::run_experimental_profile_fixed_support_batch_plan_many`
- Support-atlas persistence:
  `benchmarks/fixed_support_v2/support_atlas_sweep.py`

## Required Checks

- [x] Fixed-support profile/batch execution uses an explicit non-increasing
  epsilon continuation schedule, rather than treating the final epsilon solve
  as the PD-IPM algorithm.
- [x] HEAD route lifecycle initializes and recenters the log dual at the same
  epsilon used by the primary continuation policy's first outer barrier.
- [x] `epsilon` used in complementarity equals the requested solver epsilon
  unless the legacy capacity epsilon flag is explicitly enabled.
- [x] `residual_crit` is derived from the same requested solver epsilon.
- [x] Active condensate stationarity is never masked in strict fixed-support
  residuals.
- [x] `final_residual` reconstructs from the reported component norms.
- [x] `exp(r) * exp(rho)` matches `exp(epsilon)` when the complementarity norm
  is small.
- [x] Element budget residuals reconstruct directly from returned amounts.
- [x] Gas stationarity source has a single gauge convention within one residual
  evaluation.
- [x] Step diagnostics expose enough information to distinguish convergence,
  line-search rejection, fallback acceptance, restoration, and tiny-step
  stagnation.

## Minimal Analytic Test Cases

1. One element, one gas, one active condensate.

   Purpose:

   - Verify budget and complementarity directly.
   - Verify `final_residual` equals the component reconstruction.
   - Verify strict active condensate stationarity is present.

2. Two elements, two gases, one active condensate.

   Purpose:

   - Verify element-potential dimensions and stationarity equations.
   - Catch accidental broadcasting or row/column transposition errors.
   - Status: covered by
     `test_pdipm_rgie_v11_fixed_support_batch_residual_contract_two_element`.

3. One support with requested epsilon and legacy epsilon toggled.

   Purpose:

   - Verify the two modes are intentionally different.
   - Prevent silent reintroduction of capacity-derived epsilon in the default
     path.
   - Status: covered by
     `test_pdipm_rgie_v11_fixed_support_batch_legacy_epsilon_is_opt_in`.

4. Epsilon continuation smoke path, for example `[-2, -4, -6, -8, -10]`.

   Purpose:

   - Verify each stage's complementarity target changes as expected.
   - Separate barrier-continuation failures from single-stage globalization
     failures.
   - Status: smoke-covered by
     `test_pdipm_rgie_v11_fixed_support_batch_continuation_smoke`.

5. Prepared profile/batch plan execution path.

   Purpose:

   - Verify the support-atlas/profile path calls the continuation wrapper, not
     only the single fixed-epsilon stage solver.
   - Verify the returned arrays expose the full epsilon schedule and stage
     diagnostics.
   - Status: covered by
     `test_condensate_profile_experimental_fixed_support_batch_path`.

6. HEAD route lifecycle continuation input.

   Purpose:

   - Verify `ln_mk + rho` equals the primary continuation policy's
     `initial_epsilon`.
   - Prevent the lifecycle from constructing a final-barrier centered dual and
     then starting continuation at a looser barrier.
   - Status: covered by
     `test_lifecycle_centers_initial_dual_at_primary_policy_epsilon`.

## Immediate Next Work

1. [x] Add unit tests for component reconstruction and complementarity consistency
   on a tiny fixed-support batch problem.
2. [x] Add a continuation wrapper only after the fixed-epsilon contract tests are
   passing.
3. [x] Wire the continuation wrapper into the prepared profile/batch execution
   path and persist stage diagnostics.
4. [ ] Re-run the strict1000 diagnostic subset and inspect which residual component
   dominates each failure.

## Contract Tests Added

- `tests/unittests/optimize/minimize_cond_api_test.py::test_pdipm_rgie_v11_fixed_support_batch_matches_layer_core`
  now checks that `final_residual` reconstructs from the five reported residual
  component norms.
- The same test now exposes and checks final `rho` through the experimental
  payload key `final_log_activity_correction`, verifying the log
  complementarity vector `ln_mk + rho - epsilon` against the reported
  complementarity norm.
- `test_pdipm_rgie_v11_fixed_support_batch_residual_contract_two_element`
  checks a two-element/two-gas/one-condensate case by reconstructing all five
  residual component norms from returned `ln_nk`, `ln_mk`, `ln_ntot`, final
  `lambda`, and final `rho`.
- `test_pdipm_rgie_v11_fixed_support_batch_legacy_epsilon_is_opt_in`
  verifies the default requested-epsilon complementarity and the explicit
  legacy capacity-epsilon mode are distinguishable.
- `test_pdipm_rgie_v11_fixed_support_batch_continuation_smoke` verifies the
  internal continuation wrapper runs the explicit schedule
  `[-2, -4, -6, -8, -10]` and returns the final-stage epsilon.
- `test_condensate_profile_experimental_fixed_support_batch_path` verifies the
  prepared profile fixed-support batch path returns the default epsilon schedule
  `(0, -1, -2, -4, -6, -8, -10)` and stage diagnostics for both single-plan and
  many-plan execution.
- `test_lifecycle_centers_initial_dual_at_primary_policy_epsilon` verifies the
  HEAD route lifecycle builds the continuation input at the primary policy's
  actual initial barrier.

## Step Stop Diagnostics

The fixed-support batch payload now includes:

- `stop_reason_code` with labels:
  `converged`, `max_iter`, `max_iter_tiny_step`, `no_accepted_trial`,
  `nonfinite_residual`, `unknown_not_converged`.
- `dominant_residual_component_index` with labels:
  `gas`, `condensate_stationarity`, `budget`, `complementarity`,
  `total_density`.

These are propagated through both single-plan and many-plan fixed-support
batch APIs and persisted by the support-atlas runner.
