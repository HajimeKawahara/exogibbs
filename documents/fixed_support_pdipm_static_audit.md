# Fixed-Support PD-IPM Static Audit

This checklist records static-reading findings for the GPU fixed-support
PD-IPM/R-GIE batch solver.  The goal is strict fixed-support KKT convergence,
not loose rescue acceptance.

## Findings

- [x] Unify the barrier parameter used by the batch solver.  The public
  `epsilon=-10` path must use the same epsilon in the complementarity equation
  and in the convergence tolerance.  The legacy capacity-based per-species
  epsilon should not silently replace the requested solver epsilon.
- [x] Do not mask active-support condensate stationarity in strict
  fixed-support KKT residuals.  Activity-based masks are useful for support
  selection/inactive diagnostics, but every fixed active variable must appear
  in the strict residual.
- [x] Use scalar fraction-to-boundary step control for the batch PD-IPM path.
  Component-wise clipping can break the coupled Newton/R-GIE direction and
  should be a comparison/debug mode, not the strict default.
- [x] Replace residual-only candidate selection diagnostics with step
  diagnostics that expose accepted step size, rejected trial count, and the
  accepted step family.
- [x] Wire fixed-support profile/batch execution through explicit epsilon
  continuation.  The single fixed-epsilon solve remains the inner stage solve,
  but the support-atlas/profile route must follow the barrier path by default.
- [ ] Keep line-search/globalization semantics distinct from Ipopt-style
  Armijo/filter restoration.  The current batch path is residual-grid
  globalization; if strict convergence remains blocked, add a true merit/filter
  layer rather than treating the current selector as Ipopt-equivalent.

## Initial Fix Order

1. [x] Make `epsilon` semantics consistent in fixed-support batch solves.
2. [x] Include all active-support condensate stationarity components in strict
   residuals.
3. [x] Enable scalar fraction-to-boundary in the batch core.
4. [x] Record real final step size and trial rejection counts.
5. [x] Connect the prepared fixed-support profile/batch route to epsilon
   continuation and expose per-stage diagnostics.
6. [ ] Re-run the selected strict1000 diagnostic cases and compare residual
   component breakdowns.

## Changes Applied

- The fixed-support batch core now uses requested solver epsilon by default for
  complementarity and convergence tolerance.  Legacy capacity-based epsilon is
  available only through
  `EXOGIBBS_FIXED_SUPPORT_BATCH_USE_LEGACY_CAPACITY_EPSILON=1`.
- Strict fixed-support residuals include all active condensate stationarity
  components.
- Batch step control defaults to scalar fraction-to-boundary.  The previous
  component clipping behavior is available with
  `EXOGIBBS_FIXED_SUPPORT_BATCH_STEP_CONTROL=component_clip`.
- Batch diagnostics now report final accepted step size, rejected trial count,
  second-order-correction accepted count, and residual components consistent
  with the returned final residual.
- Prepared fixed-support profile/batch execution now uses the default
  path-following schedule `(0, -1, -2, -4, -6, -8, -10)` unless overridden by
  `EXOGIBBS_FIXED_SUPPORT_BATCH_EPSILON_SCHEDULE`.  Stage diagnostics are
  returned by the API and persisted by the support-atlas runner.
