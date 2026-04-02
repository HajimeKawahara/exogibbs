# IPM Audit Summary

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
