# Amount-space restoration design

> **Architecture status:** This document records the original amount-space
> restoration mathematics, staged experiments, and their results.  The
> persistent restoration architecture and forward implementation contract are
> now defined in `documents/fixed_support_solver_v2_design.md`.  Where lifecycle
> or migration instructions differ, the v2 design is authoritative.

## Experimental implementation status

The default-off independent phase is enabled with
`EXOGIBBS_FIXED_SUPPORT_BATCH_BUDGET_RESTORATION_PHASE=1` in addition to amount
coordinates and the Ipopt-linearized dual return.  It uses a fixed-shape state
per layer:

```
normal -> restoration -> cooldown -> normal
```

Entry occurs only after the normal/stationarity/SOC candidate set has no
acceptable trial.  While the phase is active, only a theta-reducing amount
candidate is selectable; normal, stationarity-restoration, and SOC candidates
are suppressed from acceptance.  The proximity reference and amount scales
remain fixed at the phase-entry point even when several restoration calls are
needed.

Exit requires the configured reduction from entry theta, dedicated budget and
total-density tolerances, and acceptance by the original persistent filter.
Only then is the Ipopt multiplier return applied once using the entry
`m/eta/qtot` state.  A cooldown blocks immediate re-entry for a configured
number of normal globalization attempts; a rejected cooldown attempt consumes
one count without changing the iterate.  The older every-iteration candidate
implementation remains available
when the phase option is zero for historical A/B reproduction.

The disabled control and amount candidate were run with the historical local
scratch runner:

```console
./volatiles_code/run_fixed_support_amount_restoration_gpu.csh
```

That runner and its raw outputs are intentionally not archived in the
repository.  Local artifacts were written below
`volatiles_artifacts/fixed_support_amount_restoration/{disabled,amount_ipopt_phase}`.

### Initial A/B result

This result used two incorrect restoration parameters and is retained only as
a wiring observation.  The amount helper inherited elastic penalty `1` rather
than Ipopt's `resto_penalty_parameter=1000`, and used a fixed restoration
barrier `1e-3` rather than the current main barrier `exp(epsilon)`.  Both are
corrected; the A/B must be rerun before comparing convergence quantitatively.
The run still exposed the missing dual-recentering lifecycle because any
selected feasibility candidate returned with unchanged duals.

The corrected-parameter rerun confirms the split behavior.  Source layer 0
reduced absolute budget residual from `7.88e-6` to `1.79e-8` and total-density
residual from `3.51e-7` to `4.07e-10`, but condensate stationarity increased
from `8.28e5` to `2.17e9`; it stopped after 402 iterations with
`tiny_step_stalled`.  Source layer 5 improved budget from `2.43e-5` to
`9.11e-6` but worsened stationarity from `1.28e7` to `4.02e7`.  Source layer 6
again converged through `-17` with final residual `5.26e-7`, while source 7 was
unchanged.  Runtime was 453 s versus the 109 s disabled control.

This validates the amount-space feasibility equations after parameter
correction and isolates the next failure to restoration return: primal
feasibility is recovered, but unchanged `lambda/rho` are incompatible with
the restored amounts.  Implement explicit amount-restoration selection counts,
dual recentering, and phase entry/exit before further numerical tuning.

The next default-off bridge experiment is implemented with
`EXOGIBBS_FIXED_SUPPORT_BATCH_BUDGET_RESTORATION_DUAL_RECENTER=1`.  After the
amount solve it sets `rho=epsilon-r`, so `m*eta=mu` holds exactly, and computes
`lambda` with the existing source-convention-consistent scaled least-squares
fit to gas and condensate stationarity.  The runner now writes this variant to
`amount_dual_recenter` and preserves the previous unrecentered `amount`
artifact.  This remains a candidate experiment; it does not yet retain a
restoration phase state or perform internal restoration-objective backtracking.

### Hard dual-recenter A/B result

Hard recentering improved convergence from 1/4 to 2/4: sources 5 and 6
converged through epsilon `-17` with final residuals about `5.07e-7` and
`5.26e-7`.  Source layer 0 reached budget residual `1.41e-13`, total-density
residual at machine precision, and reduced gas stationarity to `3.09e3`, but
still failed at epsilon `-11`; condensate stationarity remained `2.32e9`.
Source 7 was unchanged.  Runtime decreased from the unrecentered 453 s to
341 s but remained three times the disabled control.

Local Ipopt source shows that `eta=mu/m_restored` is not its restoration return
rule.  `ComputeBoundMultiplierStep` forms
`delta_eta=(mu-m_restored*eta_current)/m_current`, applies a dual
fraction-to-boundary step, and resets bound multipliers if their maximum
exceeds a threshold.  Equality multipliers are then recomputed by least
squares.  The hard recenter experiment can create extreme `eta` for amounts
that restoration moved close to zero, explaining why a single `lambda` cannot
fit all condensate stationarity equations in source layer 0.  The next return
experiment should implement this Ipopt linearized multiplier update and reset
policy before adding the full phase state machine.

The Ipopt-linearized return experiment is now implemented behind
`EXOGIBBS_FIXED_SUPPORT_BATCH_BUDGET_RESTORATION_DUAL_RECENTER_POLICY=ipopt_linearized`.
The bound-multiplier formula is isolated in
`fixed_support_ipopt_bound_multiplier_update`, including dual
fraction-to-boundary and the Ipopt default reset threshold `1000`.  After that
update, `lambda` is refit by scaled least squares.  The A/B runner writes the
new variant to `amount_ipopt_dual_return` and preserves both prior amount
artifacts.

### Ipopt-linearized return result

The run converged 0/4, but this understates a large layer-0 improvement.
Source layer 0 ended at residual `282` instead of the disabled control's
`9.06e5`: budget was `2.30e-12`, total density exact, gas stationarity `170`,
condensate stationarity `201`, and complementarity `101`.  Hard recenter had
left condensate stationarity at `2.32e9`, so the Ipopt multiplier return
removed the extreme-dual failure.  Sources 5/6 similarly reached residuals
`235/232` but lost their hard-recenter convergence and stopped at 500
iterations; source 7 was unchanged.

The final accepted candidates for sources 0/5/6 were the pre-existing
stationarity-restoration path, not amount-restoration candidates.  Mixing an
amount candidate into every normal iteration therefore causes repeated
handoffs between feasibility, multiplier return, and stationarity restoration
without an exit contract.  Implement the actual phase state next: enter only
after globalization failure, remain in amount restoration until theta exit,
perform the Ipopt multiplier return once, re-enter normal PD-IPM, and suppress
immediate restoration re-entry.  Add separate cumulative counters for amount
and stationarity restoration before evaluating that phase.

The amount candidate changed convergence from 0/4 to 1/4: source layer 6
converged through epsilon `-17` with final residual `5.26e-7`.  It did not
solve source layer 0.  Layer 0 ran 441 iterations and stopped with
`tiny_step_stalled`; condensate stationarity grew from the disabled control's
`8.28e5` to `2.05e10`.  Source layer 5 also worsened, while source 7 was
essentially unchanged.  Runtime increased from 109 s to 400 s because an
eight-pass restoration candidate was generated every iteration.

The final selected layer-0 candidate was the pre-existing stationarity
restoration path, not the amount candidate.  The amount candidate had changed
the earlier trajectory without recentering `lambda/eta`; subsequent
stationarity restoration then followed the inconsistent dual state.  This
confirms that amount feasibility can reach a useful basin (source 6), but also
confirms the design requirement: it must be an entered restoration phase with
explicit return and dual recentering, not another candidate mixed into every
normal line-search iteration.  Add a cumulative selected-amount-restoration
counter before the next run.

## Scope

This document specifies a default-off feasibility restoration phase for the
fixed-support PD-IPM/R-GIE solver.  The regular equilibrium solve continues to
use log gas amounts.  Restoration alone uses physical amounts so that element
and total-density constraints have their natural linear Jacobian.

The existing `fixed_support_full_restoration` is not this method: it introduces
Ipopt-like elastic slacks, but its primal variables remain `q`, `r`, and
`qtot`, and its constraint Jacobian remains `A*diag(amount)`.

## Restoration problem

Let

```
x = (n, m, ntot)
c_b(x) = W_b (A_g n + A_c m - b)
c_t(x) = w_t (1^T n - ntot)
c(x) = (c_b, c_t)
```

where `W_b` uses the solver's existing positive-budget scaling and `w_t`
uses a fixed reference scale captured on restoration entry.  Do not scale the
total row by the changing trial `ntot`.

Introduce positive and negative elastic variables `p, v >= 0`:

```
c(x) - p + v = 0.
```

For restoration reference `x_ref`, minimize

```
rho * 1^T(p + v)
+ zeta/2 * ||D_x (x - x_ref)||_2^2,
```

subject to `n, m, ntot, p, v >= 0`.  This mirrors Ipopt's L1 restoration
objective.  Use `zeta = proximity_weight * sqrt(mu)` in the first experiment.
The original Gibbs/barrier objective is evaluated for finiteness at candidate
points but is not the restoration objective.

## Scaling

`D_x` must not be based only on the current amount: that would make a trace
species prohibitively expensive to revive.  Use static restoration-entry
scales:

- gas: maximum of the entry amount and a configured fraction of entry `ntot`;
- condensate: maximum of the entry amount and its element-budget capacity
  times a configured floor fraction;
- total density: entry `ntot`.

Record all scales.  Initial experiments must compare at least two floor
fractions; the selection must be global, not layer- or species-name-specific.

## Newton system

The restoration constraint Jacobian is constant during a restoration call:

```
J = [[W_b A_g, W_b A_c, 0],
     [w_t 1^T, 0,       -w_t]].
```

Apply a primal-dual barrier to the amount bounds and elastic bounds.  Eliminate
the amount and slack directions using their diagonal barrier/proximity blocks,
then solve the Schur system in constraint space.  Its dimension is
`n_element + 1`, independent of gas/support count.  This preserves the current
GPU-friendly small dense solve and avoids a dense `(n_gas+n_condensate)` KKT
factorization.

Use fixed Newton passes and fraction-to-boundary steps for `x`, `p`, and `v`.
Accept a restoration Newton trial only when the elastic objective decreases
and the scaled constraint norm does not increase.  No ad hoc species clipping
is permitted.

## Entry policy

SOC remains a Maratos correction, not a feasibility recovery method.  Enter
restoration only after the regular globalization has failed:

1. the normal direction has no filter-acceptable trial;
2. first-trial SOC is rejected or kappa-stopped; and
3. the trial step is below the Ipopt-style minimum step, or the current
   implementation reaches `no_accepted_trial` with theta above the restoration
   threshold.

A later soft-restoration state may delay full restoration while ordinary KKT
steps reduce primal-dual error.  It is not required for the first experiment.

The batch state uses an enum-like fixed-shape mode (`normal`, `restoration`)
and masks, not Python branching per layer.

## Exit and return to R-GIE

Restoration may return only if all conditions hold:

- scaled theta is reduced by `kappa_resto` relative to the entry theta;
- the point is acceptable to the original persistent filter;
- amounts and the original barrier objective are finite;
- element and total-density residuals satisfy a dedicated restoration
  tolerance.

Convert amounts back with a representational floor only after acceptance:

```
q = log(max(n, n_repr_floor))
r = log(max(m, m_repr_floor))
qtot = log(max(ntot, ntot_repr_floor)).
```

The floors are serialization/coordinate floors, not feasibility floors, and
their induced inventory must be audited.

Recenter duals before resuming the regular solve:

1. treat the full restoration displacement as one primal Newton step and use
   the linearized complementarity equation to update `eta`;
2. apply the dual fraction-to-boundary step and reset all bound multipliers to
   one if the configured bound-multiplier threshold is exceeded;
3. estimate element potentials by a scaled least-squares fit to gas and
   condensate stationarity, resetting them to zero if the solve is nonfinite
   or exceeds a configured equality-multiplier threshold;
4. set `rho = log(eta)`;
5. reset the filter only according to the existing Ipopt-style restoration
   return contract, and record the reset.

For entry values `(m_0, eta_0)` and accepted restored amount `m_R`, the bound
multiplier direction is fixed by the Newton linearization of `m*eta=mu`:

```
Delta m = m_R - m_0
eta_0 Delta m + m_0 Delta eta = mu - m_0 eta_0
Delta eta = (mu - m_R eta_0) / m_0.
```

The returned multiplier is `eta_R = eta_0 + alpha_dual Delta eta`.  It is not
the hard assignment `mu/m_R`; the latter was an early experiment and produced
extreme multipliers when restoration moved an amount close to zero.

With `eta_R` fixed, define

```
M = [A_g.T]
    [A_c.T]

d = [q_R + g + qtot_reference - qtot_R]
    [h - eta_R].
```

The equality return is the column-scaled minimum-norm solution of
`min_lambda ||M lambda-d||_2`.  This is a stationarity projection, not an
inverse that can recover a unique old multiplier.  If `M` is rank deficient,
column scaling fixes the minimum-norm convention.  A feasible restoration
point need not be stationary, so a nonzero least-squares residual is expected
and must be reduced by the resumed PD-IPM.

Ipopt's default `bound_mult_reset_threshold` is `1000`.  Its default
`constr_mult_reset_threshold` is `0`, which resets equality multipliers to
zero instead of retaining a least-squares estimate.  The current ExoGibbs
experiment deliberately uses an infinite equality threshold to preserve the
tested least-squares return; a finite production threshold must be selected
globally and recorded, not tuned by layer or species.

The first experiment must export stationarity before and after dual recentering
so that feasibility improvement is not mistaken for full KKT improvement.

## Inactive condensates

Restoration does not remove support members.  An inactive condensate remains
an amount variable and may move toward zero.  Regular PD-IPM continuation
subsequently enforces `m*eta=mu` and determines the active/inactive boundary.
Support pruning is a separate lifecycle optimization and is not part of this
design.

## Diagnostics

Per layer and restoration call, record:

- entry/exit theta and each budget row;
- elastic L1 norm and proximity term;
- minimum amount/slack and fraction-to-boundary alpha;
- Newton/Schur residual and condition estimate;
- amount displacement by gas and condensate block;
- representational-floor inventory injection;
- pre/post recenter gas stationarity, condensate stationarity, and
  complementarity;
- return-to-filter acceptance and restoration stop reason.

Layer 0 diagnostics must explicitly include Si and Mg budget histories.

## Experimental stages

1. Implement a pure amount-space restoration helper and synthetic tests for
   exact feasibility, positivity, elastic decrease, and Schur/full-KKT
   equivalence.
2. Generate one shadow restoration candidate at the first layer-0
   `no_accepted_trial`; do not allow selection.  Verify that the Si budget
   defect decreases without extreme log reconstruction.
3. Enable candidate selection under the original filter, still at epsilon
   `-11`, and compare with the current disabled-restoration control.
4. Add dual recentering and continue through `-13/-15/-17`.
5. Run lower-temperature profiles and audit trace gas abundances against the
   log-only control and FastChem4.

## Acceptance criteria

The experiment is successful only if:

- source layer 0 passes epsilon `-11`, then strict `-17` continuation;
- sources 5/6/7 do not regress;
- no layer accepts a restoration point with worse scaled theta;
- budget closure is not obtained by material representational-floor inventory;
- final trace gas log abundances remain finite and are reported down to the
  existing comparison floor;
- runtime and restoration call counts remain compatible with batched GPU use.
