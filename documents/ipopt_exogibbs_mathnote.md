# Ipopt SOC and exogibbs R-GIE: mathematical mapping note

## 1. Status and scope

This note defines the mathematical contract required to map the filter
line-search second-order correction (SOC) used by Ipopt to the fixed-support
PD-IPM / reduced GIE (R-GIE) variables used by exogibbs.

The intended result is not a literal copy of Ipopt. The variable spaces differ:

- Ipopt applies its primal-dual system to the original NLP variables, slack
  variables, and bound multipliers.
- exogibbs uses logarithmic gas and condensate amounts and a logarithmic
  condensate dual variable.
- Ipopt's relaxed bound-complementarity blocks and exogibbs' logarithmic
  complementarity equation are therefore not identical equations.

The correct target is:

1. preserve Ipopt's SOC control flow and constraint-RHS construction;
2. define the replacement SOC direction using the full exogibbs KKT system;
3. eliminate variables algebraically to obtain an equivalent R-GIE solve;
4. verify the reduced direction against the full exogibbs linearization before
   running GPU experiments.

Unless explicitly stated otherwise, all Jacobians below are evaluated at the
current iterate, not at a trial point.

## 2. Exogibbs variables and dimensions

Let

```text
Ng = number of gas species
Nc = number of active condensates in the fixed support
Ne = number of element constraints
```

The state is

```text
q       in R^Ng     log gas amounts
r       in R^Nc     log condensate amounts
lambda  in R^Ne     element equality multipliers
rho     in R^Nc     log condensate duals
qtot    in R        log total gas amount
```

Define the positive quantities

```text
n    = exp(q)
m    = exp(r)
eta  = exp(rho)
Ntot = exp(qtot)
mu   = exp(epsilon)
```

and the stoichiometric matrices

```text
Ag in R^(Ne x Ng)     gas formula matrix
Ac in R^(Ne x Nc)     active-condensate formula matrix
b  in R^Ne            target element inventory
```

The fixed-support core keeps a current `qtot_reference`. At the current point
this equals `qtot`; during trial evaluation it is held fixed. This produces the
`-delta_qtot` term in the gas-stationarity linearization.

## 3. Full exogibbs KKT residual

With `g` denoting the fixed gas stationarity source and `h` the condensate
standard-state vector, define

```text
Fg = q + g + qtot_reference - qtot - Ag.T lambda
Fc = h - Ac.T lambda - eta
Cb = Ag n + Ac m - b
T  = r + rho - epsilon
Ct = sum(n) - Ntot
```

The full residual is

```text
F(z) = (Fg, Fc, Cb, T, Ct).
```

`Cb` and `Ct` are primal equality constraints. They belong in the filter
infeasibility `theta`. `Fg`, `Fc`, and `T` are dual/centrality equations. They
do not belong in `theta`.

This definition matches `residual_components` in
`src/exogibbs/optimize/minimize_cond.py` and the audit ordering in
`src/exogibbs/optimize/fixed_support_kkt.py`.

## 4. Full linearized system with generic RHS blocks

Let

```text
Delta = (dq, dr, dlambda, drho, dqtot).
```

For generic residual/RHS blocks

```text
Rg, Rc, Rb, RT, Rt,
```

consider the replacement Newton system

```text
dq - Ag.T dlambda - 1 dqtot                  = -Rg        (4.1)
-Ac.T dlambda - diag(eta) drho               = -Rc        (4.2)
Ag diag(n) dq + Ac diag(m) dr                = -Rb        (4.3)
dr + drho                                    = -RT        (4.4)
n.T dq - Ntot dqtot                          = -Rt.       (4.5)
```

For the ordinary Newton direction,

```text
(Rg, Rc, Rb, RT, Rt) = (Fg, Fc, Cb, T, Ct).
```

The word "full" in this note means that all five equation blocks are present.
It does not require solving a dense `(Ng + 2 Nc + Ne + 1)` matrix directly.

## 5. Exact R-GIE elimination for generic RHS blocks

Equations (4.1), (4.2), and (4.4) give

```text
dq    = Ag.T dlambda + 1 dqtot - Rg                         (5.1)
drho  = diag(eta)^(-1) (Rc - Ac.T dlambda)                  (5.2)
dr    = -RT - diag(eta)^(-1) Rc
        + diag(eta)^(-1) Ac.T dlambda.                      (5.3)
```

Define

```text
j     = m / eta
bn    = Ag n
Qhat  = Ag diag(n) Ag.T + Ac diag(j) Ac.T
sigma = sum(n) - Ntot.
```

Substitution into (4.3) and (4.5) yields the reduced system

```text
[ Qhat   bn      ] [dlambda] = [yb]                         (5.4)
[ bn.T   sigma   ] [dqtot  ]   [yt]
```

with

```text
yb = -Rb
     + Ag (n * Rg)
     + Ac (m * RT)
     + Ac (j * Rc),                                        (5.5)

yt = -Rt + dot(n, Rg).                                     (5.6)
```

After solving (5.4), the remaining components must be reconstructed with
(5.1)--(5.3). Equations (5.1)--(5.6) are the required generic R-GIE mapping.

### 5.1 Check against the existing ordinary step

The current core solves for an absolute multiplier candidate `pi`, where

```text
pi = lambda + dlambda.
```

Using

```text
Rg = q + g - Ag.T lambda
Rc = h - Ac.T lambda - eta
Rb = Ag n + Ac m - b
RT = r + rho - epsilon
Rt = sum(n) - Ntot,
```

and substituting `dlambda = pi - lambda` into (5.4)--(5.6), all terms containing
the current `lambda` cancel. The resulting right-hand side is

```text
yb_absolute =
    Ag (n * (q + g))
  + Ac ((m / eta) * h + m * RT - m)
  + (b - Ag n - Ac m),

yt_absolute = dot(n, q + g) - (sum(n) - Ntot),
```

which matches `rhs_top` and `rhs_bottom` in the current fixed-support core.
This is an internal sign and elimination check for the generic derivation.

## 6. What Ipopt constructs for SOC

Let the current primal point be `x`, the ordinary primal direction be `d`, and
the rejected line-search point be

```text
x_trial = x + alpha_test d.
```

For Ipopt equality constraints `c`, the first SOC constraint vector is

```text
c_soc = c(x_trial) + alpha_soc c(x).                        (6.1)
```

This follows directly from

```cpp
c_soc->Copy(*curr_c());
c_soc->AddOneVector(1.0, *trial_c(), alpha_primal_soc);
```

and `Vector::AddOneVector(a, v, c)`, which implements

```text
y <- a v + c y.
```

For repeated SOC attempts, `c_soc` is not reset. After a rejected SOC trial,
Ipopt updates it recursively as

```text
c_soc^(k+1) = c(x_trial^(k)) + alpha_soc^(k) c_soc^(k).     (6.2)
```

Ipopt then solves a new primal-dual system for a replacement direction
`delta_soc`. It does not compute a correction and add it to the old trial.
The new SOC trial is generated from the current iterate:

```text
x_soc_trial = x + alpha_soc delta_soc.                      (6.3)
```

The Ipopt acceptance test uses the original line-search test value
`alpha_test`, even though the SOC point itself is generated with `alpha_soc`.

Ipopt's `soc_method=0` uses the current damped Lagrangian-gradient blocks,
`c_soc`, and current relaxed-complementarity blocks. `soc_method=1` scales the
stationarity blocks by `alpha_soc`. The initial exogibbs mapping should use the
`soc_method=0` analogue; adding method 1 before method 0 is audited would add an
unnecessary branch.

## 7. Exogibbs SOC RHS corresponding to Ipopt method 0

Define the exogibbs primal equality vector

```text
C(z) = (Cb(z), Ct(z)).
```

For a rejected normal trial, construct

```text
Cb_soc = Cb(z_trial) + alpha_soc Cb(z),                     (7.1)
Ct_soc = Ct(z_trial) + alpha_soc Ct(z).                     (7.2)
```

The first exogibbs method-0 SOC replacement direction should solve (4.1)--(4.5)
with

```text
Rg = Fg(z)
Rc = Fc(z)
Rb = Cb_soc
RT = T(z)
Rt = Ct_soc.                                                (7.3)
```

Equations (5.4)--(5.6) then give

```text
yb_soc = -Cb_soc
         + Ag (n * Fg)
         + Ac (m * T)
         + Ac ((m / eta) * Fc),                             (7.4)

yt_soc = -Ct_soc + dot(n, Fg).                              (7.5)
```

Solving the reduced system with (7.4)--(7.5), followed by reconstruction with
(5.1)--(5.3), produces a full exogibbs replacement direction

```text
Delta_soc = (dq_soc, dr_soc, dlambda_soc, drho_soc, dqtot_soc).
```

The resulting linear audit is not `F + J Delta_soc = 0` in every block. The
correct audit is

```text
Fg + Jg Delta_soc       = 0
Fc + Jc_dual Delta_soc  = 0
Cb_soc + Jb Delta_soc   = 0
T + JT Delta_soc        = 0
Ct_soc + Jt Delta_soc   = 0.                               (7.6)
```

The distinction between `Cb` and `Cb_soc`, and between `Ct` and `Ct_soc`, is
essential.

## 8. Trial construction and the two alpha values

The implementation needs two separate step-size fields:

```text
alpha_test  original normal line-search alpha
alpha_soc   fraction-to-boundary alpha for Delta_soc
```

The SOC state is

```text
q_trial_soc      = q      + alpha_soc dq_soc
r_trial_soc      = r      + alpha_soc dr_soc
lambda_trial_soc = lambda + alpha_y   dlambda_soc
rho_trial_soc    = rho    + alpha_dual drho_soc
qtot_trial_soc   = qtot   + alpha_soc dqtot_soc.            (8.1)
```

The filter/Armijo acceptance call must retain `alpha_test`. Treating
`alpha_test` and `alpha_soc` as one candidate alpha changes the algorithm.

`alpha_y` and `alpha_dual` are also distinct in Ipopt:

- `lambda` corresponds to equality multipliers and uses `alpha_y`;
- `eta = exp(rho)` is analogous to a bound multiplier and uses the dual
  fraction-to-the-boundary step `alpha_dual`.

In original multiplier coordinates, an Ipopt-like update would be linear in
`eta`,

```text
eta_trial = eta + alpha_dual deta.
```

The existing exogibbs update is linear in `rho`,

```text
rho_trial = rho + alpha_dual drho,
eta_trial = eta * exp(alpha_dual drho).
```

These agree only to first order. Keeping the log update is consistent with the
existing exogibbs Newton/trial convention, but it is another reason to describe
the result as an Ipopt control-flow mapping rather than a literal variable
mapping.

The preliminary `alpha_y` experiment applied one scalar to both `lambda` and
`rho`. That experiment does not implement Ipopt's separation of equality-dual
and bound-dual step sizes and must not be used to select a final dual-step
policy.

The current code's fraction-to-boundary rule for log variables can be
interpreted as enforcing positivity of the first-order amount update
`n * (1 + alpha dq)` and `m * (1 + alpha dr)`. Since the actual trial uses
`exp(q + alpha dq)`, positivity is automatic. This rule is therefore an
algorithmic step limiter, not a mathematical domain requirement in log space.
That distinction should remain explicit.

## 9. Complementarity mapping is analogous, not literal

Ipopt supplies relaxed bound-complementarity residuals in the original bound
variables and multipliers. Exogibbs currently uses

```text
T = r + rho - epsilon = log(m eta / mu).                    (9.1)
```

The original product complementarity residual would be

```text
K = m eta - mu.                                             (9.2)
```

Linearizing (9.2) and dividing by `m eta` gives

```text
dr + drho = -(1 - mu / (m eta)),                            (9.3)
```

whereas linearizing (9.1) gives

```text
dr + drho = -(r + rho - epsilon).                           (9.4)
```

Equations (9.3) and (9.4) agree to first order near the central path but are
not globally identical. To remain consistent with the existing exogibbs
ordinary Newton system, the first SOC implementation should use (9.4). It
should be described as an Ipopt-control-flow mapping onto the exogibbs log-KKT
system, not as Ipopt's literal bound-complementarity equation.

Changing from (9.4) to (9.3) would be a separate Newton-system experiment and
must not be bundled into the SOC mapping.

## 10. Difference from the current exogibbs SOC helper

The current helper solves a pure nonlinear primal-defect correction. For a
normal trial `z_trial`, it forms approximately

```text
Db = Cb(z_trial) - [Cb(z) + Jb(z) (z_trial - z)]
Dt = Ct(z_trial) - [Ct(z) + Jt(z) (z_trial - z)]
```

and solves a reduced system with

```text
Rg = 0, Rc = 0, Rb = Db, RT = 0, Rt = Dt.
```

It then adds this correction to `z_trial`.

This is a valid linear correction for the supplied pure primal defect, and the
existing unit test verifies that contract. It is not the SOC replacement
direction defined by (6.1)--(7.6):

- it uses nonlinear defect rather than `C_soc`;
- it omits current stationarity and complementarity RHS blocks;
- it adds a correction to the old trial rather than replacing the direction
  from the current iterate;
- it has only one alpha field;
- its repeated-correction update is not Ipopt recurrence (6.2).

Therefore the current policy name `ipopt_first_trial` describes its trigger and
iteration guard only. It does not yet describe an Ipopt-equivalent SOC linear
system.

## 11. Required implementation contract before GPU experiments

The next implementation should be accepted only after the following CPU audits
pass.

### 11.1 Dense-versus-reduced direction

For deterministic small matrices, solve the full block system (4.1)--(4.5)
and the reduced system (5.4)--(5.6). Require agreement of all five direction
blocks to a scale-aware tolerance.

### 11.2 Generic RHS block audit

Use arbitrary nonzero `Rg`, `Rc`, `Rb`, `RT`, and `Rt`. Verify all equations
(4.1)--(4.5). A pure-primal-defect test is insufficient because it does not
exercise stationarity or complementarity elimination terms.

### 11.3 SOC RHS audit

Construct `Cb_soc` and `Ct_soc` from a known current point and trial. Verify
(7.6), specifically using `Cb_soc`/`Ct_soc` rather than current constraint
residuals.

### 11.4 Ordinary-step regression

Set the generic RHS to the current full residual. Verify that the generic
reduced solve reproduces the existing ordinary R-GIE raw direction before
clipping or optional projection.

### 11.5 Current-origin trial audit

Verify that the SOC trial is `z + alpha_soc Delta_soc`, not
`z_trial + correction`.

### 11.6 Alpha separation audit

Use different `alpha_test` and `alpha_soc`. Verify that the state is generated
with `alpha_soc` while the filter/Armijo input retains `alpha_test`.

### 11.7 Repeated SOC recurrence audit

For two rejected SOC trials, verify recurrence (6.2) and the `kappa_soc`
continuation condition. Do not recompute each SOC from only the original normal
trial defect.

### 11.8 Full linearized residual diagnostic

Export the five block norms in (7.6), not only one aggregate norm. A small
reduced solve residual can coexist with a sign error in reconstructed full
blocks if only the reduced equations are inspected.

## 12. Scrutiny findings

### 12.0 CPU contract implementation status

The non-orchestrating mathematical contracts are implemented in

```text
src/exogibbs/optimize/fixed_support_ipopt_soc.py
tests/unittests/optimize/fixed_support_ipopt_soc_test.py
```

The implemented pure-JAX functions are:

```text
fixed_support_reduced_direction_from_rhs
fixed_support_soc_constraint_rhs
fixed_support_soc_trial_from_current
fixed_support_linearized_rhs_residual_blocks
```

The tests cover:

```text
dense full direction == reduced direction
all five generic RHS blocks
ordinary absolute-pi direction regression
first and second c_soc recurrence
SOC-specific Cb_soc/Ct_soc block audit
current-origin trial and four distinct step-size roles
```

These tests passed before solver orchestration was changed. The existing
fixed-support solve path was subsequently wired to these helpers only for
`SOC_POLICY=ipopt_first_trial`; legacy and reduced-defect policies retain their
previous paths.

### 12.1 Numerical algebra audit performed for this note

A deterministic NumPy audit used dimensions

```text
Ng=4, Nc=3, Ne=2
```

with positive random `n`, `m`, `eta`, dense `Ag`, `Ac`, and arbitrary nonzero
values in every generic RHS block `Rg`, `Rc`, `Rb`, `RT`, and `Rt`.

The observed errors were

```text
max |Delta_dense - Delta_reduced|       = 8.88e-16
max |J Delta_reduced + R|               = 4.44e-16
max |RHS_delta + M[lambda,0] - RHS_pi|  = 7.22e-16
```

The third check is the conversion from the generic `dlambda` reduced system to
the existing absolute-`pi` ordinary-step RHS. These checks validate the
algebra and signs in Sections 4 and 5 for the tested nonsingular system.

They do not yet validate the complete solver orchestration in Sections 6--8.
The pure construction/reduction contracts above are implemented. The active
`ipopt_first_trial` path now includes:

```text
c_soc recurrence in a fixed max_soc loop
generic reduced replacement direction from the current iterate
separate alpha_test, alpha_soc, alpha_y, and alpha_dual values
current-origin trial generation
kappa_soc continuation and per-SOC persistent-filter checks
```

The following remain absent from exported active diagnostics or require GPU
validation:

- exported `alpha_soc`, `alpha_y`, and `alpha_dual` candidate fields;
- blockwise diagnostics for equation (7.6).

The `min_dual_infeas` active rule first fixes the `rho` trial using
`alpha_dual`, then minimizes gas/condensate stationarity along the equality
multiplier direction only. It no longer applies one scalar jointly to
`lambda` and `rho`.

### 12.2 Conclusions from source and algebra review

The derivation and source comparison support the following conclusions.

1. The generic R-GIE elimination (5.1)--(5.6) is algebraically consistent with
   the existing ordinary fixed-support direction after converting from
   `dlambda` to the absolute candidate `pi`.
2. Ipopt's `AddOneVector` call unambiguously gives the first SOC constraint RHS
   `c_trial + alpha_soc c_current`; it is not the nonlinear Taylor defect used
   by the current helper.
3. Ipopt generates a replacement direction from the current iterate. The
   current trial-plus-correction construction is a different algorithm.
4. `alpha_test` and `alpha_soc` have different roles and cannot share one field
   without changing acceptance semantics.
5. `alpha_y` for `lambda` and `alpha_dual` for `eta` are separate Ipopt
   mechanisms. Applying one `alpha_y` to both `lambda` and `rho` is not an
   exact mapping.
6. The mapping of Ipopt relaxed complementarity to exogibbs log
   complementarity is necessarily analogous rather than literal. Keeping the
   existing log equation is the minimal controlled choice.
7. The earlier GPU experiment changed SOC triggering, repetition, and dual
   stepping before implementing the correct SOC RHS. Its failure does not test
   the method defined in Sections 6--8.
8. No additional GPU parameter sweep is justified until the audits in Section
   11 pass.

## 13. Source anchors

The relevant local source locations are:

```text
Ipopt/src/Algorithm/IpBacktrackingLineSearch.cpp
  first-trial SOC trigger and alpha_test handling

Ipopt/src/Algorithm/IpFilterLSAcceptor.cpp
  c_soc recurrence, max_soc, kappa_soc, replacement delta_soc

Ipopt/src/LinAlg/IpVector.hpp
  AddOneVector semantics

src/exogibbs/optimize/minimize_cond.py
  current fixed-support residuals, ordinary R-GIE RHS, trial construction

src/exogibbs/optimize/fixed_support_kkt.py
  full five-block linearized residual audit

src/exogibbs/optimize/fixed_support_soc.py
  current pure-primal-defect correction
```
# Implementation fidelity update (2026-07-11)

The active `ipopt_first_trial` path now keeps the normal Newton directional
derivative `grad(phi)^T delta` fixed throughout SOC acceptance, as Ipopt keeps
`reference_gradBarrTDelta_` fixed for one line search.  The original
`alpha_test` is used for the switching and Armijo tests even when the SOC
fraction-to-boundary step differs.

The reduced solve now records pre-sanitization finiteness, its linear-system
residual, solution norm, smallest singular value, and a condition estimate.
Consequently, a finite direction after `nan_to_num` is no longer interpreted
as evidence that the raw solve succeeded.

The fixed-shape filter implements a bounded reset state with Ipopt's default
trigger and reset limit (five each).  Since ExoGibbs evaluates line-search
candidates as a vector, "an accepted iteration containing a history-rejected
candidate" approximates Ipopt's sequential `last_rejection_due_to_filter_`
state.  This remaining difference must be retained when interpreting results.

Filter infeasibility can be evaluated as either the historical maximum of
scaled equality residuals (`max_scaled`) or their sum (`l1_scaled`).  The
fidelity experiment uses `l1_scaled`, matching Ipopt's 1-norm structure while
retaining ExoGibbs' explicit element and total-density row scaling.
# Scaled reduced solve update (2026-07-11)

The SOC reduced system is now equilibrated without changing its symmetric
saddle-point structure.  Four fixed Ruiz iterations construct a diagonal
scale `D` and solve

```text
(D M D) z = D b,  delta = D z.
```

Two fixed iterative-refinement passes then solve for corrections using the
residual of the original unscaled system.  No SOC acceptance gate,
regularization, clipping, or fallback was added in this step.  Diagnostics
retain the unscaled condition estimate and add the scaled condition estimate
and the normalized residual of the unscaled equations.
# First-selected SOC nullspace probe (2026-07-11)

The reduced solve now exposes the smallest right singular vector of the
unscaled `(lambda, qtot)` matrix.  Diagnostics retain the solve and nullspace
quantities from the first SOC candidate that is actually selected and
accepted, rather than maxima collected after the nonlinear iterate has
already diverged.  The probe records the relative weight in the equality
multiplier block and the absolute `qtot` component, plus the dominant
multiplier index.  It does not alter the direction, acceptance rule, or
fallback behavior.

## 14. Return from amount-space restoration

The restoration NLP has its own equality multipliers for its elastic
constraints.  They are not the original R-GIE element potentials and are
discarded on return.  The original dual state `(lambda, eta)` is initialized
at the accepted restored primal point by two separate operations.

Let `(m_0, eta_0)` be the original state at restoration entry and let `m_R` be
the accepted restored condensate amount.  Treating the complete restoration
displacement as one primal Newton step gives

```text
Delta m = m_R - m_0
m_0 Delta eta + eta_0 Delta m = mu - m_0 eta_0
Delta eta = (mu - m_R eta_0) / m_0.                       (14.1)
```

Use a dual fraction-to-boundary step and return

```text
eta_R = eta_0 + alpha_dual Delta eta.                    (14.2)
```

If `max(eta_R)` exceeds the configured bound threshold, reset all components
of `eta_R` to one.  Equations (14.1)--(14.2) reproduce
`MinC_1NrmRestorationPhase::ComputeBoundMultiplierStep`; they intentionally do
not impose the nonlinear hard assignment `eta_R=mu/m_R`.

After `eta_R` is fixed, the element potentials solve the stationarity
projection

```text
min_lambda || M lambda - d ||_2,

M = [A_g.T],
    [A_c.T]

d = [q_R + g + qtot_reference - qtot_R].                (14.3)
    [h - eta_R]
```

ExoGibbs column-equilibrates `M`, computes the minimum-norm least-squares
solution, and transforms it back to the original multiplier coordinates.
Thus rank-deficient systems have an explicit numerical selection convention.
The projection is generally inexact: restoration establishes primal
feasibility, not stationarity.  The returned point is an initialization for
the resumed original PD-IPM, not a KKT solution by construction.

A nonfinite estimate, or one exceeding a finite configured equality threshold,
is reset to zero.  Ipopt defaults this threshold to zero; the current
ExoGibbs experiment uses infinity to retain the least-squares estimate while
the phase lifecycle is evaluated.  Both bound and equality resets must be
exported as restoration-return diagnostics.

Source anchors for this return are:

```text
Ipopt/src/Algorithm/IpRestoMinC_1Nrm.cpp
  restoration return and bound-multiplier Newton step

Ipopt/src/Algorithm/IpDefaultIterateInitializer.cpp
  equality-multiplier least-squares/reset policy

src/exogibbs/optimize/fixed_support_restoration.py
  amount restoration and original-dual return mapping
```

### 14.1 Phase lifecycle contract

The dual mapping above is applied only on an accepted restoration exit, never
to an intermediate amount iterate.  The phase stores its entry primal/dual
state, keeps the proximity metric fixed there, and admits only amount steps
with strict theta decrease.  An exit must satisfy

```text
theta_R <= kappa_resto theta_entry,
budget_relative_R <= budget_relative_tolerance,
total_density_R <= total_density_tolerance,
original_filter_accepts(x_R) = true.                    (14.4)
```

After (14.4), equations (14.1)--(14.3) are evaluated once from the saved entry
state to the accepted restored point.  Normal PD-IPM then resumes under a
finite cooldown during which restoration cannot immediately re-enter.
