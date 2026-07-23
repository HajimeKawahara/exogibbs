# Fixed-support PD-IPM/R-GIE solver v2 design

## 1. Status

This document is the implementation contract for the next fixed-support
condensate solver.  It supersedes further behavioral extension of the
experimental fixed-support loop in `optimize/minimize_cond.py`.

**Validation status (2026-07-23):** the M0--M6 experimental implementation and
GPU solver validation are complete.  The final exact-state matrix converged
10/10 cases, including support 128, and the separate corrected lifecycle
closed all five curated families.  The evidence, hashes, limitations, and
closure decision are recorded in
`documents/fixed_support_solver_v2_validation.md`.

The existing implementation is retained as the `v1` experimental baseline.
It may receive correctness fixes and diagnostics needed to preserve a valid
comparison, but it must not receive new globalization modes, candidate
families, layer-specific rules, or threshold tuning.

The v2 solver is built alongside v1.  It is not enabled by a preset until the
verification and migration gates in this document pass.

Experimental solver validation does not satisfy the remaining production
migration gates.  In particular, no preset promotion, clean release commit,
public-route decision, or production runtime budget has been approved.

## 2. Decision summary

The following decisions are fixed for the first v2 implementation.

1. The normal equilibrium solve retains log gas and log condensate amounts.
2. Large primal-feasibility recovery is a separate physical-amount
   restoration problem.
3. Restoration is an independent persistent solver state, not a normal line
   search candidate.
4. The normal filter, restoration globalization, and restoration return are
   separate components with explicit contracts.
5. GPU line-search trials are evaluated in parallel, but selection and filter
   reset reproduce the sequential trial order.
6. The first integrated baseline has only normal Newton, filter line search,
   and full restoration.  SOC is added only after this baseline is correct.
7. The first restoration return follows the Ipopt default policy: apply the
   linearized bound-multiplier update and reset equality multipliers to zero.
   A least-squares equality return is a later global policy experiment.
8. Condensate support is fixed inside one solve.  Support closure and expansion
   remain an outer lifecycle operation.
9. Core kernels receive typed configuration and state.  They do not read
   environment variables.
10. No decision may depend on a layer index, species name, temperature band,
    or benchmark-family name.

## 3. Why v2 is required

### 3.1 The mathematical coordinate split

The normal R-GIE state is

```text
q      = log(n)
r      = log(m)
lambda = element equality multipliers
rho    = log(eta)
qtot   = log(ntot)
mu     = exp(epsilon).
```

Log amounts are appropriate for local equilibrium convergence and for trace
gas abundances.  They are not a reliable coordinate for repairing a large
inventory defect.  The equality Jacobian contains

```text
Ag diag(n), Ac diag(m).
```

When the available Si/Mg or other inventory carriers are trace, a finite
amount-space correction maps to an enormous log step.  The reduced linear
solve can be accurate while the exponential trial invalidates its own local
model.  This is a coordinate/globalization failure, not a singular-vector or
linear-solver accuracy failure.

The amount-space restoration state uses

```text
x = (n, m, ntot)
```

so the budget and total-density Jacobian is constant.  The normal and
restoration coordinate systems solve different tasks and must not be mixed in
one candidate ladder.

The canonical original residual used throughout v2 is

```text
gamma = hgas(T) + log(P / Pref)

Fg = q + gamma - qtot - Ag.T lambda
Fc = hcond - Ac.T lambda - eta
Cb = Ag n + Ac m - target
T  = r + rho - epsilon
Ct = sum(n) - ntot.
```

The original barrier objective and filter violation are

```text
phi_mu = dot(n, gamma + q - qtot)
       + dot(m, hcond)
       - mu * sum(r)

theta = norm_1([Wb Cb, wt Ct]).
```

`Wb` and `wt` are fixed for one fixed-epsilon solve.  Convergence checks each
KKT component separately; a small combined norm cannot hide a failed budget,
stationarity, complementarity, or total-density component.

### 3.2 The missing algorithm boundary in v1

The v1 step constructs normal, joint-stationarity, SOC, soft-restoration, and
amount-restoration trials together.  Selection masks, phase state, filter
updates, and multiplier return are interleaved in one JAX loop.

The amount helper preserves only its returned primal amounts.  Its elastic
slacks and restoration equality duals are reinitialized on every call.  It
therefore does not continue one restoration NLP even when the outer carry says
that restoration mode is active.

Ipopt instead constructs a second algorithm for the restoration NLP.  That
algorithm owns its iterate, search direction, line search, barrier state, and
convergence check.  The original filter is consulted only to decide whether a
restored point may return to the original NLP.

The reusable lesson is this ownership boundary, not the accumulation of
individual Ipopt thresholds in the original loop.

## 4. Scope and non-goals

### 4.1 Goals

- Preserve trace gas accuracy and the existing R-GIE reduced solve.
- Give normal globalization and feasibility recovery unambiguous ownership.
- Make every phase transition testable without running a full atmosphere.
- Preserve fixed-shape GPU batching within a support bucket.
- Produce typed terminal statuses instead of silently choosing another
  candidate family.
- Make a failed restoration scientifically interpretable.

### 4.2 Non-goals for the first implementation

- Dynamic support changes inside the PD-IPM loop.
- Per-layer or per-species policy selection.
- Reproducing every Ipopt heuristic.
- Promoting the current neutrality retraction experiment.
- Keeping v1 result tuples or environment variables as the internal API.
- Solving ordinary dual/stationarity stagnation by invoking a feasibility
  restoration when the current primal point is already feasible.
- Changing the main variables to full amount space.

## 5. Package architecture

The v2 implementation lives in a new package.  The initial file boundaries are
part of the design contract.

```text
src/exogibbs/optimize/fixed_support_v2/
    __init__.py
    types.py             typed configuration, state, status, diagnostics
    problem.py           residuals, objectives, Jacobians, coordinate maps
    linear_solver.py     R-GIE reduced solve and numerical diagnostics
    normal.py            one normal PD-IPM direction and trial evaluation
    filter.py            persistent filter and sequential-order selection
    restoration.py       persistent elastic amount-space solver
    return_map.py        accepted restoration primal/dual return
    controller.py        phase transitions and batched super-iteration
    continuation.py      epsilon schedule outside the fixed-epsilon solver
    reference.py         small dense CPU reference equations for tests
```

Public API and profile bucketing stay outside this package.  v1 helpers may be
called temporarily only through adapters with tests proving identical
mathematical inputs and outputs.  v2 code must not import the monolithic v1
batch loop.

## 6. Typed state and configuration

All JIT-visible state is a fixed-shape PyTree.  Scalar codes are integer enums
with stable labels in report serialization.

### 6.1 Problem data

`FixedSupportProblem` is immutable within one compiled support bucket.

```text
Ag, Ac
target inventory
gas standard-state source and pressure convention
condensate standard-state source
support indices
row and variable scaling
```

The canonical gas source is

```text
gamma = hgas(T) + log(P / Pref),
Fg = q + gamma - qtot - Ag.T lambda.
```

It is independent of the current iterate.  A legacy incoming stationarity
source `g_init = hgas + log(P / Pref) - qtot_init` is converted once at the API
boundary by `gamma = g_init + qtot_init`.  Internal kernels do not branch
between external and reconstructed source meanings.  The canonical original
barrier objective uses the same `gamma` and current `qtot`.

### 6.2 Original state

```text
OriginalState:
    q, r, lambda, rho, qtot
    epsilon
    iteration
```

### 6.3 Filter state

```text
FilterState:
    phi_entries
    theta_entries
    valid_entries
    successive_filter_rejections
    reset_count
```

The arrays use a fixed capacity derived from solver limits:

```text
filter_capacity = max_normal_iterations + max_restoration_calls + 1.
```

There is at most one normal accepted-step insertion per normal iteration and
one insertion per restoration entry.  This bound prevents capacity exhaustion
without dynamic allocation.  A violated bound is an internal contract error;
entries are never silently overwritten.

### 6.4 Restoration state

```text
RestorationState:
    x                         # n, m, ntot
    positive_slack
    negative_slack
    equality_dual
    lower_bound_dual_x
    lower_bound_dual_positive
    lower_bound_dual_negative
    restoration_mu
    entry_x
    entry_original_state
    entry_phi
    entry_theta
    variable_scales
    row_scales
    iteration
    accepted_iteration_count
```

Elastic slacks, duals, scales, and the restoration barrier persist across
restoration iterations.  Reinitializing any of them is allowed only on a
`NORMAL -> RESTORATION` transition.

### 6.5 Controller state

```text
mode = NORMAL | RESTORATION | CONVERGED | FAILED
original_state
filter_state
restoration_state
normal_iteration_count
restoration_call_count
terminal_status
```

There is no baseline cooldown state.  If normal globalization fails again
after a successful restoration return, the controller may enter a new
restoration call.  A global maximum restoration-call count is a termination
safeguard, not a trajectory-selection heuristic.

### 6.6 Configuration

Configuration is grouped by ownership.

```text
FixedSupportV2Config
    normal: NormalConfig
    linear_solver: LinearSolverConfig
    filter: FilterConfig
    restoration: RestorationConfig
    continuation: ContinuationConfig
    limits: SolverLimitConfig
```

The core accepts this object explicitly.  Temporary experiment scripts may
translate environment variables into a config at their boundary, but kernels
must never call `os.environ`.

## 7. Normal PD-IPM kernel

The normal kernel performs one local operation:

1. evaluate the complete KKT residual;
2. assemble and solve the R-GIE reduced system;
3. reconstruct the full primal-dual direction;
4. compute fraction-to-boundary limits where mathematically applicable;
5. evaluate the ordered normal alpha ladder in parallel;
6. return trial values and reason masks without changing controller state.

Linear scaling, iterative refinement, and regularization belong to
`linear_solver.py`.  They may change how the same Newton equation is solved;
they may not change the acceptance rules.

The v2 normal kernel does not construct:

- `joint_stationarity_restoration_direction`;
- budget-projected fallback directions;
- amount-restoration candidates;
- relaxed combined-residual fallbacks.

If no normal trial is acceptable, the kernel reports
`NORMAL_LINE_SEARCH_FAILED`.  It does not select the smallest residual trial.

The normal variables `q,r,rho,qtot` are log or unbounded coordinates.  V2 does
not apply an amount-space fraction-to-boundary formula to their directions.
The initial normal trial is `alpha_max=1`.  A future explicit trust-region
model for exponential coordinates would be a separate globalization method,
not a boundary interpretation of a log variable.

## 8. Parallel evaluation with sequential line-search semantics

GPU parallelism does not require mixed candidate semantics.

For configured contraction `0 < beta < 1` and maximum trial count `k`, the
ordered normal ladder is

```text
alpha[i] = alpha_max * beta**i,  i = 0, ..., k-1.
```

This is a fixed-shape representation of sequential backtracking.  It avoids
duplicate trial sizes created by clipping an unrelated alpha table against a
boundary alpha.  V2 evaluates all trials in parallel and then computes:

```text
acceptable[i]
first_acceptable = first i for which acceptable[i]
rejected_prefix  = indices before first_acceptable
last_rejection_reason = reason at first_acceptable - 1
```

When no trial is acceptable, the rejected prefix is the whole ladder.  Filter
reset counters use only the last sequential rejection reason, matching the
Ipopt contract.  The existence of a history-rejected trial elsewhere in the
parallel ladder does not increment the counter.

Normal and SOC trials are never concatenated and selected by a shared
`max-alpha` rule.  SOC has its own ordered attempt after a rejected normal
trial when it is introduced in a later milestone.

## 9. Persistent original filter

The original filter owns pairs `(phi, theta)`:

- `phi` is the original Gibbs log-barrier objective;
- `theta` contains only scaled primal equality violation.

The baseline filter theta is the 1-norm of the scaled equality residual,
matching Ipopt's default line-search norm.  Componentwise max norms remain in
the convergence report and are not substituted for filter theta.

Stationarity and complementarity are convergence quantities, not components
of `theta`.

Filter operations are pure functions:

```text
accept_to_current
accept_to_history
add_margin_adjusted_entry
reset_from_sequential_rejection_history
prepare_restoration_start
```

`prepare_restoration_start` adds the pre-restoration original iterate once.
The filter is frozen while the restoration solver takes internal iterations.
It is not a hard feasibility envelope: a later original trial may be accepted
through either the phi or theta branch, as in Ipopt.

## 10. Full restoration solver

### 10.1 Restoration NLP

Use physical amounts

```text
x = (n, m, ntot)
```

and scaled original equalities

```text
c(x) = [Wb (Ag n + Ac m - target),
        wt (sum(n) - ntot)].
```

Introduce positive elastic variables `p,v`:

```text
c(x) - p + v = 0,
x,p,v >= 0.
```

The restoration objective is

```text
rho_elastic * sum(p + v)
+ zeta/2 * ||Dx (x - x_entry)||^2.
```

`rho_elastic`, `zeta`, `Wb`, `wt`, and `Dx` are global configuration or
entry-derived scaling.  They are fixed during one restoration call.  No value
depends on layer identity or a named element.

### 10.2 Restoration iteration

One restoration iteration:

1. evaluates the restoration KKT residual and restoration merit;
2. solves the constant-Jacobian Schur system;
3. reconstructs directions for `x,p,v` and their duals;
4. computes primal and dual fraction-to-boundary limits;
5. evaluates an ordered restoration alpha ladder;
6. accepts according to the restoration objective/constraint globalization;
7. updates all restoration state together.

The current fixed-pass update without a restoration acceptance test is not
part of v2.

The original filter is not used to accept each internal restoration step.  It
is used by the restoration convergence check to determine whether the current
restoration point may return.

### 10.3 Restoration exit

A restoration point may return only after at least one accepted restoration
iteration and all of the following hold:

```text
original infeasibility <= required_reduction * entry infeasibility
original filter accepts (phi_R, theta_R)
original current iterate accepts (phi_R, theta_R)
original primal values and objective are finite
```

Dedicated final budget and total-density tolerances may be stricter than the
required-reduction test, but they are convergence tolerances configured for
the entire solver, not emergency layer thresholds.

If the restoration NLP converges internally but the point is unacceptable to
the original problem, return the typed failure
`RESTORATION_FEASIBLE_BUT_UNACCEPTABLE`.  Do not relax the original filter or
return the best-theta point.

Other required failures include:

```text
RESTORATION_LINEAR_SOLVE_FAILED
RESTORATION_LINE_SEARCH_FAILED
RESTORATION_MAX_ITER
RESTORATION_LOCALLY_INFEASIBLE
RESTORATION_NONFINITE
```

## 11. Return to the original variables

The return map is evaluated once after an accepted restoration exit.

1. Convert positive restored amounts to `q,r,qtot`.
2. Audit any representational floor before accepting the conversion.
3. Apply the Ipopt linearized condensate bound-multiplier update from the saved
   entry `(m_entry, eta_entry)` to `m_R`.
4. Apply the dual fraction-to-boundary step.
5. Reset all condensate bound multipliers to one if the global bound threshold
   is exceeded.
6. Set equality multipliers to zero for the first v2 baseline, matching the
   Ipopt default `constr_mult_reset_threshold=0` behavior.
7. Return to `NORMAL` on the next controller super-iteration.

The return map is an initializer, not a claim that stationarity or
complementarity is already small.  Post-return KKT residuals are always
reported.

A later `least_squares` equality return policy may be added only as a config
enum and an A/B across the complete benchmark matrix.  It may not be selected
based on a layer or on which return has the smaller immediate residual.

## 12. SOC policy

SOC is a Maratos correction for a rejected normal trial.  It is not a general
feasibility solver.

SOC is excluded from the first integrated v2 milestone.  After normal/filter/
restoration behavior passes its gates, SOC is added with this contract:

1. start from the first sequentially rejected normal trial for which SOC is
   eligible;
2. construct the exact `c_soc` RHS from the original current iterate;
3. solve the existing generic R-GIE RHS equation;
4. apply the Ipopt `kappa_soc` recurrence for a fixed maximum number of
   corrections;
5. test corrected trials in their defined sequential order;
6. enter full restoration if the normal/SOC sequence fails.

SOC corrections do not compete with stationarity or amount-restoration
candidates in one mask.

## 13. Controller transitions

The baseline transition table is:

| Current mode | Event | Next mode | Action |
| --- | --- | --- | --- |
| NORMAL | KKT convergence | CONVERGED | freeze result |
| NORMAL | accepted normal trial | NORMAL | update original/filter state |
| NORMAL | line search failed and primal infeasibility is material | RESTORATION | initialize restoration once and augment original filter |
| NORMAL | line search failed at a sufficiently feasible point | FAILED | report `NORMAL_DUAL_STEP_FAILED` |
| RESTORATION | accepted internal trial, no exit | RESTORATION | persist complete restoration state |
| RESTORATION | accepted original exit | NORMAL | apply return map once |
| RESTORATION | restoration failure | FAILED | preserve typed restoration status |

After SOC is added, `NORMAL line search failed` first passes through the
defined SOC attempt.  No additional long-lived mode is introduced.

## 14. GPU execution model

Compilation buckets already group layers with a common support shape.  v2
keeps this model.

One compiled controller uses a fixed-shape batched `lax.while_loop`.  Each
super-iteration forms `normal_mask` and `restoration_mask`.

- The normal batched kernel updates only `normal_mask` layers.
- The restoration batched kernel updates only `restoration_mask` layers.
- A transition becomes active on the next super-iteration, so entry,
  restoration iteration, exit, and return cannot occur accidentally in one
  fused candidate selection.
- `lax.cond(any(mask), ...)` avoids a phase kernel when no layer in the bucket
  needs it.
- Fixed-shape masks freeze inactive layers; they do not change mathematical
  state.

When a bucket contains both modes, both kernels may run in one
super-iteration.  This is acceptable for the correctness implementation.
Mode-based compaction or host scheduling is a later performance optimization
and must be bitwise/within-tolerance equivalent to the masked controller.

The controller terminates when all layers are `CONVERGED` or `FAILED`, or when
global iteration safeguards are reached.

## 15. Continuation and support lifecycle

The fixed-epsilon solver is complete before epsilon continuation is applied.
Continuation owns:

- the epsilon schedule;
- warm-start transfer between epsilon stages;
- explicit multiplier centering required by a changed barrier;
- per-layer advancement only after the previous stage converges.

Continuation must not reinterpret a restoration failure as permission to
advance epsilon.

Support lifecycle remains outside both the fixed-epsilon solver and
continuation:

1. choose a candidate support using global policy;
2. solve the complete barrier schedule on that fixed support;
3. evaluate inactive KKT driving and full-budget closure;
4. expand or revise support and solve again if required.

Keeping inactive candidates in a fixed support is valid.  At finite barrier
they remain positive and approach the boundary through complementarity.
Support closure is not replaced by an oracle that supplies only the final
active set.

## 16. Diagnostics and terminal statuses

Diagnostics are structured PyTrees grouped by component.  The solver does not
return one positional tuple containing every historical experiment.

Required summaries per layer and epsilon stage:

```text
normal iterations and accepted alphas
normal line-search rejection reason counts
filter entry/reset counts
restoration calls and iterations per call
restoration entry/exit phi and theta
elastic objective, elastic L1 norm, and proximity term
restoration primal/dual residuals
Schur solve residual and conditioning estimate
return multiplier reset flags
pre/post-return KKT component norms
representational-floor inventory injection
terminal status
```

Detailed candidate arrays and named SVD decompositions belong to an opt-in
fixed-size trace buffer.  They are not permanent fields in the production
summary.

Every non-converged result has exactly one primary terminal status.  Report
formatting must not infer stop reasons from counters after the solve.

## 17. Verification strategy

### 17.1 Mathematical unit tests

- Full dense KKT versus reduced R-GIE direction.
- Original residual and barrier-objective directional derivative.
- Filter add/accept/reset against a small sequential Python reference.
- Parallel alpha selection against the same sequential reference.
- Amount restoration Schur step versus a dense restoration KKT solve.
- Restoration merit derivative and fraction-to-boundary positivity.
- Persistence: the second restoration iteration must consume the first
  iteration's slacks and duals, not reinitialize them.
- Restoration return bound-multiplier formula and reset behavior.
- Canonical gas-source convention across initialization, normal trials,
  restoration return, and final residual evaluation.

### 17.2 State-machine tests

Construct synthetic states for every row of the transition table.  Assert the
next mode, state ownership, counter update, and absence of changes to masked
layers.

### 17.3 Batch invariants

- Batched and single-layer execution agree.
- Permuting layers and undoing the permutation gives the same results.
- A converged or failed layer is bitwise frozen while other layers continue.
- Mixed NORMAL/RESTORATION buckets match separate-mode execution.

### 17.4 Captured failure-state fixtures

Store small deterministic numeric fixtures at representative globalization
failures.  Name fixtures by physical state and a content hash, not by a source
layer number.  Required fixture classes are:

- weak trace-carrier budget Jacobian;
- feasible point with ordinary stationarity stagnation;
- restoration feasible but unacceptable to the original filter;
- restoration return with and without a bound-multiplier reset.

Fixtures test subsystem contracts.  They do not authorize fixture-specific
branches.

### 17.5 Profile gates

The complete curated profile matrix is evaluated only after component and
state-machine tests pass.

Promotion requires:

- every v1-converged layer remains converged through the same epsilon stage;
- no nonfinite accepted iterate;
- no accepted restoration exit violating its global exit contract;
- no material inventory from representational floors;
- a strict reduction in non-converged layers over the frozen v1 baseline;
- final budget, stationarity, complementarity, and total-density tolerances are
  satisfied independently;
- trace gas comparisons are reported at the existing floors;
- GPU runtime is measured separately for compilation and execution.

The first correctness milestone may be slower.  Production promotion requires
an agreed runtime budget after the correctness artifact exists; performance is
not recovered by weakening solver contracts.

## 18. Migration plan and gates

### Milestone 0: freeze and extract contracts

- Mark v1 as experimental baseline.
- Record its config and artifact hashes.
- Add `fixed_support_v2/types.py` and `problem.py`.
- Move no behavior yet.

Gate: canonical residual functions reproduce v1 residuals on deterministic
fixtures, including gas-source and qtot conventions.

### Milestone 1: normal solver and filter

- Implement the normal R-GIE kernel.
- Implement parallel evaluation with sequential selection semantics.
- Exclude SOC and every fallback direction.
- Run in shadow beside v1 on iterations where v1 selects a normal step.

Gate: directions and accepted normal trials agree with the mathematical
reference; failures are typed rather than silently rescued.

### Milestone 2: standalone persistent restoration

- Implement complete restoration state.
- Implement restoration KKT/Schur direction and internal line search.
- Feed captured failed original states directly into the restoration solver.
- Do not connect it to the original controller yet.

Gate: restoration merit and feasibility follow their contracts, state is
persistent, and every fixture terminates with an expected exit or typed
failure.

### Milestone 3: controller and return map

- Implement NORMAL/RESTORATION transitions.
- Add original-filter entry augmentation and exit checks.
- Use the Ipopt-default zero equality-multiplier return.
- Run the fixed epsilon `-11` benchmark matrix.

Gate: no phase mixing, no repeated restoration initialization, no untyped
stops, and no regression of v1-converged states.

### Milestone 4: continuation

- Add epsilon scheduling around the complete fixed-epsilon controller.
- Verify independent per-layer stage advancement.

Gate: strict-stage convergence is not inferred from a looser stage and all
terminal statuses propagate to the profile report.

### Milestone 5: exact SOC

- Add SOC only through the contract in section 12.
- Compare normal+restoration with normal+SOC+restoration globally.

Gate: SOC improves the benchmark matrix without changing restoration
semantics or accepting a candidate outside sequential order.

### Milestone 6: performance and production integration

- Profile mixed-mode masked execution.
- Optimize repeated residual evaluation and linear algebra without changing
  state transitions.
- Connect support closure and the public prepared-profile path.

Gate: all unit, batch, profile, artifact-integrity, and runtime requirements
pass before preset promotion.

### Experimental validation closure

The experimental portions of M0--M6 closed on 2026-07-23:

- 69 focused mathematical, controller, continuation, SOC, restoration, and
  profile tests passed;
- the final GPU fixed-support matrix converged 10/10 exact-state cases;
- the corrected external lifecycle converged 191/191 fixed-support solves and
  closed all five curated families;
- support 128 returned from restoration after 61 accepted iterations under the
  global safeguard of 100 and converged through epsilon `-17`;
- all final independent KKT component gates passed;
- no material representational-floor inventory was accepted.

This closes solver correctness validation, not production promotion.  The
remaining M6 production work is tracked as a separate migration after the
validation report.

## 19. Features not carried from v1 baseline

The following v1 behaviors are deliberately absent from the v2 baseline:

```text
joint_stationarity_restoration_direction
relaxed_stationarity_fallback
best-residual selection after line-search failure
budget direction clipping/projection as globalization
amount restoration as a normal candidate
hard eta = mu/m restoration return
restoration cooldown as a trajectory rule
max-alpha selection across mixed candidate families
filter reset based on any rejected parallel candidate
electron-row neutrality retraction
fraction-to-boundary limits applied to log-coordinate directions
```

Removing these behaviors from v2 does not delete their v1 diagnostic history.
Any one may return only after it is expressed as a general mathematical
component with an explicit owner, contract, and global A/B result.

## 20. Deferred formulation decision

If v2 has a correct persistent restoration and exact globalization but the
normal solver still systematically stalls near inactive condensate bounds, a
single global formulation experiment is authorized:

```text
gas variables: log amounts
condensate variables: physical amounts with explicit lower bounds
```

This hybrid formulation would make the condensate boundary finite while
retaining trace-gas resolution.  It is not implemented in parallel with the
initial v2 work because that would confound architecture and coordinate
effects.  The decision is based on the complete benchmark matrix, never on a
layer-specific switch.

## 21. Source anchors

The relevant Ipopt ownership boundaries are:

```text
Ipopt/src/Algorithm/IpAlgBuilder.cpp
    constructs a separate restoration IpoptAlgorithm

Ipopt/src/Algorithm/IpBacktrackingLineSearch.cpp
    original line search, SOC entry, and restoration invocation

Ipopt/src/Algorithm/IpFilterLSAcceptor.cpp
    original filter lifecycle and restoration-start augmentation

Ipopt/src/Algorithm/IpRestoConvCheck.cpp
Ipopt/src/Algorithm/IpRestoFilterConvCheck.cpp
    restoration exit against the original infeasibility/filter/current point

Ipopt/src/Algorithm/IpRestoMinC_1Nrm.cpp
    restoration solve ownership and multiplier return
```

Existing ExoGibbs mathematical derivations remain in
`documents/ipopt_exogibbs_mathnote.md`.  The earlier restoration experiment and its
results remain in `documents/amount_space_restoration_design.md`; that file is
historical input, not the v2 architecture contract.

## 22. Post-validation next task

Do not add further ad hoc globalization behavior to this closed validation
series.  If production migration is authorized, begin from
`documents/fixed_support_solver_v2_validation.md` and:

1. archive the validated source in a clean commit;
2. reconcile or explicitly supersede the two historical v1 status mismatches;
3. define an opt-in public API/preset and rollback boundary;
4. approve compilation and execution budgets;
5. run full repository and production-profile regressions;
6. promote only after those separate migration gates pass.
