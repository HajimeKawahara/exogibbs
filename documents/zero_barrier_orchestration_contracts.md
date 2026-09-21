# Zero-barrier orchestration contracts

The refactor separates support selection, one numerical attempt on a specified
support, and independent physical assessment. Changing coordinates or an
initializer does not itself authorize changing the ordered support. The
controllers own phase additions, deletions, and candidate ordering.

## PR 1: numerical attempts and physical assessment

`FixedSupportState` carries the physical state. `FixedSupportAttempt` pairs it
with the ordered support, optimizer termination, and evaluation information.
Each coordinate kernel solves exactly that support once and returns it unchanged.
Linear-amount kernels may return signed condensate amounts: these are equation
roots to assess, not accepted physical states. Existing drop loops remain
controllers around these kernels, with the same order and evaluation budgets.

`ZeroBarrierAudit` records distinct decisions without requiring controllers to
reinterpret the exported diagnostic dictionary:

- `accepted`: all physical conditions, positivity, and inactive closure pass;
- `local_kkt_passed`: a local root can seed the existing support-addition logic;
- `root_blocks_passed`: converged equality blocks permit the existing
  nonpositive-amount deletion rule, before positivity or closure holds.

An optimizer evaluation-limit status of zero may certify a terminal result only
when the complete physical audit accepts it. It cannot authorize adding or
deleting phases. Optimizer success alone never certifies a physical solution.

## PR 2: initialization and numerical retries

The initializer controller expresses the existing dual-support, finite-barrier
homotopy, and original-state fallback order as explicit stages, replacing
recursive re-entry. A stage may propose another support; that proposal remains
subject to an exact solve and the same independent audit.

Numerical restart specifications separately identify coordinate choices,
regularization, and initial values for a specified support. Retrying that support
is a numerical operation; selecting another candidate or applying a drop rule
is a controller decision. Eligibility guards, early exits, evaluation accounting,
and the order of attempts remain unchanged.

## PR 3: finite-barrier lifecycle boundaries

The lifecycle separates batch execution, per-layer assessment, and construction
of the initializer passed to exact refinement. Batch results carry finite-barrier
solver outcomes. Layer assessment decides whether an outcome is eligible for the
existing closure or refinement path. The handoff carries the selected state and
support without declaring physical acceptance.

Finite-barrier initialization permits a nonnegative condensate state under its
existing KKT and convergence rules. The terminal zero-barrier result requires
strictly positive active amounts and all-phase closure. Their budgets and
acceptance decisions remain separate. Accepted internal results still undergo
the existing caller-gauge audit before public result construction.

## Compatibility and validation

This work changes internal ownership only. Equations, physical tolerances,
branch order, evaluation limits, public APIs, and diagnostic schemas are
preserved. Compatibility reports remain dictionaries at the reporting boundary.

Targeted regressions cover unchanged support in coordinate attempts, signed
roots and drop eligibility, terminal-only acceptance after an evaluation limit,
initializer order, numerical retries, and finite-to-exact handoff. Full unit
tests and fresh documented-example acceptance are separate requirements. A
dry-run, a subset run, or earlier successful examples do not establish full
acceptance for this refactor. The full acceptance runner includes all four NUTS
examples with 500 warmup steps and 1000 samples each.
