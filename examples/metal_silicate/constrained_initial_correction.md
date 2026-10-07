# Bounded initial correction on an explicit composition face

`constrained_initial_correction.correct_initial_amounts` is an optional example
helper for preparing an initial amount ledger at a new pressure. It consumes
the existing `LocalProblem`, absolute atom budget and full-potential callbacks.
It does not modify [`common_gibbs.minimize_gibbs`](COMMON_GIBBS.md), a material
callback, the physical alloy domain, or any final acceptance tolerance.

A conserved donor may lie on an explicit composition bound, such as the
phosphorus upper fraction. Ordinary unconstrained chemical-potential equations
omit the multiplier for that bound. The helper instead selects only declared
homogeneous composition rows `C n >= 0` whose phase-relative slack is below
`1e-8`. It seeks stationarity tangent to their intersection with the atom
constraints:

```text
x = n / sum(atom_budget)
z = log(x)
atom equations = (A x) / normalized_atom_budget - 1
active-face equations = (C_active x) / corresponding_phase_total
stationarity equations = null_space(vstack(A, C_active)).T @ mu_RT(n, T, P)
```

Every phase callback is evaluated at the current pressure, temperature and
candidate amounts. Tiny positive components remain positive log variables;
they are not rounded to zero or turned into simplex constraints. Exactly
excluded components retain zero amounts. Scaling the atom equations supports
large absolute inventories, including the `1e26` scale in the analytic tests.

The caller binds the donor's acceptance, inventory and source provenance.
Before any callback is evaluated, this helper rejects a ledger that changes
atom support, has nonpositive active amounts, fails `1e-9` relative atom
closure, or violates the existing composition domain. A failed pressure
trial with an atom residual of order `1e-6` is therefore not a valid donor.
Without an explicit active metal composition bound the helper skips correction.

```python
correction = correct_initial_amounts(
    problem, temperature_k, pressure_bar, atom_budget, callbacks,
    initial_component_amounts_mol=accepted_donor_amounts,
    phase_composition_bounds=bounds,
    max_nfev=12,
)
result = minimize_gibbs(
    problem, temperature_k, pressure_bar, atom_budget, callbacks,
    initial_component_amounts_mol=correction.component_amounts_mol,
    phase_composition_bounds=bounds,
)
```

`InitialCorrectionResult` contains `component_amounts_mol`, `adopted` and
`diagnostics`. It has no equilibrium `accepted` flag. A callback failure,
insufficient equation closure or any final gate failure returns a copy of the
original valid ledger. Adoption requires positive support, relative atom error
below `1e-9`, face/stationarity residuals below `1e-8`, every phase-relative
composition slack at least `-1e-12`, the original scalar initial-domain check,
and freshly evaluated total Gibbs energy **no greater than** its value at the
original donor under those same current callbacks. No energy allowance is
added. Euler consistency is checked using the original callback contract.
The original scalar solve and its full independent audits remain mandatory.

The least-squares budget is restricted to `1 <= max_nfev <= 12`. SciPy's `nfev`
excludes finite-difference Jacobian calls. Diagnostics separately count every
actual residual evaluation and phase callback, report the active constraints,
current energies and adoption gates, and retain the stage and exception of
invalid callbacks. An interrupted correction may have no returned optimizer
result; `solver_result_returned` distinguishes that case. The helper's provider
example imports are lazy so a consumer can load the separately pinned file
before initializing its original physical checkout.

Only analytic CPU toy tests validate this implementation: absolute scaling,
an active phosphorus upper bound with a positive trace component, unchanged
scalar acceptance, bad-donor rejection, callback failure, bounded
nonconvergence, all inequalities, and a strict energy-increase veto. No BSE
pressure solve or GPU calculation was launched. This initial-guess option does
not establish convergence for a scientific case, an accepted new equilibrium,
a material calibration, or a physical metal appearance/disappearance boundary.
