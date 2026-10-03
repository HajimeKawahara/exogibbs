# Conserved Gibbs descent for an initial ledger

[`feasible_initial_descent.descend_initial_amounts`](feasible_initial_descent.py)
prepares a lower-G initial amount ledger for the unchanged
[`common_gibbs.minimize_gibbs`](COMMON_GIBBS.md). It leaves physical callbacks,
composition bounds, species support and final acceptance tolerances unchanged.
The original scalar solve and its independent audits remain mandatory. An
adopted initial guess is not an accepted equilibrium or a new pressure root.

The caller supplies an accepted, conserved donor at the same atom inventory
and evaluates it with the existing callbacks at the current temperature and
pressure. Only explicitly declared composition rows whose phase-relative slack
is below `1e-8` select an active face. Without an active metal composition row,
the helper returns the donor without evaluating a callback.

Let `x` be the amounts divided by the total atom inventory. Stack relative atom
rows and active homogeneous composition rows into `C`. At each iteration:

```text
S = diag(sqrt(x))
Q = null_space(row_normalize(C S))
B = S Q
direction = -B B.T mu_RT
trial = x + step * direction
```

Rows are normalized by their Euclidean norms before SVD. A second projection
removes the residual constraint component of the computed null basis; this
prevents a trace element fixed by one component from accumulating relative atom
drift. No positive component is clipped, removed or turned into a zero bound.
Every direction preserves atom amounts and the donor's active-face value to
floating-point accuracy. Every trial is independently checked for positive
support, relative atom closure below `1e-9`, active-face slack below `1e-8`,
active-face drift below `1e-10`, every composition slack at least `-1e-12`, and
the original scalar's absolute initial-domain tolerance.

A fraction of `0.9` of the nearest positivity or inactive-composition boundary
limits the initial step, which is also capped at one. Up to 24 halvings seek
strictly lower, freshly evaluated current-pressure Gibbs energy satisfying an
Armijo coefficient of `1e-4`. Invalid trial callbacks are recorded and rejected.
An invalid initial callback returns the original donor. When no representable
decrease remains, a budget expires, or the line search fails, the best feasible
strictly lower-G ledger is returned if one exists; otherwise the original donor
is returned. No positive energy allowance is added. Tangent stationarity is
reported without imposing an equilibrium acceptance gate on this initializer.

```python
from feasible_initial_descent import descend_initial_amounts

initial = descend_initial_amounts(
    problem, temperature_k, pressure_bar, atom_budget, callbacks,
    initial_component_amounts_mol=accepted_donor_amounts,
    phase_composition_bounds=bounds,
    max_iterations=256, max_evaluations=1024, max_seconds=300,
    progress=record_progress,
)
result = minimize_gibbs(
    problem, temperature_k, pressure_bar, atom_budget, callbacks,
    initial_component_amounts_mol=initial.component_amounts_mol,
    phase_composition_bounds=bounds,
)
```

`FeasibleInitialDescentResult` contains the full `component_amounts_mol` ledger,
`adopted`, and `diagnostics`; it has no equilibrium `accepted` flag. Diagnostics
include all complete or partial callback-bundle evaluations, phase callback
counts, invalid callbacks, every accepted iteration, atom and face residuals,
composition slacks, current energies, tangent stationarity, elapsed time, and
the termination reason. The fixed request caps are 256 iterations, 1024 bundle
evaluations, and 1800 seconds. The default time budget is 300 seconds. A deadline
is checked between callbacks and cannot interrupt an individual blocking
provider call; a consumer that needs a hard wall-time limit must supervise the
whole process. Imports of provider example modules remain lazy so the caller
can bind its physical checkout first.

The analytic CPU tests cover scale changes from `1e-6` to `1e26`, element-potential
gauges, positive trace support, atom and active-face preservation for every
evaluated trial, inactive composition bounds, invalid inputs and callbacks,
actual evaluation counts, deadlines, and original-scalar rejection of a partial
initial descent. These tests establish numerical contracts, not convergence of
a physical low-hydrogen case. The older
[log-amount correction](constrained_initial_correction.md) is unchanged.
