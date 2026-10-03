# Initial Jacobian diagnostic on an explicit composition face

[`constrained_initial_diagnostic.py`](constrained_initial_diagnostic.py) examines
why a valid initial ledger may be difficult to correct at a new pressure. It
uses the residual equations in the
[bounded initial correction](constrained_initial_correction.md), without
running that correction or a scalar minimizer. The original physical callbacks,
composition bounds, temperature, pressure, and acceptance rules stay unchanged.
Callback construction and any internal thermodynamic solves remain provider
responsibilities.

```python
import json

from constrained_initial_diagnostic import diagnose_initialization

report = diagnose_initialization(
    problem, temperature_k, pressure_bar, atom_budget, callbacks,
    initial_component_amounts_mol=accepted_donor_amounts,
    phase_composition_bounds=bounds,
    progress=lambda event: print(json.dumps(event, allow_nan=False), flush=True),
)
```

The caller must bind the initial ledger to its accepted source and inventory,
construct callbacks at the requested current temperature and pressure, and
explicitly authorize their evaluation. Pressure-dependent callbacks must be
rebuilt for each pressure. The helper can be loaded separately before physical
source initialization: imports of `common_gibbs` and `local` occur only when
`diagnose_initialization` is called.

The initial ledger must pass the same support, atom and domain gates as the
bounded correction. An initial-input error raises before any provider call.
Without an explicit active metal composition face the diagnostic skips, with
`completed=False`. Perturbed ledgers are positive derivative probes; they need
not conserve atoms or satisfy the declared composition caps. They are never
accepted states or pressure-continuation donors.

Writing `x = n / sum(atom_budget)` and `z = log(x)`, the residual has atom,
phase-relative active-face, and tangent-stationarity blocks. For every active
component the diagnostic forms a forward difference in `z` using three grids:

- SciPy's default binary64 two-point step,
  `sqrt(eps) * sign(z) * max(1, abs(z))`, taking `sign(0) = 1`;
- an absolute log-amount step magnitude of `1e-4`;
- an absolute log-amount step magnitude of `1e-3`.

All three grids use the default step's sign, so changing scale does not also
change the side of the difference. Actual representable step sizes are saved.
With `N` active components the
complete diagnostic performs `1 + 3 * (N + 1)` residual evaluations, each
calling every supplied phase. For 47 components and three phases this is 145
residual evaluations and 435 callback invocations. Those counts do not measure
internal provider work or cache misses. In particular, the published liquid
model caches native MELTS standard-state receipts and evaluates its mixing
expression directly; its callback count is not a native MELTS solve count.

Each grid ends with a base reevaluation after its last displaced component.
This displaces the final phase's one-entry composition cache; earlier phases
return to their base compositions as later phase columns are evaluated.
The resulting base drift is recorded. Consecutive identical cache hits cannot
measure nested-solver repeatability, and a larger provider cache could also
hide reevaluation. Zero measured drift does not prove accurate derivatives.

The JSON-compatible result includes the dense Jacobians, singular spectra,
rank thresholds, condition numbers, column norms, and the three weakest right
singular vectors with component names. It compares grids by absolute and
relative column differences. Analytic atom/face derivatives and the derivative
along uniform scaling provide independent arithmetic checks. The latter is
one in the atom rows at a conserved ledger and zero in the face and chemical
potential rows. A linear Newton direction and its predicted residual are
reported using only matrix algebra; that direction is not evaluated or accepted.

Base records retain full potentials, all residual blocks, phase energies,
evaluation durations, and drift relative to the first base. The optional
`progress` callback receives `base`, `grid_start`, `column`, `grid_complete`,
`complete`, or `failure` events with real cumulative counts and elapsed time.
A provider failure returns the partial report and the actual failing phase
amounts. Failure reports preserve the completed-column count; incomplete
Jacobians are not presented as complete matrices.

Interpret scale comparisons before choosing another correction run. Agreement
of the two larger-step grids with disagreement at the default step supports a
finite-difference resolution problem. Agreement of all grids with a weak
singular direction supports local conditioning as a concern. Disagreement at
all scales calls for inspecting provider errors, base drift, and nonlinear
truncation; these patterns do not uniquely identify a physical cause.
Atmospheric callbacks may perform nested gas/condensate solves, whose stopping
criteria are not direct bounds on derivative error. No specific provider is
assumed noisy in advance. Total Gibbs differences between atom-inconsistent
probes also depend on the elemental energy gauge, so they cannot establish a
failure of thermodynamic descent.

Only deterministic analytic CPU tests validate this helper, including noisy
cached synthetic callbacks, scale-dependent differences, weak-direction
identification, invalid inputs, and exact failure accounting. The test suite
prevents calls to the equilibrium minimizers. No BSE solve or GPU calculation
was used to validate this implementation. A complete diagnostic is not a
converged equilibrium, a material validation, or a metal stability proof.
