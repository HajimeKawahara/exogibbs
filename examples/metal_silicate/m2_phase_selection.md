# Mixed-metal phase selection controls

`phase_selection.py` selects a metal-free or metal-bearing branch of the
common extensive Gibbs energy. It returns `metal_present`, `metal_absent`,
or `unresolved`, together with phase amounts and the minimizing incipient
alloy composition. This example verifies those numerical mechanisms using
the actual ExoEOS Ma Fe-Si-O-H excess scalar and explicitly synthetic
standards. **M2-A and M2-B scientific acceptance remain pending.**

The local scalar minimization contract is documented in
[`common_gibbs.py`](common_gibbs.py). The material equations, calibration
gaps, BSE input audit and conservative alloy curvature bound are documented
in ExoEOS's `documents/m2_material_contract.rst` on its accompanying provider
branch. Neither numerical phase selection nor the restricted alloy domain
supplies missing H interactions, pressure dependence, liquid stability or
common experimental reaction standards.

## Interface and evidence

`select_metal_phase(record, element_amounts_mol, temperature_k, pressure_bar,
callbacks, metal_lower, metal_upper, *, convex_phase_bounds=None, maxiter=1000,
tolerance=1e-8)` uses the same `PhaseState(mu_rt, gibbs_rt)` callbacks as
the common-energy solver. Inputs are K, bar and mol of named components;
`gibbs_rt = G/(RT)` is extensive and has units of mol. The insertion cost
per component mole divided by RT is dimensionless. ExoEOS receives the
pressure converted from bar to Pa once in the material callback.

The metal bounds follow the complete component order in the record.
`convex_phase_bounds` supplies proven global molar curvature lower bounds
for the exact phase scalars on their declared composition domains. It is
evidence supplied by the material provider or an analytic model, not an
invitation to label a numerically sampled Hessian as a global bound.

`minimize_insertion` searches the mixed-alloy composition and records an
upper bound, lower bound, their gap, the candidate composition and the
minimum-certification reason. Its certificate uses a supporting plane
and a globally valid curvature bound. Numerical starts help find favorable
alloys but cannot certify absence. `select_metal_phase` requires a certified
minimum within the stated tolerance and nonnegative curvature evidence for
every included phase before it accepts the complete assemblage. Missing
host evidence remains `unresolved`, even if the alloy minimum is certified.

The whole-metal zero branch has exactly zero amount and does not normalize
an empty phase. Its incipient composition comes from the separate insertion
search. Unsupported provider endpoints and an optimized alloy outside the
declared domain also remain unresolved. A negative insertion trial can
reject the metal-free branch without establishing an accepted metal-bearing
solution. Stability is always relative to the declared phase catalog.

## Diagnostics for inventory series

Every selection now preserves `metal_free_result`, the separately solved
zero-metal state, and `metal_free_insertion`, the minimizing incipient-alloy
cost evaluated against that state's elemental potentials. The fields remain
available after a successful metal-bearing solve or a failed attempt to obtain
one. If the absent state or insertion could not be evaluated, the corresponding
field is `None`. `dataclasses.asdict` includes both fields in saved reports.

The existing `result` and `insertion` retain their selected-state meaning.
For an accepted metal-bearing state, `insertion` tests coexistence and is
approximately zero. It must not be used as the sign-changing metal-generation
cost along a hydrogen or Mg/Si series. Use `metal_free_insertion` for that
diagnostic: a negative upper bound supplies a favorable insertion into the
absent state, while a certified nonnegative lower bound excludes favorable
insertion within the declared alloy domain, subject to the recorded numerical
tolerance. A favorable insertion followed by a failed metal-bearing solve
still remains unresolved, even though the retained reference has zero metal.

`local_metal_status(saved_selection)` returns `metal_present`, `metal_absent`,
or `unresolved` after applying the existing `local_metal_selection_accepted`
gate and checking the exact returned metal amount. It also accepts older
reports without the new diagnostic fields. This local classification does not
change `selection.status`: missing global host evidence still leaves that
status unresolved, and no experimental phase boundary is certified.

`metal_composition_domain` records the declared bounds, the effective upper
bounds after exact-zero element support is removed, and the incipient and
selected compositions' fraction slacks. `active_bounds` distinguishes
`composition_box` restrictions from `exact_zero_budget` constraints and natural
`simplex_boundary` contacts, using the recorded `contact_tolerance`. The
`composition_box_contact` flag identifies a restriction that can affect a
series response or apparent transition. Existing selected-state metal
constraint slacks and KKT multipliers are copied into
`selected_composition_constraints`; no uncomputed incipient multipliers are
invented. A composition-box contact is not a physical phase boundary or an
experimental calibration limit. Extending the box requires new valid curvature
evidence and separate material justification; compositions are never clipped
to make a series point pass.

## Composition-constrained update (2026-09-22)

Present-metal minimization now receives the same composition box as the
insertion search. `minimize_gibbs(..., phase_composition_bounds={"metal":
(lower, upper)})` uses the active phase component order. The bounds are
homogeneous linear inequalities in amounts:

```text
n_i - lower_i * sum(n_metal) >= 0
upper_i * sum(n_metal) - n_i >= 0
```

They enter the feasible-start linear program, scalar minimization and final
audit. They impose no positive metal amount: the separate absent branch has
exactly zero metal. Exact-zero element support and fixed-zero composition
bounds also retain exact zeros. No returned composition is clipped or
normalized into the domain.

At an active composition boundary, the correct stationarity condition is
`mu - A.T @ lambda - C.T @ eta = 0`, with `C @ n >= 0`, `eta >= 0`, and
complementarity. The elemental potentials remain the physical element
duals. The result adds `constrained_kkt_residual_rt` and
`composition_constraints` (phase, component, bound, fraction slack and
dimensionless multiplier). Existing `reaction_residual` and
`reduced_potentials_rt` retain their unconstrained definitions and generally
do not vanish at an active domain boundary. They must not be substituted
for the constrained KKT residual there.

An energy-decreasing metal-bearing start is constructed from the accepted
absent state and the minimizing incipient composition. An atom-conserving
linear program limits host/gas changes relative to each actual component
amount before backtracking on the true energy. These restrictions apply
only to the initial guess; the equilibrium solve uses the original atom
inventory and declared composition domain. Failed attempts remain in
`local_attempts`. If a scalar step collapses an entire phase, minimizing
energy along its existing feasible segment provides a new start without
introducing an amount floor. Stationarity refinement preserves active
composition faces and is accepted only within the domain and without
increasing energy.

The new [four-point receipt](../../results/m2_phase_selection/reference_20260922.json)
uses the same synthetic standards and real Ma excess model as the archived
reference below. Offsets 0 and 0.1 still recover metal-bearing equilibria;
4.63 still has exactly zero metal. At 4.62 the constrained solve now accepts
`metal_present`, with metal amount approximately `0.000828981121 mol` and
composition `(0.9197906523, 0.08, 2.30152577e-6, 2.07046167e-4)`.
The Si upper bound has a nonnegative multiplier and the corrected KKT
conditions pass. This is a new restricted-domain numerical result, not a
reclassification of the earlier failed solve or a physical BSE boundary.

Independent tests eliminate the amounts in a two-element ideal host/alloy
model and minimize its one-variable energy. Its known boundary solution has
metal amount 0.2 mol, composition `(0.9, 0.1)` and multiplier 2. Elemental
gauge changes and scales from `1e-9` to `1e24` retain the same composition
and multiplier. Additional controls check exact zeros, constrained phase
selection and preservation of unresolved host stability.

## Archived four-point reference (2026-09-20)

[`run_metal_selection.py`](run_metal_selection.py) constructs ideal host
and gas phases plus the actual Ma alloy mixing model at **2173.15 K and
1 bar**. The alloy domain is `Fe >= 0.86`, `Si <= 0.08`, `O <= 0.02`,
`H <= 0.04`, with nonnegative fractions summing to one. ExoEOS's interval
calculation gives a global curvature lower bound of about **7.95567654**
after eliminating Fe. The ideal host and gas have analytic nonnegative
curvature. This domain is a mathematical control, not an experimental
calibration box or a bound on every possible physical alloy.

The unshifted coexistence reference has 0.2 mol of alloy with atomic
fractions `(0.945, 0.025, 0.01, 0.02)`. Host and gas standards are set to
`-ln(x_reference)`, and alloy standards to
`-ln(x_reference) - ln(gamma_reference)`. These are standards in the Ma
formal convention, supplied directly without an additional source shift.
At the reference amounts every full component potential and every
elemental potential is zero. Convexity then supplies a common supporting
plane. The record includes all reference amounts, standards and atom
formulas, allowing independent reconstruction of this manufactured solution.

A common offset is added to all four alloy standard potentials while
the absolute element budget stays fixed. The saved
[`reference_20260920.json`](../../results/m2_phase_selection/reference_20260920.json)
contains:

| Alloy standard offset / RT | Selection | Alloy amount (mol) | Minimum insertion cost / RT |
| --- | --- | --- | --- |
| 0 | `metal_present` | 0.2000000000 | approximately 0 |
| 0.1 | `metal_present` | 0.1943541873 | approximately 0 |
| 4.62 | `unresolved` | metal-free reference only | -0.00671802590847 |
| 4.63 | `metal_absent` | exactly 0 | +0.00328197409153 |

The zero-offset solution reproduces the independent component amounts
to a maximum relative error below `4e-15`. Independent reconstruction of
all atom budgets has maximum relative error below `1.3e-15` across the
four points. The insertion lower/upper gaps are below `9e-14`.
Saved records include both present-alloy and incipient compositions,
phase energies, fresh chemical potentials, local derivative audits,
complementarity diagnostics, primitive inputs and source hashes.
JSON result phase rows always follow the explicit full `phase_order`,
including an exactly zero metal row on a metal-free branch. They are
reconstructed from the complete component vector before serialization.

At offsets 4.62 and 4.63 the incipient alloy is approximately
`(0.9197921548, 0.08, 2.27466689e-6, 2.05570532e-4)`; Si reaches its
declared upper bound. At 4.62 a negative insertion cost rejects metal
absence, but the present-branch solve encounters an unavailable endpoint
potential. Its zero metal amount belongs to the retained metal-free
reference and is **not** an accepted absence result. The adjacent offsets
bracket an insertion sign change, not an accepted coexistence boundary.
These offsets do not represent a physical hydrogen series or a BSE result.

## Reproduction and optional BSE attempt

Use matching provider branches with ExoEOS's solution-energy and interval
bound helpers, and a Python environment containing the project dependencies.
Run from the ExoGibbs source root, replacing the checkout path:

```sh
JAX_PLATFORMS=cpu JAX_PLATFORM_NAME=cpu JAX_ENABLE_X64=1 \
PYTHONPATH=/path/to/exoeos/src:src \
python examples/metal_silicate/run_metal_selection.py \
  --exoeos-checkout /path/to/exoeos \
  --output /tmp/metal-selection-reference.json
```

The output must not exist; previous records are preserved. The reference
does not require MELTS. The JSON includes both checkout commits and hashes
of every material/solver source used by this path, including uncommitted
new modules, plus JAX/NumPy/SciPy versions and the numerical acceptance
summary. Archived machine paths record the executed environment; the
command above describes reproduction with an independent checkout.

The optional `--bse-inventory` route consumes the stage-1 exported ledger
through [`build_bse_problem`](run_bse_common_gibbs.py) and requires the pinned
external MELTS runtime and its Python environment:

```sh
JAX_PLATFORMS=cpu JAX_PLATFORM_NAME=cpu JAX_ENABLE_X64=1 \
PYTHONPATH=/path/to/exoeos/src:src \
python examples/metal_silicate/run_metal_selection.py \
  --exoeos-checkout /path/to/exoeos \
  --bse-inventory /path/to/exoeos/examples/m2_material/bse_inventory.json \
  --runtime /path/to/alphamelts-py-2.3.2-ubuntu_22_04-x86_64 \
  --python /path/to/melts-env/bin/python \
  --output /tmp/bse-metal-selection-attempt.json
```

The BSE route now defaults to `--gas-model m1_shared`, using the continuous
common gas reactions from [the contact control](m2_common_gas.md).
`--gas-model source` explicitly selects the historical source gas laws.
The newly selected gas laws do not align the melt and alloy standards or
calibrate their material properties.

This route intentionally supplies no global MELTS-liquid curvature evidence.
It must remain `unresolved`; the runner rejects an accidental stable-phase
claim. BSE standards retain their documented extrapolation and calibration
limits, and exact BSE liquid stability has not been established. Native
backend failure is preserved in the returned diagnostics. This optional
command is separate from the archived four-point mathematical reference.

The [2026-09-22 native BSE receipt](../../results/m2_phase_selection/20260922/README.md)
does obtain an accepted constrained local minimum at 2173.15 K and 1 bar.
Its alloy amount is approximately `3.6196087923e22 mol`, with atomic fractions
`(Fe, Si, O, H) = (0.9817036259, 0.0002355798, 0.0170067570, 0.0010540374)`.
The alloy lies inside the declared domain, so all domain multipliers are
zero at this particular point. Its certified insertion minimum is consistent
with zero within `1e-8`. The only phase-selection reason remains
`Host global stability is not established.` This supports a conditional
local BSE connection; it establishes neither global liquid stability nor
calibrated physical metal formation.
