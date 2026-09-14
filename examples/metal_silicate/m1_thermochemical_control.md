# Milestone 1: frozen source thermochemical controls

`m1_thermochemical_control.py` supplies two explicit alternative local models
for the retained-column comparison. All local source temperatures are **2350 K**;
pressures are supplied in bar and amounts are absolute atom or component moles.
The upper catalog is the existing ExoGibbs selection of 35 neutral gases and
26 pure-condensate candidates, using the packaged FastChem4 thermochemical data.
No external FastChem executable or new equilibrium solver is used.

## Declared thermochemical data

On the exact-zero C/N/S support, the source has 23 active components, seven
independent elements, and 16 independent source reactions. With component
standard potentials `h = mu0/(RT)`, atom matrix `A`, reaction matrix `nu`, and
recorded source reaction constants `d`,

```text
nu @ h = d
A @ nu.T = 0
h -> h + A.T @ q
```

leaves all source reaction constants unchanged. Seven elemental gauge degrees
therefore remain. The example determines a unique source vector by matching the
upper standard potentials for the declared anchors **H2, O2, Fe, Mg, SiO, Na,
and He**. These seven species have independent atom vectors. The resulting
23-by-23 linear system is checked for full rank and its residual is recorded.

The shared H2O and SiH4 standard potentials then differ from the upper data.
The control replaces these two gas formation data and changes every affected
source reaction consistently:

```text
d_new = d + nu_gas @ (h_upper_shared - h_source_shared)
```

At 2350 K, the nonzero shared-gas shifts are approximately **+0.001332583393 RT
for H2O** and **+0.026788593378 RT for SiH4**. Source reaction R9 changes by the
first shift, R15 (H2O dissolution) changes by its negative, and R18 changes by
the second shift. All other active source reaction constants remain unchanged.
The silicate and metal standards and the seven anchor standards are retained.

This explicitly selects which physical data remain fixed. Choosing different
anchors generally defines a different counterfactual after the gas-data
replacement; it is not merely another gauge. A common elemental gauge change
applied to the entire upper gas/condensate setup leaves the derived reaction
constants unchanged, and a regression tests this invariance. The counterfactual
is a sensitivity control, **not a calibrated correction or an error bound**.
No temperature dependence for the frozen source standards is invented. Upper
parcels continue to use their original temperature-dependent data and phase
eligibility limits.

## Two separately identified models

`solve_standard_control(network, case, upper, ...)` passes the altered source
reaction constants to the existing finite source solver. It retains the
original nine-gas source catalog and the previous source-to-upper approximation.
This isolates the selected standard-data change; it cannot remove the extra
upper gases or bottom-condensation discrepancy.

`solve_control(network, case, upper, element_amounts_mol=..., pressure_bar=...)`
also uses the full upper gas/condensate equilibrium as the local atmospheric
reservoir touching the positive source silicate and metal. Its 21 logarithmic
unknowns are the 14 active deep component amounts and seven atmospheric atom
amounts. Each evaluation calls the existing `m1_chemistry.solve_parcel`. The
14 source reactions involving deep components and seven absolute elemental
balances close the outer root. The two gas-only source reactions already
follow from the common upper gas equilibrium and the new standards.

The shared partial pressures use the mole fractions in **all 35 gases**; the
nine-species subset is never renormalized. Atmospheric inventories include the
returned gas **plus retained pure condensates**. Condensates are separately
reported as `retained_cloud`, not assigned to the silicate reservoir. The
finite source total is the sum of silicate, metal, gas, and retained cloud.
An optional `initial_control` supplies a previous accepted local root to start
a neighboring pressure calculation; it imposes no equilibrium constraints.
When the old parcel develops a different cloud support at the requested
pressure, or its initializer-only solve fails, the original source reactions
provide a new seed instead. The full catalog is still used for every accepted
state. Finite logarithmic amount bounds `[-650, 10]`, matching the source
solver's numerical domain, prevent overflowing or underflowing trial amounts.
The existing least-squares solver uses its bounded dogleg method (`dogbox`);
the reflective method stalled at a cloud transition in the low-O control.
These are solver safeguards; accepted states must satisfy the same finite
budgets and chemical residuals without a trace-element floor.

The returned arrays retain the full original source component order for deep
amounts, with exact zeros in all gas and absent-C/N/S slots. The additional
upper gas/condensate arrays remain in the normal `bottom_parcel` report, and
`source_formula_matrix`, `source_phase_species`, and phase atom totals expose
the mappings needed for a consumer's independent reservoir ledger.

## Acceptance and scope

`audit_control(network, original_case, upper, report)` rebuilds the changed
reaction data and recounts the primitive deep and upper component arrays. It
checks all **16 active source reactions** below `1e-8 RT`, all finite element
residuals below `1e-9`, and the existing independent upper gas/condensate audits.
Saved acceptance flags, phase budgets, or reaction residuals do not establish
acceptance. The consumer may replace `report['bottom_parcel']` with a fresh
parcel at the same T/P and atmospheric atom amounts before calling this audit.

`BOUNDARY_CONTRACT` describes common partial pressures and the declared
mass-action contact conditions. The original empirical source metal activities
are retained; they do not supply a jointly integrable excess Gibbs energy.
Positive silicate and metal remain a declared branch. No stability search among
alternative magma/alloy phases, enthalpy/entropy match, thermal balance,
experimental calibration, or M1 observational acceptance is certified.

ExoInventory must close the atmospheric pressure and global reservoir budgets
again for each control before comparing spectra. A local parcel substitution
at the original pressure root is insufficient. These two controls separate a
specific standard-data choice from the additional gas/condensate contact
condition; neither exhausts uncertainty in the underlying thermochemistry.

Run the focused regressions with the existing local environment:

```bash
JAX_PLATFORMS=cpu JAX_ENABLE_X64=1 PYTHONPATH=src:$PYTHONPATH \
python -m pytest -q tests/unittests/examples/m1_thermochemical_control_test.py
```

The pressure-step regression starts from a vendored accepted local 10 bar
source and evaluates 16, 25, 16.3, and 16.3 bar in sequence. The final pressure
is a nearby test probe, not a planetary root claimed by this provider. This
covers large forward/reverse pressure changes and a fresh repeated solve.
The source archive hash and original provider commit accompany the seed.
