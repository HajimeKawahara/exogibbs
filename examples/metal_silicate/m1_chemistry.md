# Milestone 1: source-to-upper chemistry diagnostic

`m1_chemistry.py` compares the frozen 2350 K sulfur/nitrogen source with the
neutral gas and pure-condensate equilibria supplied by ExoGibbs. It uses the
existing ideal-He extension with **C=N=S=0 and He/H=0.1**. No external FastChem
executable is used; FastChem4 is the provenance of the packaged thermochemical
tables.

Run from the repository root, offline:

```bash
JAX_PLATFORMS=cpu JAX_PLATFORM_NAME=cpu JAX_ENABLE_X64=1 PYTHONPATH=src \
python examples/metal_silicate/m1_chemistry.py \
    --pressure-bar 100 --point 1400 0.1 \
    --output results/m1_chemistry/central.json
```

Pressure is required explicitly in bar. The example above specifies local
diagnostic points, not an atmosphere or a pressure root. In particular, neither
the raw source's 10000 bar default nor a previous Inventory pressure root is
adopted automatically. Repeat `--point T_K P_BAR` for additional independent
parcels. Existing output files are never overwritten. A nonzero exit status
reports failed local solves/audits; inspect the saved `failures` and residuals.

The three boundary comparisons use exactly the same gas atom budget:

1. The shared nine gases: H2, O2, H2O, Fe, Mg, SiO, Na, SiH4, and He.
2. The explicit 35-species neutral gas catalog compatible with the seven elements.
3. The same expanded gas catalog plus 26 pure-condensate candidates, retaining
   their packaged temperature upper bounds. Ineligible phases cannot be present
   and are excluded from the absent-phase insertion audit.

The report retains mole fractions, partial pressures, H2O/H2, gas molar mass,
gas/cloud atom amounts, gas mass fraction, and source-reaction differences.
Source-reaction differences are **model comparisons**, separate from each
solver's own equilibrium residuals. A source that passes its equations can
still differ from the upper model or supply material that immediately condenses.
The diagnostic does not automatically accept such a boundary for Inventory.

`--oxygen-factor 0.9` or `1.1` changes O/Si while keeping the local total mass
and all other elemental ratios fixed. The local reference mass is set by the
pinned seven-element source ratios with added He; it is an arbitrary amount
scale, not the milestone's planetary participating mass. ExoInventory selects
participating masses and closes its pressure roots, then supplies trial local
conditions and gas atom budgets to the chemistry functions.

## Amount contract and independent acceptance

`build_setups()` returns the shared gas setup and the expanded gas/condensate
bundle. `solve_parcel(setup, temperature, pressure_bar, element_amounts_mol)`
uses the element order `H, He, O, Mg, Si, Fe, Na`. Each active budget must be
positive; C/N/S species and charge are absent from these setups. Use
`upper.gas_setup` for the expanded gas-only control.

The solver receives atom amounts normalized to a sum of one; output component
moles are restored to the caller's original amount basis. No chemical budget
is floored. Atomic molar masses are recorded in kg/mol. `total_element_mol_per_kg`
uses **gas plus retained cloud mass**, and `gas_mass_fraction` includes He and
all gases. Only gas moles enter partial pressures and mean gas molar mass.

`audit_parcel` independently reconstructs every element and total mass with
relative residuals below `1e-9`. Using ideal-gas chemical potentials and a
least-squares elemental potential, it separately tests gas stationarity,
present-phase equilibrium, and absent eligible-phase insertion at `1e-8`.
Solver convergence is also required. The solver's default element floor does
not relax this independent per-element audit. The existing source solver audits
its fixed positive gas/silicate/metal assemblage and exact-zero CNS separately.

Temperature upper bounds alone do not establish lower-temperature or pressure
validity, or a jointly calibrated source/upper model. The source remains frozen
at 2350 K. A change to standards, calibration, or phase potentials requires a
separate physical-model change and validation.

## Scope and records

This is the chemical diagnostic part of M1-A. It supplies a local callback and
reproducible comparison records; it does not complete M1-A's opacity checks or
its selection of an acceptable coupled input domain. Inventory owns retained
columns, absolute layer masses, global elemental closure and outer pressure
roots. ExoEOS owns density/other properties; ExoJAX owns opacity and radiation.
Rainout, transport, RCE, kinetics and a planetary retrieval are outside this example.

Each JSON record identifies source inputs, species/element mappings, actual
import paths, example/provider git state, Python-source/data hashes, versions,
local residuals, phase eligibility, failures, setup time and per-solve time.
The first solve includes compilation; later solves in that process can reuse
compiled code. No saved timing is a prediction of coupled-column cost.

Run the focused regression tests with:

```bash
JAX_PLATFORMS=cpu JAX_PLATFORM_NAME=cpu JAX_ENABLE_X64=1 PYTHONPATH=src \
python -m pytest tests/unittests/examples/m1_chemistry_test.py
```
