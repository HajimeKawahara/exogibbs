Finite CO2 substitution, 2026-11-04
==================================

``original.npz`` retains the 75 saved curated-tio-v1 cases and 75 new NIST626
cold solutions. ``hydrogen_poor.npz`` adds current/candidate solutions for six
controlled gas cases: 1500/2500/4000 K at 0.1/100 bar, C/O=.5, H/O=.01,
He/O=.001, with all other heavy-element/O ratios retained from the preset.
Both use the same 537-species ion-inclusive catalog and exact fixed budgets.
Only the CO2 formation potential changes.

NPZ files retain all logarithmic amounts, potentials, budgets, the formula
matrix, tighter solutions and same-model Gibbs differences including the
first-order feasibility correction. ``path.npz`` preserves a five-point
finite substitution. ``local_sensitivities.json`` contains nine local solves
for the explicitly scoped isotope thermal mean and plus/minus one standard
source-Q uncertainty. These are diagnostics, not a physical confidence band.
``summary.json`` and ``diagnostics.json`` preserve results and closure.

``receipt.json`` identifies the actual execution, numerical sources, outputs
and independent FastChem binary. This README was added after execution and
is pinned by the MCEG subject rather than the execution receipt.

Recompute in a new directory (requires JAX; optional independent comparator)::

   PYTHONPATH=src python mceg/reviews/co2_equilibrium_impact.py --output-dir /tmp/co2-impact-new --fastchem-executable /path/to/fastchem

Offline tests and the figure read the saved records without solving again::

   PYTHONPATH=src pytest tests/model_evidence/co2_equilibrium_impact_test.py
   PYTHONPATH=src python mceg/reviews/plot_co2_candidate.py

The original hot hydrogen-rich grid and the controlled hydrogen-poor cases
are not a survey or validation of planetary atmospheres. Production curated
is unchanged. Numerical verification does not determine complete isotope,
continuum, condensation, transport or observational physical accuracy.
