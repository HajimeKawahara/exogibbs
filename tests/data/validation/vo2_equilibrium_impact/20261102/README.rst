VO2 energy and electronic-state sensitivity
===========================================

This packet records four endpoints at the same 75 gas-only conditions:
current ``curated-c2h-v1``, an energy-only shift, an electronic-only term,
and both terms together. All use the same 537-species ion-inclusive network.
Only VO2 changes; the elemental budget and other chemical potentials are
fixed within each comparison. C/H alone changes between the three C/O cases.

The energy shift raises G and H by 59.342 kJ/mol and preserves S and Cp.
Its magnitude is the historical JANAF-minus-Balducci atomization gap
(1236.342 - 1177 kJ/mol). This is a source-gap diagnostic, not an adopted
D0: the current fit is not extrapolated to 0 K and the historical auxiliary
references and thermal reductions have not been reconciled.

The electronic term uses the three Kim (2014) doublet states at 0,
0.1845 and 0.8130 eV and divides their sum by the existing ground weight 2.
All states share the same rovibrational factor in this diagnostic. G/H/S/Cp
come from the same electronic sum; omitted states, state-specific vibration
and rotation, and the disputed assignments preclude a complete thermal model.
The combined case adds these two corrections without fitting new constants.

Files
-----

* ``inputs.json`` freezes the small factual input set and source identities.
  The source-packet hash identifies the copy used for extraction, even if
  the separate source review later gains new material.
* ``equilibria.npz`` stores all four endpoints, primary and tighter cold
  log amounts, mole fractions, budgets, formula matrix, chemical potentials,
  and V/C/O/N allocations. Model order is baseline, energy, electronic,
  combined. Deltas compare each of the latter three against baseline.
* ``diagnostics.json`` records all 600 primary/tighter cold solves.
* ``summary.json`` contains source scope, caloric corrections, all V species
  at the largest baseline VO2-share point, and tracked-species extrema.
* ``receipt.json`` pins executed source/runtime inputs and numerical outputs,
  tolerances, environment, command and standalone FastChem binary hash.

The grid uses 1500, 2000, 2500, 3000 and 4000 K; 1e-6, 1e-4, 0.01, 1 and
100 bar; and C/O = reference, 1 and 1.5. Primary tolerance is 1e-13 and each
endpoint is independently cold-started again at 5e-14. Standalone FastChem
checks all four endpoints at 1500 K, 100 bar and reference C/O using identical
instantaneous chemical potentials and budgets. Neither numerical agreement
nor the finite gas grid establishes atmospheric accuracy or an uncertainty
interval. Condensation, nonideality, quenching and spectra are not evaluated.
All species remain in the archive without concentration floors.

Reproduce locally into an ignored directory::

   OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 PYTHONPATH=src JAX_PLATFORMS=cpu \
     python mceg/reviews/vo2_equilibrium_impact.py \
     --output-dir results/vo2-impact-new \
     --fastchem-executable /path/to/fastchem

Offline tests read this packet and do not run FastChem or require MCEG.
The output path must not already exist. The default curated coefficients
and previous equilibrium archives are unchanged.
