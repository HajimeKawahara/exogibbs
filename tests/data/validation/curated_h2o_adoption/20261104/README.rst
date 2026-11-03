Joint CO2/H2O production adoption (2026-11-04)
==============================================

This packet verifies the actual ``curated-h2o-v1`` default, which inherits
``curated-co2-v1``. Relative to ``curated-tio-v1``, only CO2 and H2O formation
functions change. The old single-species candidate endpoints retain their
original meanings and are not relabeled as this joint calculation.

``ions.npz`` and ``neutral.npz`` retain 75 new cold solutions each, with
537 and 421 species respectively. Species, formula matrices, budgets and
non-target potentials match the earlier archived production model. The
new potentials also match an independent assembly of the frozen CO2 and
H2O candidates on 2501 temperatures, with maximum difference 4.27e-14.

Every recorded endpoint converges at tolerance 5e-14. Six tighter cold
controls use the independently assembled candidate model at 2e-14; the
largest CO2/H2O log-fraction difference is 5.33e-14 and largest absolute
fraction difference 3.11e-15. A diagnostic 1e-14 request at 1500 K, 1e-6 bar
reached a float64 stopping-residual floor of 1.54e-14 without convergence.
The saved summary records that distinction. Two fresh ion-inclusive
FastChem comparisons agree within 1.61e-8 dex.

``summary.json``, ``diagnostics.json`` and ``receipt.json`` retain the
joint response, numerical checks, executed inputs and numerical outputs.
Any ``source/`` files preserve earlier default-dependent evidence before
its resolver paths changed; those earlier execution receipts are not
rewritten. Latest production pins are checked directly against current
files. Full physical isotope/scattering uncertainty, condensation,
transport, radiative transfer and observational validation remain outside
this ideal-gas numerical verification.

Offline replay::

   PYTHONPATH=src pytest tests/model_evidence/curated_h2o_adoption_test.py
   PYTHONPATH=src pytest tests/unittests/presets/curated_h2o_test.py

Optional full execution, into a new directory::

   JAX_PLATFORMS=cpu JAX_ENABLE_X64=1 PYTHONPATH=src \
     python mceg/reviews/curated_h2o_adoption.py \
     --output-dir results/curated-h2o-adoption-replay \
     --fastchem-executable /path/to/fastchem
