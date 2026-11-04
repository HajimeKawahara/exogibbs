OH production adoption (2026-11-05)
==================================

This packet verifies the actual ``curated-oh-v1`` public factory on
``curated-h2o-v1``. Only OH thermochemistry changes: the exact reviewed
NASA9 reaction with balanced ATcT 1.172 atomization replaces inherited
JANAF. Species, budgets, atomic references and earlier adopted molecules
remain fixed. This is bounded to 1500--4000 K and 1 bar standard pressure.

``ions.npz`` and ``neutral.npz`` retain 75 fresh cold solutions each,
with 537 and 421 species. Production h agrees with the frozen candidate
on 2501 temperatures to 1.07e-14. Six tighter independently assembled
candidate cases agree to 2.85e-13 in OH ln(x) and 3.40e-14 in any
absolute mole fraction. Two FastChem comparisons agree to 1.19e-8 dex.
Full-grid candidate reproduction, conservation, charge, mass action and
the resolved finite OH response are in ``summary.json`` and the arrays.

``diagnostics.json`` records 156 converged cases. Production tolerance
is 5e-14; tighter controls use 2e-14. ``receipt.json`` records the actual
executed sources and numerical outputs. An earlier development run
stopped on an archive-key error before completing its packet; only the
successful run is archived. ``source/`` preserves earlier H2O-default
source bytes without changing old receipts or numerical results.

Full physical isotope/continuum/energy-covariance uncertainty and
atmospheric validation remain unknown. See ``documents/presets/curated_oh.rst``.

Offline checks::

   PYTHONPATH=src pytest tests/model_evidence/curated_oh_adoption_test.py
   PYTHONPATH=src pytest tests/unittests/presets/curated_oh_test.py

Optional fresh execution::

   JAX_PLATFORMS=cpu PYTHONPATH=src \
     python mceg/reviews/curated_oh_adoption.py \
     --output-dir results/curated-oh-adoption-replay \
     --fastchem-executable /path/to/fastchem
