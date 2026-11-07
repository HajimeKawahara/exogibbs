H2O candidate finite-substitution evidence
=========================================

This packet compares the 2025 evaluated and PW2024 discrete thermal models
against the current ``curated-tio-v1`` production gas model. Only the H2O
formation potential changes. The balanced ATcT energy, NASA atomic functions,
537-species ion-inclusive catalog and elemental inventories stay fixed.

``equilibria.npz`` retains all log amounts and potentials for the two sets of
75 new cold endpoints and the 75 verified production-reference endpoints.
It also retains tighter cold controls for all three models, a five-point
finite substitution, and same-potential Gibbs exchange diagnostics.
``summary.json`` and the diagnostic files record the measured response and
two independent FastChem comparisons for each of the three models.
``receipt.json`` pins the executed review inputs and output bytes.

The original grid covers 1500, 2000, 2500, 3000 and 4000 K, five pressures
from 1e-6 to 100 bar, and three C/O ratios. It is an ideal gas calculation,
not a validation of condensation, transport, opacity or a real atmosphere.
The source-table/discrete-state contrast is a controlled scope sensitivity,
not a complete physical uncertainty interval or a unique scattering model.

The tighter controls leave water log fractions stable within 1.21e-13 and
the evaluated/discrete water response within 1.57e-13. The largest absolute
change in any mole fraction is 3.13e-14. A trace-fluorine conservation
residual gives a larger relative change in F and extremely small fluorides:
the largest displayed-species change is 1.35e-6 in ln(x), at x(F) = 2.32e-8.
The 2e-6 displayed-species tolerance therefore accompanies separate bounds
of 1e-7 for water and 1e-9 for absolute mole fractions; it is not assigned as
the precision of the water response.

Offline replay::

   pytest tests/model_evidence/h2o_equilibrium_impact_test.py

Optional full rerun, using an installed local FastChem 4.0.3 executable::

   PYTHONPATH=src python mceg/reviews/h2o_equilibrium_impact.py \
       --output-dir results/h2o-candidate-impact-rerun \
       --fastchem-executable /path/to/fastchem

The output directory must be new. The recorded run uses float64 CPU JAX;
the independent FastChem binary hash is stored in the receipt. No download
is needed for offline replay, and this packet does not adopt a new curated
dataset.
