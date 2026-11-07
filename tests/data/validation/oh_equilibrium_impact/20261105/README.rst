OH NASA/ATcT finite-substitution evidence
========================================

This packet changes OH alone from the inherited JANAF/FastChem column to
NASA thermal functions plus balanced ATcT 1.172 energy. The current
``curated-h2o-v1`` catalogs, inventories and every other potential remain
fixed. Production curated data are unchanged.

``ions.npz`` and ``neutral.npz`` each contain 75 reused baseline endpoints
and 75 new cold candidate endpoints, on 537-species ion-inclusive and
421-species neutral networks. They retain the full log amounts, potentials,
species, formula matrix, elemental budgets and conditions without a trace
abundance floor. The grid covers 1500/2000/2500/3000/4000 K, five pressures
from 1e-6 to 100 bar and three C/O ratios. It is a prescribed hot
hydrogen-rich ideal gas, not a validation of any particular planet.

The files also retain 12 tighter cold endpoints (both models at three
cases per network), 10 potential-interpolation points (five per network),
six selected direct-MYTHOS controls, and same-potential Gibbs exchange
diagnostics. ``summary.json`` records actual abundance and oxygen-budget
responses, conservation, numerical stability and four independent FastChem
4.0.3 comparisons (both models at two ion-inclusive cases).
``diagnostics.json`` and ``selected_diagnostics.json`` preserve endpoint
and control convergence. The original ``receipt.json`` pins the executed
command, runtime inputs, output bytes and FastChem executable.

The maximum absolute OH response occurs at 4000 K, 1 bar and reference
C/O: x(OH) rises from 6.0911629e-5 to 6.3275666e-5 in the ion network,
while its oxygen-inventory share rises from 11.2390648% to 11.6752507%.
The neutral network gives almost the same response. Selected MYTHOS
controls change OH by at most 0.0975211% relative to NASA/ATcT, and change
any mole fraction by at most 4.3973e-8. These controls use identical energy
and atomic conventions; they are not a full physical uncertainty band.

Numerical tolerances and the rejected preliminary run
----------------------------------------------------

The completed run uses primary tolerance 5e-14 and tighter tolerance 2e-14.
OH log fractions are stable to 2.85e-13, and absolute mole fractions to
3.40e-14. The largest visible-species log-fraction change is 1.463e-6
in F at x=2.318e-8; the unrestricted maximum 1.463e-5 belongs to extremely
scarce F10S2. Visible means at least one compared x is >=1e-12.
H/C/O residuals are below 9.04e-13, separate from the trace-fluorine
maximum elemental residual 1.465e-6.

A preliminary primary tolerance 1e-13 produced a visible-species change
2.5713e-6, exceeding the declared 2e-6 stability threshold. That run was
rejected, and the endpoints rerun at 5e-14. This packet contains the
completed run only; its execution receipt was not rewritten to describe
later documentation or testing. The new OH candidate MCEG contract
separately pins this README, the documentation and offline tests.

Reproduction
-------------

Offline replay::

   PYTHONPATH=src pytest tests/model_evidence/oh_equilibrium_impact_test.py

Optional full rerun with the local FastChem 4.0.3 executable::

   PYTHONPATH=src python mceg/reviews/oh_equilibrium_impact.py \
       --output-dir results/oh-candidate-impact-rerun \
       --fastchem-executable /path/to/fastchem

The output directory must be new. The recorded run uses float64 CPU JAX.
Offline replay needs no network access or external FastChem binary.
Finite substitution, tolerance controls and cross-solver agreement are
numerical checks; no condensation, transport, photochemical or observational
validation is claimed.
