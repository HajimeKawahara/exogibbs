NaH source review and candidate priorities
==========================================

Retain production provisionally; the main next calculation is a version-explicit thermal candidate, not an unqualified energy replacement.

This is a scoped source review for issues #308/#309. It adds reproducible
evidence and follow-ups without changing curated data. Full physical accuracy
remains unknown. Missing ancestral full texts are recorded in MCEG; they do
not block merging a review which candidly preserves its evidential limits.

Identity, construction and energy separation
--------------------------------------------

The species is neutral gas-phase NaH, identified as
``H1Na1``; neither a solid hydride/oxide nor a different oxide
stoichiometry is interchangeable. The reaction is H(g) + Na(g) -> NaH(g), with a
one-bar standard pressure. Current production uses the unchanged FastChem
BC2016 formation fit. Its source line and all 75 saved chemical potentials
are replayed directly. Natural elemental masses are used for the molecular
comparison; any isotope-mass ratio is a proxy, not an isotope-mixture model.

Balanced NASA and JANAF D0 values are 180.999972 and
197.371000 kJ/mol. The calculation subtracts each source's
H(298)-H(0) before balancing neutral atoms against the molecule. NASA's native
R=8.314510 is preserved for its source evaluation; common-R caloric comparisons
use 8.31446261815324. Atomic-reference and molecular-thermal effects are
reported separately from energy; coefficient integration constants alone
are not dissociation energies.

.. list-table:: Source and thermal diagnostics
   :header-rows: 1

   * - T (K)
     - Current/NASA formation K
     - JANAF/NASA energy contribution to ln K
     - JANAF/NASA molecular thermal contribution to ln K
   * - 1500
     - 1.077877
     - +1.312737
     - -0.004949
   * - 2500
     - 1.058066
     - +0.787642
     - -0.018196
   * - 3000
     - 1.063632
     - +0.656369
     - -0.025659
   * - 4000
     - 1.093899
     - +0.492277
     - -0.029622

Primary evidence and scientific interpretation
----------------------------------------------

The present FastChem curve already uses Barklem-Collet 2016, including
D0=1.886 +/-0.003 eV (181.971336 kJ/mol); it does not inherit JANAF's substantially
larger D0. Walji et al. (2015) provide the improved potential depth
De=15797.4 +/-4.3 cm-1. That number must not replace D0 directly: a consistent
ground zero-point energy must first be subtracted. The full paper was read;
its supplementary potential input was not obtained, so no new D0 is invented.

All 3339 current Rivlin states are retained and their nuclear-spin product 8
is removed once. The state sum reproduces the current PF, including its
quasi-bound-inclusive scope: at 4000 K Q=17363.285 including spin, matching
the paper's inclusive 17363.28 rather than its bound-only 16941.48. This
2.49 percent scope sensitivity is not automatically an upstream error.
The 2015 text describes bound-only line lists, while the current distribution
contains the broader sum. The current definition's 9000 K maximum, paper's
approximately 7000 K line-list claim and PF ending at 5000 K are separately
recorded; the audit evaluates only through 4000 K.

At 1500/4000 K, Q/NASA is 0.999342/1.012276; at 4000 K the molecular Cp
increase is 1.74982 J/mol/K and Q/BC2016 is 0.954337. Similar K values do not
prove caloric agreement. BC molecular and atomic Q are also used to reconstruct
the original one-bar formation constant and expose the separate FastChem fit
error. A future thermal candidate should compare bound-only and resonance
policies with the same atomic anchors before a finite substitution.

Current footprint and bounded decision
--------------------------------------

The curated-oh-v1 archive contains 75 converged, 537-species gas-only cases.
The maximum Na allocation to NaH is 5.0402269 percent,
at 3000 K, 100 bar, C/O=0.588843655
(case 19); its mole fraction there is
1.425397e-07. The largest relative Na conservation
residual is 2.67e-11. This is an existing allocation
footprint, not a measured abundance response to a new thermodynamic model.
It excludes condensation, photochemistry, transport and observational validation.

* Use the Walji 2015 supplementary potential/LEVEL input to obtain the same-isotope ground zero and D0; do not substitute De for D0.

* Compare derivative-consistent Rivlin bound-only and quasi-bound-inclusive candidates with consistent atomic thermal anchors, then run finite Na redistribution.

* No historical-paper acquisition is a condition for merging the scoped audit; supplement DOI 10.1063/1.4906086 is needed only for a refined energy candidate.

Reproduction and source access
------------------------------

Run ``python mceg/reviews/nah_source_review.py`` offline and
``pytest tests/model_evidence/nah_source_review_test.py`` for independent
energy, caloric, conservation and provenance checks. The evidence packet is
``tests/data/validation/nah_source_review/20261105``; original source hashes,
inspection limits, frozen numerical results and a receipt accompany it.
``--extract /path/to/originals`` verifies all original state/PF/definition
hashes before regenerating the compact state packet. Every distributed state
is preserved; no network access is needed by replay or CI.

* `DOI 10.1093/mnras/stv979 <https://doi.org/10.1093/mnras/stv979>`_: Full published paper including Tables 1-3 and sections 2-4; current official states/PF/definition inspected.

* `DOI 10.1063/1.4906086 <https://doi.org/10.1063/1.4906086>`_: Full published paper, particularly Table II/III and potential/confidence-limit discussion. Supplementary potential solver input not obtained.

* `DOI 10.1051/0004-6361/201526961 <https://doi.org/10.1051/0004-6361/201526961>`_: Existing pinned BC2016 numerical excerpts and curated source record reused; original article ancestry not freshly reaudited.

* Official JANAF tables and the pinned NASA CEA source are linked individually
  in ``sources.json``. The evaluated entries are inspected; upstream book
  originals are not represented as freshly audited.
