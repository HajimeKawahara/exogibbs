KH source review and candidate priorities
=========================================

Retain production in this source PR; prioritize a spectroscopic-D0 candidate because modern evidence favors a smaller dissociation energy D0, while thermal/state selection must be explicit.

This is a scoped source review for issues #308/#309. It adds reproducible
evidence and follow-ups without changing curated data. Full physical accuracy
remains unknown. Missing ancestral full texts are recorded in MCEG; they do
not block merging a review which candidly preserves its evidential limits.

Identity, construction and energy separation
--------------------------------------------

The species is neutral gas-phase KH, identified as
``H1K1``; neither a solid hydride/oxide nor a different oxide
stoichiometry is interchangeable. The reaction is H(g) + K(g) -> KH(g), with a
one-bar standard pressure. Current production uses the unchanged FastChem
JANAF formation fit. Its source line and all 75 saved chemical potentials
are replayed directly. Natural elemental masses are used for the molecular
comparison; any isotope-mass ratio is a proxy, not an isotope-mixture model.

Balanced NASA and JANAF D0 values are 177.999972 and
180.390000 kJ/mol. The calculation subtracts each source's
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
     - 1.201515
     - +0.191718
     - -0.009216
   * - 2500
     - 1.087896
     - +0.115031
     - -0.030419
   * - 3000
     - 1.053841
     - +0.095859
     - -0.043737
   * - 4000
     - 1.009885
     - +0.071894
     - -0.063609

Primary evidence and scientific interpretation
----------------------------------------------

The inspected 1988 publisher abstract gives De=14772.7 +/-0.6 cm-1 and
D0=14282.6 cm-1. The latter converts to 170.857839 kJ/mol, within 2 J/mol of
BC2016's 1.7708 eV and about 9.532 kJ/mol below JANAF. Khan et al. (2026)
independently adopt that De in their published model. The quoted uncertainty
belongs to De; a complete D0 uncertainty is not inferred from the abstract.
The same-thermal energy-only diagnostic multiplies JANAF K by 0.4657 at
1500 K (see exact values in the packet). This motivates a spectroscopic-energy
candidate even though production remains unchanged in this source PR.

All 13081 current Khan states are retained, including X/A/B/C labels and
bound/quasi-bound levels. The nuclear-spin product 8 is removed once. At
4000 K the C label contributes 0.1043 percent to Q. The paper's PF discussion
names X/A/B, while the distributed state file also includes C; we report
that scope difference without silently deleting states. Published PF knots
are noninteger and rounded (1500.2, 4000.4 K near our samples). Replays use
the actual tabulated temperature. Residual small state/PF differences are
recorded and are not claimed to be fully explained by rounding alone.

At 1500/4000 K Q/NASA is 1.000119/1.018201, while the 4000 K Cp difference
is +6.60348 J/mol/K. The top 10 percent of J carries 0.397 percent of Q;
that is a sensitivity diagnostic, not a completeness proof. The raw BC2016
Q is about twice the spin-normalized low-temperature sum. Khan explicitly
discusses and divides this unexplained extra factor two in its comparison.
We retain the raw BC data in diagnostics. Production KH is JANAF, so no
factor-two correction is applied to it. Metadata's 5000 K maximum and the
published table ending at 10001 K do not independently prove physical
completeness or thermochemical suitability.

Current footprint and bounded decision
--------------------------------------

The curated-oh-v1 archive contains 75 converged, 537-species gas-only cases.
The maximum K allocation to KH is 5.1015005 percent,
at 2500 K, 100 bar, C/O=1
(case 39); its mole fraction there is
1.0275729e-08. The largest relative K conservation
residual is 4.23e-10. This is an existing allocation
footprint, not a measured abundance response to a new thermodynamic model.
It excludes condensation, photochemistry, transport and observational validation.

* Build a provisional energy-only candidate using the inspected D0=14282.6 cm-1, explicitly retaining abstract-only uncertainty limits; use the full 1988 paper if a precision uncertainty claim is needed.

* Reconcile Khan PF generation, MARVELization and X/A/B/C bound/quasi-bound selection, then compare Q/H/S/Cp with production and NASA thermal candidates.

* Run fixed-network K redistribution, conserving K/H and recording convergence. Source-review merger does not require completing candidate adoption.

Reproduction and source access
------------------------------

Run ``python mceg/reviews/kh_source_review.py`` offline and
``pytest tests/model_evidence/kh_source_review_test.py`` for independent
energy, caloric, conservation and provenance checks. The evidence packet is
``tests/data/validation/kh_source_review/20261105``; original source hashes,
inspection limits, frozen numerical results and a receipt accompany it.
``--extract /path/to/originals`` verifies all original state/PF/definition
hashes before regenerating the compact state packet. Every distributed state
is preserved; no network access is needed by replay or CI.

* `DOI 10.1093/mnras/stag979 <https://doi.org/10.1093/mnras/stag979>`_: Full published 15-page paper, sections 2.3/3.1/4 and current official states/PF/definition.

* `DOI 10.1016/0009-2614(88)87016-7 <https://doi.org/10.1016/0009-2614(88)87016-7>`_: Publisher abstract inspected; original full text not acquired. Khan 2026 separately confirms adopted De.

* `DOI 10.1051/0004-6361/201526961 <https://doi.org/10.1051/0004-6361/201526961>`_: Pinned numerical excerpts reused; older spectroscopic originals not reaudited.

* Official JANAF tables and the pinned NASA CEA source are linked individually
  in ``sources.json``. The evaluated entries are inspected; upstream book
  originals are not represented as freshly audited.
