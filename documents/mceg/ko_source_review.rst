KO source review and candidate priorities
=========================================

Retain production provisionally; prioritize coupled-state thermochemistry over selecting NASA from the smaller apparent discrepancy.

This is a scoped source review for issues #308/#309. It adds reproducible
evidence and follow-ups without changing curated data. Full physical accuracy
remains unknown. Missing ancestral full texts are recorded in MCEG; they do
not block merging a review which candidly preserves its evidential limits.

Identity, construction and energy separation
--------------------------------------------

The species is neutral gas-phase KO, identified as
``K1O1``; neither a solid hydride/oxide nor a different oxide
stoichiometry is interchangeable. The reaction is K(g) + O(g) -> KO(g), with a
one-bar standard pressure. Current production uses the unchanged FastChem
JANAF formation fit. Its source line and all 75 saved chemical potentials
are replayed directly. Natural elemental masses are used for the molecular
comparison; any isotope-mass ratio is a proxy, not an isotope-mixture model.

Balanced NASA and JANAF D0 values are 269.999972 and
263.789000 kJ/mol. The calculation subtracts each source's
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
     - 0.421581
     - -0.497882
     - -0.365526
   * - 2500
     - 0.519522
     - -0.298729
     - -0.358146
   * - 3000
     - 0.543788
     - -0.248941
     - -0.362452
   * - 4000
     - 0.566114
     - -0.186706
     - -0.382950

Primary evidence and scientific interpretation
----------------------------------------------

The source difference has both energy and thermal causes. At 1500 K, JANAF
minus NASA shifts ln K by -0.497882 from balanced energy, -0.365526 from
molecular thermal functions, and +0.000605 from atomic thermal functions.
The molecular term remains about -0.383 at 4000 K, so an energy-only adjustment
cannot reconcile the two curves.

Lee, Soldan and Wright (2002) find strong spin-orbit mixing of nearby Sigma
and Pi states: X Omega=1/2 and two excited Omega components are the useful
labels. Their theoretical D0=66 +/-1 kcal/mol corresponds to 276.144 +/-4.184
kJ/mol, but their treatment neglects the atomic fine-structure correction;
the printed O fine-structure ordering is not used in our atomic anchors.
Their paper also calls for multireference treatment of the long-range region.
The 2009 Mintz et al. publisher abstract reports hf(298)=55.9 kJ/mol and a
spin-free Sigma/Pi separation of 1.1 kJ/mol. Full text and supporting tables
were not acquired. Te gaps cannot be used as rovibronic origins in an
unqualified three-level replacement: different rovibrational ladders and
couplings matter. No such replacement or invented state sum is presented.

Current footprint and bounded decision
--------------------------------------

The curated-oh-v1 archive contains 75 converged, 537-species gas-only cases.
The maximum K allocation to KO is 0.0002350569 percent,
at 4000 K, 100 bar, C/O=0.588843655
(case 24); its mole fraction there is
4.4168645e-13. The largest relative K conservation
residual is 4.23e-10. This is an existing allocation
footprint, not a measured abundance response to a new thermodynamic model.
It excludes condensation, photochemistry, transport and observational validation.

* Acquire DOI 10.1021/jp9034826 full text and supporting Tables S1/S3/S6 to reconcile hf298, spin-orbit convention and neutral-atom asymptotes.

* Build and validate derivative-consistent coupled-state thermal functions before a modern energy-plus-thermal candidate and finite substitution.

* The missing follow-up source is not a merge gate for the scoped retain-and-record source audit.

Reproduction and source access
------------------------------

Run ``python mceg/reviews/ko_source_review.py`` offline and
``pytest tests/model_evidence/ko_source_review_test.py`` for independent
energy, caloric, conservation and provenance checks. The evidence packet is
``tests/data/validation/ko_source_review/20261105``; original source hashes,
inspection limits, frozen numerical results and a receipt accompany it.

* `DOI 10.1063/1.1511179 <https://doi.org/10.1063/1.1511179>`_: Full seven-page paper read through primary institutional copy; Table I and sections III-V inspected.

* `DOI 10.1021/jp9034826 <https://doi.org/10.1021/jp9034826>`_: Publisher abstract only; full article and supporting numerical tables unavailable in this review.

* Official JANAF tables and the pinned NASA CEA source are linked individually
  in ``sources.json``. The evaluated entries are inspected; upstream book
  originals are not represented as freshly audited.
