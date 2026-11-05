NaO source review and candidate priorities
==========================================

Retain production provisionally; prioritize a complete NaOUCMe-based thermal candidate after resolving version and 2500-4000 K completeness.

This is a scoped source review for issues #308/#309. It adds reproducible
evidence and follow-ups without changing curated data. Full physical accuracy
remains unknown. Missing ancestral full texts are recorded in MCEG; they do
not block merging a review which candidly preserves its evidential limits.

Identity, construction and energy separation
--------------------------------------------

The species is neutral gas-phase NaO, identified as
``Na1O1``; neither a solid hydride/oxide nor a different oxide
stoichiometry is interchangeable. The reaction is Na(g) + O(g) -> NaO(g), with a
one-bar standard pressure. Current production uses the unchanged FastChem
JANAF formation fit. Its source line and all 75 saved chemical potentials
are replayed directly. Natural elemental masses are used for the molecular
comparison; any isotope-mass ratio is a proxy, not an isotope-mixture model.

Balanced NASA and JANAF D0 values are 246.999972 and
269.298000 kJ/mol. The calculation subtracts each source's
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
     - 6.151838
     - +1.788004
     - +0.022749
   * - 2500
     - 2.763962
     - +1.072802
     - -0.062375
   * - 3000
     - 2.233242
     - +0.894002
     - -0.097227
   * - 4000
     - 1.675085
     - +0.670501
     - -0.160842

Primary evidence and scientific interpretation
----------------------------------------------

The modern NaOUCMe list treats the coupled X doublet-Pi and A doublet-Sigma
states. All 36,120 distributed states are retained. Dividing gtot by the
common nuclear-spin product 4 preserves the electronic/fine-structure weights;
no additional doublet divisor is used. At 1500 K the normalized state sum is
0.853348 of NASA's molecular Q, and A contributes 7.07 percent. At 2500 K
Q/NASA is 0.836858. The 4000 K ratio 0.793268 is only a finite-list diagnostic:
the paper and metadata certify a 2500 K scope, and the top 10 percent of J
already carries 0.872 percent of the sum at 4000 K. Boundary population is
not a global missing-state bound.

The current 300 K PF is 6632.5136 including nuclear spin, reproduced by the
state sum to better than 8e-7 relative. Paper Table 4 instead gives 7027.4727.
Both values are recorded; no inferred energy shift is applied to hide the
revision difference. The paper's potential depths are De, not neutral-atom
D0 determinations. The Mintz et al. publisher abstract gives hf(298)=86.4
kJ/mol, much nearer JANAF's 83.680 than NASA's 106.505317, but an adopted D0
requires consistent thermal and atomic references. Its full text is a
candidate-selection follow-up, not a prerequisite for this scoped audit.

Current footprint and bounded decision
--------------------------------------

The curated-oh-v1 archive contains 75 converged, 537-species gas-only cases.
The maximum Na allocation to NaO is 0.0009104318 percent,
at 4000 K, 100 bar, C/O=0.588843655
(case 24); its mole fraction there is
2.4165094e-11. The largest relative Na conservation
residual is 2.67e-11. This is an existing allocation
footprint, not a measured abundance response to a new thermodynamic model.
It excludes condensation, photochemistry, transport and observational validation.

* Reconcile current NaOUCMe distribution with paper Table 4 and converge coupled electronic/rovibrational thermal moments to 4000 K before fitting a candidate.

* Inspect the full neutral NaO thermochemistry in DOI 10.1021/jp9034826 if selecting a new energy; balance to the same neutral-atom ground references.

* Run a fixed-network finite substitution only after the candidate is defined; historical source full texts are not a condition for merging this scoped audit.

Reproduction and source access
------------------------------

Run ``python mceg/reviews/nao_source_review.py`` offline and
``pytest tests/model_evidence/nao_source_review_test.py`` for independent
energy, caloric, conservation and provenance checks. The evidence packet is
``tests/data/validation/nao_source_review/20261105``; original source hashes,
inspection limits, frozen numerical results and a receipt accompany it.
``--extract /path/to/originals`` verifies all original state/PF/definition
hashes before regenerating the compact state packet. Every distributed state
is preserved; no network access is needed by replay or CI.

* `DOI 10.1093/mnras/stab3357 <https://doi.org/10.1093/mnras/stab3357>`_: Full published paper, sections 2-3 and Tables 2-6; official current states/PF/definition inspected.

* `DOI 10.1021/jp9034826 <https://doi.org/10.1021/jp9034826>`_: Publisher abstract only; full article and supplementary tables unavailable in this review.

* Official JANAF tables and the pinned NASA CEA source are linked individually
  in ``sources.json``. The evaluated entries are inspected; upstream book
  originals are not represented as freshly audited.
