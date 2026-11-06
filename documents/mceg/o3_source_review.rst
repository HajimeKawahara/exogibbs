O3: source review on the current curated baseline
=================================================

Retain current curated in this bounded source review and record the unresolved high-temperature thermal choice. Do not replace it with NASA solely because the constants disagree; prioritize a thermal-state/dissociation review before a full ozone candidate.

Identity and inspected sources
------------------------------

Neutral bent ozone O-O-O, CAS 10028-15-6, singlet ground electronic state; not ozone anion, oxygen atom or the O2 triplet molecule.

The current source label is **JANAF 1998 (FastChem fit) [U]**. NASA CEA records cite Gurvich,1989 pt1 p101 pt2 p15. The original numerical records and gas identities are pinned; shared ancestry is not independent validation.

The energy-aligned JANAF/NASA ratio reaches 0.491 at 4000 K, whereas ATcT changes the JANAF atomization energy by only +1.099 kJ/mol. Thus a modest energy revision does not settle the roughly factor-two high-temperature thermal difference. Available low-temperature partition evidence cannot determine which full 1500-4000 K thermal model is physically superior.

The 2025 TIPS assessment and official numerical arrays limit the principal 16O3 partition sum to 1000 K. Its normalized Q/NASA is 0.98751 at that last node. All requested 1500-4000 K comparisons are explicitly unavailable for this source; no extrapolated TIPS ozone model is presented. The original JANAF/NASA evaluated functions remain fully compared and the gap is a finite high-temperature state-completeness question, not a request to obtain every historical paper.

Source comparisons
------------------

All comparisons use atom formation from neutral ground-state atoms at one bar. Native NASA results preserve R=8.314510 J/mol/K; JANAF and the reaction engine use R=8.31446261815324. Molecular H is reconstructed from one fixed 298.15 K formation anchor plus the tabulated heat-content increment, never from the temperature-dependent formation-enthalpy column. NASA H(0) is its header Hf(298)-[H(298)-H(0)], not polynomial extrapolation to zero. No extra ZPE or well depth is introduced.

Principal 16O nuclei have zero spin, so the TIPS nuclear-spin divisor is 1. Rotational symmetry/permutation restrictions are already included by the source. The molecule has a low dissociation channel to O2+O; total atomization energy to three atoms is not that first threshold. Bound/metastable populations and anharmonicity need explicit treatment before a high-temperature state replacement.

The balanced source difference is split into energy, molecular thermal and atomic thermal terms before exponentiation. Energy alignment removes the source-specific energy-over-R difference while retaining molecular and atomic thermal differences. This is a diagnostic, not a replacement. The current fit has an inherited residual; its 1/T coefficient is not a physical dissociation energy. All source-specific molecular and reaction G/H/S/Cp and analytic current-fit calorics are in ``comparison.json``.

.. list-table:: Exact source knots; K ratios are equilibrium-constant diagnostics
   :header-rows: 1

   * - T / K
     - Current / NASA
     - JANAF / NASA
     - Energy-aligned JANAF / NASA
     - ATcT energy-only / current
     - NASA thermal + ATcT / current
   * - 1500
     - 0.88945318
     - 0.88214454
     - 0.9474502
     - 1.0921187
     - 1.1432208
   * - 2000
     - 0.86370636
     - 0.86787872
     - 0.91563318
     - 1.0683225
     - 1.1723945
   * - 3000
     - 0.70873744
     - 0.7144592
     - 0.74043298
     - 1.0450448
     - 1.4227914
   * - 4000
     - 0.47995748
     - 0.47844981
     - 0.49143674
     - 1.0335969
     - 2.0966077

Balanced atomization energies (kJ/mol): janaf=595.021000, nasa=595.915105, atct_1.228=596.120000. ATcT uses version 1.228 and all participating atoms from that same version. Rounded marginal uncertainties are retained with the inputs; they do not provide reaction covariance or a complete uncertainty budget.

Current atmospheric footprint
-----------------------------

Ozone is important for UV absorption and atmospheric photochemistry in oxygen-rich environments. The extremely small abundance in the present hot H-rich equilibrium grid says nothing about those settings.

The final **curated-tio2-v1** archive contains 75 converged ion-inclusive endpoints and 537 species: T=1500, 2000, 2500, 3000, 4000 K; P=1e-6, 1e-4, 0.01, 1, 100 bar; C/O=0.5888436553555908, 1, 1.5. The maximum mole fraction is 2.19099681e-18. Its largest participating-element share is 1.21280921e-14 (1.21280921e-12%) of O, at 4000 K, 1 bar and C/O=0.588843655. Every participating element is accounted for in the packet. The archived target potential agrees with the literal current function; no new equilibrium solve was run.

These finite hot hydrogen-rich results do not bound cooler or oxidizing atmospheres, photochemistry, non-LTE populations, condensation or reaction kinetics. A small inventory here is not evidence that a molecule is unimportant in those settings.

Decision, access needs and MCEG
-------------------------------

Identify an ozone state/thermodynamic calculation with a defensible 1500-4000 K domain, first-dissociation treatment and converged Q/H/S/Cp; compare it against both old thermal models at a common ATcT atomization energy. A standalone energy-only sensitivity is possible, but is not evidence that the large thermal discrepancy is resolved. No additional DOI is mandatory to merge this scoped source-gap record.

This review changes neither curated data nor runtime behavior. It is independently mergeable under the established source-review standard: source attribution, normalized numerical comparisons and the bounded decision are supported; complete physical accuracy and complete historical ancestry remain unknown. Missing historical full texts are recorded in MCEG and are not a mandatory hold on this scoped review. Candidate construction and fixed-network substitution, if recommended, are a separate follow-up.

References and reproducibility
------------------------------

* `NIST-JANAF SRD 13 <https://doi.org/10.18434/T42S31>`_: official selected neutral-gas tables.
* `NASA CEA thermodynamic records <https://cearun.grc.nasa.gov/cea/thermo.inp>`_: NASA/TP-2002-211556; original cached full-file hash and selected exact records.
* `ATcT 1.228 <https://atct.anl.gov/Thermochemical%20Data/version%201.228/>`_: exact phase/CAS-selected formation enthalpies.
* `Partition sums for HITRAN2024 (Gamache et al., 2025) <https://www.homepages.ucl.ac.uk/~ucapsy0/pdf/25GaViTe.pdf>`_; DOI 10.1016/j.jqsrt.2025.109568. Complete author PDF and pinned official HAPI TIPS_2025 numerical arrays inspected.

Replay with ``python mceg/reviews/o3_source_review.py`` and
``pytest tests/model_evidence/o3_source_review_test.py``.
The packet is ``tests/data/validation/o3_source_review/20261106``;
the contract is ``mceg/contracts/o3_source_review.yaml``.
Source hashes resolve through the existing historical-byte resolver; output hashes refer directly to this packet. No mutable public-default alias or shared index is pinned.
