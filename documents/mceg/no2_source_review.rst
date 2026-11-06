NO2: source review on the current curated baseline
==================================================

Compare an ATcT energy-only candidate with a separately labelled NASA-thermal/ATcT candidate; retain current curated pending that evaluation. Modern ground-manifold partition information does not justify calling the entire NASA thermal model validated.

Identity and inspected sources
------------------------------

Neutral bent O-N-O nitrogen dioxide, CAS 10102-44-0, doublet ground state; not nitrite NO2-, nitronium NO2+ or its N2O4 dimer.

The current source label is **JANAF 1998 (FastChem fit) [U]**. NASA CEA records cite Gurvich,1989 pt1 p332 pt2 p207. The original numerical records and gas identities are pinned; shared ancestry is not independent validation.

The modern energy is close to NASA and lower in binding than JANAF by 1.058 kJ/mol. The energy-aligned JANAF/NASA ratio falls from 0.939 to 0.818 over 1500-4000 K, so the growing difference is mostly thermal at high temperature. This needs explicit manifold and anharmonicity accounting rather than interpreting a newer energy anchor as a complete thermal correction.

The 2025 TIPS paper uses ground-vibrational rotational sums plus harmonic vibrational factors and a switch to a fully analytic product approximation at higher temperatures for the principal isotopologues. It is not a full coupled rovibronic high-temperature spectrum. The exact principal 16O14N16O nodes after spin-product removal give Q/NASA 0.96865 at 1500 K and 0.86081 at 4000 K. The disagreement is scientifically useful but does not by itself select one full thermal model. The supplied 2025 corrigendum was checked: its symmetry correction concerns the mixed 16O14N18O isotope, not the principal table used here.

Source comparisons
------------------

All comparisons use atom formation from neutral ground-state atoms at one bar. Native NASA results preserve R=8.314510 J/mol/K; JANAF and the reaction engine use R=8.31446261815324. Molecular H is reconstructed from one fixed 298.15 K formation anchor plus the tabulated heat-content increment, never from the temperature-dependent formation-enthalpy column. NASA H(0) is its header Hf(298)-[H(298)-H(0)], not polynomial extrapolation to zero. No extra ZPE or well depth is introduced.

For 16O14N16O remove nuclear-spin product 3 once from TIPS. Equivalent-oxygen symmetry and the electronic doublet are already part of the source construction; neither an extra division by two nor multiplication by two is applied. Natural-mass translation is only an elemental proxy for principal-isotope internal states.

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
     - 1.0346162
     - 1.0237422
     - 0.93871796
     - 0.91866651
     - 0.96835387
   * - 2000
     - 0.97453451
     - 0.97787768
     - 0.91631108
     - 0.93835768
     - 1.0275732
   * - 3000
     - 0.90210561
     - 0.90909401
     - 0.87052464
     - 0.95847093
     - 1.1095562
   * - 4000
     - 0.8486025
     - 0.8453395
     - 0.81829588
     - 0.96868864
     - 1.1792359

Balanced atomization energies (kJ/mol): janaf=928.473000, nasa=927.396928, atct_1.228=927.415000. ATcT uses version 1.228 and all participating atoms from that same version. Rounded marginal uncertainties are retained with the inputs; they do not provide reaction covariance or a complete uncertainty budget.

Current atmospheric footprint
-----------------------------

NO2 participates in atmospheric NOx chemistry and radiative/photochemical cycles. Those oxidizing and illuminated environments are outside the present reducing equilibrium grid.

The final **curated-tio2-v1** archive contains 75 converged ion-inclusive endpoints and 537 species: T=1500, 2000, 2500, 3000, 4000 K; P=1e-6, 1e-4, 0.01, 1, 100 bar; C/O=0.5888436553555908, 1, 1.5. The maximum mole fraction is 6.99899256e-15. Its largest participating-element share is 9.35545725e-11 (9.35545725e-09%) of N, at 4000 K, 1 bar and C/O=0.588843655. Every participating element is accounted for in the packet. The archived target potential agrees with the literal current function; no new equilibrium solve was run.

These finite hot hydrogen-rich results do not bound cooler or oxidizing atmospheres, photochemistry, non-LTE populations, condensation or reaction kinetics. A small inventory here is not evidence that a molecule is unimportant in those settings.

Decision, access needs and MCEG
------------------------------

Construct the energy-only candidate first. Before treating a NASA or TIPS thermal replacement as preferable, compare excited electronic states, vibration-rotation treatment and high-temperature derivatives explicitly; run fixed-network substitutions for the selected continuous candidate. Full historical books are an optional ancestry follow-up, not a mandatory merge hold.

This review changes neither curated data nor runtime behavior. It is independently mergeable under the established source-review standard: source attribution, normalized numerical comparisons and the bounded decision are supported; complete physical accuracy and complete historical ancestry remain unknown. Missing historical full texts are recorded in MCEG and are not a mandatory hold on this scoped review. Candidate construction and fixed-network substitution, if recommended, are a separate follow-up.

References and reproducibility
------------------------------

* `NIST-JANAF SRD 13 <https://doi.org/10.18434/T42S31>`_: official selected neutral-gas tables.
* `NASA CEA thermodynamic records <https://cearun.grc.nasa.gov/cea/thermo.inp>`_: NASA/TP-2002-211556; original cached full-file hash and selected exact records.
* `ATcT 1.228 <https://atct.anl.gov/Thermochemical%20Data/version%201.228/>`_: exact phase/CAS-selected formation enthalpies.
* `Partition sums for HITRAN2024 (Gamache et al., 2025) <https://www.homepages.ucl.ac.uk/~ucapsy0/pdf/25GaViTe.pdf>`_; DOI 10.1016/j.jqsrt.2025.109568. Complete author PDF and pinned official HAPI TIPS_2025 numerical arrays inspected.
* `Corrigendum to HITRAN2024 partition sums <https://doi.org/10.1016/j.jqsrt.2025.109618>`_; DOI 10.1016/j.jqsrt.2025.109618. Complete local Gamache_Corr.pdf inspected.

Replay with ``python mceg/reviews/no2_source_review.py`` and
``pytest tests/model_evidence/no2_source_review_test.py``.
The packet is ``tests/data/validation/no2_source_review/20261106``;
the contract is ``mceg/contracts/no2_source_review.yaml``.
Source hashes resolve through the existing historical-byte resolver; output hashes refer directly to this packet. No mutable public-default alias or shared index is pinned.
