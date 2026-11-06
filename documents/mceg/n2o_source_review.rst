N2O: source review on the current curated baseline
==================================================

Evaluate a NASA-thermal/ATcT candidate against a retained-thermal energy-only control in a separate follow-up. Modern partition data favor investigating the high-temperature JANAF thermal deficit, but the present source review does not adopt either candidate.

Identity and inspected sources
------------------------------

Neutral nitrous oxide N-N-O, CAS 10024-97-2, linear singlet ground state; not cyclic N2O, an N-O-N connectivity, or N2O+.

The current source label is **JANAF 1998 (FastChem fit) [U]**. NASA CEA records cite Gurvich,1989 pt1 p337 pt2 p210. The original numerical records and gas identities are pinned; shared ancestry is not independent validation.

JANAF and NASA have a small balanced energy difference but the energy-aligned JANAF/NASA ratio falls to 0.912 at 4000 K. The current literal fit follows JANAF to about one percent at the sampled nodes. ATcT lowers JANAF atomization energy by 0.829 kJ/mol; its thermal-preserving effect is distinct from the temperature-dependent thermal discrepancy.

The 2025 TIPS assessment was read with its N2O direct-sum/product-approximation discussion. For the principal 14N2-16O isotopologue the direct-sum/product crossover is 4520 K, above the reviewed upper limit; other isotopologues have different limits. Exact pinned principal-isotope TIPS nodes, divided by nuclear-spin product 9 once, give Q/NASA about 1.00023 at 1500 K and 0.97053 at 4000 K. They therefore do not reproduce either older source exactly. Sparse Q nodes here are not differentiated into calorics. The 2025 ABG-IMRHT/Ames-2000K paper supplies an additional modern line-list lead; its abstract was inspected and the full numerical model was not retrieved.

Source comparisons
------------------

All comparisons use atom formation from neutral ground-state atoms at one bar. Native NASA results preserve R=8.314510 J/mol/K; JANAF and the reaction engine use R=8.31446261815324. Molecular H is reconstructed from one fixed 298.15 K formation anchor plus the tabulated heat-content increment, never from the temperature-dependent formation-enthalpy column. NASA H(0) is its header Hf(298)-[H(298)-H(0)], not polynomial extrapolation to zero. No extra ZPE or well depth is introduced.

14N2-16O has nuclear-spin product 9; its two N atoms occupy inequivalent positions, so a homonuclear-rotor symmetry division must not be added. The source tables already include molecular degeneracies. TIPS is a principal-isotope internal-state proxy combined with the same natural-mass translation as NASA; no isotope-mixture entropy is added.

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
     - 0.95888095
     - 0.94895376
     - 0.98290352
     - 0.9356905
     - 0.94211018
   * - 2000
     - 0.94388755
     - 0.94598763
     - 0.97125851
     - 0.95136937
     - 0.98170167
   * - 3000
     - 0.92090489
     - 0.92713523
     - 0.94357408
     - 0.96731096
     - 1.032092
   * - 4000
     - 0.90362547
     - 0.90035383
     - 0.91230049
     - 0.97538165
     - 1.0652742

Balanced atomization energies (kJ/mol): janaf=1102.951000, nasa=1103.395679, atct_1.228=1102.122000. ATcT uses version 1.228 and all participating atoms from that same version. Rounded marginal uncertainties are retained with the inputs; they do not provide reaction covariance or a complete uncertainty budget.

Current atmospheric footprint
-----------------------------

N2O has atmospheric radiative and chemical relevance beyond this hot hydrogen-rich grid. Its tiny equilibrium abundance here cannot evaluate cooler, biological, oxidizing or photochemical production.

The final **curated-tio2-v1** archive contains 75 converged ion-inclusive endpoints and 537 species: T=1500, 2000, 2500, 3000, 4000 K; P=1e-6, 1e-4, 0.01, 1, 100 bar; C/O=0.5888436553555908, 1, 1.5. The maximum mole fraction is 1.92764676e-13. Its largest participating-element share is 3.56546667e-09 (3.56546667e-07%) of N, at 4000 K, 100 bar and C/O=0.588843655. Every participating element is accounted for in the packet. The archived target potential agrees with the literal current function; no new equilibrium solve was run.

These finite hot hydrogen-rich results do not bound cooler or oxidizing atmospheres, photochemistry, non-LTE populations, condensation or reaction kinetics. A small inventory here is not evidence that a molecule is unimportant in those settings.

Decision, access needs and MCEG
-------------------------------

Build continuous NASA/ATcT and retained-thermal energy-only candidates, validate all caloric derivatives and perform current-network substitutions. A TIPS-based replacement would additionally need a smooth state/caloric representation and treatment of its source crossover, not extrapolation of sparse partition nodes. No unavailable ancestral book is required for the bounded candidate calculation or source-review merge.

This review changes neither curated data nor runtime behavior. It is independently mergeable under the established source-review standard: source attribution, normalized numerical comparisons and the bounded decision are supported; complete physical accuracy and complete historical ancestry remain unknown. Missing historical full texts are recorded in MCEG and are not a mandatory hold on this scoped review. Candidate construction and fixed-network substitution, if recommended, are a separate follow-up.

References and reproducibility
------------------------------

* `NIST-JANAF SRD 13 <https://doi.org/10.18434/T42S31>`_: official selected neutral-gas tables.
* `NASA CEA thermodynamic records <https://cearun.grc.nasa.gov/cea/thermo.inp>`_: NASA/TP-2002-211556; original cached full-file hash and selected exact records.
* `ATcT 1.228 <https://atct.anl.gov/Thermochemical%20Data/version%201.228/>`_: exact phase/CAS-selected formation enthalpies.
* `Partition sums for HITRAN2024 (Gamache et al., 2025) <https://www.homepages.ucl.ac.uk/~ucapsy0/pdf/25GaViTe.pdf>`_; DOI 10.1016/j.jqsrt.2025.109568. Complete author PDF and pinned official HAPI TIPS_2025 numerical arrays inspected.
* `Accurate N2O IR line lists: ABG-IMRHT and Ames-2000K <https://www.sciencedirect.com/science/article/pii/S0022407325001645>`_; DOI 10.1016/j.jqsrt.2025.109502. Publisher abstract inspected; complete numerical line-list model not retrieved.

Replay with ``python mceg/reviews/n2o_source_review.py`` and
``pytest tests/model_evidence/n2o_source_review_test.py``.
The packet is ``tests/data/validation/n2o_source_review/20261106``;
the contract is ``mceg/contracts/n2o_source_review.yaml``.
Source hashes resolve through the existing historical-byte resolver; output hashes refer directly to this packet. No mutable public-default alias or shared index is pinned.
