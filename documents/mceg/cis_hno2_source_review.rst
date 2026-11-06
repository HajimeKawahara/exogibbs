cis-HNO2: source review on the current curated baseline
=======================================================

Resolve the identity exclusion and retain curated pending a joint cis/trans energy-candidate assessment. A modern conformer-gap measurement is directly relevant, but must be reconciled with the current ATcT network before selecting an update.

Identity and inspected sources
------------------------------

Neutral cis-HONO nitrous acid, planar singlet ground state, CAS 7782-77-6 with exact ATcT suffix ``*2``. JANAF provides separate cis and trans tables and the runtime has separate species. NASA HNO2 explicitly describes a cis-trans mixture; matching only the elemental formula would combine different thermodynamic species.

The current source label is **Chase 1998 JANAF (FastChem fit) [U]**. NASA CEA records cite Cis-trans. Gurvich,1989 pt1 p363 pt2 p228. The original numerical records and gas identities are pinned; shared ancestry is not independent validation. This NASA identity is different and is not used as a replacement or numerical comparator.

The original JANAF cis evaluation page 1288 was visually inspected and explicitly describes a single conformer, with symmetry number one and its own frequencies. Current/JANAF differs by less than 1.46% at all four hot nodes. The same-isomer ATcT-minus-JANAF atomization-energy shift is +0.740 kJ/mol, giving K ratios 1.06113-1.02250 over 1500-4000 K. No NASA native or energy-aligned ratio is reported for this individual conformer.

The 2025 TIPS paper explicitly sums cis and trans, uses nuclear-spin factor six, and adopts the 2023 measured ground-level gap 95.8 +/- 9.2 cm-1. This is a ground-state energy difference, not a torsional barrier despite the 2023 title. Rounded ATcT 1.228 conformer values imply 1.610 kJ/mol (about 134.6 cm-1), while old JANAF implies 2.165 kJ/mol; the inspected leading ATcT cis contributors list older measurements. These alternatives deserve a joint energy assessment, not an undocumented direct replacement by the newest number. The 2025 spectroscopy paper improves selected interacting bands but does not establish a complete hot partition function for an individual conformer. The pinned gap replay converts the 2023 result to 1.14602 +/- 0.11006 kJ/mol. At a fixed trans reference it would raise cis K relative to the rounded ATcT-gap model by only 3.79% at 1500 K and 1.40% at 4000 K; this sensitivity is not an adopted correction.

Source comparisons
------------------

All comparisons use atom formation from neutral ground-state atoms at one bar. Native NASA results preserve R=8.314510 J/mol/K; JANAF and the reaction engine use R=8.31446261815324. Molecular H is reconstructed from one fixed 298.15 K formation anchor plus the tabulated heat-content increment, never from the temperature-dependent formation-enthalpy column. NASA H(0) is its header Hf(298)-[H(298)-H(0)], not polynomial extrapolation to zero. No extra ZPE or well depth is introduced.

Each curated isomer is a separate chemical species. A summed cis/trans partition function contains the relative ground-state Boltzmann weight and must never be copied into both rows or treated as a constant conformer multiplier. The NASA mixed function is pinned only to prove the identity mismatch; its standalone molecular G/H/S/Cp are explicitly segregated from valid single-isomer comparisons.

The exact-isomer JANAF molecular and atom-formation G/H/S/Cp are compared with the analytic current fit. A physical ATcT-minus-JANAF atomization-energy sensitivity retains the current thermal function and fitting residual. The fitted 1/T coefficient is not physical D0. No energy alignment can turn an equilibrium-isomer mixture into a single-isomer thermal function.

.. list-table:: Exact single-isomer source knots
   :header-rows: 1

   * - T / K
     - Current / exact JANAF
     - ATcT energy-only / current
   * - 1500
     - 1.0145744
     - 1.06113
   * - 2000
     - 0.99618413
     - 1.0455058
   * - 3000
     - 0.98908294
     - 1.0301116
   * - 4000
     - 1.0046293
     - 1.0224998

The NASA mixed-species record is **excluded**, rather than assigned a same-isomer ratio. Native and energy-aligned NASA comparison fields are null. Its standalone molecular G/H/S/Cp are available under ``excluded_nasa_mixture`` solely to make the rejected identity mapping auditable.

Balanced atomization energies (kJ/mol): janaf=1250.945000, nasa=unavailable, atct_1.228=1251.685000. ATcT uses version 1.228 and all participating atoms from that same version. Rounded marginal uncertainties are retained with the inputs; they do not provide reaction covariance or a complete uncertainty budget.

Current atmospheric footprint
-----------------------------

HONO is a reactive atmospheric nitrogen species and an OH photochemical source. The present hot hydrogen-rich equilibrium inventory is tiny and does not test cooler oxidizing atmospheres or photochemistry.

The final **curated-tio2-v1** archive contains 75 converged ion-inclusive endpoints and 537 species: T=1500, 2000, 2500, 3000, 4000 K; P=1e-6, 1e-4, 0.01, 1, 100 bar; C/O=0.5888436553555908, 1, 1.5. The maximum mole fraction is 1.77677255e-15. Its largest participating-element share is 1.6432013e-11 (1.6432013e-09%) of N, at 4000 K, 100 bar and C/O=0.588843655. Every participating element is accounted for in the packet. The archived target potential agrees with the literal current function; no new equilibrium solve was run.

These finite hot hydrogen-rich results do not bound cooler or oxidizing atmospheres, photochemistry, non-LTE populations, condensation or reaction kinetics. A small inventory here is not evidence that a molecule is unimportant in those settings.

Decision, access needs and MCEG
-------------------------------

For a joint candidate, choose one trans energy reference, compare the measured 2023 conformer gap with the ATcT covariance/provenance and retain separate isomer thermal functions. Test both individual changes and the combined HONO inventory. The 2023 full text, DOI 10.1016/j.jqsrt.2023.108727, would sharpen the energy assessment; the inspected 2025 primary paper already reproduces its key number. Its absence is not a hold on this identity/source review.

This review changes neither curated data nor runtime behavior. It is independently mergeable under the established source-review standard: source attribution, normalized numerical comparisons and the bounded decision are supported; complete physical accuracy and complete historical ancestry remain unknown. Missing historical full texts are recorded in MCEG and are not a mandatory hold on this scoped review. Candidate construction and fixed-network substitution, if recommended, are a separate follow-up.

References and reproducibility
------------------------------

* `NIST-JANAF SRD 13 <https://doi.org/10.18434/T42S31>`_: official selected neutral-gas tables.
* `NASA CEA thermodynamic records <https://cearun.grc.nasa.gov/cea/thermo.inp>`_: NASA/TP-2002-211556; original cached full-file hash and selected exact records.
* `ATcT 1.228 <https://atct.anl.gov/Thermochemical%20Data/version%201.228/>`_: exact phase/CAS-selected formation enthalpies.
* `Partition sums for HITRAN2024 (Gamache et al., 2025) <https://www.homepages.ucl.ac.uk/~ucapsy0/pdf/25GaViTe.pdf>`_; DOI 10.1016/j.jqsrt.2025.109568. Complete author PDF inspected, especially section 3.12; HONO is a cis-plus-trans sum, not a single-conformer table.
* `HONO far-infrared intensities and conformer energy difference (Tchana Betnga et al., 2023) <https://www.sciencedirect.com/science/article/abs/pii/S0022407323002455>`_; DOI 10.1016/j.jqsrt.2023.108727. Primary publisher abstract and numerical result as cited in the inspected 2025 TIPS paper; 2023 full text not obtained.
* `New interacting-band analysis of cis- and trans-HONO (2025) <https://www.tandfonline.com/doi/full/10.1080/00268976.2025.2561868>`_; DOI 10.1080/00268976.2025.2561868. Primary publisher abstract inspected; spectroscopy is not a complete hot thermodynamic evaluation.

Replay with ``python mceg/reviews/cis_hno2_source_review.py`` and
``pytest tests/model_evidence/cis_hno2_source_review_test.py``.
The packet is ``tests/data/validation/cis_hno2_source_review/20261106``;
the contract is ``mceg/contracts/cis_hno2_source_review.yaml``.
Source hashes resolve through the existing historical-byte resolver; output hashes refer directly to this packet. No mutable public-default alias or shared index is pinned.
