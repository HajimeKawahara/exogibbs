HNO: source review on the current curated baseline
==================================================

Prioritize an HNO retained-thermal ATcT energy candidate, and compare NASA-thermal/ATcT separately. Keep curated unchanged until continuous candidate and current-network evaluations are completed.

Identity and inspected sources
------------------------------

Neutral nitrosyl hydride H-N-O (nitroxyl), CAS 14332-28-6, singlet ground state; not the H-O-N isomer, nitrite acid HONO, or NO- anion.

The current source label is **JANAF 1998 (FastChem fit) [U]**. NASA CEA records cite Gurvich,1989 pt1 p362 pt2 p227. Jacox,1998 p151. The original numerical records and gas identities are pinned; shared ancestry is not independent validation.

ATcT 1.228 gives balanced atomization energy 823.532 kJ/mol, 7.613 kJ/mol below the JANAF reference behind the inherited fit. The thermal-preserving diagnostic reduces K to 0.543 at 1500 K and 0.795 at 4000 K. Thermal disagreement changes sign with temperature, so a NASA/ATcT replacement is not equivalent to the physical energy shift. The current fit must retain its inherited residual if only the energy is changed.

The latest ATcT HNO provenance page was inspected. Its dominant contributor is the HNO -> H + NO dissociation threshold near 16450 cm-1, with additional atomization computations and connected NO thermochemistry. This first dissociation channel is not the total atomization to H+N+O. The individual old Dixon/Dateo papers were not inspected in full, and their network contribution is not claimed as a new independent measurement. No complete modern 1500-4000 K HNO partition-function source was established in this review.

Source comparisons
------------------

All comparisons use atom formation from neutral ground-state atoms at one bar. Native NASA results preserve R=8.314510 J/mol/K; JANAF and the reaction engine use R=8.31446261815324. Molecular H is reconstructed from one fixed 298.15 K formation anchor plus the tabulated heat-content increment, never from the temperature-dependent formation-enthalpy column. NASA H(0) is its header Hf(298)-[H(298)-H(0)], not polynomial extrapolation to zero. No extra ZPE or well depth is introduced.

HNO is a neutral singlet; it must not inherit the triplet convention of the isoelectronic NO- anion. NASA/JANAF molecular degeneracies are retained without an added spin or symmetry factor. All atoms and molecular energy zeros are balanced in their own source; natural-mass functions remain elemental proxies.

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
     - 1.3266588
     - 1.3133864
     - 1.0742192
     - 0.54312146
     - 0.50053834
   * - 2000
     - 1.2292317
     - 1.2313561
     - 1.0590317
     - 0.63266342
     - 0.59843047
   * - 3000
     - 1.0272574
     - 1.0347356
     - 0.93579252
     - 0.73696775
     - 0.79326633
   * - 4000
     - 0.84991819
     - 0.84750858
     - 0.78597141
     - 0.79540142
     - 1.0091288

Balanced atomization energies (kJ/mol): janaf=831.145000, nasa=828.642727, atct_1.228=823.532000. ATcT uses version 1.228 and all participating atoms from that same version. Rounded marginal uncertainties are retained with the inputs; they do not provide reaction covariance or a complete uncertainty budget.

Current atmospheric footprint
-----------------------------

HNO is a nitrogen/oxygen reaction intermediate. The modest hot-grid abundance does not validate kinetic pathways, oxygen-rich atmospheres or aqueous nitroxyl chemistry.

The final **curated-tio2-v1** archive contains 75 converged ion-inclusive endpoints and 537 species: T=1500, 2000, 2500, 3000, 4000 K; P=1e-6, 1e-4, 0.01, 1, 100 bar; C/O=0.5888436553555908, 1, 1.5. The maximum mole fraction is 1.89588777e-10. Its largest participating-element share is 1.75336187e-06 (0.000175336187%) of N, at 4000 K, 100 bar and C/O=0.588843655. Every participating element is accounted for in the packet. The archived target potential agrees with the literal current function; no new equilibrium solve was run.

These finite hot hydrogen-rich results do not bound cooler or oxidizing atmospheres, photochemistry, non-LTE populations, condensation or reaction kinetics. A small inventory here is not evidence that a molecule is unimportant in those settings.

Decision, access needs and MCEG
-------------------------------

Construct the retained-thermal physical energy shift first and a distinct NASA/ATcT thermal alternative; validate G/H/S/Cp and derivative continuity, then run fixed-network substitutions. The already available ATcT values and NASA records suffice for bounded candidate work. Original HNO dissociation papers would improve ancestry, but are not a mandatory source-review merge hold.

This review changes neither curated data nor runtime behavior. It is independently mergeable under the established source-review standard: source attribution, normalized numerical comparisons and the bounded decision are supported; complete physical accuracy and complete historical ancestry remain unknown. Missing historical full texts are recorded in MCEG and are not a mandatory hold on this scoped review. Candidate construction and fixed-network substitution, if recommended, are a separate follow-up.

References and reproducibility
------------------------------

* `NIST-JANAF SRD 13 <https://doi.org/10.18434/T42S31>`_: official selected neutral-gas tables.
* `NASA CEA thermodynamic records <https://cearun.grc.nasa.gov/cea/thermo.inp>`_: NASA/TP-2002-211556; original cached full-file hash and selected exact records.
* `ATcT 1.228 <https://atct.anl.gov/Thermochemical%20Data/version%201.228/>`_: exact phase/CAS-selected formation enthalpies.
* `Nitroxyl and its anion: spin states and reactivity (2002) <https://pmc.ncbi.nlm.nih.gov/articles/PMC124232/>`_; DOI 10.1073/pnas.112202099. Primary open article inspected for neutral/anion spin distinction; solution thermochemistry is not transferred.
* `ATcT1.228 HNO value and source-network provenance <https://atct.anl.gov/Thermochemical%20Data/version%201.228/species/?species_number=196>`_; DOI 10.1021/jp047912y. Current gas-species value/provenance page inspected; DOI identifies ATcT methodology, not a standalone new HNO measurement.

Replay with ``python mceg/reviews/hno_source_review.py`` and
``pytest tests/model_evidence/hno_source_review_test.py``.
The packet is ``tests/data/validation/hno_source_review/20261106``;
the contract is ``mceg/contracts/hno_source_review.yaml``.
Source hashes resolve through the existing historical-byte resolver; output hashes refer directly to this packet. No mutable public-default alias or shared index is pinned.
