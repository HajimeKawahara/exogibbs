N2H4: source review on the current curated baseline
===================================================

Investigate a retained-thermal ATcT energy candidate and a separate NASA-thermal alternative, with torsion/inversion accounting as the central thermal check. A factor-two source disagreement alone does not prove the NASA thermal function is better.

Identity and inspected sources
------------------------------

Neutral gas-phase hydrazine H2N-NH2, CAS 302-01-2, singlet molecule with torsional/inversion motion; not liquid hydrazine, an ionic state, NH2NH radical or a chosen rigid planar conformer.

The current source label is **JANAF 1998 (FastChem fit) [U]**. NASA CEA records cite Gurvich,1989 pt1 p360 pt2 p225. The original numerical records and gas identities are pinned; shared ancestry is not independent validation.

The energy-aligned JANAF/NASA ratio rises from 1.499 at 1500 K to 2.263 at 4000 K, although their native atomization energies differ by only 85 J/mol. The large source difference is therefore thermal. ATcT lowers the JANAF physical atomization energy by 2.739 kJ/mol, giving a separate retained-thermal ratio 0.803 to 0.921. At 4000 K the molecular Cp differs by about 4.10 J/mol/K and entropy by 10.50 J/mol/K.

The ATcT 1.228 gas-hydrazine page, including its 2026 computational-network inputs, was inspected. The 2017 primary formation-enthalpy study reports gas Hf 298=97.0+/-3.0 kJ/mol from isogyric reactions, consistent with the current ATcT97.69+/-0.29 value; its abstract was inspected, not the full calculation tables. The 2001 hydrazine thermodynamic survey is an identified optional resource for internal-motion ancestry. Its full text was not obtained, so no detailed torsion model is attributed to it here.

Source comparisons
------------------

All comparisons use atom formation from neutral ground-state atoms at one bar. Native NASA results preserve R=8.314510 J/mol/K; JANAF and the reaction engine use R=8.31446261815324. Molecular H is reconstructed from one fixed 298.15 K formation anchor plus the tabulated heat-content increment, never from the temperature-dependent formation-enthalpy column. NASA H(0) is its header Hf(298)-[H(298)-H(0)], not polynomial extrapolation to zero. No extra ZPE or well depth is introduced.

Internal rotation/inversion and equivalent-well counting must be treated within one consistent molecular partition function. Do not add a conformer multiplicity or symmetry divisor to already evaluated NASA/JANAF entropy. The gas CAS record is separate from liquid hydrazine; no phase-change enthalpy is mistaken for gas atomization energy.

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
     - 1.5265705
     - 1.4898555
     - 1.4988876
     - 0.80282596
     - 0.52273262
   * - 2000
     - 1.6643329
     - 1.6730218
     - 1.680623
     - 0.84813709
     - 0.50729099
   * - 3000
     - 1.9472979
     - 1.9880478
     - 1.9940649
     - 0.89600556
     - 0.45873918
   * - 4000
     - 2.2736254
     - 2.2580055
     - 2.2631291
     - 0.92094359
     - 0.40413808

Balanced atomization energies (kJ/mol): janaf=1696.355000, nasa=1696.440048, atct_1.228=1693.616000. ATcT uses version 1.228 and all participating atoms from that same version. Rounded marginal uncertainties are retained with the inputs; they do not provide reaction covariance or a complete uncertainty budget.

Current atmospheric footprint
-----------------------------

Hydrazine belongs to reduced nitrogen chemistry and can matter in non-equilibrium nitrogen conversion. Its extremely low hot H-rich equilibrium share does not constrain production or storage in other environments.

The final **curated-tio2-v1** archive contains 75 converged ion-inclusive endpoints and 537 species: T=1500, 2000, 2500, 3000, 4000 K; P=1e-6, 1e-4, 0.01, 1, 100 bar; C/O=0.5888436553555908, 1, 1.5. The maximum mole fraction is 3.140385e-14. Its largest participating-element share is 5.8086047e-10 (5.8086047e-08%) of N, at 4000 K, 100 bar and C/O=0.588843655. Every participating element is accounted for in the packet. The archived target potential agrees with the literal current function; no new equilibrium solve was run.

These finite hot hydrogen-rich results do not bound cooler or oxidizing atmospheres, photochemistry, non-LTE populations, condensation or reaction kinetics. A small inventory here is not evidence that a molecule is unimportant in those settings.

Decision, access needs and MCEG
-------------------------------

A retained-thermal ATcT candidate is directly calculable. For a full thermal replacement, reconcile the evaluated torsional/inversion models and density of accessible states before selecting NASA or constructing a new function; then run fixed-network substitutions. DOI 10.1021/je000301o is useful optional ancestry, not a prerequisite for merging this numerical source review.

This review changes neither curated data nor runtime behavior. It is independently mergeable under the established source-review standard: source attribution, normalized numerical comparisons and the bounded decision are supported; complete physical accuracy and complete historical ancestry remain unknown. Missing historical full texts are recorded in MCEG and are not a mandatory hold on this scoped review. Candidate construction and fixed-network substitution, if recommended, are a separate follow-up.

References and reproducibility
------------------------------

* `NIST-JANAF SRD 13 <https://doi.org/10.18434/T42S31>`_: official selected neutral-gas tables.
* `NASA CEA thermodynamic records <https://cearun.grc.nasa.gov/cea/thermo.inp>`_: NASA/TP-2002-211556; original cached full-file hash and selected exact records.
* `ATcT 1.228 <https://atct.anl.gov/Thermochemical%20Data/version%201.228/>`_: exact phase/CAS-selected formation enthalpies.
* `Enthalpies of Formation of Hydrazine and Its Derivatives (2017) <https://pubs.acs.org/doi/10.1021/acs.jpca.7b04914>`_; DOI 10.1021/acs.jpca.7b04914. Primary publisher abstract inspected; full numerical tables not obtained.
* `Survey of the Thermodynamic Properties of Hydrazine (Giordano2001) <https://pubs.acs.org/doi/10.1021/je000301o>`_; DOI 10.1021/je000301o. Primary publisher abstract inspected; full text optional and not obtained.

Replay with ``python mceg/reviews/n2h4_source_review.py`` and
``pytest tests/model_evidence/n2h4_source_review_test.py``.
The packet is ``tests/data/validation/n2h4_source_review/20261106``;
the contract is ``mceg/contracts/n2h4_source_review.yaml``.
Source hashes resolve through the existing historical-byte resolver; output hashes refer directly to this packet. No mutable public-default alias or shared index is pinned.
