NO3: source review on the current curated baseline
==================================================

Evaluate a retained-thermal ATcT energy-only candidate, while keeping full thermal replacement as a distinct vibronic-state follow-up. Near-exact NASA/JANAF agreement is shared-source consistency, not independent proof of thermal accuracy.

Identity and inspected sources
------------------------------

Neutral nitrate radical NO3, CAS 12033-49-7, associated with the X2A2-prime electronic state; not nitrate anionNO3-. ATcT calls this formula ON(O)O nitroxyl, which must not be confused with neutral HNO.

The current source label is **JANAF 1998 (FastChem fit) [U]**. NASA CEA records cite Chase,1998 p1607. The original numerical records and gas identities are pinned; shared ancestry is not independent validation.

NASA explicitly cites Chase 1998 for NO3; native NASA/JANAF atomization energy differs by only 1.286 J/mol and their aligned K ratio is within 0.021% of unity. ATcT reduces physical atomization energy by 1.917 kJ/mol, giving a retained-thermal K ratio 0.858 at 1500 K and 0.944 at 4000 K. That bounded energy direction is accessible even though complete vibronic thermochemistry remains unresolved.

Larsson and Viel 2024 was inspected in full. Its coupled five-electronic-state potential and roughly 2500 vibronic eigenstates establish substantial nonadiabatic effects and assignments through roughly 6000 cm-1; the authors note possible missing states above 5800 cm-1. This is not a complete rotational/vibronic partition function over 1500-4000 K. It gives a concrete source for a future thermal study, not permission to splice 2500 vibrational levels into a complete molecular Q without rotation, spin and convergence work.

Source comparisons
------------------

All comparisons use atom formation from neutral ground-state atoms at one bar. Native NASA results preserve R=8.314510 J/mol/K; JANAF and the reaction engine use R=8.31446261815324. Molecular H is reconstructed from one fixed 298.15 K formation anchor plus the tabulated heat-content increment, never from the temperature-dependent formation-enthalpy column. NASA H(0) is its header Hf(298)-[H(298)-H(0)], not polynomial extrapolation to zero. No extra ZPE or well depth is introduced.

Use the neutral radical CAS and doublet manifold, not the stable nitrate anion. Coupled electronic surfaces must not be counted once as vibronic states and again as an independent electronic partition factor. Atomic balances and conventional evaluated entropy are retained; no extra nuclear-spin or symmetry multiplier is added to the existing source functions.

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
     - 1.0151977
     - 1.0004976
     - 1.0000826
     - 0.85752232
     - 0.84503559
   * - 2000
     - 0.99256723
     - 1.0004607
     - 1.0001494
     - 0.89111566
     - 0.89806812
   * - 3000
     - 0.98752241
     - 1.0004011
     - 1.0001936
     - 0.92602501
     - 0.93792011
   * - 4000
     - 1.0065492
     - 1.000354
     - 1.0001983
     - 0.94398923
     - 0.93799303

Balanced atomization energies (kJ/mol): janaf=1133.661000, nasa=1133.662286, atct_1.228=1131.744000. ATcT uses version 1.228 and all participating atoms from that same version. Rounded marginal uncertainties are retained with the inputs; they do not provide reaction covariance or a complete uncertainty budget.

Current atmospheric footprint
-----------------------------

NO3 is an atmospheric oxidant in nighttime chemistry. Its tiny abundance in the present hot H-rich equilibrium grid cannot validate an illuminated or nocturnal oxidizing chemical network.

The final **curated-tio2-v1** archive contains 75 converged ion-inclusive endpoints and 537 species: T=1500, 2000, 2500, 3000, 4000 K; P=1e-6, 1e-4, 0.01, 1, 100 bar; C/O=0.5888436553555908, 1, 1.5. The maximum mole fraction is 1.90219476e-23. Its largest participating-element share is 1.75919472e-19 (1.75919472e-17%) of N, at 4000 K, 100 bar and C/O=0.588843655. Every participating element is accounted for in the packet. The archived target potential agrees with the literal current function; no new equilibrium solve was run.

These finite hot hydrogen-rich results do not bound cooler or oxidizing atmospheres, photochemistry, non-LTE populations, condensation or reaction kinetics. A small inventory here is not evidence that a molecule is unimportant in those settings.

Decision, access needs and MCEG
-------------------------------

Construct a physical ATcT-minus-JANAF energy shift with fixed inherited thermal function and fit residual; validate and run current-network substitutions. A new thermal candidate needs coupled vibronic levels, rotational treatment and high-energy completeness beyond the 2024 finite spectrum. The 2024 DOI is available; no extra paper is mandatory for the bounded energy candidate or review merge.

This review changes neither curated data nor runtime behavior. It is independently mergeable under the established source-review standard: source attribution, normalized numerical comparisons and the bounded decision are supported; complete physical accuracy and complete historical ancestry remain unknown. Missing historical full texts are recorded in MCEG and are not a mandatory hold on this scoped review. Candidate construction and fixed-network substitution, if recommended, are a separate follow-up.

References and reproducibility
------------------------------

* `NIST-JANAF SRD 13 <https://doi.org/10.18434/T42S31>`_: official selected neutral-gas tables.
* `NASA CEA thermodynamic records <https://cearun.grc.nasa.gov/cea/thermo.inp>`_: NASA/TP-2002-211556; original cached full-file hash and selected exact records.
* `ATcT 1.228 <https://atct.anl.gov/Thermochemical%20Data/version%201.228/>`_: exact phase/CAS-selected formation enthalpies.
* `2500 vibronic eigenstates of the NO3 radical (Larsson and Viel2024) <https://pubs.rsc.org/en/content/articlelanding/2024/cp/d4cp02653e>`_; DOI 10.1039/D4CP02653E. Complete author preprint inspected; raw hash recorded outside repo, no full paper or state list committed.
* `NO3 vibronic level structure I (Stanton2007) <https://pubmed.ncbi.nlm.nih.gov/17430034/>`_; DOI 10.1063/1.2715547. Primary abstract inspected; original full text not obtained.

Replay with ``python mceg/reviews/no3_source_review.py`` and
``pytest tests/model_evidence/no3_source_review_test.py``.
The packet is ``tests/data/validation/no3_source_review/20261106``;
the contract is ``mceg/contracts/no3_source_review.yaml``.
Source hashes resolve through the existing historical-byte resolver; output hashes refer directly to this packet. No mutable public-default alias or shared index is pinned.
