NCO: source review on the current curated baseline
==================================================

Prioritize a retained-thermal ATcT energy candidate for NCO. Its large outdated atomization-energy offset is clear after exact identity matching, while a thermal replacement requires a separate rovibronic assessment.

Identity and inspected sources
------------------------------

Neutral N-C-O radical, CAS 22400-26-6, linear X2Pi ground electronic manifold. It is not cyanate anion NCO-, CON/CNO connectivity variants, or the closed-shell HNCO molecule. JANAF C-087, NASA NCO and the ATcT gas record refer to the same neutral radical, resolving the appendix identity exclusion.

The current source label is **Chase 1998 JANAF (FastChem fit) [U]**. NASA CEA records cite Hf:East,1993. Jacox,1998 p184. The original numerical records and gas identities are pinned; shared ancestry is not independent validation.

The original JANAF page 635 was visually inspected: the 1970 evaluation adopts Hf(0) about 159.0 kJ/mol from early photodissociation and competing estimates. The current ATcT value is 127.03 kJ/mol. Balanced against same-version atoms, the atomization-energy increase is 31.984 kJ/mol: retained-thermal K rises by 12.9945 at 1500 K and 2.61613 at 4000 K. The inherited fit reproduces JANAF to within 1.25%; aligned JANAF/NASA thermal ratios are only 1.02286-1.03818, so this is primarily an energy problem.

NASA cites East 1993 and moves the energy substantially toward later experiment/theory, but remains 4.411 kJ/mol below the current balanced ATcT atomization energy. The 2004 focal-point study and the 1996 HNCO threshold paper are present in the current ATcT provenance; their abstracts were inspected, not all original derivations. Current ATcT includes direct NCO dissociation and HNCO-to-NCO thresholds, so the update is not justified merely by the date of the database.

Source comparisons
------------------

All comparisons use atom formation from neutral ground-state atoms at one bar. Native NASA results preserve R=8.314510 J/mol/K; JANAF and the reaction engine use R=8.31446261815324. Molecular H is reconstructed from one fixed 298.15 K formation anchor plus the tabulated heat-content increment, never from the temperature-dependent formation-enthalpy column. NASA H(0) is its header Hf(298)-[H(298)-H(0)], not polynomial extrapolation to zero. No extra ZPE or well depth is introduced.

The JANAF page explicitly includes X2Pi spin-orbit components at 0 and 95.6 cm-1 with weights two each, plus higher electronic entries. Do not multiply its thermochemistry by another doublet or Pi degeneracy. A future thermal model must retain spin-orbit and vibronic contributions consistently; principal-isotope spin weights and translational masses require explicit normalization.

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
     - 0.11357999
     - 0.11217778
     - 1.0228632
     - 12.994515
     - 12.54722
   * - 2000
     - 0.19562351
     - 0.1959422
     - 1.0281608
     - 6.8441584
     - 6.6675452
   * - 3000
     - 0.3401208
     - 0.34265183
     - 1.0346856
     - 3.6047905
     - 3.5098736
   * - 4000
     - 0.45517228
     - 0.45321688
     - 1.0381799
     - 2.6161342
     - 2.5090984

Balanced atomization energies (kJ/mol): janaf=1269.854000, nasa=1297.427281, atct_1.228=1301.838000. ATcT uses version 1.228 and all participating atoms from that same version. Rounded marginal uncertainties are retained with the inputs; they do not provide reaction covariance or a complete uncertainty budget.

Current atmospheric footprint
-----------------------------

NCO is an intermediate connecting carbon and reactive nitrogen in combustion and atmospheric chemistry. The current finite hot reducing equilibrium grid contains very little NCO; this does not estimate non-equilibrium reaction fluxes.

The final **curated-tio2-v1** archive contains 75 converged ion-inclusive endpoints and 537 species: T=1500, 2000, 2500, 3000, 4000 K; P=1e-6, 1e-4, 0.01, 1, 100 bar; C/O=0.5888436553555908, 1, 1.5. The maximum mole fraction is 1.93898057e-11. Its largest participating-element share is 1.79368361e-07 (1.79368361e-05%) of N, at 4000 K, 100 bar and C/O=1. Every participating element is accounted for in the packet. The archived target potential agrees with the literal current function; no new equilibrium solve was run.

These finite hot hydrogen-rich results do not bound cooler or oxidizing atmospheres, photochemistry, non-LTE populations, condensation or reaction kinetics. A small inventory here is not evidence that a molecule is unimportant in those settings.

Decision, access needs and MCEG
-------------------------------

Construct the physical ATcT-minus-JANAF energy shift, preserving inherited thermal shape and fit residual, then run fixed-network substitutions. Include a joint HNCO/NCO diagnostic after separate tests. If replacing thermal functions later, investigate the few-percent aligned thermal difference independently of the much larger energy correction.

This review changes neither curated data nor runtime behavior. It is independently mergeable under the established source-review standard: source attribution, normalized numerical comparisons and the bounded decision are supported; complete physical accuracy and complete historical ancestry remain unknown. Missing historical full texts are recorded in MCEG and are not a mandatory hold on this scoped review. Candidate construction and fixed-network substitution, if recommended, are a separate follow-up.

References and reproducibility
------------------------------

* `NIST-JANAF SRD 13 <https://doi.org/10.18434/T42S31>`_: official selected neutral-gas tables.
* `NASA CEA thermodynamic records <https://cearun.grc.nasa.gov/cea/thermo.inp>`_: NASA/TP-2002-211556; original cached full-file hash and selected exact records.
* `ATcT 1.228 <https://atct.anl.gov/Thermochemical%20Data/version%201.228/>`_: exact phase/CAS-selected formation enthalpies.
* `NCO and CHNO isomer focal-point thermochemistry (Schuurman et al., 2004) <https://pubmed.ncbi.nlm.nih.gov/15268193/>`_; DOI 10.1063/1.1707013. Primary author abstract inspected; original full numerical derivation not obtained.
* `Jet-cooled HNCO photodissociation thermochemistry (Zyrianov et al., 1996) <https://experts.arizona.edu/en/publications/competitive-photodissociation-channels-in-jet-cooled-hnco-thermoc/>`_; DOI 10.1063/1.472665. Author institutional abstract and current ATcT provenance inspected; full original paper not obtained.
* `NCO photodissociation threshold (Cyr et al., 1992) <https://doi.org/10.1063/1.463989>`_; DOI 10.1063/1.463989. Primary bibliographic identity and current ATcT threshold provenance inspected; full paper not obtained.

Replay with ``python mceg/reviews/nco_source_review.py`` and
``pytest tests/model_evidence/nco_source_review_test.py``.
The packet is ``tests/data/validation/nco_source_review/20261106``;
the contract is ``mceg/contracts/nco_source_review.yaml``.
Source hashes resolve through the existing historical-byte resolver; output hashes refer directly to this packet. No mutable public-default alias or shared index is pinned.
