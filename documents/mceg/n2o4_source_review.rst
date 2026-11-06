N2O4: source review on the current curated baseline
===================================================

Keep current curated during this source review; a low-priority retained-thermal ATcT energy candidate can be evaluated together with NO2 when oxidizing NOx equilibria are targeted. Do not import a recent reactive-fluid equation of state as a single ideal-gas molecular standard potential.

Identity and inspected sources
------------------------------

Neutral gas-phase molecular dinitrogen tetroxide O2N-NO2, CAS 10544-72-6, singlet NO2 dimer; not liquid/reactive-fluid bulk NTO, nitrate/nitrosonium ion pairs or a different asymmetric connectivity.

The current source label is **JANAF 1998 (FastChem fit) [U]**. NASA CEA records cite Gurvich,1989 pt1 p342 pt2 p212. The original numerical records and gas identities are pinned; shared ancestry is not independent validation.

ATcT lowers JANAF atomization energy by 1.691 kJ/mol, whereas NASA and modern ATcT agree to about 2 J/mol at the rounded source precision. A retained-thermal correction gives K ratio 0.873 at 1500 K and 0.950 at 4000 K. Thermal alignment changes the JANAF/NASA ratio from 0.904 to 1.009 across the range, so the small raw differences partly conceal compensating energy and thermal changes.

The 2024 Polat et al. primary reactive-fluid paper was read; its partition-function sensitivities concern near-critical NO2/N2O4 phase behavior. The 2025 Samukov et al. primary NIST/publisher description and results were inspected: it combines a cubic equation of state with reaction equilibrium and VLE data near the fluid region. Neither establishes superior isolated-molecule thermal functions at 1500-4000 K. This is an explicit domain distinction, not dismissal of the newer experimental evidence.

Source comparisons
------------------

All comparisons use atom formation from neutral ground-state atoms at one bar. Native NASA results preserve R=8.314510 J/mol/K; JANAF and the reaction engine use R=8.31446261815324. Molecular H is reconstructed from one fixed 298.15 K formation anchor plus the tabulated heat-content increment, never from the temperature-dependent formation-enthalpy column. NASA H(0) is its header Hf(298)-[H(298)-H(0)], not polynomial extrapolation to zero. No extra ZPE or well depth is introduced.

Treat the neutral molecular dimer separately from two NO2 molecules and from a bulk dissociating fluid. Do not double count the dimer as an internal electronic or conformer state of NO2. The tabulated symmetry/torsional conventions are preserved; natural-mass evaluated functions are used without additional spin factors.

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
     - 1.0608789
     - 1.0358184
     - 0.90382603
     - 0.87320316
     - 0.94329658
   * - 2000
     - 0.99484692
     - 1.0052704
     - 0.90757771
     - 0.90330928
     - 1.0057251
   * - 3000
     - 0.99754268
     - 1.0180048
     - 0.95093415
     - 0.9344534
     - 1.0028259
   * - 4000
     - 1.0712077
     - 1.0617222
     - 1.0088146
     - 0.95042584
     - 0.93377901

Balanced atomization energies (kJ/mol): janaf=1910.083000, nasa=1908.393856, atct_1.228=1908.392000. ATcT uses version 1.228 and all participating atoms from that same version. Rounded marginal uncertainties are retained with the inputs; they do not provide reaction covariance or a complete uncertainty budget.

Current atmospheric footprint
-----------------------------

N2O4 controls NO2 association in cooler oxidizing gases and reactive fluids. The negligible current hot H-rich inventory does not validate low-temperature dimerization, liquid density or phase equilibrium.

The final **curated-tio2-v1** archive contains 75 converged ion-inclusive endpoints and 537 species: T=1500, 2000, 2500, 3000, 4000 K; P=1e-6, 1e-4, 0.01, 1, 100 bar; C/O=0.5888436553555908, 1, 1.5. The maximum mole fraction is 1.62186739e-34. Its largest participating-element share is 2.99988268e-30 (2.99988268e-28%) of N, at 4000 K, 100 bar and C/O=0.588843655. Every participating element is accounted for in the packet. The archived target potential agrees with the literal current function; no new equilibrium solve was run.

These finite hot hydrogen-rich results do not bound cooler or oxidizing atmospheres, photochemistry, non-LTE populations, condensation or reaction kinetics. A small inventory here is not evidence that a molecule is unimportant in those settings.

Decision, access needs and MCEG
-------------------------------

If lower-temperature or oxidizing NOx modeling is needed, first extend the model domain explicitly and compare measured NO2/N2O4 equilibrium with a consistent monomer/dimer standard-state model. Within 1500-4000 K, an energy-only candidate can be built directly from existing ATcT values and tested on the current network. Neither step requires every ancestral paper before this review can merge.

This review changes neither curated data nor runtime behavior. It is independently mergeable under the established source-review standard: source attribution, normalized numerical comparisons and the bounded decision are supported; complete physical accuracy and complete historical ancestry remain unknown. Missing historical full texts are recorded in MCEG and are not a mandatory hold on this scoped review. Candidate construction and fixed-network substitution, if recommended, are a separate follow-up.

References and reproducibility
------------------------------

* `NIST-JANAF SRD 13 <https://doi.org/10.18434/T42S31>`_: official selected neutral-gas tables.
* `NASA CEA thermodynamic records <https://cearun.grc.nasa.gov/cea/thermo.inp>`_: NASA/TP-2002-211556; original cached full-file hash and selected exact records.
* `ATcT 1.228 <https://atct.anl.gov/Thermochemical%20Data/version%201.228/>`_: exact phase/CAS-selected formation enthalpies.
* `Scaling towards the critical point in the combined reaction/Gibbs ensemble (2024) <https://omoultosethtudelft.github.io/web/assets/publications/89.Polat_FPE_2024_582_114084.pdf>`_; DOI 10.1016/j.fluid.2024.114084. Complete author-hosted primary paper inspected.
* `Accurate thermodynamic model for dissociating N2O4 VLE states (2025) <https://www.nist.gov/publications/accurate-thermodynamic-model-characterise-dissociating-n2o4-vapour-liquid-equilibrium>`_; DOI 10.1007/s10765-025-03565-x. Primary NIST abstract and publisher methodological/results sections inspected.

Replay with ``python mceg/reviews/n2o4_source_review.py`` and
``pytest tests/model_evidence/n2o4_source_review_test.py``.
The packet is ``tests/data/validation/n2o4_source_review/20261106``;
the contract is ``mceg/contracts/n2o4_source_review.yaml``.
Source hashes resolve through the existing historical-byte resolver; output hashes refer directly to this packet. No mutable public-default alias or shared index is pinned.
