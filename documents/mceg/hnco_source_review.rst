HNCO: source review on the current curated baseline
===================================================

Prioritize a retained-thermal ATcT energy candidate for HNCO. The major discrepancy is an obsolete energy anchor; the available evidence does not require replacing the inherited thermal function at the same time.

Identity and inspected sources
------------------------------

Neutral isocyanic acid H-N=C=O, CAS 75-13-8, singlet ground state and bent near-prolate asymmetric rotor. It is distinct from HOCN, HCNO, HONC, cyanate anion and condensed acid. JANAF C-053, NASA HNCO and exact ATcT gas identity agree; the appendix identity exclusion is resolved without changing the frozen 93-species screen.

The current source label is **Chase 1998 JANAF (FastChem fit) [U]**. NASA CEA records cite Hf:East,1993. Jacox,1998 p236. The original numerical records and gas identities are pinned; shared ancestry is not independent validation.

The original JANAF evaluation, printed page 601, was visually inspected. Its 1970 energy selection combines older photodissociation and solution-derived arguments and gives Hf(0) about -98.7 kJ/mol. ATcT 1.228 instead gives -115.86 kJ/mol, supported by later threshold experiments and a thermochemical network. With consistent atoms, the physical atomization-energy increase is 17.260 kJ/mol, raising the retained-fit K by 3.99057 at 1500 K and 1.68030 at 4000 K. Current/JANAF is within 1.65% at the four nodes; energy-aligned JANAF/NASA differs by only 0.59-0.69%, isolating energy as the dominant issue.

The NASA energy cites East 1993 and is already much closer to ATcT than JANAF. The 2004 focal-point abstract and 1996 threshold abstract support the direction of the later energy revision; they are not represented as independent confirmation of all ATcT inputs. The full 2021 author paper computes HNCO internal partition functions only through 500 K, using resolved rotational levels plus harmonic vibrations. That is useful convention evidence but does not establish a new 1500-4000 K thermal model.

Source comparisons
------------------

All comparisons use atom formation from neutral ground-state atoms at one bar. Native NASA results preserve R=8.314510 J/mol/K; JANAF and the reaction engine use R=8.31446261815324. Molecular H is reconstructed from one fixed 298.15 K formation anchor plus the tabulated heat-content increment, never from the temperature-dependent formation-enthalpy column. NASA H(0) is its header Hf(298)-[H(298)-H(0)], not polynomial extrapolation to zero. No extra ZPE or well depth is introduced.

JANAF explicitly uses ground-state weight one and symmetry number one. The 2021 partition values already set nuclear-spin degeneracy to one, so no second division or arbitrary electronic factor is allowed. A future state model must handle the quasilinear bending spectrum and natural-mass versus principal-isotope convention.

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
     - 0.27155564
     - 0.26716003
     - 0.99414962
     - 3.9905693
     - 3.9490786
   * - 2000
     - 0.37063465
     - 0.37085157
     - 0.9935971
     - 2.8234243
     - 2.843282
   * - 3000
     - 0.51000555
     - 0.51494736
     - 0.99335152
     - 1.9976409
     - 2.0304972
   * - 4000
     - 0.60957176
     - 0.60671349
     - 0.99309033
     - 1.6803048
     - 1.6840622

Balanced atomization energies (kJ/mol): janaf=1743.502000, nasa=1759.900332, atct_1.228=1760.762000. ATcT uses version 1.228 and all participating atoms from that same version. Rounded marginal uncertainties are retained with the inputs; they do not provide reaction covariance or a complete uncertainty budget.

Current atmospheric footprint
-----------------------------

HNCO participates in nitrogen/carbon chemistry and is observed in molecular environments; its trace equilibrium inventory on this hot hydrogen-rich grid is separate from its potentially important kinetic and cooler-cloud roles.

The final **curated-tio2-v1** archive contains 75 converged ion-inclusive endpoints and 537 species: T=1500, 2000, 2500, 3000, 4000 K; P=1e-6, 1e-4, 0.01, 1, 100 bar; C/O=0.5888436553555908, 1, 1.5. The maximum mole fraction is 1.3761476e-10. Its largest participating-element share is 1.27302637e-06 (0.000127302637%) of N, at 4000 K, 100 bar and C/O=1. Every participating element is accounted for in the packet. The archived target potential agrees with the literal current function; no new equilibrium solve was run.

These finite hot hydrogen-rich results do not bound cooler or oxidizing atmospheres, photochemistry, non-LTE populations, condensation or reaction kinetics. A small inventory here is not evidence that a molecule is unimportant in those settings.

Decision, access needs and MCEG
-------------------------------

Build the exact ATcT-minus-JANAF atomization-energy correction while retaining the current thermal shape and fitting residual; verify G/H/S/Cp, then perform fixed-network substitutions. Evaluate HNCO and NCO separately and together because their energy network and interconversion are coupled. Original 1996/2004 full texts are useful optional ancestry, not a prerequisite for this bounded source review.

This review changes neither curated data nor runtime behavior. It is independently mergeable under the established source-review standard: source attribution, normalized numerical comparisons and the bounded decision are supported; complete physical accuracy and complete historical ancestry remain unknown. Missing historical full texts are recorded in MCEG and are not a mandatory hold on this scoped review. Candidate construction and fixed-network substitution, if recommended, are a separate follow-up.

References and reproducibility
------------------------------

* `NIST-JANAF SRD 13 <https://doi.org/10.18434/T42S31>`_: official selected neutral-gas tables.
* `NASA CEA thermodynamic records <https://cearun.grc.nasa.gov/cea/thermo.inp>`_: NASA/TP-2002-211556; original cached full-file hash and selected exact records.
* `ATcT 1.228 <https://atct.anl.gov/Thermochemical%20Data/version%201.228/>`_: exact phase/CAS-selected formation enthalpies.
* `NCO and CHNO isomer focal-point thermochemistry (Schuurman et al., 2004) <https://pubmed.ncbi.nlm.nih.gov/15268193/>`_; DOI 10.1063/1.1707013. Primary author abstract inspected; original full numerical derivation not obtained.
* `Jet-cooled HNCO photodissociation thermochemistry (Zyrianov et al., 1996) <https://experts.arizona.edu/en/publications/competitive-photodissociation-channels-in-jet-cooled-hnco-thermoc/>`_; DOI 10.1063/1.472665. Author institutional abstract and current ATcT provenance inspected; full original paper not obtained.
* `HNCO observations and CHNO partition functions (Canelo et al., 2021) <https://arxiv.org/abs/2104.10605>`_; DOI 10.1093/mnras/stab1163. Complete author manuscript inspected, especially Table 2 and Appendix A; range ends at 500 K.

Replay with ``python mceg/reviews/hnco_source_review.py`` and
``pytest tests/model_evidence/hnco_source_review_test.py``.
The packet is ``tests/data/validation/hnco_source_review/20261106``;
the contract is ``mceg/contracts/hnco_source_review.yaml``.
Source hashes resolve through the existing historical-byte resolver; output hashes refer directly to this packet. No mutable public-default alias or shared index is pinned.
