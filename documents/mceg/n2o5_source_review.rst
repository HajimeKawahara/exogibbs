N2O5: source review on the current curated baseline
===================================================

A bounded NASA-thermal/ATcT candidate is available for later evaluation; retain curated now. Do not invent a JANAF0 K anchor or infer physical D0 from the inherited fit intercept, since the official JANAF table omits 0 K caloric entries.

Identity and inspected sources
------------------------------

Neutral gas-phase covalent dinitrogen pentoxide O2N-O-NO2, CAS 10102-03-1, singlet molecule; not solid nitronium nitrate, aqueous nitric acid or an aggregate of independent NO2+ and NO3- ions.

The current source label is **JANAF 1998 (FastChem fit) [U]**. NASA CEA records cite Gurvich,1989 pt1 p343 pt2 p213. The original numerical records and gas identities are pinned; shared ancestry is not independent validation.

The native current/NASA ratio falls from its low-temperature value to 0.812 at 4000 K; NASA/ATcT gives 1.174 relative to current there. The official JANAF0 K row is blank, so the source alignment here deliberately removes balanced reaction enthalpy at 298.15 K and leaves JANAF D0 unavailable. NASA header and ATcT0 K energies remain independently usable. This makes the follow-up finite and actionable without a fabricated energy-only correction.

The latest ATcT species/provenance page was inspected and the covalent gas identity was checked separately from its condensed entry. It is strongly linked to the NO3 dissociation threshold and the measured NO2+NO3/N2O5 equilibrium. Osthoff et al.2007 is a concrete primary equilibrium source at atmospheric temperatures; the publisher abstract was inspected, not the full paper. That cold equilibrium evidence is not extrapolated as a validated 4000 K thermal function.

Source comparisons
------------------

All comparisons use atom formation from neutral ground-state atoms at one bar. Native NASA results preserve R=8.314510 J/mol/K; JANAF and the reaction engine use R=8.31446261815324. Molecular H is reconstructed from one fixed 298.15 K formation anchor plus the tabulated heat-content increment, never from the temperature-dependent formation-enthalpy column. NASA H(0) is its header Hf(298)-[H(298)-H(0)], not polynomial extrapolation to zero. No extra ZPE or well depth is introduced.

The covalent molecular gas is not represented as a charged ion pair. Internal torsion and symmetry belong to the molecular evaluated function; no separate conformer/spin multiplier is appended. The 298 K alignment is labelled throughout and is not called a zero-temperature atomization alignment. Native NASA R and fixed H298/H0 gauges are preserved.

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
     - 0.63443756
     - 0.62069009
     - 0.52824216
     - unavailable
     - 1.3860544
   * - 2000
     - 0.66786319
     - 0.6760084
     - 0.59899167
     - unavailable
     - 1.3596883
   * - 3000
     - 0.73997352
     - 0.75424719
     - 0.69581346
     - unavailable
     - 1.2672678
   * - 4000
     - 0.81155594
     - 0.80441316
     - 0.75720506
     - unavailable
     - 1.1742079

Balanced atomization energies (kJ/mol): janaf=unavailable, nasa=2152.710569, atct_1.228=2151.095000. ATcT uses version 1.228 and all participating atoms from that same version. Rounded marginal uncertainties are retained with the inputs; they do not provide reaction covariance or a complete uncertainty budget.

JANAF N2O5 has blank 0 K entries. Its energy-aligned column therefore uses a balanced **298.15 K enthalpy alignment**, not D0 alignment. No JANAF D0 or retained-fit physical energy update is invented. The NASA/ATcT column still uses their available 0 K references.

Current atmospheric footprint
-----------------------------

N2O5 is a reservoir in nighttime oxidizing nitrogen chemistry and can participate in heterogeneous conversion. The present hot reducing ideal-gas inventory does not include those kinetic or aerosol processes.

The final **curated-tio2-v1** archive contains 75 converged ion-inclusive endpoints and 537 species: T=1500, 2000, 2500, 3000, 4000 K; P=1e-6, 1e-4, 0.01, 1, 100 bar; C/O=0.5888436553555908, 1, 1.5. The maximum mole fraction is 5.83537383e-41. Its largest participating-element share is 1.07933836e-36 (1.07933836e-34%) of N, at 4000 K, 100 bar and C/O=0.588843655. Every participating element is accounted for in the packet. The archived target potential agrees with the literal current function; no new equilibrium solve was run.

These finite hot hydrogen-rich results do not bound cooler or oxidizing atmospheres, photochemistry, non-LTE populations, condensation or reaction kinetics. A small inventory here is not evidence that a molecule is unimportant in those settings.

Decision, access needs and MCEG
-------------------------------

Construct a NASA-thermal/ATcT candidate with explicit energy zero and compare G/H/S/Cp and current-network substitutions. A retained-thermal energy-only model would first need an independently justified original 0 K energy reference; do not manufacture it by extrapolating JANAF or the fitted 1/T term. DOI10.1039/B709193A is optional useful cold-equilibrium validation, not a mandatory hold on this scoped review.

This review changes neither curated data nor runtime behavior. It is independently mergeable under the established source-review standard: source attribution, normalized numerical comparisons and the bounded decision are supported; complete physical accuracy and complete historical ancestry remain unknown. Missing historical full texts are recorded in MCEG and are not a mandatory hold on this scoped review. Candidate construction and fixed-network substitution, if recommended, are a separate follow-up.

References and reproducibility
------------------------------

* `NIST-JANAF SRD 13 <https://doi.org/10.18434/T42S31>`_: official selected neutral-gas tables.
* `NASA CEA thermodynamic records <https://cearun.grc.nasa.gov/cea/thermo.inp>`_: NASA/TP-2002-211556; original cached full-file hash and selected exact records.
* `ATcT 1.228 <https://atct.anl.gov/Thermochemical%20Data/version%201.228/>`_: exact phase/CAS-selected formation enthalpies.
* `NO3 absorption and NO3+NO2/N2O5 equilibrium (Osthoff et al.2007) <https://pubs.rsc.org/en/content/articlelanding/2007/cp/b709193a>`_; DOI 10.1039/B709193A. Primary publisher abstract and its ATcT provenance link inspected; full original paper not obtained.
* `ATcT1.228 dinitrogen pentoxide value and source-network provenance <https://atct.anl.gov/Thermochemical%20Data/version%201.228/species/?species_number=882>`_; DOI 10.1021/jp047912y. Current gas identity, formation values and listed provenance contributors inspected; DOI identifies ATcT methodology.

Replay with ``python mceg/reviews/n2o5_source_review.py`` and
``pytest tests/model_evidence/n2o5_source_review_test.py``.
The packet is ``tests/data/validation/n2o5_source_review/20261106``;
the contract is ``mceg/contracts/n2o5_source_review.yaml``.
Source hashes resolve through the existing historical-byte resolver; output hashes refer directly to this packet. No mutable public-default alias or shared index is pinned.
