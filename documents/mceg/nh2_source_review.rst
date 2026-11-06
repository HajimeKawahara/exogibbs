NH2: source review on the current curated baseline
==================================================

Prioritize a bounded NH2 candidate comparison: retain the literal inherited thermal fit and its residual while applying the balanced ATcT-minus-JANAF physical D0 shift; compare a NASA-thermal/ATcT alternative separately. Do not silently combine the energy and thermal changes. Keep curated unchanged in this source-review PR.

Identity and inspected sources
------------------------------

Neutral gas-phase amidogen NH2, CAS 13770-40-6, bent X 2B1 radical; not NH2- or an NH3 fragment treated as a separate isomer.

The current source label is **JANAF 1998 (FastChem fit) [U]**. NASA CEA records cite Hf:Anderson,1989. Gurvich,1989 pt1 p349.Jacox,1998 p133. The original numerical records and gas identities are pinned; shared ancestry is not independent validation.

The inherited JANAF energy is less binding than ATcT 1.228 by 4.089 kJ/mol after all H/N atoms are balanced. The JANAF/NASA thermal difference also grows with temperature, so the two candidate constructions are scientifically different. NH2 has the largest elemental footprint among the 33 remaining species, making this a meaningful next replacement calculation. The IUPAC radical assessment (2005) independently reports a much lower molecular formation enthalpy than the older JANAF value, in the direction of ATcT; the accessible NIST CCCBDB transcription was inspected, not the complete IUPAC article.

Andre, Maouhoub-Andre and Rat's June 2026 public thermochemical dataset was inspected in full. It lists three vibrational frequencies, a moment-of-inertia product, symmetry number 2 and a ground electronic weight of 1, with S(298.15 K)=188.940 J/mol/K. That value is about R ln(2) below the NASA/JANAF entropy. The dataset does not make a resolved total-J/fine-structure state-sum convention explicit; a weight of 1 is not by itself proof of an error if spin is included elsewhere. Its electron-spin normalization therefore needs clarification before use. The newer date alone does not establish a better thermal model; the printed energy choices are not a new experimental formation enthalpy. Independent replay of the printed classical-rotation, harmonic-vibration and electronic inputs reproduces Q and S, so an unreported resolved-spin factor is not evident in those printed outputs. A ground-weight-two sensitivity is temperature dependent, and is not an adopted correction.

Source comparisons
------------------

All comparisons use atom formation from neutral ground-state atoms at one bar. Native NASA results preserve R=8.314510 J/mol/K; JANAF and the reaction engine use R=8.31446261815324. Molecular H is reconstructed from one fixed 298.15 K formation anchor plus the tabulated heat-content increment, never from the temperature-dependent formation-enthalpy column. NASA H(0) is its header Hf(298)-[H(298)-H(0)], not polynomial extrapolation to zero. No extra ZPE or well depth is introduced.

The evaluated tables already include their electronic/rotational conventions. No extra electronic factor, symmetry divisor or nuclear-spin factor is applied to JANAF/NASA. A new NH2 state sum must distinguish total J from spin-free rotational N and handle the equivalent H nuclei consistently. Natural-mass evaluated functions are used here; no isotope-resolved mixture is introduced.

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
     - 0.83860569
     - 0.83421218
     - 0.92123067
     - 1.387998
     - 1.4987845
   * - 2000
     - 0.83650676
     - 0.83623202
     - 0.90083591
     - 1.2787676
     - 1.4190683
   * - 3000
     - 0.82271909
     - 0.82498519
     - 0.86694607
     - 1.1781333
     - 1.3626895
   * - 4000
     - 0.81495356
     - 0.81445429
     - 0.84532976
     - 1.1308261
     - 1.3369141

Balanced atomization energies (kJ/mol): janaf=709.647000, nasa=710.888526, atct_1.228=713.736000. ATcT uses version 1.228 and all participating atoms from that same version. Rounded marginal uncertainties are retained with the inputs; they do not provide reaction covariance or a complete uncertainty budget.

Current atmospheric footprint
-----------------------------

NH2 connects ammonia and other nitrogen species in hot gas chemistry. The present ideal-gas equilibrium footprint does not assess ammonia photolysis, reaction rates or atmospheric transport.

The final **curated-tio2-v1** archive contains 75 converged ion-inclusive endpoints and 537 species: T=1500, 2000, 2500, 3000, 4000 K; P=1e-6, 1e-4, 0.01, 1, 100 bar; C/O=0.5888436553555908, 1, 1.5. The maximum mole fraction is 1.32550785e-06. Its largest participating-element share is 0.0122586103 (1.22586103%) of N, at 4000 K, 100 bar and C/O=0.588843655. Every participating element is accounted for in the packet. The archived target potential agrees with the literal current function; no new equilibrium solve was run.

These finite hot hydrogen-rich results do not bound cooler or oxidizing atmospheres, photochemistry, non-LTE populations, condensation or reaction kinetics. A small inventory here is not evidence that a molecule is unimportant in those settings.

Decision, access needs and MCEG
-------------------------------

Construct the two continuous NH2 candidates with fixed atoms and explicit physical energy/fit-residual bookkeeping; validate G/H/S/Cp and scalar/JAX derivatives, then run current-base network substitutions. Ask the 2026 dataset authors for the ground-spin/rotation convention only if pursuing that thermal alternative; it is not required for the ATcT energy candidate or source-review merge.

This review changes neither curated data nor runtime behavior. It is independently mergeable under the established source-review standard: source attribution, normalized numerical comparisons and the bounded decision are supported; complete physical accuracy and complete historical ancestry remain unknown. Missing historical full texts are recorded in MCEG and are not a mandatory hold on this scoped review. Candidate construction and fixed-network substitution, if recommended, are a separate follow-up.

References and reproducibility
------------------------------

* `NIST-JANAF SRD 13 <https://doi.org/10.18434/T42S31>`_: official selected neutral-gas tables.
* `NASA CEA thermodynamic records <https://cearun.grc.nasa.gov/cea/thermo.inp>`_: NASA/TP-2002-211556; original cached full-file hash and selected exact records.
* `ATcT 1.228 <https://atct.anl.gov/Thermochemical%20Data/version%201.228/>`_: exact phase/CAS-selected formation enthalpies.
* `IUPAC critical evaluation of selected radicals (2005), as indexed/transcribed by NIST CCCBDB <https://cccbdb.nist.gov/hf0kx.asp>`_; DOI 10.1063/1.1724828. NIST numerical transcription and bibliographic identity inspected; complete original article not obtained.
* `NH2: Amino radical (Andre et al., 2026) <https://zenodo.org/records/20662187>`_; DOI 10.5281/zenodo.20662187. Complete numerical PDF inspected; raw file hash recorded, PDF not committed.

Replay with ``python mceg/reviews/nh2_source_review.py`` and
``pytest tests/model_evidence/nh2_source_review_test.py``.
The packet is ``tests/data/validation/nh2_source_review/20261106``;
the contract is ``mceg/contracts/nh2_source_review.yaml``.
Source hashes resolve through the existing historical-byte resolver; output hashes refer directly to this packet. No mutable public-default alias or shared index is pinned.
