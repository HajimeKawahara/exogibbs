cis-H2N2: source review on the current curated baseline
=======================================================

Resolve the identity exclusion and prepare a retained-thermal ATcT energy candidate for cis-HNNH as a separate follow-up. Do not replace the cis row with the NASA equilibrium-isomer mixture.

Identity and inspected sources
------------------------------

Neutral cis-HNNH (Z-diazene), CAS 15626-42-3, singlet C2v ground state. It is distinct from trans-HNNH and isodiazene H2NN. The JANAF H-061 evaluation and FastChem record explicitly identify cis. NASA N2H2 explicitly combines trans, cis and 1,1 isomers in equilibrium, and therefore cannot be a single-cis comparator.

The current source label is **Chase 1998 JANAF (FastChem fit) [U]**. NASA CEA records cite Gurvich,1989 pt1 p356.Trans,cis,& 1,1 in equilibrium. The original numerical records and gas identities are pinned; shared ancestry is not independent validation. This NASA identity is different and is not used as a replacement or numerical comparator.

The original JANAF page 1321 was visually inspected. Its 1965 evaluation uses estimated geometry, partly estimated vibrational frequencies and a broad energy uncertainty, and states that cis thermochemical properties are adopted. ATcT 1.228 gives Hf(0)=228.92 kJ/mol versus the old JANAF value near 220.00 kJ/mol. Consistent atomic balancing yields a physical atomization-energy decrease of 9.259 kJ/mol, reducing retained-fit K to 0.47597 at 1500 K and 0.75699 at 4000 K. Current/JANAF agrees within 1.68%, isolating source energy rather than a fit transcription failure.

The full Martin-Taylor benchmark author manuscript was inspected. Its high-level anharmonic calculation gives a cis-minus-trans 0 K gap of 5.21 kcal/mol and explicitly separates three isomers; it is a major contributor to the current ATcT cis value, not independent validation of the same network. The full 2024 isotope-spectroscopy paper distinguishes cis/trans/iso IR bands at the CCSD level; it is not a complete high-temperature thermodynamic evaluation and its qualitative equilibrium discussion does not remove cis from the 1500-4000 K network.

Source comparisons
------------------

All comparisons use atom formation from neutral ground-state atoms at one bar. Native NASA results preserve R=8.314510 J/mol/K; JANAF and the reaction engine use R=8.31446261815324. Molecular H is reconstructed from one fixed 298.15 K formation anchor plus the tabulated heat-content increment, never from the temperature-dependent formation-enthalpy column. NASA H(0) is its header Hf(298)-[H(298)-H(0)], not polynomial extrapolation to zero. No extra ZPE or well depth is introduced.

JANAF uses electronic weight one and rotational symmetry number two for the cis molecule. No trans or isodiazene statistical multiplier belongs in that single-isomer function. A future state-based thermal model must handle the strong resonances described in the benchmark paper and explicitly normalize nuclear-spin weights.

The exact-isomer JANAF molecular and atom-formation G/H/S/Cp are compared with the analytic current fit. A physical ATcT-minus-JANAF atomization-energy sensitivity retains the current thermal function and fitting residual. The fitted 1/T coefficient is not physical D0. No energy alignment can turn an equilibrium-isomer mixture into a single-isomer thermal function.

.. list-table:: Exact single-isomer source knots
   :header-rows: 1

   * - T / K
     - Current / exact JANAF
     - ATcT energy-only / current
   * - 1500
     - 1.0167543
     - 0.47596966
   * - 2000
     - 0.99687525
     - 0.57303936
   * - 3000
     - 0.98697387
     - 0.68990555
   * - 4000
     - 1.0047211
     - 0.75699363

The NASA mixed-species record is **excluded**, rather than assigned a same-isomer ratio. Native and energy-aligned NASA comparison fields are null. Its standalone molecular G/H/S/Cp are available under ``excluded_nasa_mixture`` solely to make the rejected identity mapping auditable.

Balanced atomization energies (kJ/mol): janaf=1153.717000, nasa=unavailable, atct_1.228=1144.458000. ATcT uses version 1.228 and all participating atoms from that same version. Rounded marginal uncertainties are retained with the inputs; they do not provide reaction covariance or a complete uncertainty budget.

Current atmospheric footprint
-----------------------------

Diazene participates in nitrogen-hydrogen reaction chemistry. Its cis inventory is tiny on this hot equilibrium grid; reaction intermediates and non-equilibrium populations are outside the inventory assessment.

The final **curated-tio2-v1** archive contains 75 converged ion-inclusive endpoints and 537 species: T=1500, 2000, 2500, 3000, 4000 K; P=1e-6, 1e-4, 0.01, 1, 100 bar; C/O=0.5888436553555908, 1, 1.5. The maximum mole fraction is 6.98294068e-12. Its largest participating-element share is 1.29159775e-07 (1.29159775e-05%) of N, at 4000 K, 100 bar and C/O=0.588843655. Every participating element is accounted for in the packet. The archived target potential agrees with the literal current function; no new equilibrium solve was run.

These finite hot hydrogen-rich results do not bound cooler or oxidizing atmospheres, photochemistry, non-LTE populations, condensation or reaction kinetics. A small inventory here is not evidence that a molecule is unimportant in those settings.

Decision, access needs and MCEG
-------------------------------

Build an exact cis ATcT-minus-JANAF physical energy shift while retaining the inherited thermal function and fitting residual; then perform current-network substitutions. A full thermal replacement needs a cis-resolved rovibrational model, not the NASA mixture. The available original evaluation and benchmark paper suffice for this scoped source review without a mandatory new literature request.

This review changes neither curated data nor runtime behavior. It is independently mergeable under the established source-review standard: source attribution, normalized numerical comparisons and the bounded decision are supported; complete physical accuracy and complete historical ancestry remain unknown. Missing historical full texts are recorded in MCEG and are not a mandatory hold on this scoped review. Candidate construction and fixed-network substitution, if recommended, are a separate follow-up.

References and reproducibility
------------------------------

* `NIST-JANAF SRD 13 <https://doi.org/10.18434/T42S31>`_: official selected neutral-gas tables.
* `NASA CEA thermodynamic records <https://cearun.grc.nasa.gov/cea/thermo.inp>`_: NASA/TP-2002-211556; original cached full-file hash and selected exact records.
* `ATcT 1.228 <https://atct.anl.gov/Thermochemical%20Data/version%201.228/>`_: exact phase/CAS-selected formation enthalpies.
* `Benchmark thermochemistry of diazene isomers (Martin and Taylor, 1999) <https://arxiv.org/abs/physics/9808014>`_; DOI 10.1080/002689799165530. Complete author manuscript inspected, including isomerization energies, anharmonic force fields and limits.
* `Infrared spectra of nitrogen-15-labelled diazene isomers (Breza and Gatial, 2024) <https://reference-global.com/download/article/10.2478/acs-2024-0008.pdf>`_; DOI 10.2478/acs-2024-0008. Complete primary PDF inspected; finite isotope spectroscopy, not a hot equilibrium thermodynamic replacement.

Replay with ``python mceg/reviews/cis_h2n2_source_review.py`` and
``pytest tests/model_evidence/cis_h2n2_source_review_test.py``.
The packet is ``tests/data/validation/cis_h2n2_source_review/20261106``;
the contract is ``mceg/contracts/cis_h2n2_source_review.yaml``.
Source hashes resolve through the existing historical-byte resolver; output hashes refer directly to this packet. No mutable public-default alias or shared index is pinned.
