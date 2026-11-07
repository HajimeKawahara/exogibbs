NaCN source geometry and thermal review
=======================================

Retain current production while prioritizing a modern thermal candidate.
The present JANAF-derived fit inherits a demonstrably obsolete linear-molecule
assumption; agreement with NASA, which cites the same JANAF evaluation, does not
validate that geometry. This source-review PR records the conflict and actionable
next calculation. It does not adopt an incomplete nonlinear model or new energy.

Identity and inspected sources
--------------------------------

The species is the neutral gas monomer ``NaCN``, canonical ``C1N1Na1``,
not solid ionic salt, a charged molecule or a dimer. Gas-phase spectroscopy
establishes a nearly T-shaped singlet asymmetric top. The conventional cyanide
name should not be read as proof of a linear carbon-end bond; separate linear
isomers must not be counted again without a resolved potential/state treatment.

The official JANAF 1998 printed page 634 actually retains a March-1966
evaluation. Its heat-capacity/entropy discussion explicitly assumes a linear
structure by analogy with HCN, with C-infinity-v symmetry and a 180-degree angle.
It uses B0=.182724 cm-1, frequencies 400, 239 (twice), 2176 cm-1, electronic weight 1,
symmetry number 1 and mass 49.00747 u. The doubly represented bend makes four
oscillators; a nonlinear triatomic instead has three normal modes.
The printed source, gas label and numerical table were inspected, rather than
inferring this structure from a formula match.

The primary 1980 KCN gas-phase rotational paper,
`DOI 10.1063/1.440016 <https://doi.org/10.1063/1.440016>`_, was inspected in the
author's archived thesis chapter. The full 2012 NaCN manuscript,
`DOI 10.1016/j.jms.2011.12.005 <https://doi.org/10.1016/j.jms.2011.12.005>`_,
confirms the T-shaped motif and corrects incomplete Hamiltonian diagonalization.
Its Table 1 provides the NaCN rotational constants used here:
A/B/C=57921.935397000/8368.494770200/7271.885083800 MHz.
These are low-vibrational-state principal-isotope measurements, not a complete
1500--4000 K rovibrational or electronic partition function.

The potential-energy study DOI 10.1063/1.2749504 was inspected only at the
publisher-indexed abstract level. DOI 10.1111/j.1365-2966.2011.20132.x is a
specific vibrational/potential follow-up; its complete numerical tables were
not inspected here. No uninspected numerical energy is adopted from either.

Native energy and thermal comparison
--------------------------------------

Both sources use neutral atoms and 1 bar. NASA9 uses its native
R=8.314510 J/mol/K; JANAF/current use modern SI R. Physical D0, molecular thermal
functions and atomic-reference thermals are separated. The source physical
atomization energies are 1195.582000 kJ/mol (JANAF)
and 1195.789221 kJ/mol (NASA). A fit intercept is not
an independent measured D0. The JANAF printed Hf298 uncertainty is
2.1 kJ/mol; its historical
vaporization and monomer/dimer assumptions are recorded, not revalidated by a fit.

.. list-table:: Formation K ratios
   :header-rows: 1

   * - T (K)
     - Current / NASA
     - JANAF / NASA
     - D0-aligned JANAF / NASA
   * - 1500
     - 0.99640269
     - 0.99008976
     - 1.00612793
   * - 2000
     - 0.99509022
     - 0.99380869
     - 1.00585822
   * - 3000
     - 0.99664663
     - 0.99794415
     - 1.00599438
   * - 4000
     - 1.00158182
     - 0.99993204
     - 1.00597566

The packet includes molecular and reaction G/H/S/Cp, the literal fit derivatives,
raw JANAF formation-column closure and current-vs-source residuals. Rebuilding
the explicitly printed old linear RRHO approximation reproduces JANAF thermal H
within 3 J/mol, S within 0.003 J/mol/K and Cp within 0.001 J/mol/K at the four
nodes. This independently identifies the old model behind the table.

For scale only, measured nonlinear rotation gives qrot/qrot(old linear) between
28.912 and
47.213 over these nodes,
and the classical rotational Cp differs by R/2. This is **not a full partition
ratio, formation-K correction or abundance response**: changing rotation while
retaining the old four vibrational oscillators would be inconsistent. Reliable
three-mode or large-amplitude thermal functions are needed before a candidate.

Current importance and next step
----------------------------------

The final ``curated-tio2-v1`` archive gives maximum Na share
0.23376801% and maximum mole fraction
6.65719181e-09, at 1500 K,
0.01 bar and C/O=1.5. All 75 ion-inclusive,
537-species endpoints converge and reproduce the unchanged literal potential.
This hot H-rich grid is not a limit for cool carbon-rich envelopes, other
atmospheric compositions or non-LTE chemistry.

Prioritize nonlinear/large-amplitude thermal evaluation with measured rotation
and trustworthy gas or variational vibrational data. Keep energy selection
separate and preserve the old physical-energy versus fit-residual distinction.
A derivative-consistent candidate and finite substitution are follow-ups;
missing ancestral full text does not block this scoped source review.
Full physical uncertainty, anharmonicity, high-temperature state completeness
and planetary abundance accuracy remain unknown.

Replay
--------

Run ``python mceg/reviews/nacn_source_review.py`` and
``pytest tests/model_evidence/nacn_source_review_test.py``.
The packet is ``tests/data/validation/nacn_source_review/20261106``;
MCEG is ``mceg/contracts/nacn_source_review.yaml``. Input hashes use the existing
historical resolver; outputs are checked directly. No full paper or huge state
list is committed, and no new equilibrium solve is claimed.
