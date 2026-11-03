PO2 energy and thermal-source investigation
===========================================

This 2026-11-03 follow-up to #308 and #309 completes the initial source
assessment of neutral gaseous PO2, network ID ``O2P1``. The current curated
fit is retained provisionally. The evidence favors investigating a **weaker
binding energy than the inherited JANAF value**, with a separate thermal-model
review. It does not yet select a complete replacement. Remaining scientific
questions are recorded in MCEG and do not block this evidence-only update.

Why the sources differ
----------------------

The reaction is P(g) + 2 O(g) -> PO2(g), with dimensionless K at 1 bar.
The species is the bent, neutral X2A1 radical, CAS 12164-97-5, with C2v
symmetry. Neither PO2- nor PO3- nor a condensed oxide is interchangeable
with it. Total atomization into P + O + O also differs from PO + O dissociation.

* The inherited FastChem record cites JANAF 1998, but the inspected
  `printed PO2 page 1748
  <https://janaf.nist.gov/pdf/JANAF-FourthEd-1998-Oxygen.pdf>`_
  retains the **September 1962** evaluation. Its energy comes from a
  270 +/- 25 kcal/mol estimate attributed to Brewer and Rosenblatt (1961).
  Its geometry and vibrational frequencies are also estimates. Publication
  in the 1998 compilation is not a new measurement.
* The current official `O-032 table
  <https://janaf.nist.gov/tables/O-032.txt>`_ gives Hf(298.15)=-314.524 kJ/mol.
  Its inspected H/S nodes agree with the printed page and reproduce the
  frozen source-comparison baseline. The curated fit agrees with the
  atom-balanced JANAF H/S reconstruction within 0.010471 in ln K across
  26 nodes from 1500 to 4000 K, approximately 1.1% in K. This establishes
  numerical consistency, not the historical fitting procedure.
* `NASA/TP-2002-211556
  <https://ntrs.nasa.gov/citations/20020085330>`_, Appendix D page 170,
  identifies the neutral PO2 record as Gurvich (1989), volume 1 part 1
  page 408 and part 2 page 260. Its header Hf(298.15)=-281.527237 kJ/mol
  and H(298.15)-H(0)=10.512866 kJ/mol accompany two NASA9 intervals,
  200-1000 and 1000-6000 K. All displayed PO2/P/O coefficients match the
  frozen baseline. The original Gurvich discussion was not obtained.

The NIST CCCBDB value, -281.53 +/- 10 kJ/mol, cites the same Gurvich
evaluation. It is not an independent measurement supporting NASA. Likewise,
the experimental value quoted by the later paper below may overlap the
Gurvich energy basis; that genealogy has not been established.

Energy, thermal functions and atomic references
-----------------------------------------------

The molecular Hf(298.15) difference is about -32.997 kJ/mol. After subtracting
each source's P and two O atoms, the reaction difference is -32.889 kJ/mol.
The decomposition uses a common R=8.31446261815324 J/mol/K and retains the
298 K energy, thermal enthalpy increment, 298 K entropy and entropy increment
as separate terms. Atomic-reference contributions to the full free-energy
difference are only 0.093-0.107 kJ/mol over the six inspected temperatures.

.. list-table:: Atom-formation equilibrium-constant ratios
   :header-rows: 1

   * - T, K
     - Raw JANAF / NASA
     - H298-aligned JANAF / NASA
     - Current curated / NASA
   * - 1500
     - 17.4954
     - 1.25213
     - 17.6795
   * - 2000
     - 9.47060
     - 1.31045
     - 9.45900
   * - 3000
     - 5.15287
     - 1.37852
     - 5.11867
   * - 4000
     - 3.79512
     - 1.41171
     - 3.80712

The energy anchor dominates the large disagreement, but a 25-41% residual
in K remains after that diagnostic alignment. Neither the aligned curve nor
raw NASA is adopted here. At 3000 K the four JANAF-minus-NASA reaction
components are -32.8890, +2.5817, -0.0090 and -10.5798 kJ/mol respectively.
The entropy increment makes the old curve more favorable to PO2 formation.

Ground energies require another distinction. For each source, first compute
H(0)=Hf(298.15)-[H(298.15)-H(0)] in its 298 K reference gauge, then subtract
the same source's atoms. This gives total atomization D0=1120.275 kJ/mol
for JANAF and 1087.241875 kJ/mol from the NASA headers. Directly subtracting
the two molecular Hf(0) values would give a different, incorrect reaction
correction. The NASA header convention uses R=8.31451 J/mol/K; the common-R
polynomial comparison above is intentionally kept separate.

NASA explicitly labels its elemental P(cr) as white phosphorus. The JANAF
P(ref) record follows the white-P convention discussed in
:doc:`ph2_source_review`; the gas-atom tables are P-008 and O-001, not
P-001, which is the elemental reference phase. No additional white/red-P
shift is applied to the balanced gas reaction. The printed historical
270 kcal/mol estimate equals 1129.68 kJ/mol, **not** the current balanced
JANAF D0. Its original atomic/reference cycle has not been recovered, so
the note is not treated as an exact reconstruction of the current tables.

Later evidence and the preferred direction
------------------------------------------

The inspected open-access `Chen et al. (2023)
<https://doi.org/10.5194/acp-23-10643-2023>`_ ground-state reaction-dynamics
study provides a useful modern comparison in Table 3. It quotes an
experimental total D0 of 11.26 eV from Drowart et al. (1972) and a
ZPE-corrected theoretical atomization energy of about 11.40 eV from
Bauschlicher (1999). Its own electronic PES minimum is -11.435 eV relative
to separated atoms. The last quantity is an electronic well depth and
cannot be equated directly with D0 without a consistent zero-point and
atomic fine-structure treatment. These later values motivate weaker
binding than the balanced JANAF value; proximity to the quoted experiment
supports evaluating NASA as a diagnostic candidate, not declaring it exact.

The same table quotes experimental ground-state vibrational fundamentals
near 1075.4, 397.3 and 1327.5 cm^-1. JANAF used estimates of 980, 515 and
1044 cm^-1. Fundamentals, harmonic frequencies and fitted PES frequencies
are different objects; substituting three observed numbers does not establish
an anharmonic high-temperature partition function. CCCBDB also lists
Gurvich excited levels at 15447 and 19000 cm^-1 whose source assignments
were not inspected. A ground-state reaction PES alone does not resolve
electronic-state completeness in a thermal model.

Chen's citations were checked in that paper, **not** in the original
experimental or theoretical numerical tables. RSC access to the Drowart
article returned 403; only publisher-deposited bibliography was obtained.
The Bauschlicher publisher abstract establishes its computational scope,
but its numerical tables and associated thermal fits were not obtained.
These are recorded access gaps, not reasons to declare all later values
unsupported or to postpone this initial investigation indefinitely.

Practical importance in the saved model
----------------------------------------

The existing 75-case, 537-species ``curated-c2h-v1`` ion-inclusive grid has
PO2 holding at most **0.0743313% of the phosphorus budget**, at 1500 K,
1e-6 bar and reference C/O=0.5888436553555908. Its mole fraction there is
3.25297e-10. The budget fraction is 0.000743313, not 0.0743313.

This calculation reuses the saved converged endpoints and checks their
PO2 formation potentials against the current fit. It is a finite baseline
footprint, not a new equilibrium solve, an abundance response to replacing
the source, an observation, or evidence of irrelevance in every atmosphere.
The grid spans 1500-4000 K, five pressures from 1e-6 to 100 bar, and C/O
values of 0.5888436553555908, 1 and 1.5. No new substitution is needed to
complete this source-assessment scope.

MCEG follow-ups and reproduction
--------------------------------

``mceg/contracts/po2_source_review.yaml`` records supported numerical source
attribution and the conditional direction of the energy review. Quantified
accuracy of a complete replacement and of atmospheric predictions remain
unknown. Retaining the old fit is provisional, not an endorsement of its
1962 estimates.

For a future adoption, select an auditable modern energy evaluation and
balance its atomic, thermal and zero-point conventions. The priority
originals are `Drowart et al. (1972), 10.1039/F29726801749
<https://doi.org/10.1039/F29726801749>`_ and
`Bauschlicher (1999), 10.1021/jp992409k
<https://doi.org/10.1021/jp992409k>`_; they offer experimental and theoretical
routes, and both are not mandatory prerequisites to merge this record.
Gurvich's cited book pages can resolve the existing NASA construction;
`Lei et al. (2001), 10.1021/jp011778p
<https://doi.org/10.1021/jp011778p>`_ is a targeted spectroscopy follow-up.
Assess a named energy/thermal candidate over 1500-4000 K and then measure
its fixed-network equilibrium response before adopting it.

The compact packet is ``tests/data/validation/po2_source_review/20261103``.
It contains public numerical tables, factual source notes, a coefficient
excerpt and a hashed receipt; full research papers are not redistributed.
Replay and regression checks are offline and do not import MCEG::

   PYTHONPATH=src python mceg/reviews/po2_source_review.py
   PYTHONPATH=src pytest tests/model_evidence/po2_source_review_test.py

No production coefficient, runtime dependency, historical baseline or
equilibrium endpoint is changed.
