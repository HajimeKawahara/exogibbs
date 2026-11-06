O2 source review and next candidate
===================================

Retain the current JANAF-derived formation fit in this source-review PR.
A bounded next candidate should combine the evaluated 2025 natural-isotope
thermal functions with balanced ATcT 1.228 energy, followed by finite substitution.
The pointwise differences are below one percent; source quality and explicit
conventions motivate that candidate, rather than closeness to NASA.
No production coefficient, candidate factory or equilibrium solve changes here.

Identity and reference conventions
-----------------------------------

The target is neutral gas O2, canonical ``O2``, with an X triplet ground
manifold and thermally accessible electronic states. JANAF ``O2(ref)`` is the
elemental thermochemical reference: its zero formation enthalpy does not mean
zero atomization energy. Its ideal-gas functions apply over the reviewed
1500--4000 K range. Atomic O is the ground gas reference, not singlet excited O.
The ATcT singlet-atom page (species 446) is explicitly excluded; ground O is
species 22 and O2 is species 6 in version 1.228.

Both source functions use 1 bar. NASA9 retains its native R=8.314510 J/mol/K;
JANAF and the current fit use modern SI R=8.31446261815324 J/mol/K. Native-source
energy and molecular/atomic thermal contributions are separated before comparing
D0-aligned functions. The formal FastChem fit intercept is not substituted for
physical D0, and a future energy-only increment must preserve that fit residual.

The 2025 functions already include allowed rotational/electronic statistical
weights. Pure 16O has zero nuclear spin; no extra triplet multiplier or second
homonuclear symmetry division is applied. Published natural-isotope entropy
already includes the mass-translation adjustment. It receives no further natural
mass or isotope-mixing correction in this diagnostic.

Inspected primary evidence
---------------------------

* Official JANAF O-029/O-001 numerical tables and NASA CEA selected O2/O records
  were inspected. The NASA coefficients match the frozen Issue #308 records.
  Current production retains the JANAF-derived FastChem fit; source agreement
  does not establish independent ancestry.
* Furtenbacher, Harvey and Csaszar (2025),
  `DOI 10.1063/5.0273661 <https://doi.org/10.1063/5.0273661>`_: the full accepted
  manuscript and official supplemental natural/pure-isotope thermochemical and
  partition tables were inspected. The 15946 O2 levels are unchanged from 2019;
  the update revises SI constants and uncertainty treatment. Natural-mixture
  corrections use Woolley heat-capacity increments with 30% relative expanded
  uncertainty, integrated for H/S. The natural entropy includes about
  0.0034 J/mol/K of mass translation. Unbound-state uncertainty is not quantified.
* ATcT 1.228 ground-species pages give Hf0(O)=246.843 kJ/mol and Hf0(O2)=0.
  Their balanced central D0 is 493.686 kJ/mol, compared with JANAF 493.579.
  The rounded atomic uncertainty is 0.0021 kJ/mol; full network covariance is
  not reconstructed or claimed here.

The 2019 predecessor is DOI 10.1063/1.5083135. It is an ancestry reference;
its complete state file has not been independently resummed in this packet.
No missing ancestral text is a merge gate for this scoped review.

Numerical source separation
----------------------------

.. list-table:: Formation K ratios, before rounding
   :header-rows: 1

   * - T (K)
     - Current / NASA
     - JANAF / NASA
     - D0-aligned JANAF / NASA
     - 2025 natural + ATcT / current
   * - 1500
     - 1.00333367
     - 0.99929320
     - 0.99909220
     - 1.00508314
   * - 2000
     - 0.99784704
     - 0.99916735
     - 0.99901662
     - 1.00824060
   * - 3000
     - 0.99571872
     - 0.99805281
     - 0.99795244
     - 1.00830562
   * - 4000
     - 0.99757771
     - 0.99631847
     - 0.99624332
     - 1.00521316

The last column retains NASA atomic thermals and is a pointwise source diagnostic,
not an implemented candidate or an abundance response. The +107 J/mol physical
energy increment alone multiplies K by 1.008616 at 1500 K and 1.003222 at 4000 K.
At common D0, the 2025 natural molecular thermals/NASA K factor is
0.999466--0.999618 at these four nodes. The molecular Cp difference at 4000 K
is -0.04706 J/mol/K, within the published expanded 2025 Cp uncertainty of
0.55 J/mol/K. Shared spectroscopic inputs limit independence.

The packet records native molecular and formation-reaction G/H/S/Cp, literal
fit derivatives, raw-JANAF formation-column closure and physical D0 separately.
Independent pure-isotope Q/Q'/Q'' reconstruction closes the rounded published
thermal nodes within 3 J/mol in H, 0.002 J/mol/K in S and 0.003 J/mol/K in Cp.
These are rounding checks, not uncertainty bounds on the full state model.

Current-grid importance and decision
-------------------------------------

The final ``curated-tio2-v1`` archive contains 75 converged ion-inclusive,
537-species endpoints at 1500--4000 K, 1e-6--100 bar and three C/O values.
O2 reaches 0.00489408% of oxygen and mole fraction 1.32080945e-8, both at
2000 K, 1e-6 bar and reference C/O=0.5888436554. This is a hot H-rich
inventory, not a limit for oxidizing atmospheres, photochemistry or low-temperature
biology-related oxygen. O2 remains an important atmospheric species despite its
small inventory in this particular grid.

The next calculation can use the openly available 2025 tables and ATcT anchors:
construct derivative-consistent thermals, retain the physical energy versus fit
residual distinction, and compare both networks with the unchanged named base.
No additional paper is required to begin that scoped candidate. Full physical
accuracy, unbound-state effects and observational atmospheric validation remain
unknown; they are not converted into an indefinite source-review merge gate.

Replay and record
-------------------

Run ``python mceg/reviews/o2_source_review.py`` and
``pytest tests/model_evidence/o2_source_review_test.py``.
The compact source packet is ``tests/data/validation/o2_source_review/20261106``;
input hashes use the existing historical-source resolver, while output hashes
refer directly to this packet. The contract is ``mceg/contracts/o2_source_review.yaml``.
The archived footprint is a read-only replay, with zero new equilibrium solves.
