C2H: source investigation
=========================

The 2026-11-02 review traces the large C2H difference in
`issue #308 <https://github.com/HajimeKawahara/exogibbs/issues/308>`_
to an old JANAF energy estimate and different thermal functions. The intended
species is neutral, gaseous, linear ethynyl (HCC, CAS 2122-48-7), with ground
state X doublet Sigma-plus. The baseline's composition match is now supported
by source identification; identical thermal populations are not implied.
This closes the initial literature/attribution review with explicit access
gaps. Production coefficients and the historical baseline are unchanged.

The subsequent :doc:`C2H correction <../presets/curated_c2h>` implements the
conditional NASA/ATcT choice as a separate opt-in dataset and records a
75-pair abundance comparison. The report below describes the initial source
review; its historical inputs and original curated preset remain frozen.

Sources and identity
--------------------

The official `JANAF C-124 table <https://janaf.nist.gov/tables/C-124.html>`_
and the carbon volume, printed p.672, retain a March 1967 evaluation. The
tabulated formation enthalpy at 298.15 K is 476.976 kJ/mol; the commentary
quotes about 477.0 +/- 29 kJ/mol. Its uncertainty is not treated as a modern
one-sigma error. The estimate combines bounds from Cotter (1958), Benson
(1965), mass-spectrometric results of Coats and Anderson (1957), and comparison
with Tsang's 1964 private communication. These original papers were not
individually inspected. The thermal model estimates frequencies and electronic
levels by analogy: bending 640 cm^-1 and an excited level at 9000 cm^-1.
Both the thermochemistry and the partition function need renewed assessment.

The official `NASA report <https://ntrs.nasa.gov/citations/20020085330>`_,
Appendix D p.88, lists C2H ``g 6/01``, Hf(298.15 K) = 566.200482 kJ/mol,
with NASA9 intervals 200--1000 and 1000--6000 K. All 18 selected coefficients
match the frozen record. The C and H records on pp.74 and 126 also match;
the selected excerpt archives all 72 coefficients. This authenticates these
records, not the acquisition
history of the entire local NASA copy. Its references are:

* Ervin et al. (1990), *Bond strengths of ethylene and acetylene*,
  JACS 112, 5750--5759, DOI ``10.1021/ja00171a013``.
* Jacox (1998), *Vibrational and Electronic Energy Levels of Polyatomic
  Transient Molecules. Supplement A*, JPCRD 27, 115--393,
  DOI ``10.1063/1.556017``.
* Peric, Peyerimhoff and Buenker (1990), vibronic C2H spectrum, **part III**,
  Molecular Physics 71, 693--719, DOI ``10.1080/00268979000102071``.
* Kanamori and Hirota (1988), infrared CCH vibronic bands,
  JCP 89, 3962--3969, DOI ``10.1063/1.454877``.

Jacox's inspected HCC entry, pp.142--143, identifies the ground state and
reports a bending frequency of 371.60 cm^-1 and an approximate A doublet Pi
origin at 3692.61 cm^-1, with strong vibronic mixing. Those values differ
substantially from JANAF's estimates. NASA's citation list does not establish
how every PAC input was assigned. Ervin's full numerical tables and Peric's
detailed calculation remain access gaps; bibliographic identification is not
full-text inspection. The saved literature record distinguishes these levels.

`ATcT 1.172 CCH <https://atct.anl.gov/Thermochemical%20Data/version%201.172/species/?species_number=103>`_
gives 563.73 kJ/mol at 0 K and 567.85 +/- 0.15 kJ/mol at 298.15 K, with
estimated 95% confidence limits. It supports a much higher energy anchor than
JANAF, but does not supply a replacement high-temperature Cp/S model. The
network includes Mordaunt and Ashfold's 1994 HCC--H photodissociation result
(DOI ``10.1063/1.467635``), with an adjusted input uncertainty. It is not a
second independent measurement of that result. CCH and the C-atom formation
enthalpies are correlated (47.9%); reaction uncertainty must use covariance.
No independent-error quadrature or ATcT runtime correction is performed here.

Numerical attribution
---------------------

The reaction is ``2 C(g) + H(g) -> CCH(g)``, with dimensionless Kp at 1 bar.
Fresh official C-124, C-003 and H-001 tables reproduce every selected baseline
H/S/G node. C and H are neutral gas atoms, not graphite or molecular hydrogen.
Their subtracted contributions are included explicitly. The calculations use
the baseline R = 8.31446261815324 J/mol/K, including NASA evaluation; NASA's
printed energy normalization uses 8.31451, so its header energy and re-evaluated
polynomial energy differ by a few J/mol. This small convention difference
does not explain the discrepancy.

.. list-table:: Fixed-node formation constants
   :header-rows: 1

   * - T (K)
     - JANAF / NASA
     - Curated / NASA
     - Energy-aligned JANAF / NASA
   * - 500
     - 9.76073e8
     - Outside curated range
     - 0.46760
   * - 1000
     - 18580.48
     - Outside curated range
     - 0.40668
   * - 1500
     - 471.67635
     - 475.86838
     - 0.36907
   * - 3000
     - 10.71031
     - 10.66803
     - 0.29959

The last column is a diagnostic: shift the JANAF reaction H(298.15 K) to the
NASA value while retaining JANAF thermal functions. It is not a proposed
physical model. The remaining factor 2.7 at 1500 K establishes that replacing
the energy anchor alone cannot reproduce NASA.

For delta = JANAF minus NASA, molecule minus atoms, at 1500 K the reaction
Gibbs-energy difference decomposes into:

.. list-table:: Contributions to delta G (kJ/mol)
   :header-rows: 1

   * - Term
     - Value
   * - H(298.15 K) anchor
     - -89.21084
   * - Enthalpy increment from 298.15 K
     - -3.36019
   * - -T delta S(298.15 K)
     - +8.78609
   * - -T delta entropy increment
     - +7.00553

The sum is -76.77940 kJ/mol and gives ln(K_JANAF/K_NASA) = -delta G/(RT).
Only +0.006875 kJ/mol of the sum comes from subtracting different atomic
models; this is a molecular-source discrepancy. The archived table also
contains Cp, H, S and the atomic/molecular split at every temperature.

Decision and remaining evidence
-------------------------------

The old JANAF energy anchor is unsuitable as the preferred new C2H reference.
NASA thermal functions with a consistently balanced modern energy anchor are
a candidate for the next adoption study. Retain the current runtime only
pending that study; this is not a physical endorsement of its inherited
JANAF fit. Before adoption, inspect the remaining original source details,
assess vibronic partition functions and run the one-species finite substitution
on the fixed 537-species network. No equilibrium abundance or spectral impact
is inferred from the K ratios. HCN's existing NASA+ATcT correction is covered
by the baseline regression and must not be applied twice.

The MCEG contract ``mceg/contracts/c2h_source_review.yaml`` separates supported
identity/numerical attribution from unknown high-temperature physical accuracy
and unexecuted abundance impact. Source hashes, inspection levels and follow-up
criteria live in ``tests/data/validation/c2h_source_review/20261102/``.
The large source PDFs are acquisition references, not required replay inputs.

Offline replay::

   python mceg/reviews/c2h_source_review.py --output results/c2h_review/comparison.json
   pytest tests/model_evidence/c2h_source_review_test.py

This creates a new result without overwriting baseline-001 or requiring MCEG,
network access, a sibling checkout or production thermochemical changes.
