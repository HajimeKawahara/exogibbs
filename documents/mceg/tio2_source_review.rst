TiO2 gas source review
========================

Decision and atmospheric relevance
------------------------------------

Retain the current JANAF/FastChem fit provisionally. This review supplies a
reproducible thermal diagnostic and identifies a useful candidate calculation;
it does not establish a complete replacement or change ``curated-cos-v1``.
Evidence review can merge while historical-source completeness and physical
accuracy remain explicitly unknown in MCEG, following the #311--317 policy.

In the existing 75-case, 537-species ion-inclusive archive, TiO2 contains up to
50.0872% of Ti, at 1500 K, 100 bar and reference C/O. Its mole fraction there is
8.03171e-8. This is an allocation in a gas-only network, not a prediction including
condensation, cloud nucleation or disequilibrium. No new equilibrium solve was run.

Source conventions and shared ancestry
----------------------------------------

All reactions balance ``Ti(g) + 2 O(g) -> TiO2(g)`` at one bar. The JANAF species
is gaseous bent TiO2, not rutile, anatase or a multi-monomer cluster. The NASA
record cites Chase 1998, page 1762, December 1973, and Jacox 1998. Its 18 printed
coefficients reproduce the frozen record. JANAF Hf298 is -305.432 kJ/mol; NASA's
native-R header gives -305.430 kJ/mol. Source comparison uses the same modern R
for dimensional H/S; the NASA header is checked separately with R=8.31451.

.. list-table:: Balanced source and fit comparison
   :header-rows: 1

   * - T (K)
     - Curated / NASA K
     - JANAF / NASA K
   * - 1500
     - 1.077321
     - 1.060675
   * - 3000
     - 1.025457
     - 1.036149
   * - 4000
     - 1.035478
     - 1.030101

The balanced H298 anchor difference is -0.634271 kJ/mol; the molecular anchor
alone is only about -0.004 kJ/mol with common R. Thus atomic references matter.
The agreement largely shares an old energy source, rather than independently
confirming its accuracy. Direct JANAF formation-G and H/S reconstructions agree
within source rounding. No extrapolation of the WebBook 4000--6000 K Shomate
interval is used.

What the primary records establish
------------------------------------

The printed `JANAF oxygen volume
<https://janaf.nist.gov/pdf/JANAF-FourthEd-1998-Oxygen.pdf>`_, pages 1762 and 1815,
was visually inspected. It gives Hf298 uncertainty 12.6 kJ/mol, S298 uncertainty
4.2 J/mol/K, an estimated bend of 270 +/- 80 cm-1, estimated geometry, and an
assumed triplet at 18880 cm-1. Its energy evaluation reconciles several
high-temperature experiments and depends on TiO auxiliary data. The individual
ancestral experiments were not all obtained. These are inherited uncertainties,
not a new reaction covariance estimate.

`Brunken et al. 2008 <https://doi.org/10.1086/528934>`_ provides measured
48Ti16O2 rotational constants, a closed-shell ground state and symmetry number
two. `Wang et al. 2009 <https://doi.org/10.1039/B821849H>`_ reports a ground-state
bend of 322 +/- 6 cm-1 in its primary abstract. The other fundamentals, 959 and
944 cm-1, are gas measurements compiled by
`NIST WebBook <https://webbook.nist.gov/cgi/cbook.cgi?ID=C13463677&Mask=3EE9&Units=SI>`_.
The excited-state bond angle in Wang is not used as ground-state geometry.

`Hildenbrand 1976 <https://doi.org/10.1016/0009-2614(76)80510-6>`_ reports
D0(atomization)=1271.9 +/- 11.7 kJ/mol in the primary publisher abstract. The
balanced JANAF zero-K value is 1267.776 kJ/mol, not its D298=1277.407 kJ/mol.
The +4.124 kJ/mol central difference is within the quoted uncertainty scales;
its energy-only K factors are 1.39190 and 1.17979 at 1500 and 3000 K. These are
sensitivity factors, not an adopted correction, significance test or abundance
response. Full reaction-cycle and thermal-assumption verification remains open.

`Sindel et al. 2022 <https://doi.org/10.1051/0004-6361/202243306>`_, inspected
through its `author manuscript <https://arxiv.org/pdf/2207.04987>`_, selects the
DFT method using JANAF monomer binding energy (section 2.3 and Table 2).
The reported 0.22 kJ/mol agreement is therefore not independent energy validation.
Its 4 kJ/mol selection tolerance must not replace the printed JANAF Hf uncertainty
of 12.6 kJ/mol. Cluster results are not automatically a gas-monomer replacement.

Restricted thermal diagnostic and next calculation
----------------------------------------------------

The offline RRHO diagnostic uses the measured rotational constants and gas
fundamentals as harmonic frequencies, one singlet ground state, symmetry two,
and nuclear-spin weight one. JANAF mean mass 79.8788 u is retained deliberately
to isolate thermal structure; this is not an isotope-mixture calculation.
It omits anharmonicity, centrifugal distortion and electronic completion.
At fixed molecular H298 and unchanged JANAF atoms, its K/JANAF ratios are
0.888439, 0.883169 and 0.879142 at 1500, 3000 and 4000 K. S298 becomes
259.32984 J/mol/K versus JANAF 260.148. The old estimated electronic state's
relative Boltzmann weight reaches 0.00337 at 4000 K under equal rovibrational
functions; this alone does not bound omitted real states.

The useful next step is a continuous thermal candidate, retaining the present
energy anchor first, with explicit treatment of measured fundamentals versus
harmonic frequencies, electronic levels and isotope conventions. Test its H/S/Cp
derivatives and finite fixed-network substitution separately from an energy
update. If energy replacement is pursued, inspect Hildenbrand's complete cycle
and `Balducci et al. 1985 <https://doi.org/10.1063/1.449377>`_ (metadata located;
full text not obtained). These missing papers do not block this bounded review.

Reproduction and MCEG
-----------------------

Run ``python mceg/reviews/tio2_source_review.py`` and
``pytest tests/model_evidence/tio2_source_review_test.py`` offline. Inputs,
source access, hashes, decomposition, RRHO diagnostics and archive checks are in
``tests/data/validation/tio2_source_review/20261106``. The contract
``mceg/contracts/tio2_source_review.yaml`` records source attribution, replay
and scoped retention separately from candidate adoption and physical accuracy.
