CP source and thermochemical review
===================================

Retain the Barklem-Collet-labelled CP function; the raw JANAF energy discrepancy is not a reason to revert the evaluated current source.

Identity and source scope
-------------------------

Neutral carbon phosphide is heteronuclear CP (C1P1), with X2Sigma+ ground state and an A2Pi excited manifold. Its rotational symmetry number is 1; thermochemical partition functions omit nuclear-spin statistical factors. An explicit spectroscopic state sum would have to remove the 31P spin factor consistently and count resolved electronic components once.

The retained source is the Barklem and Collet (2016), as explicitly attributed by the FastChem row. Fresh official
`JANAF C-097 <https://janaf.nist.gov/tables/C-097.html>`_ and neutral-atom
tables are pinned, together with selected records from the versioned
`NASA CEA database <https://github.com/nasa/cea/blob/2f79a647e85737742fb08a127da00d1ea1d069da/data/thermo.inp>`_.
The NASA header reads ``Gurvich,1991 pt1 p231 pt2 p219.``.
All coefficients match the screening snapshot; original interval bounds are kept.

Balanced comparison
-------------------

The reaction is ``1 C(g) + 1 P(g) -> CP(g)`` at 1 bar. Native NASA calorics use
R=8.314510; equilibrium normalization uses R=8.31446261815324 J/mol/K.
H0 is derived from printed Hf298 and H298-H0, never polynomial extrapolation
to zero. The balanced source atomization energies are 508.999977 kJ/mol (NASA) and 579.149000 kJ/mol (JANAF).
These are physical source anchors, not the literal fit coefficient of 1/T.

.. list-table:: Dimensionless formation-constant ratios
   :header-rows: 1

   * - T (K)
     - Current / NASA
     - Current / JANAF
     - H0-aligned JANAF / NASA
   * - 1500
     - 1.03566
     - 0.003749
     - 0.996646
   * - 2000
     - 1.02925
     - 0.0154172
     - 0.982764
   * - 3000
     - 1.02156
     - 0.0670115
     - 0.915665
   * - 4000
     - 1.02022
     - 0.14983
     - 0.826152

``comparison.json`` records G/H/S/Cp for each molecule and atom, balanced
reaction properties, and JAX derivatives of the literal current function.
The following molecular differences are JANAF minus NASA; they must not be
confused with the balanced reaction or a change in mixture abundances.

.. list-table:: Molecular caloric differences
   :header-rows: 1

   * - T (K)
     - Delta H (J/mol)
     - Delta S (J/mol/K)
     - Delta Cp (J/mol/K)
   * - 1500
     - -70549.127
     - -0.218494
     - -1.193214
   * - 2000
     - -71700.897
     - -0.865161
     - -3.503810
   * - 3000
     - -77224.884
     - -3.056236
     - -6.991099
   * - 4000
     - -84575.377
     - -5.169123
     - -7.294483

Interpretation and modern evidence
----------------------------------

The current function agrees with NASA within 2.1–3.6 percent at the reviewed nodes, while raw JANAF differs by factors of roughly 7–267. Source-balanced D0 differs by 70.149 kJ/mol between NASA and JANAF. Unlike PN, no phosphorus phase-reference correction is inferred here: the discrepancy is recorded as a source-energy disagreement. After H0 alignment the JANAF/NASA ratio still decreases to 0.826 at 4000 K, so an energy offset alone is not the whole thermal comparison. The current fit's 1/T coefficient is not asserted to be the independent physical dissociation energy of Barklem-Collet.

Barklem and Collet explicitly evaluate diatomic partition functions and dissociation equilibria. The inspected Ram et al. study supplies the A2Pi-X2Sigma+ spectroscopic system; transition strengths and partial band coverage do not establish an independent complete high-temperature formation function. No newly evaluated CP atomization measurement has been established in this scoped review. Retention rests on the current evaluated-source attribution and close NASA agreement, not on treating an uninspected ancestral D0 as exact.

`Barklem and Collet (2016) <https://doi.org/10.1051/0004-6361/201526961>`_: Primary paper abstract/method scope inspected; complete source-by-source CP ancestral constants not independently reconstructed.

`Ram et al. (2014) <https://doi.org/10.1016/j.jqsrt.2014.01.030>`_: Author publication record and spectroscopic-system scope inspected; no full new CP thermochemical model inferred.

Importance and bounded decision
-------------------------------

The final ``curated-tio2-v1`` ion-inclusive archive contains 75 cases and
537 species. Maximum P allocation is 0.13456737 percent at
3000 K, 0.01 bar, C/O=1.5.
The maximum mole fraction is 5.1484594e-10; its own condition is
recorded separately in the packet. All archived cases converged and the target
function reproduces the named current factory exactly. These are saved
allocations, not replacement solves. A small value in this hot hydrogen-rich
grid does not establish irrelevance in cold, oxidizing or photochemical regimes.

For a future thermal candidate, inspect the CP-specific Barklem-Collet constants and a complete electronic-state inventory before a normalized Ram/MoLLIST diagnostic; require a separately justified D0 if changing energy. No ancestral-paper acquisition is a gate for this bounded retain review.

Curated is unchanged. MCEG supports bounded source construction, numerical
diagnostics and this scoped decision; candidate adoption and complete physical
accuracy remain unknown. Missing ancestral full texts do not indefinitely gate
merging this source review.
