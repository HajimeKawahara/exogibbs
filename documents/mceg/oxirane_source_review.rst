oxirane source and thermochemical review
========================================

Retain oxirane after resolving its identity; the evaluated atomization energy already agrees closely with the current source.

Identity and source scope
-------------------------

The C2H4O1 runtime record is Oxirane, the cyclic ethylene oxide O(CH2CH2), not acetaldehyde CH3CHO or vinyl alcohol. Original JANAF page 677 specifies the gas molecule (C2v, singlet, rotational symmetry 2). The exact NASA record C2H4O,ethylen-o is selected; formula-only matching is deliberately excluded.

The retained source is the JANAF-labelled FastChem fit. Fresh official
`JANAF C-129 <https://janaf.nist.gov/tables/C-129.html>`_ and neutral-atom
tables are pinned, together with selected records from the versioned
`NASA CEA database <https://github.com/nasa/cea/blob/2f79a647e85737742fb08a127da00d1ea1d069da/data/thermo.inp>`_.
The NASA header reads ``Ethylene Oxide. Shimanouchi,1972. Chase,1998 9/65.``.
Selected records are checked against the literal official NASA coefficients and
original interval bounds. Existing screening records are cross-checked where
available; this identity appendix does not require a prior formula-only screen.

Balanced comparison
-------------------

The reaction is ``2 C(g) + 4 H(g) + 1 O(g) -> oxirane(g)`` at 1 bar. Native NASA calorics use
R=8.314510; equilibrium normalization uses R=8.31446261815324 J/mol/K.
H0 is derived from printed Hf298 and H298-H0, never polynomial extrapolation
to zero. The balanced source atomization energies are 2573.408745 kJ/mol (NASA) and 2573.422000 kJ/mol (JANAF).
These are physical source anchors, not the literal fit coefficient of 1/T.

.. list-table:: Dimensionless formation-constant ratios
   :header-rows: 1

   * - T (K)
     - Current / NASA
     - Current / JANAF
     - H0-aligned JANAF / NASA
   * - 1500
     - 1.07611
     - 1.04488
     - 1.02879
   * - 2000
     - 1.0209
     - 0.989639
     - 1.03077
   * - 3000
     - 0.995959
     - 0.964723
     - 1.03183
   * - 4000
     - 1.04731
     - 1.01443
     - 1.03199

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
     - 45.032
     - 0.277128
     - -0.041008
   * - 2000
     - 26.861
     - 0.266727
     - -0.032530
   * - 3000
     - -0.815
     - 0.254512
     - -0.021358
   * - 4000
     - -17.098
     - 0.250378
     - -0.012616

Interpretation and modern evidence
----------------------------------

ATcT 1.228 gives Hf0=-39.72 and Hf298=-52.28 +/- 0.33 kJ/mol. The same-version atomic cycle yields D0=2573.439 kJ/mol, only +0.017 kJ/mol relative to JANAF and +0.030255 kJ/mol relative to NASA. The evaluated-energy K change is at most 0.137 percent at the four source nodes. NASA and JANAF share important ancestral data and have a roughly 3 percent aligned thermal offset; the inherited fit adds its own percent-level residual. These are not grounds for adopting another isomer or claiming independent complete thermal validation.

Inspected ATcT provenance still gives a major role to the Pell 1965 combustion datum, combined with modern isomerization and atomization calculations. Modern epoxide literature also distinguishes oxirane reference thermochemistry from new measurements on substituted epoxides; a study of ethyloxirane or butene oxides is not a new measurement of this target. The full 2013 epoxide paper was inspected for that distinction. The original gas entry remains trace in the hot hydrogen-rich grid, but this does not constrain cold chemistry or kinetic pathways.

The exact molecule and atomic rows from `ATcT 1.228
<https://atct.anl.gov/Thermochemical%20Data/version%201.228/index.php>`_ are
archived with identifiers and both 0/298 K values. No isotope ZPE correction is
added to an already ground-referenced evaluated energy. Marginal uncertainties
are retained; reaction covariance is not invented by independent quadrature.

`Morgan et al., thermochemistry of epoxides (2013) <https://doi.org/10.1021/jo4002867>`_: Full primary author manuscript inspected; new substituted-epoxide experiments are not mislabeled as new oxirane measurements.

Importance and bounded decision
-------------------------------

The final ``curated-tio2-v1`` ion-inclusive archive contains 75 cases and
537 species. Maximum C allocation is 6.764799e-10 percent at
3000 K, 100 bar, C/O=1.5.
The maximum mole fraction is 4.2339921e-15; its own condition is
recorded separately in the packet. All archived cases converged and the target
function reproduces the named current factory exactly. These are saved
allocations, not replacement solves. A small value in this hot hydrogen-rich
grid does not establish irrelevance in cold, oxidizing or photochemical regimes.

No priority energy replacement is indicated. If oxirane becomes important in a colder or chemically distinct application, audit the gas vibrational model and inherited fit residual separately. Preserve the cyclic identity and do not import acetaldehyde/vinyl-alcohol parameters.

Curated is unchanged. MCEG supports bounded source construction, numerical
diagnostics and this scoped decision; candidate adoption and complete physical
accuracy remain unknown. The frozen main-table grades are unchanged; this resolves the separate identity
appendix entry. Missing ancestral full texts do not indefinitely gate
merging this source review.
