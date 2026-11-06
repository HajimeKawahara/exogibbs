CNN source and thermochemical review
====================================

Record a combined energy/thermal follow-up candidate for CNN, explicitly distinct from NCN; NASA is not a justified wholesale replacement.

Identity and source scope
-------------------------

CNN is linear C-N-N (diazomethylene, CAS 2468-81-7), canonical C1N2_cnn, with triplet ground multiplicity and rotational symmetry 1. NCN is a different connectivity with symmetry 2 and has a separate runtime row. Original JANAF page 636 and the exact NASA CNN header establish the intended species independently of the shared CN2 formula.

The retained source is the JANAF-labelled FastChem fit. Fresh official
`JANAF C-088 <https://janaf.nist.gov/tables/C-088.html>`_ and neutral-atom
tables are pinned, together with selected records from the versioned
`NASA CEA database <https://github.com/nasa/cea/blob/2f79a647e85737742fb08a127da00d1ea1d069da/data/thermo.inp>`_.
The NASA header reads ``Gurvich,1991 pt1 p220 pt2 p208. Jacox,1994.``.
Selected records are checked against the literal official NASA coefficients and
original interval bounds. Existing screening records are cross-checked where
available; this identity appendix does not require a prior formula-only screen.

Balanced comparison
-------------------

The reaction is ``1 C(g) + 2 N(g) -> CNN(g)`` at 1 bar. Native NASA calorics use
R=8.314510; equilibrium normalization uses R=8.31446261815324 J/mol/K.
H0 is derived from printed Hf298 and H298-H0, never polynomial extrapolation
to zero. The balanced source atomization energies are 1020.002853 kJ/mol (NASA) and 1068.973000 kJ/mol (JANAF).
These are physical source anchors, not the literal fit coefficient of 1/T.

.. list-table:: Dimensionless formation-constant ratios
   :header-rows: 1

   * - T (K)
     - Current / NASA
     - Current / JANAF
     - H0-aligned JANAF / NASA
   * - 1500
     - 37.7973
     - 1.00839
     - 0.738881
   * - 2000
     - 12.9672
     - 1.00018
     - 0.682063
   * - 3000
     - 4.36622
     - 0.99547
     - 0.615812
   * - 4000
     - 2.53147
     - 1.00224
     - 0.579336

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
     - -52445.989
     - -4.837545
     - -2.521295
   * - 2000
     - -53486.787
     - -5.442153
     - -1.703791
   * - 3000
     - -54712.134
     - -5.950612
     - -0.878338
   * - 4000
     - -55426.487
     - -6.158208
     - -0.616800

Interpretation and modern evidence
----------------------------------

ATcT 1.228 gives Hf0=577.9 and Hf298=578.6 +/- 2.2 kJ/mol. Its balanced D0=1074.780 kJ/mol is +5.807 kJ/mol relative to JANAF, but +54.777147 kJ/mol relative to NASA. The old JANAF formation enthalpy is only an estimate with +/-126 kJ/mol uncertainty. The evaluated-energy retained-thermal control would raise K by 1.5930 at 1500 K and 1.1908 at 4000 K. The common-H0 JANAF/NASA thermal ratio also falls from 0.7389 to 0.5793, so their very large native disagreement is not an isomer mismatch and cannot be corrected by energy alignment alone.

Original JANAF assumes a 2847 cm-1 asymmetric stretching fundamental. The later NIST/Jacox compilation reports about 1425 +/- 120 cm-1 from gas-phase photoelectron spectroscopy, with a nearby 1419 cm-1 matrix result; those media and uncertainties remain distinct. The same spectroscopy identifies low singlet terms near 6830 and 10690 cm-1. The primary Clifford et al. and Bise et al. studies are documented as state/photodissociation evidence. Inspected current ATcT provenance is strongly theoretical: eight listed contributors account for 94.2 percent and include W1RO cycles and CNN-NCN energy differences. Thus it is not described as a new direct precise calorimetric measurement. A corrected thermal model needs full state weighting and a separately chosen energy scale.

The exact molecule and atomic rows from `ATcT 1.228
<https://atct.anl.gov/Thermochemical%20Data/version%201.228/index.php>`_ are
archived with identifiers and both 0/298 K values. No isotope ZPE correction is
added to an already ground-referenced evaluated energy. Marginal uncertainties
are retained; reaction covariance is not invented by independent quadrature.

`Clifford et al., diazocarbene and diazomethyl (1998) <https://doi.org/10.1021/jp9802735>`_: Primary identity/metadata and NIST gas photoelectron assignments inspected; full experimental inversion not reconstructed.

`Bise et al., CNN photodissociation (2000) <https://doi.org/10.1063/1.1288267>`_: Primary metadata and NIST electronic-spectrum context inspected; no full thermal model inferred.

Importance and bounded decision
-------------------------------

The final ``curated-tio2-v1`` ion-inclusive archive contains 75 cases and
537 species. Maximum N allocation is 9.9882268e-07 percent at
4000 K, 100 bar, C/O=1.5.
The maximum mole fraction is 5.3986259e-13; its own condition is
recorded separately in the packet. All archived cases converged and the target
function reproduces the named current factory exactly. These are saved
allocations, not replacement solves. A small value in this hot hydrogen-rich
grid does not establish irrelevance in cold, oxidizing or photochemical regimes.

Compare the ATcT +5.807 kJ/mol retained-thermal control with a source-resolved thermal correction, auditing the old 2847 versus modern approximately 1425 cm-1 mode and the low singlet manifolds. Keep CNN and NCN independent; assess fixed-network substitution before adoption.

Curated is unchanged. MCEG supports bounded source construction, numerical
diagnostics and this scoped decision; candidate adoption and complete physical
accuracy remain unknown. The frozen main-table grades are unchanged; this resolves the separate identity
appendix entry. Missing ancestral full texts do not indefinitely gate
merging this source review.
