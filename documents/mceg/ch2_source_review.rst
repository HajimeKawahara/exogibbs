CH2 source and thermochemical review
====================================

Prioritize an equilibrated-methylene thermal plus ATcT energy candidate; triplet-only methylene is a distinct control, not an interchangeable equilibrium species.

Identity and source scope
-------------------------

The C1H2 runtime entry is Methylene with a triplet X3B1 ground state and thermally accessible singlets. Original JANAF page 605 includes excited electronic terms, so the equilibrium species is not restricted to the triplet surface. The exact NASA record CH2 and ATcT regular CH2 (ID 2465-56-7*0) are selected explicitly. A separate ATcT singlet value must not replace the ground energy merely because the formula is the same.

The retained source is the JANAF-labelled FastChem fit. Fresh official
`JANAF C-057 <https://janaf.nist.gov/tables/C-057.html>`_ and neutral-atom
tables are pinned, together with selected records from the versioned
`NASA CEA database <https://github.com/nasa/cea/blob/2f79a647e85737742fb08a127da00d1ea1d069da/data/thermo.inp>`_.
The NASA header reads ``D0(H2C-H): Ruscic,1999. Bunker,1983. Jacox,1998.``.
Selected records are checked against the literal official NASA coefficients and
original interval bounds. Existing screening records are cross-checked where
available; this identity appendix does not require a prior formula-only screen.

Balanced comparison
-------------------

The reaction is ``1 C(g) + 2 H(g) -> CH2(g)`` at 1 bar. Native NASA calorics use
R=8.314510; equilibrium normalization uses R=8.31446261815324 J/mol/K.
H0 is derived from printed Hf298 and H298-H0, never polynomial extrapolation
to zero. The balanced source atomization energies are 753.409805 kJ/mol (NASA) and 757.340000 kJ/mol (JANAF).
These are physical source anchors, not the literal fit coefficient of 1/T.

.. list-table:: Dimensionless formation-constant ratios
   :header-rows: 1

   * - T (K)
     - Current / NASA
     - Current / JANAF
     - H0-aligned JANAF / NASA
   * - 1500
     - 1.28404
     - 1.01031
     - 0.927396
   * - 2000
     - 1.17708
     - 1.00063
     - 0.928732
   * - 3000
     - 1.08117
     - 0.992667
     - 0.930386
   * - 4000
     - 1.05231
     - 1.00187
     - 0.933276

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
     - -3898.483
     - -0.604183
     - 0.173518
   * - 2000
     - -3863.636
     - -0.583290
     - 0.001150
   * - 3000
     - -3793.578
     - -0.558534
     - 0.233666
   * - 4000
     - -3330.337
     - -0.428461
     - 0.682778

Interpretation and modern evidence
----------------------------------

ATcT 1.228 gives regular-methylene Hf0=391.046 and Hf298=391.592 kJ/mol, marginal uncertainty 0.089 kJ/mol. The same-version C/H cycle gives D0=752.392 kJ/mol: -4.948 kJ/mol relative to JANAF, or -1.017805 kJ/mol relative to NASA. Retaining the current thermal fit would multiply K by 0.6725 at 1500 K and 0.8618 at 4000 K. The original JANAF singlet gap was estimated as 2600 cm-1, while the inspected IUPAC evaluation uses singlet origins 3147 +/- 5 and 11497 +/- 10 cm-1 with distinct molecular constants. Energy-only replacement would leave this thermal issue untouched.

The full IUPAC 2005 CH2 section provides separate equilibrated and triplet-only RRHO caloric tables. Three exact nodes from each are pinned. At a common H0, equilibrated IUPAC/NASA K is 1.1060, 1.1539 and 1.2192 at 1500, 2000 and 3000 K, respectively; triplet-only ratios instead are 0.9726, 0.9460 and 0.8956. At 3000 K the equilibrated thermal enthalpy exceeds the triplet-only one by 6.779 kJ/mol. Therefore the archived NASA function is numerically identical to neither evaluated table. The source tables stop at 3000 K, and their RRHO construction does not itself validate anharmonic or dissociative behavior at 4000 K. ATcT 2015 also explicitly distinguishes regular, triplet-only and singlet-only thermodynamic species.

The exact molecule and atomic rows from `ATcT 1.228
<https://atct.anl.gov/Thermochemical%20Data/version%201.228/index.php>`_ are
archived with identifiers and both 0/298 K values. No isotope ZPE correction is
added to an already ground-referenced evaluated energy. Marginal uncertainties
are retained; reaction covariance is not invented by independent quadrature.

`Ruscic et al., IUPAC radical evaluation (2005) <https://doi.org/10.1063/1.1724828>`_: Full CH2 section, pages 593 and 597–599, inspected; both caloric tables and electronic conventions transcribed at three nodes.

`Ruscic, ATcT sequential bond dissociation (2015) <https://doi.org/10.1021/acs.jpca.5b01346>`_: Primary discussion of regular versus triplet-only/singlet-only methylene inspected.

Importance and bounded decision
-------------------------------

The final ``curated-tio2-v1`` ion-inclusive archive contains 75 cases and
537 species. Maximum C allocation is 2.6037917 percent at
4000 K, 100 bar, C/O=1.5.
The maximum mole fraction is 3.0585976e-05; its own condition is
recorded separately in the packet. All archived cases converged and the target
function reproduces the named current factory exactly. These are saved
allocations, not replacement solves. A small value in this hot hydrogen-rich
grid does not establish irrelevance in cold, oxidizing or photochemical regimes.

Construct the evaluated equilibrated IUPAC thermal model with the ATcT ground-energy cycle and preserve the current fit residual; audit its extension to 4000 K explicitly. Compare retained-thermal and triplet-only controls, then perform fixed-network substitution. Carbon allocation above 2.6 percent makes this a high-priority follow-up.

Curated is unchanged. MCEG supports bounded source construction, numerical
diagnostics and this scoped decision; candidate adoption and complete physical
accuracy remain unknown. The frozen main-table grades are unchanged; this resolves the separate identity
appendix entry. Missing ancestral full texts do not indefinitely gate
merging this source review.
