HCO source and thermochemical review
====================================

Prioritize an ATcT energy candidate for formyl, with retained-thermal and NASA-thermal controls kept separate; leave curated unchanged until substitution is assessed.

Identity and source scope
-------------------------

Formyl H-C=O is the bent doublet HCO radical (C1H1O1), distinct from the HOC isomer and from formaldehyde CH2O. Its rotational symmetry number is 1. Molecular thermodynamic tables already include electronic degeneracy; no additional doublet factor or nuclear-spin entropy is added.

The retained source is the JANAF-labelled FastChem fit. Fresh official
`JANAF C-054 <https://janaf.nist.gov/tables/C-054.html>`_ and neutral-atom
tables are pinned, together with selected records from the versioned
`NASA CEA database <https://github.com/nasa/cea/blob/2f79a647e85737742fb08a127da00d1ea1d069da/data/thermo.inp>`_.
The NASA header reads ``Hf:Terentis,1996. Jacox,1998 p146.``.
All coefficients match the screening snapshot; original interval bounds are kept.

Balanced comparison
-------------------

The reaction is ``1 H(g) + 1 C(g) + 1 O(g) -> HCO(g)`` at 1 bar. Native NASA calorics use
R=8.314510; equilibrium normalization uses R=8.31446261815324 J/mol/K.
H0 is derived from printed Hf298 and H298-H0, never polynomial extrapolation
to zero. The balanced source atomization energies are 1131.986705 kJ/mol (NASA) and 1130.859000 kJ/mol (JANAF).
These are physical source anchors, not the literal fit coefficient of 1/T.

.. list-table:: Dimensionless formation-constant ratios
   :header-rows: 1

   * - T (K)
     - Current / NASA
     - Current / JANAF
     - H0-aligned JANAF / NASA
   * - 1500
     - 0.964805
     - 1.01344
     - 1.0421
   * - 2000
     - 0.973516
     - 0.999245
     - 1.04261
   * - 3000
     - 0.988362
     - 0.991866
     - 1.04255
   * - 4000
     - 1.01141
     - 1.00397
     - 1.04215

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
     - 1133.066
     - 0.349143
     - -0.014304
   * - 2000
     - 1123.873
     - 0.344130
     - -0.021885
   * - 3000
     - 1099.068
     - 0.333442
     - -0.024766
   * - 4000
     - 1070.779
     - 0.325123
     - -0.035181

Interpretation and modern evidence
----------------------------------

ATcT 1.228 gives Hf0(HCO)=41.384 and Hf298=41.758 kJ/mol, with the listed 0.078 kJ/mol marginal uncertainty. The same-version C/H/O balance yields D0=1132.863 kJ/mol, 2.004 kJ/mol above the JANAF source anchor. Keeping the inherited thermal fit would change K by 1.1743 at 1500 K and 1.0621 at 4000 K. NASA's anchor is closer, but the aligned JANAF/NASA thermal K difference remains about 4.2 percent. Thus an energy-only candidate and a NASA-thermal plus ATcT candidate answer different questions.

The inspected ATcT provenance and the primary Terentis/Kable threshold study connect HCO with the accurate H2CO -> HCO + H dissociation threshold. The NASA header already cites Terentis1996, so modern evaluation and historical NASA are not independent copies of entirely disjoint experiments. ATcT strengthens the balanced energy assessment; it does not supply a complete 4000 K molecular partition function. High-temperature bound-level and dissociation-state treatment remain distinct thermal uncertainties.

The exact molecule and atomic rows from `ATcT 1.228
<https://atct.anl.gov/Thermochemical%20Data/version%201.228/index.php>`_ are
archived with identifiers and both 0/298 K values. No isotope ZPE correction is
added to an already ground-referenced evaluated energy. Marginal uncertainties
are retained; reaction covariance is not invented by independent quadrature.

`Terentis and Kable (1996) <https://doi.org/10.1016/0009-2614(96)00705-1>`_: Publisher abstract and ATcT provenance connection inspected; full experimental reanalysis was not performed.

`Ruscic (2015) <https://doi.org/10.1021/acs.jpca.5b01346>`_: Primary ATcT thermochemistry article/provenance discussion inspected; used as context for correlated network evaluation.

Importance and bounded decision
-------------------------------

The final ``curated-tio2-v1`` ion-inclusive archive contains 75 cases and
537 species. Maximum C allocation is 0.047560128 percent at
4000 K, 100 bar, C/O=0.58884366.
The maximum mole fraction is 3.7063299e-07; its own condition is
recorded separately in the packet. All archived cases converged and the target
function reproduces the named current factory exactly. These are saved
allocations, not replacement solves. A small value in this hot hydrogen-rich
grid does not establish irrelevance in cold, oxidizing or photochemical regimes.

Build the +2.004 kJ/mol retained-thermal physical D0 control preserving the inherited fit residual; compare a separately normalized NASA-thermal/ATcT control and perform fixed-network substitution before adoption.

Curated is unchanged. MCEG supports bounded source construction, numerical
diagnostics and this scoped decision; candidate adoption and complete physical
accuracy remain unknown. Missing ancestral full texts do not indefinitely gate
merging this source review.
