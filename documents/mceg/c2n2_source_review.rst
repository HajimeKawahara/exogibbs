C2N2 source and thermochemical review
=====================================

Retain the cyanogen function for this source review; resolve the high-temperature thermal discrepancy before proposing a replacement.

Identity and source scope
-------------------------

Cyanogen is linear N-C-C-N (NCCN), the singlet D-infinity-h species with rotational symmetry 2, not the CNCN isomer. Nuclear-spin factors and rotational symmetry are already included or excluded consistently in the tabulated thermodynamic convention; no additional factor is appended.

The retained source is the JANAF-labelled FastChem fit. Fresh official
`JANAF C-134 <https://janaf.nist.gov/tables/C-134.html>`_ and neutral-atom
tables are pinned, together with selected records from the versioned
`NASA CEA database <https://github.com/nasa/cea/blob/2f79a647e85737742fb08a127da00d1ea1d069da/data/thermo.inp>`_.
The NASA header reads ``Gurvich,1979 pt1 p195 pt2 p220.``.
All coefficients match the screening snapshot; original interval bounds are kept.

Balanced comparison
-------------------

The reaction is ``2 C(g) + 2 N(g) -> C2N2(g)`` at 1 bar. Native NASA calorics use
R=8.314510; equilibrium normalization uses R=8.31446261815324 J/mol/K.
H0 is derived from printed Hf298 and H298-H0, never polynomial extrapolation
to zero. The balanced source atomization energies are 2056.868507 kJ/mol (NASA) and 2056.794000 kJ/mol (JANAF).
These are physical source anchors, not the literal fit coefficient of 1/T.

.. list-table:: Dimensionless formation-constant ratios
   :header-rows: 1

   * - T (K)
     - Current / NASA
     - Current / JANAF
     - H0-aligned JANAF / NASA
   * - 1500
     - 0.909473
     - 1.01674
     - 0.899862
   * - 2000
     - 0.876931
     - 1.00074
     - 0.880221
   * - 3000
     - 0.831001
     - 0.992048
     - 0.840169
   * - 4000
     - 0.800951
     - 1.00485
     - 0.798876

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
     - -731.047
     - -1.413595
     - -1.127245
   * - 2000
     - -1424.596
     - -1.809555
     - -1.647853
   * - 3000
     - -3592.749
     - -2.675125
     - -2.692907
   * - 4000
     - -6833.069
     - -3.599018
     - -3.802036

Interpretation and modern evidence
----------------------------------

NASA and JANAF balanced D0 differ by only 0.074507 kJ/mol. ATcT 1.228 gives D0=2056.500 kJ/mol, a -0.294 kJ/mol change from JANAF and energy-only K factors 0.97670 at 1500 K and 0.99120 at 4000 K. Yet H0-aligned JANAF/NASA K decreases from 0.89986 to 0.79888. The principal disagreement is therefore thermal, not an atomization-energy correction. Agreement of low-temperature energy evaluations cannot select the high-temperature caloric model.

The NASA source header cites Gurvich (1979); the inspected primary literature includes a 1956 high-temperature cyanogen thermodynamic treatment and the 1969 cyanogen/dicyanoacetylene statistical-thermodynamics paper. Their metadata and scope identify useful finite follow-up, while their complete state accounting has not been reconstructed here. A low-frequency bending and anharmonicity audit is needed before assuming the later table is better at 4000 K. ATcT provides the energy anchor but no complete thermal replacement.

The exact molecule and atomic rows from `ATcT 1.228
<https://atct.anl.gov/Thermochemical%20Data/version%201.228/index.php>`_ are
archived with identifiers and both 0/298 K values. No isotope ZPE correction is
added to an already ground-referenced evaluated energy. Marginal uncertainties
are retained; reaction covariance is not invented by independent quadrature.

`Cyanogen thermodynamic functions (1956) <https://doi.org/10.1063/1.1700841>`_: Primary publication metadata and 1000–4500 K equilibrium scope inspected; full numerical ancestry not reconstructed.

`Statistical thermodynamic functions of cyanogen and cyanopolyynes (1969) <https://doi.org/10.1021/je60041a030>`_: Primary publisher title and thermodynamic species scope inspected; full text not acquired.

Importance and bounded decision
-------------------------------

The final ``curated-tio2-v1`` ion-inclusive archive contains 75 cases and
537 species. Maximum N allocation is 0.00013304275 percent at
4000 K, 100 bar, C/O=1.5.
The maximum mole fraction is 7.1909466e-11; its own condition is
recorded separately in the packet. All archived cases converged and the target
function reproduces the named current factory exactly. These are saved
allocations, not replacement solves. A small value in this hot hydrogen-rich
grid does not establish irrelevance in cold, oxidizing or photochemical regimes.

Compare the JANAF and Gurvich bending-mode frequencies, degeneracies and anharmonic corrections, then evaluate a thermal-only control at a common physical D0. The small ATcT energy refinement can be included as a separate control. Missing ancestral full text is not a gate for this bounded source review.

Curated is unchanged. MCEG supports bounded source construction, numerical
diagnostics and this scoped decision; candidate adoption and complete physical
accuracy remain unknown. Missing ancestral full texts do not indefinitely gate
merging this source review.
