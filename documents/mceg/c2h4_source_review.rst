C2H4 source and thermochemical review
=====================================

Retain the current ethylene function; a small ATcT energy refinement is optional follow-up, while MaYTY does not cover the production temperature range completely.

Identity and source scope
-------------------------

Ethylene CH2=CH2 is the planar D2h singlet ground species, not another C2H4 isomer. The rotational symmetry number is 4. The tabulated thermodynamic functions already contain symmetry and electronic conventions; no new spin or symmetry factors are appended.

The retained source is the JANAF-labelled FastChem fit. Fresh official
`JANAF C-128 <https://janaf.nist.gov/tables/C-128.html>`_ and neutral-atom
tables are pinned, together with selected records from the versioned
`NASA CEA database <https://github.com/nasa/cea/blob/2f79a647e85737742fb08a127da00d1ea1d069da/data/thermo.inp>`_.
The NASA header reads ``TRC(4/88) w2600. Chao,1975. Knippers,1985.``.
All coefficients match the screening snapshot; original interval bounds are kept.

Balanced comparison
-------------------

The reaction is ``2 C(g) + 4 H(g) -> C2H4(g)`` at 1 bar. Native NASA calorics use
R=8.314510; equilibrium normalization uses R=8.31446261815324 J/mol/K.
H0 is derived from printed Hf298 and H298-H0, never polynomial extrapolation
to zero. The balanced source atomization energies are 2225.512499 kJ/mol (NASA) and 2225.527000 kJ/mol (JANAF).
These are physical source anchors, not the literal fit coefficient of 1/T.

.. list-table:: Dimensionless formation-constant ratios
   :header-rows: 1

   * - T (K)
     - Current / NASA
     - Current / JANAF
     - H0-aligned JANAF / NASA
   * - 1500
     - 1.03919
     - 1.03638
     - 1.00154
   * - 2000
     - 0.996559
     - 0.994494
     - 1.0012
   * - 3000
     - 0.973243
     - 0.972385
     - 1.0003
   * - 4000
     - 1.01053
     - 1.01065
     - 0.99944

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
     - -58.654
     - -0.013854
     - -0.041738
   * - 2000
     - -75.368
     - -0.024105
     - -0.029862
   * - 3000
     - -101.652
     - -0.034817
     - -0.020446
   * - 4000
     - -116.731
     - -0.038756
     - -0.012195

Interpretation and modern evidence
----------------------------------

NASA and JANAF agree closely after balancing their atoms: the H0-aligned K ratio stays within 0.16 percent of unity. The literal inherited fit has larger, sign-changing percent-level residuals, which must not be misidentified as a physical D0 change. ATcT 1.228 gives a +0.470 kJ/mol source D0 refinement, corresponding to energy-only K factors 1.0384 at 1500 K and 1.0142 at 4000 K. This small refinement is worth keeping explicit, but it does not establish a compelling thermal replacement or a large abundance error.

The inspected MaYTY publication describes a ground-electronic-state line list designed for completeness near 700 K. Its approximate vibrational cross-sections extend to 1500 K; that is not evidence of a complete rovibrational partition sum at 1500–4000 K. No huge line list or high-temperature extrapolation is adopted. NASA's Chao/Zwolinski ancestry is a separate evaluated thermochemical source; the numerical NASA records, rather than reconstructed unpublished corrections, are the replay inputs.

The exact molecule and atomic rows from `ATcT 1.228
<https://atct.anl.gov/Thermochemical%20Data/version%201.228/index.php>`_ are
archived with identifiers and both 0/298 K values. No isotope ZPE correction is
added to an already ground-referenced evaluated energy. Marginal uncertainties
are retained; reaction covariance is not invented by independent quadrature.

`Chao and Zwolinski (1975) <https://doi.org/10.1063/1.555518>`_: Primary thermodynamic paper metadata and stated scope inspected; full ancestry is not independently reconstructed.

`Mant et al., MaYTY (2018) <https://doi.org/10.1093/mnras/sty1239>`_: Primary paper abstract and completeness/conclusion text inspected; full line list was not downloaded.

Importance and bounded decision
-------------------------------

The final ``curated-tio2-v1`` ion-inclusive archive contains 75 cases and
537 species. Maximum C allocation is 0.17130504 percent at
2500 K, 100 bar, C/O=1.5.
The maximum mole fraction is 1.0790406e-06; its own condition is
recorded separately in the packet. All archived cases converged and the target
function reproduces the named current factory exactly. These are saved
allocations, not replacement solves. A small value in this hot hydrogen-rich
grid does not establish irrelevance in cold, oxidizing or photochemical regimes.

If percent-level energy refinements become a priority, build an ATcT retained-thermal +0.470 kJ/mol D0 control preserving the fit residual and test substitution. Do not replace high-temperature thermal functions with an incomplete MaYTY sum.

Curated is unchanged. MCEG supports bounded source construction, numerical
diagnostics and this scoped decision; candidate adoption and complete physical
accuracy remain unknown. Missing ancestral full texts do not indefinitely gate
merging this source review.
