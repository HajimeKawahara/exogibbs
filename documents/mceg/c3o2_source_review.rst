C3O2 source and thermochemical review
=====================================

Retain carbon suboxide while recording a thermal source gap and a modest evaluated energy refinement; a soft-bending model needs explicit validation.

Identity and source scope
-------------------------

Carbon suboxide is O=C=C=C=O, a singlet with nominal linear centrosymmetric structure and symmetry 2. Its unusually soft central bending coordinate makes a naive rigid linear harmonic model a substantive approximation; it is not another C3O2 isomer.

The retained source is the JANAF-labelled FastChem fit. Fresh official
`JANAF C-142 <https://janaf.nist.gov/tables/C-142.html>`_ and neutral-atom
tables are pinned, together with selected records from the versioned
`NASA CEA database <https://github.com/nasa/cea/blob/2f79a647e85737742fb08a127da00d1ea1d069da/data/thermo.inp>`_.
The NASA header reads ``Hf:Chase,1998 p690 6/68. Shimanouchi,1977 p1083.``.
All coefficients match the screening snapshot; original interval bounds are kept.

Balanced comparison
-------------------

The reaction is ``3 C(g) + 2 O(g) -> C3O2(g)`` at 1 bar. Native NASA calorics use
R=8.314510; equilibrium normalization uses R=8.31446261815324 J/mol/K.
H0 is derived from printed Hf298 and H298-H0, never polynomial extrapolation
to zero. The balanced source atomization energies are 2724.054498 kJ/mol (NASA) and 2723.954000 kJ/mol (JANAF).
These are physical source anchors, not the literal fit coefficient of 1/T.

.. list-table:: Dimensionless formation-constant ratios
   :header-rows: 1

   * - T (K)
     - Current / NASA
     - Current / JANAF
     - H0-aligned JANAF / NASA
   * - 1500
     - 0.913345
     - 1.02694
     - 0.896577
   * - 2000
     - 0.885229
     - 0.998748
     - 0.891711
   * - 3000
     - 0.869682
     - 0.984992
     - 0.886498
   * - 4000
     - 0.888291
     - 1.00839
     - 0.883569

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
     - -204.208
     - -1.096961
     - -0.053255
   * - 2000
     - -225.424
     - -1.109474
     - -0.035378
   * - 3000
     - -250.898
     - -1.119808
     - -0.015115
   * - 4000
     - -261.650
     - -1.122570
     - -0.010672

Interpretation and modern evidence
----------------------------------

The NASA and JANAF physical D0 anchors differ by only 0.100498 kJ/mol, but their H0-aligned thermal K ratio is 0.89658 at 1500 K and 0.88357 at 4000 K. ATcT 1.228 yields D0=2722.256 kJ/mol, 1.698 kJ/mol below JANAF, giving separate energy-only K factors 0.87271 and 0.95023. Thermal and energy changes are therefore both visible and must not be collapsed into a single fitted 1/T coefficient.

The inspected primary bending-potential study explicitly treats the low-frequency nu7 mode and its vibrational dependence. A standard harmonic replacement would need to establish how that soft coordinate and its state counting are treated. NASA cites Chase and Shimanouchi; the caloric difference is not proof that one represents an independent modern complete spectrum. The ATcT evaluated energy cycle does not resolve bending anharmonicity. The older formation/polymerization experiment is documented as ancestry, not reanalyzed from inaccessible details.

The exact molecule and atomic rows from `ATcT 1.228
<https://atct.anl.gov/Thermochemical%20Data/version%201.228/index.php>`_ are
archived with identifiers and both 0/298 K values. No isotope ZPE correction is
added to an already ground-referenced evaluated energy. Marginal uncertainties
are retained; reaction covariance is not invented by independent quadrature.

`Carbon suboxide nu7 bending potential (1980) <https://doi.org/10.1016/0022-2852(80)90297-0>`_: Primary author-repository abstract and vibrational-potential scope inspected; full numerical potential not reconstructed.

`Heats of formation and polymerization of carbon suboxide (1965) <https://doi.org/10.1021/j100894a060>`_: Primary publisher metadata inspected; no independent calorimetric reanalysis.

Importance and bounded decision
-------------------------------

The final ``curated-tio2-v1`` ion-inclusive archive contains 75 cases and
537 species. Maximum C allocation is 6.4945138e-09 percent at
4000 K, 100 bar, C/O=1.5.
The maximum mole fraction is 2.5429714e-14; its own condition is
recorded separately in the packet. All archived cases converged and the target
function reproduces the named current factory exactly. These are saved
allocations, not replacement solves. A small value in this hot hydrogen-rich
grid does not establish irrelevance in cold, oxidizing or photochemical regimes.

Audit the nu7 bending potential and partition convention, then compare a common-D0 thermal control and the -1.698 kJ/mol ATcT retained-thermal control separately. Production substitution is lower priority for this hot hydrogen-rich grid but may matter in other compositions.

Curated is unchanged. MCEG supports bounded source construction, numerical
diagnostics and this scoped decision; candidate adoption and complete physical
accuracy remain unknown. Missing ancestral full texts do not indefinitely gate
merging this source review.
