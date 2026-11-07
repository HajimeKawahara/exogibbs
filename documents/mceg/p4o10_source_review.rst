P4O10 source and thermochemical review
======================================

Retain gaseous P4O10 with an explicit gas-phase source gap; native source agreement contains cancellation between energy and thermal differences.

Identity and source scope
-------------------------

The target is the molecular O10P4 gas cage, tetraphosphorus decaoxide, with tetrahedral molecular symmetry (rotational symmetry 12) and singlet ground state. It is not condensed rhombohedral P4O10 and not the half-formula P2O5. Gas and solid measurements require a phase-change cycle before comparison.

The retained source is the JANAF-labelled FastChem fit. Fresh official
`JANAF O-095 <https://janaf.nist.gov/tables/O-095.html>`_ and neutral-atom
tables are pinned, together with selected records from the versioned
`NASA CEA database <https://github.com/nasa/cea/blob/2f79a647e85737742fb08a127da00d1ea1d069da/data/thermo.inp>`_.
The NASA header reads ``Gurvich,1989 pt1 p419 pt2 p271.``.
All coefficients match the screening snapshot; original interval bounds are kept.

Balanced comparison
-------------------

The reaction is ``4 P(g) + 10 O(g) -> P4O10(g)`` at 1 bar. Native NASA calorics use
R=8.314510; equilibrium normalization uses R=8.31446261815324 J/mol/K.
H0 is derived from printed Hf298 and H298-H0, never polynomial extrapolation
to zero. The balanced source atomization energies are 6603.026288 kJ/mol (NASA) and 6600.284000 kJ/mol (JANAF).
These are physical source anchors, not the literal fit coefficient of 1/T.

.. list-table:: Dimensionless formation-constant ratios
   :header-rows: 1

   * - T (K)
     - Current / NASA
     - Current / JANAF
     - H0-aligned JANAF / NASA
   * - 1500
     - 0.995785
     - 1.07251
     - 1.1568
   * - 2000
     - 0.948485
     - 0.978945
     - 1.14259
   * - 3000
     - 0.956124
     - 0.947559
     - 1.1263
   * - 4000
     - 1.05499
     - 1.02485
     - 1.11789

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
     - 1605.566
     - 0.733270
     - -0.050024
   * - 2000
     - 1586.207
     - 0.722693
     - -0.032290
   * - 3000
     - 1564.137
     - 0.713283
     - -0.012511
   * - 4000
     - 1554.086
     - 0.709714
     - -0.010106

Interpretation and modern evidence
----------------------------------

NIST lists gaseous Hf298=-2904.09 kJ/mol and S298=403.96 J/mol/K from the JANAF evaluation last reviewed in 1965. NASA cites Gurvich (1989) and uses Hf298=-2906.223447 kJ/mol. The balanced source D0 differs by +2.742288 kJ/mol, while H0-aligned JANAF/NASA K is 1.15680 at 1500 K and 1.11789 at 4000 K. Native current/NASA values near unity therefore partly reflect cancellation; they do not prove identical energy or thermal construction.

The inspected combustion/formation paper concerns phosphorus and its oxide/acid thermochemistry; condensed formation measurements alone do not establish the gaseous cage energy without vaporization or sublimation information. Modern calculations of the rhombohedral solid likewise cannot replace the gas molecular thermal function. The current review pins the actual gas tables and their provenance, but does not claim a newly established complete gaseous spectrum or uncertainty budget.

`Phosphorus combustion and P4O10/H3PO4 formation (1962) <https://doi.org/10.1039/TF9625801916>`_: Primary RSC bibliographic record inspected; no condensed-to-gas cycle is inferred from metadata alone.

Importance and bounded decision
-------------------------------

The final ``curated-tio2-v1`` ion-inclusive archive contains 75 cases and
537 species. Maximum P allocation is 3.9454425e-41 percent at
1500 K, 100 bar, C/O=0.58884366.
The maximum mole fraction is 4.3562928e-50; its own condition is
recorded separately in the packet. All archived cases converged and the target
function reproduces the named current factory exactly. These are saved
allocations, not replacement solves. A small value in this hot hydrogen-rich
grid does not establish irrelevance in cold, oxidizing or photochemical regimes.

For oxidizing or cooler applications, reconstruct the gas/condensed phase-change energy cycle and inspect a gas-cage vibrational model before selecting a common-D0 thermal control. Keep molecular P4O10 distinct from empirical P2O5 and condensed phases. This finite source gap does not gate merging the bounded review.

Curated is unchanged. MCEG supports bounded source construction, numerical
diagnostics and this scoped decision; candidate adoption and complete physical
accuracy remain unknown. Missing ancestral full texts do not indefinitely gate
merging this source review.
