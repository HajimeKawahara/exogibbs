P4 source and thermochemical review
===================================

Retain gaseous P4; record the W5 atomization result as a low-priority energy control rather than an already validated replacement.

Identity and source scope
-------------------------

The species is tetrahedral gaseous P4 (Td, singlet ground state, rotational symmetry 12), not four independent P atoms or condensed white/red phosphorus. A balanced gas atomization cycle removes elemental phase-reference ambiguity; a one-sided phosphorus phase correction would be invalid.

The retained source is the JANAF-labelled FastChem fit. Fresh official
`JANAF P-013 <https://janaf.nist.gov/tables/P-013.html>`_ and neutral-atom
tables are pinned, together with selected records from the versioned
`NASA CEA database <https://github.com/nasa/cea/blob/2f79a647e85737742fb08a127da00d1ea1d069da/data/thermo.inp>`_.
The NASA header reads ``Gurvich,1989 pt1 p402 pt2 p257.``.
All coefficients match the screening snapshot; original interval bounds are kept.

Balanced comparison
-------------------

The reaction is ``4 P(g) -> P4(g)`` at 1 bar. Native NASA calorics use
R=8.314510; equilibrium normalization uses R=8.31446261815324 J/mol/K.
H0 is derived from printed Hf298 and H298-H0, never polynomial extrapolation
to zero. The balanced source atomization energies are 1196.418936 kJ/mol (NASA) and 1195.998000 kJ/mol (JANAF).
These are physical source anchors, not the literal fit coefficient of 1/T.

.. list-table:: Dimensionless formation-constant ratios
   :header-rows: 1

   * - T (K)
     - Current / NASA
     - Current / JANAF
     - H0-aligned JANAF / NASA
   * - 1500
     - 0.992538
     - 1.01088
     - 1.01556
   * - 2000
     - 0.993325
     - 1.00176
     - 1.017
   * - 3000
     - 0.991686
     - 0.990876
     - 1.01785
   * - 4000
     - 1.00609
     - 1.00069
     - 1.0182

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
     - 28.394
     - 0.151582
     - 0.002953
   * - 2000
     - 29.358
     - 0.152841
     - 0.001641
   * - 3000
     - 29.987
     - 0.152775
     - 0.000118
   * - 4000
     - 30.153
     - 0.152122
     - -0.000127

Interpretation and modern evidence
----------------------------------

NASA and JANAF thermal functions agree within about 1.8 percent after H0 alignment. The inspected W5 Table 4 lists P4 atomization energy at 0 K as 286.61 kcal/mol (preliminary 1), 286.51 (preliminary 2), and W4-17 285.96; the P4 ATcT column is empty. W5 preliminary 2 corresponds to 1198.75784 kJ/mol, 2.75984 kJ/mol above the balanced JANAF source. This is a possible energy refinement with K factors about 1.248 at 1500 K and 1.087 at 4000 K. The 0.10 kcal/mol difference between W5 variants is a method spread, not a statistical uncertainty.

The full W5 article and its Table 4 were inspected; the compact packet pins only the relevant numerical entries and source identification. The earlier W4 study concluded that experimental P2/P4 thermochemistry was fundamentally sound. These theory results are useful controls, but they do not justify silently relabeling P4 as ATcT or replacing the phosphorus atom reference throughout the network. Current source agreement and very small allocation support scoped retention while keeping the newer energy result visible.

`Barman et al., W5 (2026) <https://doi.org/10.1021/acs.jpca.6c00467>`_: Full primary article and Table 4 inspected; P4 numerical atomization entries transcribed, ATcT column empty.

`Karton and Martin, W4 P2/P4 (2007) <https://doi.org/10.1080/00268970701543572>`_: Primary abstract and thermochemical conclusion inspected; complete original calculation not rerun.

Importance and bounded decision
-------------------------------

The final ``curated-tio2-v1`` ion-inclusive archive contains 75 cases and
537 species. Maximum P allocation is 1.8902421e-06 percent at
1500 K, 1 bar, C/O=0.58884366.
The maximum mole fraction is 2.0849863e-15; its own condition is
recorded separately in the packet. All archived cases converged and the target
function reproduces the named current factory exactly. These are saved
allocations, not replacement solves. A small value in this hot hydrogen-rich
grid does not establish irrelevance in cold, oxidizing or photochemical regimes.

If P-rich regimes or percent-level energy refinements become important, test the +2.75984 kJ/mol W5 preliminary-2 retained-thermal D0 control. Preserve the fit residual, distinguish theoretical spread from uncertainty, and assess fixed-network impact before adoption.

Curated is unchanged. MCEG supports bounded source construction, numerical
diagnostics and this scoped decision; candidate adoption and complete physical
accuracy remain unknown. Missing ancestral full texts do not indefinitely gate
merging this source review.
