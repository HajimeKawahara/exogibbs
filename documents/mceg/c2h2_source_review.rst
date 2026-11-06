C2H2 source and thermochemical review
=====================================

Prioritize an evaluated energy control for acetylene and keep thermal alternatives separate; its large carbon allocation warrants substitution testing before adoption.

Identity and source scope
-------------------------

The appendix target C2H2 is linear acetylene HCCH (ethyne), not vinylidene. The exact NASA record is C2H2,acetylene, and the original JANAF page 675 specifies a singlet D-infinity-h molecule with symmetry 2. The runtime record is explicitly Ethyne. No formula-only match to the adjacent NASA vinylidene record is used.

The retained source is the JANAF-labelled FastChem fit. Fresh official
`JANAF C-127 <https://janaf.nist.gov/tables/C-127.html>`_ and neutral-atom
tables are pinned, together with selected records from the versioned
`NASA CEA database <https://github.com/nasa/cea/blob/2f79a647e85737742fb08a127da00d1ea1d069da/data/thermo.inp>`_.
The NASA header reads ``Hf:TRC(10/93) w-3040. Gurvich,1991 pt1 p47 pt2 p39.``.
Selected records are checked against the literal official NASA coefficients and
original interval bounds. Existing screening records are cross-checked where
available; this identity appendix does not require a prior formula-only screen.

Balanced comparison
-------------------

The reaction is ``2 C(g) + 2 H(g) -> C2H2(g)`` at 1 bar. Native NASA calorics use
R=8.314510; equilibrium normalization uses R=8.31446261815324 J/mol/K.
H0 is derived from printed Hf298 and H298-H0, never polynomial extrapolation
to zero. The balanced source atomization energies are 1625.696849 kJ/mol (NASA) and 1627.153000 kJ/mol (JANAF).
These are physical source anchors, not the literal fit coefficient of 1/T.

.. list-table:: Dimensionless formation-constant ratios
   :header-rows: 1

   * - T (K)
     - Current / NASA
     - Current / JANAF
     - H0-aligned JANAF / NASA
   * - 1500
     - 1.17061
     - 1.01691
     - 1.02429
   * - 2000
     - 1.13028
     - 1.00067
     - 1.03482
   * - 3000
     - 1.10681
     - 0.989458
     - 1.05517
   * - 4000
     - 1.1273
     - 1.00416
     - 1.07453

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
     - -1098.678
     - 0.444223
     - 0.487835
   * - 2000
     - -822.751
     - 0.602698
     - 0.629164
   * - 3000
     - -59.051
     - 0.908988
     - 0.889067
   * - 4000
     - 882.167
     - 1.178389
     - 0.944880

Interpretation and modern evidence
----------------------------------

ATcT 1.228 gives Hf0=228.79 and Hf298=228.23 kJ/mol with a marginal uncertainty of 0.12 kJ/mol. Its same-version atomic cycle yields D0=1626.018 kJ/mol, 1.135 kJ/mol below JANAF and 0.321151 kJ/mol above NASA. The retained-thermal energy control would multiply K by 0.9130 at 1500 K and 0.9664 at 4000 K. These factors are not abundance ratios. The common-H0 JANAF/NASA thermal ratio grows from 1.0243 to 1.0745; the old JANAF source already includes explicit anharmonic and rotation-vibration corrections, so it must not be described as a bare harmonic model.

The primary aCeTY paper and official definition describe 5,160,803 states, J up to 99, a 22000 cm-1 upper-state limit and a 12000 cm-1 lower-state transition cutoff. Its quoted 2200 K line-list completeness uses a greater-than-90-percent lower-state-sum criterion, not proof of a complete 4000 K thermodynamic partition function. The distributed PF is pinned at four nodes and divided by the total nuclear-spin factor 4 once; the allowed ortho/para weights 3:1 already encode symmetry. Relative to the NASA internal Q, the normalized ratios are 0.9861, 0.9535, 0.7301 and 0.4409. That high-temperature deficit rules out promoting this finite PF directly to the full production range. The isotopic translation entropy correction is only -0.01036 J/mol/K and is recorded separately; the full state file was not downloaded or independently summed.

The exact molecule and atomic rows from `ATcT 1.228
<https://atct.anl.gov/Thermochemical%20Data/version%201.228/index.php>`_ are
archived with identifiers and both 0/298 K values. No isotope ZPE correction is
added to an already ground-referenced evaluated energy. Marginal uncertainties
are retained; reaction covariance is not invented by independent quadrature.

`Chubb, Tennyson and Yurchenko, aCeTY (2020) <https://doi.org/10.1093/mnras/staa229>`_: Primary full-text completeness and nuclear-spin discussion, official definition and distributed PF inspected; direct five-million-level state replay not performed.

Importance and bounded decision
-------------------------------

The final ``curated-tio2-v1`` ion-inclusive archive contains 75 cases and
537 species. Maximum C allocation is 30.122595 percent at
1500 K, 0.0001 bar, C/O=1.5.
The maximum mole fraction is 0.00018971381; its own condition is
recorded separately in the packet. All archived cases converged and the target
function reproduces the named current factory exactly. These are saved
allocations, not replacement solves. A small value in this hot hydrogen-rich
grid does not establish irrelevance in cold, oxidizing or photochemical regimes.

Build an ATcT -1.135 kJ/mol retained-thermal D0 control preserving the fit residual, alongside a separate NASA-thermal/ATcT control. Test fixed-network substitution because the current grid allocates about 30 percent of carbon to acetylene. Do not adopt the incomplete aCeTY PF across 1500–4000 K.

Curated is unchanged. MCEG supports bounded source construction, numerical
diagnostics and this scoped decision; candidate adoption and complete physical
accuracy remain unknown. The frozen main-table grades are unchanged; this resolves the separate identity
appendix entry. Missing ancestral full texts do not indefinitely gate
merging this source review.
