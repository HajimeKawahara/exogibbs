C4N2 source and thermochemical review
=====================================

Record a low-priority corrected-energy candidate for dicyanoacetylene; the 1963 combustion correction supports reassessing the older JANAF anchor.

Identity and source scope
-------------------------

Dicyanoacetylene is linear N-C-C-C-C-N (NC4N), the singlet centrosymmetric chain with rotational symmetry 2, not another C4N2 structural isomer. The NIST gas entry and original spectroscopic ancestry identify this connectivity.

The retained source is the JANAF-labelled FastChem fit. Fresh official
`JANAF C-145 <https://janaf.nist.gov/tables/C-145.html>`_ and neutral-atom
tables are pinned, together with selected records from the versioned
`NASA CEA database <https://github.com/nasa/cea/blob/2f79a647e85737742fb08a127da00d1ea1d069da/data/thermo.inp>`_.
The NASA header reads ``Hf:TRC(12/93) w8992. Khanna,1987. Brown,1989.``.
All coefficients match the screening snapshot; original interval bounds are kept.

Balanced comparison
-------------------

The reaction is ``4 C(g) + 2 N(g) -> C4N2(g)`` at 1 bar. Native NASA calorics use
R=8.314510; equilibrium normalization uses R=8.31446261815324 J/mol/K.
H0 is derived from printed Hf298 and H298-H0, never polynomial extrapolation
to zero. The balanced source atomization energies are 3262.140965 kJ/mol (NASA) and 3257.782000 kJ/mol (JANAF).
These are physical source anchors, not the literal fit coefficient of 1/T.

.. list-table:: Dimensionless formation-constant ratios
   :header-rows: 1

   * - T (K)
     - Current / NASA
     - Current / JANAF
     - H0-aligned JANAF / NASA
   * - 1500
     - 0.693668
     - 1.02897
     - 0.956179
   * - 2000
     - 0.739172
     - 1.00018
     - 0.96053
   * - 3000
     - 0.800793
     - 0.984827
     - 0.968401
   * - 4000
     - 0.86084
     - 1.00848
     - 0.973143

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
     - 4503.412
     - -0.265639
     - 0.309189
   * - 2000
     - 4631.979
     - -0.191115
     - 0.209065
   * - 3000
     - 4782.451
     - -0.128661
     - 0.109714
   * - 4000
     - 4866.970
     - -0.103725
     - 0.062863

Interpretation and modern evidence
----------------------------------

NIST explicitly lists the JANAF Hf298=533.46 kJ/mol as a 1961 evaluation, and 529.3 kJ/mol from Armstrong and Marantz (1963) as a correction to their 1960 combustion result. NASA adopts 529.2 kJ/mol from the later TRC evaluation. The corresponding NASA/JANAF balanced D0 change is +4.358965 kJ/mol, with an energy-only K factor about 1.418 at 1500 K and 1.140 at 4000 K. The raw current/NASA discrepancy therefore has a specific corrected-energy explanation. The remaining H0-aligned thermal ratio is 0.95618–0.97314, so NASA thermal and retained thermal should still be separate controls.

The correction DOI and publication metadata were verified on the primary ACS record; its numerical consequence is reported by NIST. The correction text itself has not been reanalyzed. The Brown et al. electron-diffraction/coherent-Raman study directly addresses molecular structure, while the older statistical-thermodynamics study supplies relevant thermal ancestry. NASA is not adopted merely because its database file is newer: its energy choice is supported by the documented correction, whereas thermal completeness remains a separate question.

`Armstrong and Marantz combustion correction (1963) <https://doi.org/10.1021/j100806a601>`_: Primary correction metadata inspected; corrected gas Hf298 is taken from the NIST evaluation, full correction text not reanalyzed.

`Armstrong and Marantz original combustion study (1960) <https://doi.org/10.1021/j100840a514>`_: Primary metadata inspected and explicitly distinguished from its 1963 correction.

`Brown et al., dicyanoacetylene structure (1989) <https://doi.org/10.1021/j100352a010>`_: Primary structure-study metadata inspected; not interpreted as a complete thermochemical evaluation.

`Statistical thermodynamics of cyanopolyynes (1969) <https://doi.org/10.1021/je60041a030>`_: Primary publisher metadata and species scope inspected; full thermal derivation not reconstructed.

Importance and bounded decision
-------------------------------

The final ``curated-tio2-v1`` ion-inclusive archive contains 75 cases and
537 species. Maximum N allocation is 5.4369371e-09 percent at
3000 K, 1 bar, C/O=1.5.
The maximum mole fraction is 2.940277e-15; its own condition is
recorded separately in the packet. All archived cases converged and the target
function reproduces the named current factory exactly. These are saved
allocations, not replacement solves. A small value in this hot hydrogen-rich
grid does not establish irrelevance in cold, oxidizing or photochemical regimes.

Build a corrected-energy retained-thermal control using an explicitly chosen balanced anchor (NASA +4.358965 kJ/mol relative to JANAF is the available numerical control), and compare a separate NASA-thermal control. Preserve the fit residual and run fixed-network substitution before adoption. The small hot-grid abundance places this after the more important candidates.

Curated is unchanged. MCEG supports bounded source construction, numerical
diagnostics and this scoped decision; candidate adoption and complete physical
accuracy remain unknown. Missing ancestral full texts do not indefinitely gate
merging this source review.
