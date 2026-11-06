C2O source and thermochemical review
====================================

Prioritize a same-state ATcT energy candidate for triplet CCO; independently test the thermal model before any curated adoption.

Identity and source scope
-------------------------

The C2O row is linear C-C-O in its triplet X3Sigma ground electronic state, not a cyclic isomer or singlet CCO. The original JANAF printed page 684 explicitly specifies C-infinity-v, ground degeneracy 3 and rotational symmetry 1. That structural and spin identification, rather than a CAS alias alone, establishes the match to ATcT triplet CCO.

The retained source is the JANAF-labelled FastChem fit. Fresh official
`JANAF C-136 <https://janaf.nist.gov/tables/C-136.html>`_ and neutral-atom
tables are pinned, together with selected records from the versioned
`NASA CEA database <https://github.com/nasa/cea/blob/2f79a647e85737742fb08a127da00d1ea1d069da/data/thermo.inp>`_.
The NASA header reads ``Hf:Gurvich,1991 pt1 p31. Jacox,1998.``.
All coefficients match the screening snapshot; original interval bounds are kept.

Balanced comparison
-------------------

The reaction is ``2 C(g) + 1 O(g) -> C2O(g)`` at 1 bar. Native NASA calorics use
R=8.314510; equilibrium normalization uses R=8.31446261815324 J/mol/K.
H0 is derived from printed Hf298 and H298-H0, never polynomial extrapolation
to zero. The balanced source atomization energies are 1382.184862 kJ/mol (NASA) and 1386.585000 kJ/mol (JANAF).
These are physical source anchors, not the literal fit coefficient of 1/T.

.. list-table:: Dimensionless formation-constant ratios
   :header-rows: 1

   * - T (K)
     - Current / NASA
     - Current / JANAF
     - H0-aligned JANAF / NASA
   * - 1500
     - 1.35415
     - 1.01292
     - 0.939441
   * - 2000
     - 1.23946
     - 1.00107
     - 0.950277
   * - 3000
     - 1.14874
     - 0.993086
     - 0.969667
   * - 4000
     - 1.11766
     - 1.00343
     - 0.975803

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
     - -4069.989
     - -0.289900
     - 0.901921
   * - 2000
     - -3619.405
     - -0.028866
     - 0.797969
   * - 3000
     - -3399.790
     - 0.077317
     - -0.450092
   * - 4000
     - -4449.192
     - -0.217148
     - -1.550342

Interpretation and modern evidence
----------------------------------

ATcT 1.228 gives Hf0=377.08 and Hf298=381.13 kJ/mol, with a listed marginal uncertainty of 0.83 kJ/mol. The same-version C/O cycle gives D0=1292.503 kJ/mol, 94.082 kJ/mol below JANAF and 89.681862 kJ/mol below NASA. This is a physical source-energy difference, not a fit-anchor correction. The JANAF page identifies its 1966 evaluation and a broad 63 kJ/mol uncertainty on Hf298=286.604 kJ/mol; the old value must not be presented as a precise modern determination. Native NASA/JANAF agreement cannot rule out a common obsolete energy scale.

The old JANAF thermal model includes estimated electronic terms at 4050 cm-1 (weight 2) and 6450 cm-1 (weight 1), above a triplet ground state. The NIST/Jacox compilation instead lists low singlet terms near 5276 and 8205 cm-1. Its identity evidence and numerical values are pinned separately; they are not silently converted into a new partition function. The inspected ATcT provenance combines the ketene dissociation network, triplet/singlet CCO photodissociation and theory, with uncertainty multipliers for some measurements. Choi et al. and Zengin et al. establish the state-resolved context. A modern energy cycle does not automatically validate old electronic thermal populations. The source-aligned thermal K discrepancy is a few percent at the four nodes, distinct from the very large evaluated energy shift.

The exact molecule and atomic rows from `ATcT 1.228
<https://atct.anl.gov/Thermochemical%20Data/version%201.228/index.php>`_ are
archived with identifiers and both 0/298 K values. No isotope ZPE correction is
added to an already ground-referenced evaluated energy. Marginal uncertainties
are retained; reaction covariance is not invented by independent quadrature.

`Choi et al. (1998), triplet and singlet CCO photodissociation <https://doi.org/10.1063/1.475839>`_: Publisher metadata, NIST state assignments and ATcT provenance inspected; full experimental reanalysis not performed.

`Zengin et al. (1996), low-lying CCO electronic states <https://doi.org/10.1063/1.473000>`_: Primary metadata and NIST electronic term assignments inspected; no complete high-temperature state sum inferred.

Importance and bounded decision
-------------------------------

The final ``curated-tio2-v1`` ion-inclusive archive contains 75 cases and
537 species. Maximum C allocation is 0.00085651078 percent at
4000 K, 100 bar, C/O=1.5.
The maximum mole fraction is 5.0305901e-09; its own condition is
recorded separately in the packet. All archived cases converged and the target
function reproduces the named current factory exactly. These are saved
allocations, not replacement solves. A small value in this hot hydrogen-rich
grid does not establish irrelevance in cold, oxidizing or photochemical regimes.

Construct the -94.082 kJ/mol retained-thermal D0 control preserving the literal fit residual, plus a separately normalized NASA-thermal/evaluated-energy control. Audit the obsolete low electronic terms explicitly, then run fixed-network substitution. Keep energy and thermal conclusions separate.

Curated is unchanged. MCEG supports bounded source construction, numerical
diagnostics and this scoped decision; candidate adoption and complete physical
accuracy remain unknown. Missing ancestral full texts do not indefinitely gate
merging this source review.
