CNC source and thermochemical review
====================================

Prioritize an ATcT energy candidate for CNC after resolving its identity; audit its strongly coupled thermal spectrum separately before adoption.

Identity and source scope
-------------------------

The runtime C2N1 entry is explicitly CNC_Radical: linear C-N-C, isocyanomethylidyne, CAS 53590-27-5. It is not the CCN isomer. Original JANAF page 681 specifies D-infinity-h and rotational symmetry 2; modern spectroscopy assigns X2Pi-g. NASA has distinct CNC and CCN records. The target uses CNC explicitly, and a pinned CCN record serves only as a negative identity control.

The retained source is the JANAF-labelled FastChem fit. Fresh official
`JANAF C-133 <https://janaf.nist.gov/tables/C-133.html>`_ and neutral-atom
tables are pinned, together with selected records from the versioned
`NASA CEA database <https://github.com/nasa/cea/blob/2f79a647e85737742fb08a127da00d1ea1d069da/data/thermo.inp>`_.
The NASA header reads ``CNC radical. Gurvich,1991 pt1 p221 pt2 p209.``.
Selected records are checked against the literal official NASA coefficients and
original interval bounds. Existing screening records are cross-checked where
available; this identity appendix does not require a prior formula-only screen.

Balanced comparison
-------------------

The reaction is ``2 C(g) + 1 N(g) -> CNC(g)`` at 1 bar. Native NASA calorics use
R=8.314510; equilibrium normalization uses R=8.31446261815324 J/mol/K.
H0 is derived from printed Hf298 and H298-H0, never polynomial extrapolation
to zero. The balanced source atomization energies are 1213.212834 kJ/mol (NASA) and 1341.362000 kJ/mol (JANAF).
These are physical source anchors, not the literal fit coefficient of 1/T.

.. list-table:: Dimensionless formation-constant ratios
   :header-rows: 1

   * - T (K)
     - Current / NASA
     - Current / JANAF
     - H0-aligned JANAF / NASA
   * - 1500
     - 24274.1
     - 1.01035
     - 0.828336
   * - 2000
     - 1885.69
     - 0.999515
     - 0.848855
   * - 3000
     - 148.093
     - 0.995044
     - 0.8739
   * - 4000
     - 42.0188
     - 1.0036
     - 0.888097

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
     - -127046.789
     - -0.825682
     - 0.547145
   * - 2000
     - -126835.347
     - -0.701951
     - 0.325083
   * - 3000
     - -126614.072
     - -0.610426
     - 0.152109
   * - 4000
     - -126492.633
     - -0.574760
     - 0.105831

Interpretation and modern evidence
----------------------------------

ATcT 1.228 gives CNC Hf0=672.2 and Hf298=676.5 +/- 1.7 kJ/mol. The same-version C/N cycle yields D0=1221.195 kJ/mol, 120.167 kJ/mol below JANAF and 7.982166 kJ/mol above NASA CNC. A retained-thermal energy control would multiply K by 6.5388e-5 at 1500 K and 0.026966 at 4000 K. These are analytic formation-constant factors, not solved abundance ratios. The original JANAF enthalpy is an estimate with +/-126 kJ/mol uncertainty, so the disagreement is not presented as a precise old measurement overturned by a huge number of standard deviations. The H0-aligned JANAF/NASA ratio remains 0.8283–0.8881; thermal assumptions do not become equivalent merely by updating energy.

The original page image shows two low components at 0 and 26.41 cm-1, each of weight 2. They already count the ground orbital/spin manifold: adding another Pi factor would double count. The stretches 1200 and 900 cm-1 were estimated, and strong Renner–Teller coupling limits a simple bending harmonic description. Garcia et al. (2017) derived a CNC formation enthalpy of 676 +/- 12 kJ/mol, but its 2020 erratum changes this to 678 +/- 12 after correcting the energy scale. The corrected result is compatible with ATcT 1.228; the obsolete uncorrected number is not used as the corroborating value. Current ATcT provenance combines many theoretical atomization/isomerization cycles and an entry labelled Garcia2017. The current network and the corrected experiment are recorded separately; propagation of every erratum correction inside the network has not been independently reconstructed.

The exact molecule and atomic rows from `ATcT 1.228
<https://atct.anl.gov/Thermochemical%20Data/version%201.228/index.php>`_ are
archived with identifiers and both 0/298 K values. No isotope ZPE correction is
added to an already ground-referenced evaluated energy. Marginal uncertainties
are retained; reaction covariance is not invented by independent quadrature.

`Merer and Travis, CNC absorption (1966) <https://doi.org/10.1139/p66-029>`_: Original JANAF ancestry and primary-paper bibliographic/abstract context inspected; no complete new Renner–Teller thermal model derived.

`Garcia et al., threshold photoelectron spectroscopy (2017) <https://doi.org/10.1063/1.4978336>`_: Primary abstract and distinct CNC/CCN assignment inspected; its formation enthalpy must be read with the 2020 erratum.

`Garcia et al., corrected energy scale (2020) <https://doi.org/10.1063/5.0010035>`_: Indexed text of the published erratum inspected, including corrected CNC Hf298=678 +/-12 kJ/mol. Full PDF download unavailable; spectra not refitted.

Importance and bounded decision
-------------------------------

The final ``curated-tio2-v1`` ion-inclusive archive contains 75 cases and
537 species. Maximum N allocation is 0.086815648 percent at
3000 K, 0.01 bar, C/O=1.5.
The maximum mole fraction is 6.5053825e-08; its own condition is
recorded separately in the packet. All archived cases converged and the target
function reproduces the named current factory exactly. These are saved
allocations, not replacement solves. A small value in this hot hydrogen-rich
grid does not establish irrelevance in cold, oxidizing or photochemical regimes.

Build the -120.167 kJ/mol ATcT retained-thermal physical D0 control while preserving the fit residual, alongside a separately justified NASA-thermal or state-resolved thermal control. Audit Renner–Teller state weighting and the 2020 energy-scale correction explicitly, then assess fixed-network substitution. Do not substitute CCN parameters or duplicate the ground-state orbital factor.

Curated is unchanged. MCEG supports bounded source construction, numerical
diagnostics and this scoped decision; candidate adoption and complete physical
accuracy remain unknown. The frozen main-table grades are unchanged; this resolves the separate identity
appendix entry. Missing ancestral full texts do not indefinitely gate
merging this source review.
