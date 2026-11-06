NCN source and thermochemical review
====================================

Prioritize a modern NCN energy candidate within the trace-species group; the 2026 threshold experiment corroborates the lower evaluated formation enthalpy.

Identity and source scope
-------------------------

NCN is linear N-C-N (cyanonitrene), canonical C1N2_ncn, CAS 2669-76-3, with X3Sigma-g-minus ground state and rotational symmetry 2. It is not CNN (C-N-N). Original JANAF page 637 and the explicit NASA NCN record identify this connectivity and ground state.

The retained source is the JANAF-labelled FastChem fit. Fresh official
`JANAF C-089 <https://janaf.nist.gov/tables/C-089.html>`_ and neutral-atom
tables are pinned, together with selected records from the versioned
`NASA CEA database <https://github.com/nasa/cea/blob/2f79a647e85737742fb08a127da00d1ea1d069da/data/thermo.inp>`_.
The NASA header reads ``Hf:Gurvich,1991 pt1 p219. Jacox,1998 p180.``.
Selected records are checked against the literal official NASA coefficients and
original interval bounds. Existing screening records are cross-checked where
available; this identity appendix does not require a prior formula-only screen.

Balanced comparison
-------------------

The reaction is ``1 C(g) + 2 N(g) -> NCN(g)`` at 1 bar. Native NASA calorics use
R=8.314510; equilibrium normalization uses R=8.31446261815324 J/mol/K.
H0 is derived from printed Hf298 and H298-H0, never polynomial extrapolation
to zero. The balanced source atomization energies are 1152.832853 kJ/mol (NASA) and 1180.558000 kJ/mol (JANAF).
These are physical source anchors, not the literal fit coefficient of 1/T.

.. list-table:: Dimensionless formation-constant ratios
   :header-rows: 1

   * - T (K)
     - Current / NASA
     - Current / JANAF
     - H0-aligned JANAF / NASA
   * - 1500
     - 9.8255
     - 1.01022
     - 1.05313
   * - 2000
     - 5.58037
     - 0.99818
     - 1.05527
   * - 3000
     - 3.19607
     - 0.994118
     - 1.05791
   * - 4000
     - 2.44697
     - 1.00377
     - 1.05913

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
     - -27615.440
     - 0.499327
     - 0.003732
   * - 2000
     - -27614.418
     - 0.500177
     - -0.000654
   * - 3000
     - -27614.851
     - 0.499206
     - 0.000696
   * - 4000
     - -27610.102
     - 0.500885
     - 0.012107

Interpretation and modern evidence
----------------------------------

ATcT 1.228 gives Hf0=451.33 and Hf298=451.74 +/- 0.31 kJ/mol, yielding same-version D0=1201.350 kJ/mol. That is +20.792 kJ/mol relative to JANAF and +48.517147 kJ/mol relative to NASA. Retained-thermal K factors are 5.2970 at 1500 K and 1.8686 at 4000 K. The old JANAF value, about 473 +/- 21 kJ/mol at 298 K, and the NASA Gurvich energy cannot be selected just by their widespread availability. The aligned source thermal difference remains about 5–6 percent, separate from this energy revision.

The inspected 2026 primary iPEPICO abstract reports Hf0=450.5 +/- 3.5 and Hf298=451.0 +/- 3.5 kJ/mol, obtained from NCN/NCN3 ionization and dissociation cycles. This supports the lower evaluated energy and the 2018 HEAT result, while the complete experimental analysis has not been independently rerun. Current ATcT provenance prominently includes a 2025 reaction-energy result and Harding 2018; its value is not obtained by copying the old 2013 recommendation of 457.8 kJ/mol. Original JANAF also estimates a singlet electronic term at 38000 cm-1, so modern state treatment remains a thermal follow-up. The 2013 source is useful thermochemical ancestry, but its old ATcT version is not the current energy recommendation.

The exact molecule and atomic rows from `ATcT 1.228
<https://atct.anl.gov/Thermochemical%20Data/version%201.228/index.php>`_ are
archived with identifiers and both 0/298 K values. No isotope ZPE correction is
added to an already ground-referenced evaluated energy. Marginal uncertainties
are retained; reaction covariance is not invented by independent quadrature.

`Fassheber et al., NCN threshold photoelectron thermochemistry (2026) <https://doi.org/10.1016/j.proci.2026.106251>`_: Primary publisher abstract and numerical experimental result inspected; full measurement reanalysis not performed.

`Goos et al., NCN thermochemistry in flames (2013) <https://doi.org/10.1016/j.proci.2012.06.128>`_: Primary author manuscript inspected for historical thermal/energy separation; its 2012 ATcT energy is not adopted.

Importance and bounded decision
-------------------------------

The final ``curated-tio2-v1`` ion-inclusive archive contains 75 cases and
537 species. Maximum N allocation is 2.2750255e-05 percent at
4000 K, 100 bar, C/O=1.5.
The maximum mole fraction is 1.2296488e-11; its own condition is
recorded separately in the packet. All archived cases converged and the target
function reproduces the named current factory exactly. These are saved
allocations, not replacement solves. A small value in this hot hydrogen-rich
grid does not establish irrelevance in cold, oxidizing or photochemical regimes.

Build an ATcT +20.792 kJ/mol retained-thermal D0 control and a separately justified modern thermal control, then test substitution. Retain the 2026 measurement as corroborating energy evidence with its stated uncertainty and inspection level. Do not reuse CNN parameters or an outdated 2013 energy anchor.

Curated is unchanged. MCEG supports bounded source construction, numerical
diagnostics and this scoped decision; candidate adoption and complete physical
accuracy remain unknown. The frozen main-table grades are unchanged; this resolves the separate identity
appendix entry. Missing ancestral full texts do not indefinitely gate
merging this source review.
