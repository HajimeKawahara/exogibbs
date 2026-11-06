PN source and thermal review
============================

Retain the current Lodders-labelled PN function. Its large difference from the
uncorrected JANAF table is understood; replacing it with that table would restore
an obsolete energy evaluation. The modern PaiN thermal diagnostic does not
justify a production replacement. This review changes no curated data.

Identity and evaluated sources
------------------------------

Neutral phosphorus mononitride is ``N1P1``, with a singlet X1Sigma+ ground state.
The retained FastChem row cites Lodders1999. `Lodders (1999)
<https://doi.org/10.1063/1.556046>`_, Table 2 and section 2.3, selects
Hf298 = 171.488 +/- 15 kJ/mol on the white-phosphorus reference, following
Gurvich's assessment of the later Gingerich experiment. This is also the
171487.305 J/mol NASA header value within 0.695 J/mol.

The JANAF web table instead retains 104.780 kJ/mol. A red-V to white-P shift
alone would give 87.320 kJ/mol; that is **not** Lodders' selected value.
Lodders also identifies a 0 K/298 K conversion problem in the older evaluation.
Thus neither a second -17.460 kJ/mol shift nor adoption of the uncorrected JANAF
row is warranted. The inspected evaluation supports this bounded choice even
though its ancestral experimental uncertainty has not been independently
re-evaluated here.

The packet contains freshly obtained official `JANAF N-010
<https://janaf.nist.gov/tables/N-010.html>`_ and neutral-atom tables, and selected
records from the pinned `NASA CEA database
<https://github.com/nasa/cea/blob/2f79a647e85737742fb08a127da00d1ea1d069da/data/thermo.inp>`_.
NASA attributes PN to Gurvich1989. The literal NASA coefficients match the
screening snapshot; the original NASA interval bounds are retained.

Balanced source comparison
--------------------------

The reaction is P(g) + N(g) -> PN(g), at 1 bar. NASA's dimensional functions use
its native R = 8.314510 J/mol/K; equilibrium normalization uses
R = 8.31446261815324. The source H0 values come from the printed Hf298 and
H298-H0 headers, never a polynomial extrapolation to zero. Balanced atomization
energies are 613.999944 kJ/mol (NASA) and 680.602 kJ/mol (uncorrected JANAF).
The latter is retained as a diagnostic, not a corrected physical recommendation.
A fitted coefficient of 1/T is not silently treated as an independent D0.

.. list-table:: Dimensionless formation-constant ratios
   :header-rows: 1

   * - T (K)
     - Current / NASA
     - Current / raw JANAF
     - H0-aligned JANAF / NASA
   * - 1500
     - 1.00050
     - 0.00478
     - 1.00385
   * - 2000
     - 0.99800
     - 0.01811
     - 1.00432
   * - 3000
     - 0.99628
     - 0.06867
     - 1.00461
   * - 4000
     - 0.99754
     - 0.13414
     - 1.00381

``comparison.json`` records molecular and balanced reaction G/H/S/Cp, the H0,
thermal-enthalpy and entropy decomposition, and JAX derivatives of the literal
current function. H0 alignment changes only a reaction energy offset. Close
aligned ratios do not make the absolute JANAF energy preferable, nor prove
that the finite FastChem fit is an exact reconstruction of Lodders' table.

Modern thermal diagnostic
-------------------------

`Semenov et al., PaiN (2025) <https://doi.org/10.1093/mnras/stae2610>`_
replaces YYLT. The inspected paper and distributed data contain X1Sigma+ and
A1Pi states. All 30327 distributed levels through J=210 are retained in a compact
numerical packet; transition intensities are unnecessary. Energies are relative
to the lowest rovibronic level, with gtot = 6(2J+1). Divide by the nuclear-spin
product 6 once; the heteronuclear symmetry number is 1 and no additional singlet
or electronic factor is inserted. A potential-well De or its zero-point energy
must not be substituted for the source D0. PaiN fixes its ground potential De to
51297 cm-1 from Ushakov et al.; the spectroscopic fit is not a new calorimetric
energy measurement.

The state sum agrees with the separately distributed partition nodes within
6.2e-7 relative. Against NASA, Q ratios are 0.999450--0.999327 across the four
nodes; thermal H differences range from -0.276 to -30.340 J/mol and Cp differences
from -0.09358 to +0.00338 J/mol/K. Natural-mass translation is used consistently;
the 31P14N isotope translation correction is -0.001005 J/mol/K in S.

The top 20 percent of distributed energies contributes at most 1.64e-9 of Q.
This checks only the supplied tail. Missing dark electronic states and the E
state discussed by the authors are not bounded by this number. Some distributed
A-state energies exceed the ground dissociation limit; all-distributed sums are
a labelled diagnostic, not a new bound-state definition or proof of complete
high-temperature thermodynamics.

Importance, decision and finite follow-up
----------------------------------------

In the final ``curated-tio2-v1`` 75-case ion-inclusive archive (537 species), PN
reaches mole fraction 5.311372e-9 and 1.204730 percent of phosphorus, both at
2000 K, 1 bar and C/O=0.588844. The archived PN function matches the named
factory exactly. These are saved allocations, not replacement calculations;
they say nothing about kinetic interstellar chemistry or cold atmospheres.

Retain now. Revisit an energy candidate if a demonstrably improved balanced
PN atomization evaluation becomes available; first reconcile its ZPE and atomic
references with the 171.488 +/- 15 kJ/mol evaluation. A PaiN thermal candidate
would additionally need a justified electronic-state and dissociation treatment
and fixed-network substitution. No unavailable ancestral article is a prerequisite
for merging this source review. Full physical accuracy and adoption remain
unknown in MCEG, separately from reproducible source and caloric checks.
