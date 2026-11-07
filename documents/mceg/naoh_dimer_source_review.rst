Sodium hydroxide dimer source investigation
===========================================

For neutral gas (NaOH)2, the **Gurvich1996 evaluation represented by NASA is the
scientifically preferred improvement direction** among the inspected sources.
It revises both the molecular thermal model and the analysis of older vapor
experiments. The roughly 16--19-fold larger formation constant than curated at
1500 and 3000 K is not explained by a single energy correction. This investigation
identifies the difference and records the remaining evidence in MCEG. It leaves
production coefficients unchanged; that is not an endorsement of JANAF as the
more accurate choice.

Identity and source ancestry
----------------------------

The unchanged FastChem row ``H2Na2O2 Sodium_Hydroxide`` denotes the neutral,
six-atom hydroxide dimer, also written Na2(OH)2 or NASA ``Na2O2H2``. It is distinct
from Na2O2 peroxide, NaOH monomer, hydroxide ions, hydrates and condensed NaOH.
Both reviewed constructions assign electronic statistical weight one. Their
C2h and D2h geometries are alternative empirical representations of the same
nominal dimer, not evidence that two measured isomers were accidentally merged.

The row cites JANAF1998. The `official H-062 table
<https://janaf.nist.gov/tables/H-062.html>`_ still dates its evaluation to
December1970; a compilation date does not imply new molecular measurements.
The inspected `JANAF1974 supplement, p.449
<https://doi.org/10.1063/1.3253143>`_ describes the adopted C2h structure,
estimated vibrational modes, symmetry number two and formation-energy analysis.
Its Hf298 = -145.2 +/-6.0 kcal/mol has central value -607.5168 kJ/mol,
consistent with the modern table's -607.517. The historical table is in cal and
uses 1 atm; all present Gibbs comparisons use the modern 1-bar table.

`Gurvich et al. (1996), pp.1268--1271
<https://doi.org/10.1063/1.555982>`_ replaces that construction by an estimated
D2h planar rhomboid, symmetry number four, and revised vibrational frequencies.
Its preferred geometry follows analogy with related dimers. The paper does not
supply a directly measured complete set of dimer molecular constants. Both
sources use older mass-spectrometric and molecular-beam results; their comparison
is not an independent experimental validation.

The frozen NASA record explicitly cites ``Gurvich,1996b.``. At seven selected
298.15--4000 K nodes it reproduces the printed Table3-27 absolute functions to
within 6 J/mol in H, 0.003 J/mol/K in S, and 0.014 J/mol/K in Cp. These small
residuals support attribution and numerical representation, not accuracy of the
underlying estimated molecular physics.

What produces the discrepancy
-----------------------------

The replay uses the balanced atom-formation reaction at 1 bar,
2 Na(g) + 2 O(g) + 2 H(g) -> (NaOH)2(g). The exact official JANAF dimer nodes
reproduce the frozen screening when combined with its archived atomic functions.
The current polynomial differs from JANAF by +0.01539 and -0.01167 in ln K at
1500 and 3000 K respectively, much smaller than the NASA discrepancy.

.. list-table:: Formation-constant ratios and diagnostic energy alignment
   :header-rows: 1

   * - T (K)
     - Curated/NASA
     - JANAF/NASA
     - JANAF/NASA after aligning reaction H298 only
   * - 1500
     - 0.05163
     - 0.05084
     - 0.19674
   * - 3000
     - 0.06085
     - 0.06157
     - 0.12112

Energy alignment is a diagnostic, not an adopted model. At 1500 K the
JANAF-minus-NASA reaction Gibbs difference is 37.154 kJ/mol: molecular H298
contributes +16.479, thermal enthalpy -10.911, entropy +31.346, and atomic
references +0.241 kJ/mol. At 3000 K the corresponding terms are +16.479,
-13.317, +66.289 and +0.081, totaling 69.532 kJ/mol. Molecular entropy is
20.897 and 22.096 J/mol/K larger in NASA at these temperatures. Atomic reference
differences are small here, but are retained explicitly.

The twelve printed frequencies explain the thermal change. Four old estimated
1250 cm^-1 modes become 300--500 cm^-1 modes in the revised construction; other
modes also change. Their harmonic heat-capacity difference reproduces the
printed Cp difference to within 0.002 J/mol/K at the seven reviewed nodes.
The rotor change supplies an entropy offset

.. math::

   \Delta S_{\rm rot}
     = R\left[\tfrac12\ln(7000/8156.8)-\ln(4/2)\right]
     = -6.399\ {\rm J\,mol^{-1}\,K^{-1}}.

Combining it with the harmonic vibrational entropy change reproduces the
absolute-table entropy difference within 0.003 J/mol/K. Equal mass and pressure
make translation cancel in this diagnostic. This explains the source difference;
it does not validate harmonic treatment of soft modes at high temperature.

Two printed inconsistencies are preserved in the packet. Gurvich Table3-24
lists a 1000 K entropy difference of 15.049 J/mol/K, while Table3-27 and the
official JANAF absolute values imply 19.052. NASA and the molecular-constant
replay support the latter comparison. The inertia-product uncertainty is
+/-5000 in the prose and +/-500 in the following list, in units of
10^-117 g^3 cm^6. No cause is invented and neither uncertainty is propagated.

The separate association reaction 2 NaOH(g) -> (NaOH)2(g) cancels atomic
references. Using each source's own monomer gives JANAF/NASA association ratios
0.01898 and 0.03806 at 1500 and 3000 K. These differ from atom-formation ratios
because the monomer source also changes. They must not be reported as the
abundance response to replacing only the dimer.

Experimental support and practical next work
--------------------------------------------

Gurvich selects Hf298 = -624 +/-10 kJ/mol after reassessing earlier experiments
and auxiliary thermal functions. It considers the monomer/dimer pressure ratio
from `Rao and Schoonmaker (1960) <https://doi.org/10.1063/1.1731491>`_ more
reliable than absolute pressures, which can be affected by condensed-phase
composition and creeping molten hydroxide. Its selected intermediate reaction
is NaOH(l) + NaOH(g) -> dimer(g), with extrapolated H0 = -4.9 +/-6 kJ/mol.
This is not the gas association energy or total atomization D0.

`Schoonmaker and Porter (1958) <https://doi.org/10.1063/1.1744157>`_ is another
important ancestor. The 1996 review reports its gas association enthalpy
-54 +/-5 kcal/mol at 660 K, separately from vaporization out of liquid NaOH.
Original full texts for these two experiments were not obtained: publisher
requests returned HTTP403. Their DOI metadata is confirmed; experimental
interpretation here is explicitly the inspected evaluation's interpretation.
Reported uncertainties are not treated as independent Gaussian errors.

Both evaluations also cite `Buechler et al. (1967)
<https://doi.org/10.1063/1.1840711>`_ for nonpolarity of the dimer. Its inspected
publisher title and abstract emphasize Cs compounds. The unread full text may
also discuss NaOH; the abstract alone does not establish a citation error.
Nonpolarity would in any case not uniquely locate H atoms or distinguish the
two centrosymmetric structure models.

The existing ``curated-c2h-v1`` archive spans 75 ideal-gas cases, 537 species,
1500--4000 K, 10^-6--100 bar, and three C/O ratios. Its maximum dimer sodium
share is **7.07391e-9 of the Na inventory**, at 1500 K, 100 bar and reference
C/O = 0.5888436554. The mole fraction there is 1.00858e-14. Two Na atoms per
dimer are counted, and the archived potentials match the current fit. This
supports low priority within that finite grid; it says nothing definitive about
cooler, sodium-rich conditions or a replacement response. No new equilibrium
solve was run.

The next adoption task can construct a named Gurvich/NASA dimer candidate with
explicit common atomic references and check thermodynamic consistency and a
focused fixed-network substitution. It need not wait for every old paper.
A joint monomer/dimer update is a separate choice. Original-paper confirmation,
independent molecular constants and any required high-temperature uncertainty
assessment remain MCEG follow-ups rather than conditions for merging this
completed investigation.

``mceg/contracts/naoh_dimer_source_review.yaml`` records supported numerical
attribution and the conditional scientific preference. Complete original-source
confirmation, quantified physical accuracy and atmospheric accuracy remain
unknown. The compact packet in
``tests/data/validation/naoh_dimer_source_review/20261103`` records numerical
excerpts, full-source hashes, inspected pages, access gaps and offline replay.
Full copyrighted articles are not committed.
