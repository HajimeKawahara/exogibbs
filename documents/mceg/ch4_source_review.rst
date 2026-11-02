Methane energy and thermal-source investigation
===============================================

This 2026-11-02 follow-up to #308 and #309 reviews CH4 (network ID
``C1H4``). Keep the existing implementation provisionally. Its balanced
ATcT energy remains justified, but the evidence does not select a complete
replacement thermal model over 1500-4000 K. No preset or coefficients change.
This is a completed initial source assessment with explicit follow-ups,
not a declaration that methane thermochemistry is physically validated.

Why this molecule matters
-------------------------

Replaying the existing 75-case ``curated-c2h-v1`` ion-inclusive grid finds
CH4 holding up to 97.1975% of the carbon: mole fraction 0.00122784 at
1500 K, 100 bar and C/O=1.5. This is a gas-only model footprint, not an
observed atmospheric fraction. The original source-selection review already
flagged high-temperature thermal disagreement and requested an independent
rovibrational audit. No new equilibrium substitution is run here because a
preferred replacement has not been established.

Identity, references and existing implementation
------------------------------------------------

The species is neutral gaseous tetrahedral methane, with a singlet electronic
ground state. The reaction is C(g) + 4 H(g) -> CH4(g), at standard pressure
1 bar. It is distinct from methane ions, dissolved methane and individual
minor isotopologues.

* Printed JANAF page 615, table C-067, retains the March 1961 evaluation.
  It specifies electronic weight 1, rotational symmetry number 12 and molar
  mass 16.04276 g/mol. Its frequencies and geometry trace to Jones and
  McDowell (1959) and Stevenson and Ibers (1960); its formation energy
  traces to Rossini's combustion work, with updated water/CO2 references.
  The inspected printed nodes agree with the existing frozen JANAF excerpt.
* The frozen NASA9 methane record cites Gurvich (1991), part 1 page 44 and
  part 2 page 36, and covers 200-6000 K. Its header gives
  Hf(298.15 K)=-74.600 kJ/mol and H(298.15)-H(0)=10.016202 kJ/mol.
  The Gurvich primary discussion was not obtained; a newer compilation date
  alone is not evidence for better high-temperature accuracy.
* The existing curated model uses JANAF **molecular** thermal functions,
  continuous quintic Hermite free energy, NASA **atomic** thermal functions,
  and the balanced ATcT 1.172 atomization energy. It is not simply the raw
  JANAF reaction with the molecular formation enthalpy changed.

The primary printed JANAF page is archived with its parent-PDF hash. Fresh
direct downloads of C-067.txt returned 403, so this study reuses the pinned
official excerpt and verifies selected printed nodes instead of claiming a
new full-table download. The original source-comparison baseline is preserved.

Energy and thermal differences
------------------------------

ATcT 1.172 gives CH4 Hf(0)=-66.547 and Hf(298.15)=-74.516 +/- 0.043 kJ/mol,
with estimated 95% confidence limits. The same network's C and H references
are retained. Methane's leading provenance includes Smith's equilibrium
measurement with a third-law reduction and Schley's combustion measurement.
The network also shares evidence with other species; the reported CH3
correlation is 0.851. Neither independent-error quadrature nor independent
high-temperature validation is inferred from these data.

.. list-table:: Balanced ground-energy reaction, C(g) + 4 H(g) -> CH4(g)
   :header-rows: 1

   * - Source
     - Delta H(0), kJ/mol
   * - JANAF
     - -1642.239
   * - NASA headers
     - -1641.965907
   * - ATcT 1.172
     - -1642.064

For JANAF and NASA, H(0) is first assigned in the source's 298 K elemental
reference gauge, then the molecular and atomic terms are balanced. An isolated
molecular Hf(0) subtraction would give the wrong correction. NASA header
conversions use its stated R=8.31451 J/mol/K; the source-comparison decomposition
uses a common R=8.31446261815324 J/mol/K.

.. list-table:: Dimensionless atom-formation equilibrium constants
   :header-rows: 1

   * - T, K
     - Raw JANAF / raw NASA
     - H298-aligned JANAF / NASA
     - Current curated / raw NASA
   * - 1500
     - 0.947991
     - 0.927453
     - 0.934394
   * - 2000
     - 0.884566
     - 0.870154
     - 0.874602
   * - 3000
     - 0.738873
     - 0.730825
     - 0.733047
   * - 4000
     - 0.591423
     - 0.586585
     - 0.587868

The middle column shifts only the **reaction** H298 anchor as a diagnostic.
It does not replace entropy or Cp. The growing difference survives this
alignment: methane Cp at 4000 K is 104.183 J/mol/K in JANAF and 124.804 in
NASA. At 6000 K the values are 106.306 and 146.214, respectively; this last
point is outside the curated domain. Atomic contributions to the JANAF/NASA
Gibbs difference stay below 46 J/mol at the eight inspected nodes.

What modern partition functions establish
-----------------------------------------

For each thermochemical source, an effective internal partition function is
reconstructed from its thermal free energy:

.. math::

   Q_{\rm int}(T)=\frac{\exp[-(G^\circ-H^\circ(0))/(RT)]}
   {(2\pi m k_B T/h^2)^{3/2} k_B T/p^\circ}.

ExoMol and HITRAN tabulate the main isotopologue, 12CH4, including nuclear
spin. The Td weights are 5, 5, 2, 3, 3; four proton spins give a total product
of 16. We compare Q/16 with conventional nuclear-spin-free thermochemistry.
There is no second division by rotational symmetry 12. HITRAN's
state-independent ``gi=1`` does not remove the symmetry-dependent spin weights.
Thermochemical effective Q uses each source's natural molar mass; this
comparison is not a complete treatment of isotope mixtures or fractionation.

.. list-table:: Spectroscopic Q/16 divided by thermochemical effective Q
   :header-rows: 1

   * - T, K
     - MM / JANAF
     - MM / NASA
     - TIPS2021 / JANAF
     - TIPS2025 / JANAF
   * - 1500
     - 1.05023
     - 0.972886
     - 1.04759
     - 1.04395
   * - 2000
     - 1.05377
     - 0.915671
     - 1.07447
     - 1.00560
   * - 2500
     - 0.975006
     - 0.781577
     - 1.09851
     - 0.864682

MM is the ExoMol methane line list of Yurchenko et al. (2024), using
variational rovibrational states and empirical energy replacements. Its paper
estimates completeness only through about 2000 K. The currently downloaded
``.pf`` file reaches 5000 K, while ``.def`` still states 2000 K; tabulation
does not establish completeness. The 2500 K MM row is therefore a diagnostic
outside that estimate. Its falling ratios at 3000-4000 K cannot be used to
select JANAF or NASA. Cp needs energy-moment convergence, beyond Q alone.

Version labels also matter. The paper compares against TIPS2021 and shows MM
below that table above 2000 K. We reproduce that ordering with the frozen
official HAPI TIPS2021 arrays. Current HITRAN q32/TIPS2025 instead gives
Q2025/Q2021=0.935896 at 2000 K and 0.787143 at 2500 K. These changes exceed
mere physical-constant rounding. Their cause is unresolved here; no error,
truncation mechanism or improved accuracy is asserted. The inspected 2025
corrigendum concerns an NO2 isotopologue and does not explain the CH4 change.
Both TIPS tables stop at 2500 K; missing higher-temperature values stay missing.

Wenger, Champion and Boudon (2008) provide a historical alternative to 3000 K,
combining lower states with statistical extrapolation toward dissociation.
Their Table 4 gives Q(3000)=4.19e6 with an estimated 8.3% uncertainty, including
spin. Later Nikitin et al. (2015) question the high-energy extrapolation and
report a different treatment. Its 2017 erratum restores sufficient precision
to fitted coefficients. Those corrected coefficients and full convergence
analysis were not obtained, so the older table is not elevated to a definitive
3000 K benchmark.

Decision and remaining evidence
-------------------------------

Modern calculations demonstrate that the existing thermal choice needs review,
but they do not supply a reconciled, complete 1500-4000 K replacement. Retain
the current implementation operationally and preserve the ATcT energy choice.
This is not a finding that JANAF is physically superior. Before adoption:

1. Reconcile the versioned TIPS methane arrays and their generators, and read
   the Nikitin 2015/2017 corrected numerical treatment.
2. Establish bound-state and near-dissociation coverage through 4000 K with
   common spin/isotope conventions and first/second energy moments.
3. Construct consistent G, H, S and Cp, then evaluate one-molecule finite
   substitutions with conserved elemental budgets.

The MCEG contract separates supported source bookkeeping and numerical
reproduction from unknown complete thermal accuracy and atmospheric accuracy.
The footprint is a reused ideal-gas numerical result. Condensation, nonideal
gas effects, chemistry out of equilibrium and spectra are not assessed.

Reproduction and sources
------------------------

All numerical checks are offline::

   python mceg/reviews/ch4_source_review.py --output /tmp/ch4-comparison.json
   pytest -q tests/model_evidence/ch4_source_review_test.py
   PYTHONPATH=/path/to/mceg/src python -m mceg check \
       mceg/contracts/ch4_source_review.yaml \
       --assessment source_attribution_review --root . --plain

The fixed packet is ``tests/data/validation/ch4_source_review/20261102/``.
Its ``sources.json`` records access limits, URLs, parent-download hashes and
existing-input hashes; ``literature.json`` records the interpreted evidence.

* `JANAF C-067 <https://janaf.nist.gov/tables/C-067.html>`_ and
  `printed Carbon volume <https://janaf.nist.gov/pdf/JANAF-FourthEd-1998-Carbon.pdf>`_.
* `ATcT 1.172 methane <https://atct.anl.gov/Thermochemical%20Data/version%201.172/species/?species_number=22>`_.
* `Yurchenko et al. 2024 <https://doi.org/10.1093/mnras/stae148>`_.
* `Gamache et al. 2025 <https://doi.org/10.1016/j.jqsrt.2025.109568>`_ and
  `HITRAN partition tables <https://hitran.org/docs/iso-meta/>`_.
* `Wenger et al. 2008 <https://doi.org/10.1016/j.jqsrt.2008.06.006>`_.
* `Nikitin et al. 2015 <https://doi.org/10.1016/j.jqsrt.2015.08.004>`_ and
  `2017 correction <https://doi.org/10.1016/j.jqsrt.2017.05.026>`_.
