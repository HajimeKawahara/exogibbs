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

The supplied Gamache et al. (2021) full paper closes the earlier TIPS2021
text-access gap. Section 3.6 (pp. 6-7) distinguishes two calculations for
12CH4: a direct rovibrational sum through 1300 K, using levels up to J=67
and complete to about 15000 cm^-1, and a product approximation
``Qrot * Qvib`` from 1301 to 2500 K. The latter combines a ground-vibrational
rotational Hamiltonian through J=80 with a separately summed variational
vibrational model. Thus every TIPS2021 point in the present 1500-2500 K
comparison uses the product approximation. It is not a complete coupled
rovibrational sum at those temperatures. Its potential-energy ancestry and
comparison with Nikitin are recorded explicitly; agreement does not supply
an independent high-temperature experimental constraint.

The supplied published MM paper confirms that its approximately 2000 K
completeness estimate is inferred from comparison with this TIPS2021 curve
(section 5.2 and Figure 4, p. 3725). MM uses a different, empirically refined
Owens et al. potential and MARVEL energy replacements, making it a useful
model comparison. That comparison does not quantify physical errors in both
models or establish convergence of enthalpy and Cp. The next audit must
therefore test the TIPS product approximation and its 1300/1301 K transition,
alongside independent state-sum energy moments.

Version labels also matter. The paper compares against TIPS2021 and shows MM
below that table above 2000 K. We reproduce that ordering with the frozen
official HAPI TIPS2021 arrays. Current HITRAN q32/TIPS2025 instead gives
Q2025/Q2021=0.935896 at 2000 K and 0.787143 at 2500 K. These changes exceed
mere physical-constant rounding. Their cause is unresolved here; no error,
truncation mechanism or improved accuracy is asserted. The supplied
``Gamache_Corr.pdf`` is the 2025 correction to the HITRAN2024 paper, not a
correction to the supplied 2021 paper. It restores omitted symmetry labels
for 16O14N18O and does not explain the CH4 change. The 2025 main paper was
already inspected separately from a public author copy.
Both TIPS tables stop at 2500 K; missing higher-temperature values stay missing.

Wenger, Champion and Boudon (2008) provide a historical alternative to 3000 K,
combining lower states with statistical extrapolation toward dissociation.
Their Table 4 gives Q(3000)=4.19e6 with an estimated 8.3% uncertainty, including
spin. Nikitin et al. (2015) question that high-energy extrapolation. The
user-supplied full paper and its 2017 correction are now inspected; the
previous full-text access blocker is closed.

Nikitin's contact-transformation calculation uses the NRT potential, directly
diagonalized symmetry blocks, symmetry-ratio approximations and extrapolation
in polyad number P and angular momentum J. Its Table 3 is therefore not a
fully direct, converged state sum. Expanded-basis variational calculations
approach the contact-transformation results at low J, but share the same
potential. The authors estimate Q accuracy of less than 0.1% through 1000 K
and about 1-2% at 2000 K. They explicitly leave higher-temperature error bars
unquantified because of extrapolation and near-dissociation uncertainties.
The 3000 K endpoint does not provide evidence for 4000 K completeness.

The 2017 correction adds digits to Table 4; it does not report a new physical
state calculation. We evaluate its three intervals using
``log10(Q) = sum(B[n] * log10(T)**n)`` with T in kelvin. Both the corrected
coefficients and selected original table values are retained separately in
``literature.json``. The full supplied PDFs remain local; their hashes and
page counts identify the inspected copies, and offline replay needs only the
transcribed numerical inputs.

.. list-table:: Nikitin table values and corrected printed fit, spin included
   :header-rows: 1

   * - T, K
     - 2015 Table 3 Q
     - 2017 fit Q
     - Fit minus table, %
     - Table Q/16 / JANAF
     - Table Q/16 / NASA
   * - 1500
     - 40424.32
     - 40688.64
     - +0.654
     - 1.05258
     - 0.975059
   * - 2000
     - 179153.26
     - 180413.75
     - +0.704
     - 1.09017
     - 0.947302
   * - 2500
     - 693422.17
     - 697662.19
     - +0.611
     - 1.13135
     - 0.906907
   * - 3000
     - 2346982.58
     - 2365872.02
     - +0.805
     - 1.16331
     - 0.849073

The printed corrected fit exceeds all seven Table 3 totals by 0.61-0.80%,
larger than the approximately 0.1% interpolation accuracy stated in Section
4.3. Table 5 also lists 40382 at 1500 K, differing from Table 3 by 0.1048%.
The table images and coefficient transcription were checked; the cause of
these representation differences remains unresolved. We preserve them without
silently changing coefficients or treating a fit residual as a physical error
bar. A good Q fit alone would also not validate its energy derivatives or Cp.

At 2000 K the Table 3 Q/16 lies 9.0% above JANAF and 5.3% below NASA's
effective Q. This strengthens the evidence for revisiting the thermal curve,
while the natural-mass/isotopologue caveat still applies. At 3000 K Nikitin's
table is only about 56% of Wenger's historical Q, and neither work establishes
a complete model through 4000 K. The corrected polynomial is not extrapolated
beyond 3000 K in this audit.

Decision and remaining evidence
-------------------------------

Modern calculations demonstrate that the existing thermal choice needs review,
but they do not supply a reconciled, complete 1500-4000 K replacement. Retain
the current implementation operationally and preserve the ATcT energy choice.
This is not a finding that JANAF is physically superior. Before adoption:

1. Reconcile the versioned TIPS methane arrays and their generators, and
   resolve the Nikitin corrected-fit/table residual using original numerical
   data or author clarification. Inspect the TIPS2021 product approximation
   and its transition from the direct sum; the supplied 2021 full paper,
   2025 NO2 correction, MM paper and both Nikitin papers are now inspected.
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
* `Gamache et al. 2021 <https://doi.org/10.1016/j.jqsrt.2021.107713>`_.
* `Gamache et al. 2025 <https://doi.org/10.1016/j.jqsrt.2025.109568>`_ and
  `its NO2 correction <https://doi.org/10.1016/j.jqsrt.2025.109618>`_;
  `HITRAN partition tables <https://hitran.org/docs/iso-meta/>`_.
* `Wenger et al. 2008 <https://doi.org/10.1016/j.jqsrt.2008.06.006>`_.
* `Nikitin et al. 2015 <https://doi.org/10.1016/j.jqsrt.2015.08.004>`_ and
  `2017 correction <https://doi.org/10.1016/j.jqsrt.2017.05.026>`_.
