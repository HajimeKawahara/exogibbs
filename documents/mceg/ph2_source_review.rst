PH2 source investigation and equilibrium sensitivity
====================================================

PH2 warrants review because it can hold a large part of the phosphorus budget,
even when its mole fraction is small. In the saved curated-c2h-v1 grid it holds
64.25% of P at 2500 K, 100 bar and the reference C/O ratio. This investigation
separates record identity, inconsistencies between JANAF editions, the origin
of the current fit, and the effect of one diagnostic source substitution.

Decision: retain the current PH2 fit pending a consistent energy/thermal review.
The NASA alternative is a sensitivity probe, not a newly recommended default.
No production coefficients, dataset identities or historical receipts change.

Three different records must be distinguished
---------------------------------------------

The species is neutral gaseous phosphino, PH2 (CAS 13765-43-0), rather than PH3,
an ion or a condensed phosphorus compound. The formation reaction used in
equilibrium is P(g) + 2 H(g) -> PH2(g), with dimensionless Kp at 1 bar.

.. list-table:: Formation enthalpy at 298.15 K
   :header-rows: 1

   * - Record
     - kJ/mol
     - Interpretation
   * - Printed JANAF Fourth Edition, PH2 page 1339
     - 108.499
     - Old estimated anchor; the accompanying note gives 108.5 +/- 96.2.
   * - Current official online JANAF H-079
     - 125.938
     - Different from print despite matching thermal functions at inspected nodes.
   * - Frozen NASA PH2 polynomial
     - 119.55
     - Gurvich-derived polynomial evaluated with the adopted R; comparison candidate.

The printed JANAF explanation describes a PH3-based formation-energy estimate
and assumed PH3-like geometry/frequencies for thermal functions. Its large
stated uncertainty cannot be turned into a calibrated atmospheric error bar.
`Lodders (1999), p. 1705 <https://srd.nist.gov/jpcrdreprint/1.556046.pdf>`_
explicitly identifies PH2 as correctly converted from red-V to white phosphorus
in the printed Fourth Edition. This establishes the history of that correction;
it does not establish the origin of every discrepancy in the online record.
Later theoretical and experimental energy evaluations also differ from
NASA/Gurvich. Their reference conventions, source hierarchy and access limits
are recorded in the literature packet; agreement with NASA alone is not a
selection rule.

The printed PDF and online page cannot be treated as interchangeable copies.
The online formation-H/G columns have the numerical signature of subtracting
1.5 H2, omitting P from the elemental reference. This is a reproducible numerical
attribution, not proof of the historical software error or transcription process.
It holds at all 64 shared PH2/H2 nodes from 100 to 6000 K within the tables'
rounding. The five inspected printed nodes instead close with P(ref) + H2;
the atomic H/P tables also close with their proper elemental references.
Source-specific reference states must be reconciled before comparing formation
quantities.

The current FastChem/curated PH2 fit is consistent with the printed 108.499
kJ/mol anchor and the JANAF H/S thermal functions: the maximum absolute ln K
residual is 0.01145 across 26 nodes from 1500 to 4000 K (about 1.2% in K).
This agreement does not establish how the original fit was generated.
The discrepancy with the online table therefore does not
establish that the production fit inherited the online column problem.
Current/raw-NASA K ratios are 2.53534 at 1500 K and 1.62516 at 3000 K.

After aligning the reaction enthalpy at 298.15 K, JANAF/NASA K ratios are only
1.04187 at 1500 K and 1.05770 at 3000 K. The energy anchor dominates this
comparison; the large thermal discrepancy found for C2H does not carry over.
This comparison is not an independent validation of either thermal model.

Later source leads and the remaining decision
----------------------------------------------

The inspected `Matus et al. correction table (2009)
<https://doi.org/10.1021/jp811432s>`_ reports a PH2 CCSD(T)/CBS estimate of
133.05 +/- 4.18 kJ/mol at 298 K and quotes an experimental value of
138.49 +/- 2.51 kJ/mol from Berkowitz and Cho (1989). The correction and the
original 2007 calculation are one study. The archived Burcat record selects a
separate G3B3 estimate of 135.47 kJ/mol. These values are useful leads, but their
white/red phosphorus reference choices have not been established from the
inspected material. They must not be ranked against the printed JANAF value
as if all reference states had already been harmonized. The original Gurvich
pages and the full experimental cycle were not obtained.

`Jakubek et al. (2006) <https://doi.org/10.1063/1.2168155>`_ provide measured
ground-state bending and symmetric-stretch levels of 1102 +/- 3 and
2298 +/- 3 cm^-1, compared with the JANAF estimates of 1020 and 2327 cm^-1.
These measurements motivate a thermal-model review; measured fundamentals
cannot simply replace harmonic frequencies without an explicit state model.
The neutral excited-state origin is distinct from the PH2+ cation splitting
discussed in the photoionization literature.

Thus, retaining the current fit is a provisional decision, not a new validation
of the old estimate. A preferred replacement needs a balanced molecular/atomic
energy choice and a separately justified thermal model. NASA and CCCBDB share
Gurvich ancestry, so their agreement is not independent corroboration.

Raw-NASA substitution experiment
--------------------------------

The experiment changes only PH2 to the frozen NASA PH2-P-2H reaction
coefficients. It preserves all other formation functions, the 537-species
ion-inclusive network, formula matrix and elemental budget. The baseline is
the C2H-corrected curated default, not the earlier curated-gas-v1 dataset.

The 75 paired conditions use temperatures 1500, 2000, 2500, 3000 and 4000 K;
pressures 1e-6, 1e-4, 0.01, 1 and 100 bar; and C/O = 0.5888436553555908, 1, 1.5.
Only C/H changes between compositions; other element/H ratios remain fixed
before normalization. Both endpoint models are cold-solved at 1e-13 and 5e-14:
300 accepted solves, with full log amounts, mole fractions, thermochemistry,
budgets and elemental fractions retained.

.. list-table:: Reference C/O, 2500 K, 100 bar
   :header-rows: 1

   * - Quantity
     - Current curated
     - Raw-NASA PH2 probe
   * - PH2 mole fraction
     - 2.83170e-7
     - 2.21963e-7
   * - Fraction of P held in PH2
     - 64.2509%
     - 50.3632%
   * - Fraction of P held in PH3
     - 14.8918%
     - 20.6648%

PH3 rises by a factor of about 1.3877 although its own formation function is
unchanged: the conserved phosphorus redistributes. At 1500 K and 1 bar the
largest absolute PH2 mole-fraction change in the grid is about -9.34e-8,
with a probe/current ratio of 0.4962. These are coupled equilibrium responses,
not a direct conversion of the PH2 formation-K ratio into an abundance ratio.

Tighter tolerance changes any mole fraction by less than 9.2e-14; the maximum
change in ln(x), including trace species, is below 2.5e-5. At a separate
independent-solver check (reference C/O, 3000 K, 100 bar), identical constants
and budgets in standalone FastChem agree within 5.7e-7 dex at both endpoints.
That comparison is not at the highlighted 2500 K point.

Scope, MCEG and reproduction
----------------------------

The source review and substitution have separate MCEG Subjects/Contexts in
``mceg/contracts/ph2_source_review.yaml``. Numerical attribution and source
sensitivity have recorded support. A preferred replacement with quantified
high-temperature thermal accuracy, and physical atmospheric accuracy, remain
unknown. Condensation, nonideality, quenching and photochemistry are outside
these ideal-gas calculations; neutral-only abundance sensitivity was not run.

The next physical-source decision requires the original modern PH2 energy
calculations, their atomic reference conventions and uncertainty definitions,
and independent scrutiny of the thermal state model. The online table mismatch
and the raw-NASA abundance response alone do not justify a default change.

The offline packets are
``tests/data/validation/ph2_source_review/20261102`` and
``tests/data/validation/ph2_equilibrium_impact/20261102``. Their READMEs give
reproduction commands and source/inspection provenance. Tests run separately
from the package unit suite::

   PYTHONPATH=src pytest tests/model_evidence/ph2_source_review_test.py \
     tests/model_evidence/ph2_equilibrium_impact_test.py

MCEG is optional and is not imported by numerical replay or ordinary tests.
