CN source review (2026-11-06)
=============================

Decision and molecular evidence
-------------------------------

The recommended next calculation is a **same-thermal ATcT energy candidate**.
The inherited JANAF atomization energy is 750.156 kJ/mol; the same-version
ATcT 1.228 balance C + N - CN gives 745.355 kJ/mol. The 4.801 kJ/mol decrease
is scientifically material even though current/NASA discrepancies are moderate.
It changes K by factors 0.68048, 0.82491 and 0.86558 at 1500, 3000 and 4000 K,
respectively, before any equilibrium solve. Keep curated unchanged in this source
PR; construct the candidate and run fixed-network substitutions before adoption.

The published Trihybrid model (`Syme and McKemmish 2021
<https://doi.org/10.1093/mnras/stab1551>`_) supplies three electronic manifolds,
X2Sigma+, A2Pi and B2Sigma+. The downloaded 28,004 levels start at zero,
reach J = 120.5, and have total weights 3(2J+1). Divide Q by the 14N nuclear-spin
factor 3; electron spin is already represented and must remain. Heteronuclear
rotational symmetry is one. No extra electronic or symmetry divisor is applied.
Energies are measured from the lowest rovibronic level, so adding ZPE to the
ATcT ground-level atomization energy would double count it.

The replay computes Q and the first two energy moments, hence H-H0, S and Cp.
At 4000 K the spin-normalized Q is 6348.6565, 0.0575% below NASA. At the same
translation mass, H-H0, S and Cp differ by -29.9 J/mol, -0.0123 J/mol/K and
-0.1019 J/mol/K. At 1500 K the Cp difference is +0.0504 J/mol/K. The published
PF and direct sum agree within 0.7 parts per million, consistent with rounding.
This is appreciably stronger thermal evidence than comparing Q alone.

Completeness is still scoped. At 4000 K, levels above 40,000 cm-1 contribute
0.00190% of Q, while J > 100 contributes 0.1531%; excited electronic states
contribute 8.457%. These internal tail diagnostics do not bound electronic
manifolds omitted by the three-state model. The NASA thermal model is therefore
well supported as an energy-only candidate basis within 1500-4000 K, without
claiming a complete uncertainty or authorizing replacement by a new state-sum
production fit. The ExoMol definition lists a natural-element-like mass of
26.0174 u despite its 12C14N identity; the isotope translation proxy instead
uses 26.00307400443 u. This approximately 0.083% translation effect is kept
separate from the internal partition comparison.

Source comparison and normalization
-----------------------------------

The current ``curated-cos-v1`` entry is the inherited FastChem JANAF 1998 fit,
marked unreviewed in the source screening. Its literal coefficients and exact
molecular identity are checked against the named production preset. Official
JANAF molecular/atomic tables and selected NASA CEA records are pinned. NASA
uses native R = 8.314510 J/mol/K; JANAF and current fits use
R = 8.31446261815324. Both use one bar. The official C, CN and CH NASA lower
bound is 300 K; this review does not copy the modified screening file's
200 K extension. Selected polynomial coefficients do agree exactly.

NASA H(298)-H(0) and JANAF H(T)-H(298) produce compatible fixed-reference
zero-energy offsets. Their individual offsets are not relabelled as physical
formation enthalpies at 0 K; only the **atom-balanced difference** is a D0.
ATcT molecule and atoms all come from version 1.228. Marginal network
uncertainties are not combined as independent errors because covariance is
unavailable. Agreement with ATcT inputs is not independent physical validation.

At exact 300-4000 K source knots, the replay decomposes JANAF minus NASA ln K
into balanced energy, molecular thermal and atomic thermal contributions.
Current fits are evaluated only at 1500-4000 K. The following dimensionless
factors are source diagnostics, not abundance responses:

.. list-table::
   :header-rows: 1

   * - T (K)
     - Current / NASA K
     - Energy-aligned JANAF / NASA K
     - ATcT energy-only / current K
   * - 1500
     - 1.330400
     - 0.998949
     - 0.680483
   * - 3000
     - 1.153278
     - 0.999021
     - 0.824914
   * - 4000
     - 1.112192
     - 0.998902
     - 0.865579

The last column means an additive D0 shift applied to the inherited current
thermal function; it is an algebraic diagnostic, not a saved candidate model.
The fit/source residual is retained separately, rather than interpreting it
as an atomic thermochemical measurement. Principal-isotope Q is compared with
NASA at a common translation mass, with the isotope translation effect also
reported separately. Neither comparison is a natural-isotope mixture model.

Importance, reproducibility and limits
--------------------------------------

The frozen 75-case, 537-species, ion-inclusive ``curated-cos-v1`` grid places
at most 8.7374% of N in CN, at
3000 K, 0.01 bar and C/O = 1.5.
Its maximum mole fraction is 6.54726e-06. Archived target
potentials match the current literal fit. These endpoints establish investigation
priority; they are not a new substitution calculation, universal importance
bound, or validation of photochemical/cool atmospheres.

Replay offline with::

   python mceg/reviews/cn_source_review.py
   pytest tests/model_evidence/carbon_radical_source_review_test.py

The packet is ``tests/data/validation/cn_source_review/20261106``.
It contains source URLs/hashes, exact selected NASA records, JANAF and PF tables,
selected state fields where used, input facts, numerical output and dependency
receipt. PDFs are not committed. MCEG records source attribution, normalized
comparisons and the scoped next-step decision as supported; full physical
accuracy and complete historical source recovery remain unknown. The existing
source's literature remains an MCEG provenance matter. Recovering every original
Gurvich/JANAF antecedent or FastChem fit weight is not a gate for this evidence PR.
No runtime model, curated payload or equilibrium result is changed.

Primary evaluated sources include the `JANAF database
<https://janaf.nist.gov/>`_, `pinned official NASA CEA coefficients
<https://github.com/nasa/cea/blob/2f79a647e85737742fb08a127da00d1ea1d069da/data/thermo.inp>`_
and `ATcT 1.228 <https://atct.anl.gov/Thermochemical%20Data/version%201.228/index.php>`_.
Individual access limits and the inspected modern-paper DOI are in ``sources.json``.
