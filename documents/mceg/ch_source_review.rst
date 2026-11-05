CH source review (2026-11-06)
=============================

Decision and molecular evidence
-------------------------------

The recommended next calculation is an **ATcT energy-only candidate**, retaining
the current thermal function initially. The JANAF atomization energy is
336.538 kJ/mol; ATcT 1.228 C + H - CH gives 334.443 kJ/mol, a decrease of
2.095 kJ/mol. The corresponding K factors are 0.84537, 0.91944 and 0.93895
at 1500, 3000 and 4000 K. Curated stays unchanged pending fixed-network
substitution and a comparison of thermal alternatives.

`Masseron et al. 2014 <https://doi.org/10.1051/0004-6361/201423956>`_,
section 4.3, reports D0 = 27960 +/- 10 cm-1 (334.476 +/- 0.120 kJ/mol),
consistent with the balanced ATcT value at the stated precision. This is a
physical ground-level dissociation threshold, not an electronic potential De;
no ZPE is added. Its partition calculation includes the low-lying quartet
alongside the observed doublet states. The printed quadratic ln Q fit is
replayed only over the inspected 1000-4000 K portion of the plotted range.
Its weights exclude nuclear spin and retain electronic/angular degeneracies.

The current MoLLIST PF download, spanning 1000-8000 K, divided by the proton
spin factor 2, agrees with that printed quadratic within rounding at every
reviewed knot. At 1500 K it gives Q/NASA = 1.04349; at 4000 K, 0.98983.
A derivative check prevents overinterpreting the latter agreement: differentiating
the printed fit gives H-H0 about 4.796 kJ/mol lower and Cp about
2.408 J/mol/K lower than NASA at 4000 K. These are **analytic-fit diagnostics**,
not independently verified caloric measurements. The 4000 K energy-aligned
JANAF/NASA K ratio is 1.04200, so thermal choice is also material.

The downloaded MoLLIST **states artifact is not used as a thermodynamic sum**.
Its 2680 actual rows differ from the definition's 2681; the minimum row labelled
B2Sigma- is 14.27603 cm-1 and no quartet label occurs. This conflicts with a
common ground-energy interpretation (the 2014 paper places B, v=0 near
25712.5 cm-1). We archive numerical fields and inventory without relabelling
or adjusting zeros. This artifact mismatch does not establish an error in the
published PF, the original spectroscopy, or the paper's partition calculation.

`Sokolov et al. 2025 <https://doi.org/10.1093/mnras/staf1563>`_, section 2.2,
uses thirteen electronic states and identifies quartet population above roughly
3000 K. Its description of a previously patched MoLLIST PF is recorded as a
different source construction; it is not imposed on this inspected download,
which matches Masseron's formula. Their bound/quasibound/continuum distinction
also matters: a photodissociation calculation's discretized continuum cannot
simply be added to a bound-molecule equilibrium partition function. The ATcT
quartet-doublet gap is 72.049 kJ/mol, an identity check rather than a complete
quartet partition model. The precise next thermal task is to obtain or reproduce
that model's bound-state Q, H, S and Cp with an explicit cutoff and energy zero.
No additional full text is mandatory for merging this bounded source review.

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
     - 1.310575
     - 0.998931
     - 0.845370
   * - 3000
     - 1.176325
     - 1.031806
     - 0.919440
   * - 4000
     - 1.151274
     - 1.042003
     - 0.938950

The last column means an additive D0 shift applied to the inherited current
thermal function; it is an algebraic diagnostic, not a saved candidate model.
The fit/source residual is retained separately, rather than interpreting it
as an atomic thermochemical measurement. Principal-isotope Q is compared with
NASA at a common translation mass, with the isotope translation effect also
reported separately. Neither comparison is a natural-isotope mixture model.

Importance, reproducibility and limits
--------------------------------------

The frozen 75-case, 537-species, ion-inclusive ``curated-cos-v1`` grid places
at most 1.8837% of C in CH, at
4000 K, 1 bar and C/O = 1.5.
Its maximum mole fraction is 1.53077e-05. Archived target
potentials match the current literal fit. These endpoints establish investigation
priority; they are not a new substitution calculation, universal importance
bound, or validation of photochemical/cool atmospheres.

Replay offline with::

   python mceg/reviews/ch_source_review.py
   pytest tests/model_evidence/carbon_radical_source_review_test.py

The packet is ``tests/data/validation/ch_source_review/20261106``.
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
