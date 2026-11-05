PH3 electronic-supplement follow-up
===================================

Retain the current Lodders-labelled PH3 fit. The electronic tables missing
from the :doc:`original PH3 source review <ph3_source_review>` have now been
retrieved. This separate audit supersedes that access limitation; the original
review, source packet and receipt remain unchanged as historical evidence.
The phosphorus-reference correction is already present in curated.

Recovered electronic supplement and consistency audit
------------------------------------------------------

The public publisher supplements ``mmc8`` through ``mmc14`` were retrieved on
2026-11-05. They contain ``PCP.DAT``, ``PGEF.DAT``, ``PHCF.DAT``, ``PJQ.DAT``,
``PJQP.DAT``, ``PJQPP.DAT`` and ``PS.DAT``. For example, the
`second-moment archive
<https://ars.els-cdn.com/content/image/1-s2.0-S0022407314001289-mmc13.zip>`_
is directly accessible even though the article landing page returned HTTP 403.
Archive/member hashes, exact-token extracts and the extraction method are in
``tests/data/validation/ph3_thermal_supplement/20261105``.

Each partition/moment file has 100 rows, labels 1--100, and 10000 temperature
columns at 1 K intervals. There are no header or separator rows to shift the
data. Appendix A describes cumulative state sums. This audit preserves the
published row labels without assuming whether physical J starts at 0 or 1.
The final-row Q(296 K)=3249.5 agrees with the paper's direct 3249.46 within the
source's five-significant-digit rounding. The printed polynomial is unnecessary
for this direct value, but recovering Q alone does not validate its derivatives.

Under the stated energy-moment interpretation, the second moment decreases
from 15010 to 5437.1 between row labels 41 and 42 at 300 K. Adding nonnegative
state contributions cannot do this; the drop exceeds rounding by more than
9572. The final row has Q=3319.5 and first moment 5285, so
``Q2/Q - (Q1/Q)**2 = -0.896878``. This violates nonnegative energy variance even
after source rounding. The same test fails at 500, 1000, 1500 and 2000 K.
At 300 K it would give total Cp=13.3291 J/mol/K, whereas ``PCP.DAT`` gives
37.306. Nuclear-spin normalization cancels from these ratios; an energy-zero
shift or translational convention cannot repair the moment inequality.

The caloric files also need clarification. At 1 K, the tabulated Cp=20.819
J/mol/K is already close to the translational 5R/2=20.7862, despite Appendix A
describing internal functions. At 3000 K, the tabulated H increment gives
dH/dT=62.000 J/mol/K with both 30 and 60 K five-point stencils; their rounding
bounds are 0.250 and 0.125. The entropy derivative gives T dS/dT=62.000 and
61.958. The supplied Cp is only 57.938. At 4000 K, dH/dT is about 51.18 while
Cp is 41.402. A consistent translation conversion leaves this closure residual
unchanged. Step agreement is a sensitivity check, not a rigorous derivative
error bound; the exact moment-positivity failures need no differentiation.

These findings concern the retrieved distributed files under their documented
interpretation. They do not establish that the underlying state calculations,
method or all published results are wrong. No patched second moment, refitted
printed polynomial, thermal candidate or replacement calculation is accepted
from this packet.

Decision and follow-up
-----------------------

The earlier request to obtain the electronic supplement is complete. The
minimum remaining input is a corrected ``PJQPP.DAT`` with its cumulative-row
definition, and a mutually consistent set of Q, moments and H/S/Cp tables or an
author clarification that resolves the documented discrepancies. An independently
recomputed complete state sum with checked moments is another route. This does
not require recovering every older experimental paper before merging the review.
Then reconstruct H/S/Cp with the same ground zero, spin convention, one-bar
translation and declared isotope convention. Retain
or independently reassess a consistently balanced energy; a phase-reference
conversion is not a new energy measurement. Test a continuous thermal candidate
and one-species fixed-network substitution. Full physical accuracy, older
calorimetry recovery and disequilibrium atmospheric behavior remain unknown.
This source-review PR can be assessed independently of later candidate adoption.

Offline reproduction
--------------------

The MCEG contract ``mceg/contracts/ph3_source_review.yaml`` records the recovered
files and supported consistency-failure claim separately from unknown physical
accuracy. No production potential, species set, abundance or default changes.

.. code-block:: bash

   python mceg/reviews/ph3_thermal_supplement_review.py --output results/ph3_supplement.json
   pytest tests/model_evidence/ph3_thermal_supplement_review_test.py
