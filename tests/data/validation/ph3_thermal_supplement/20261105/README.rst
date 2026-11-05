PH3 electronic-supplement follow-up
===================================

This separate packet preserves the original PH3 source review unchanged.
``sources.json`` pins the seven public publisher archives and their ASCII
members, records their dimensions, and describes exact-token extraction.
``moments.csv`` retains selected published row labels, without assigning a
physical J offset; ``calorics.csv`` retains the same temperature nodes.
Complete archives and the author-hosted article were inspected externally;
only the small numerical extracts required by this audit are redistributed.

``comparison.json`` checks nonnegative moment variance, monotonic cumulative
second moments, the direct 296 K partition sum, and dH/dT = T dS/dT = Cp.
Five significant digits give conservative source-rounding bounds. Two finite
difference steps assess numerical sensitivity, not a rigorous truncation or
physical error bound. The exact moment inequalities do not depend on finite
differences, energy-zero shifts, nuclear-spin normalization or translation.

The retrieved files fail these checks under their stated interpretation. This
does not establish an error in the underlying method or every result in the
paper. No repaired moments, fitted thermal candidate or equilibrium response
is asserted. Reproduce with:

.. code-block:: bash

   python mceg/reviews/ph3_thermal_supplement_review.py --output results/ph3_supplement.json
   pytest tests/model_evidence/ph3_thermal_supplement_review_test.py
