C3H source and identity review
=============================

This packet freezes the 2026-11-02 investigation. It does not supply a new
production species or thermochemical replacement. Run from the repository root::

   PYTHONPATH=src python mceg/reviews/c3h_source_review.py
   pytest tests/model_evidence/c3h_source_review_test.py

``inputs.json`` records the visually transcribed Tsuji Table 4 row, its displayed
precision, ATcT molecular/atomic energies and the caloric comparison conventions.
``comparison.json`` is regenerated offline. ``receipt.json`` pins its inputs and
output, including the existing 75-case ``ions.npz``; that grid is not duplicated.

``sources.json`` identifies the acquired originals, SHA256 values, inspected
locations and unsuccessful Tsuji 1964 access. The Tsuji excerpt contains original
PDF pages 2, 6, 7, 10 and 21, corresponding to printed pages 412, 416, 417, 420 and
431. The Burcat excerpt contains original PDF pages 23, 106 and 325, corresponding
to printed pages 19, 102 and 321. Numerical records are retained with their source
annotations. The NASA9 high-temperature a7 is negative in this inspected PDF;
the sign has been checked against the page image.

The ATcT pages are version 1.172 and distinguish CCCH from CH(CC). Their linked
bibliography pages document shared experimental/theoretical ancestry and ATcT's
uncertainty adjustments; they do not replace inspection of the full originals.
CDMS entry 037501 is a partial low-temperature spectroscopy model, not a complete
hot partition function. No entropy, Gibbs-energy or equilibrium-constant
comparison to Burcat is made while its pressure convention remains unchecked.

The numerical review verifies source attribution, analytic caloric identities
and the archived carbon budget. It does not establish the original Tsuji isomer,
physical accuracy, or the response to adopting either modern isomer.
