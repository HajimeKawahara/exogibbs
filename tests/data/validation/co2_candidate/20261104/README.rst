Bounded NIST CO2 candidate, 2026-11-04
=====================================

The primary candidate uses 626 internal thermodynamics, NASA mass 44.0095 u,
unchanged ATcT 1.172 atomization and NASA atomic thermal functions at 1 bar.
It is restricted to 1500-4000 K. The isotope-average function is an explicitly
limited thermal-reference diagnostic, not natural-isotope chemistry.

``candidate.json`` contains C2 quintic Gibbs coefficients. ``nodes.json``
and ``off_knot_reference.json`` preserve the source thermodynamics used at
11 knots and 90 withheld nodes. ``nist_samples.json`` retains the actual
source rows from all 12 isotope tables so the arithmetic and conventions
remain independently inspectable offline. Original table hashes are pinned;
full original data and papers remain outside git.

``validation.json`` reports interpolation errors. ``scientific_scope.json``
records isotope entropy conventions, actual conservation/precision checks,
and independent peer checks against the full 2501-row source and analytic
formation derivatives. The whole-table check is acquisition-time evidence;
the compact offline checks use the retained independent source rows.
``receipt.json`` pins the candidate inputs and outputs.

Rebuild from previously downloaded official tables::

   PYTHONPATH=src python mceg/reviews/co2_candidate.py --extract /path/to/originals --output-dir /tmp/co2-candidate

Evaluate the shipped candidate and validation offline::

   PYTHONPATH=src python mceg/reviews/co2_candidate.py
   PYTHONPATH=src pytest tests/model_evidence/co2_candidate_test.py

The source c2 and internal moments retain their original conventions; common
modern R and reconstructed translation are explicit. No production change,
isotope D0 model or complete physical uncertainty is asserted.
