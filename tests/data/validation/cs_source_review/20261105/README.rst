CS source audit, 2026-11-05
==========================

Offline replay::

   PYTHONPATH=src python mceg/reviews/cs_source_review.py
   PYTHONPATH=src pytest tests/model_evidence/cs_source_review_test.py

``states.npz`` keeps energy, degeneracy, J and v for every current JnK state.
``state_inventory.json`` retains actual and declared limits and selected
official PF values. ``sources.json`` pins the original downloads and records
their URLs. ``literature.json`` separates inspected primary text, official
displayed values and incomplete inspections. ``comparison.json`` is the
replay result; ``receipt.json`` pins all selected inputs and the result.

Original large files and publisher papers are not redistributed. Download
the original JnK states.bz2, pf and def.json into a separate directory under
the names in ``sources.json``; then regenerate the compact packet with::

   PYTHONPATH=src python mceg/reviews/cs_source_review.py --extract RAW_DIRECTORY --packet OUTPUT_DIRECTORY

Extraction rejects changed original hashes. It performs no network access.
The default replay is also fully offline. Above 3000 K the finite state sum
is a diagnostic beyond the published JnK range, not a validated extrapolation.
The Te-only electronic model is an illustrative sensitivity, not a replacement.
The 75-case footprint reuses the saved current model and runs no new solve.
