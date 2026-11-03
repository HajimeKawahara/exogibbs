N3 source-review evidence, 2026-11-03
====================================

This packet preserves official JANAF numerical tables, a NASA report
coefficient excerpt, factual literature notes and an offline replay.
``sources.json`` records inspected locations and download hashes; full
research papers remain outside the repository. The ATcT 1.172 facts were
inspected on its official pages, with no claim of full-page byte archival.

``comparison.json`` balances 3 N(g) -> N3(g) at 1 bar, decomposes six
JANAF/NASA temperature nodes, compares the current fit at 26 exact nodes,
and reuses the existing 75-case equilibrium archive. ``receipt.json`` pins
inputs and outputs. No new equilibrium solve or curated adoption is made.

Run from the repository root::

   PYTHONPATH=src python mceg/reviews/n3_source_review.py
   PYTHONPATH=src pytest tests/model_evidence/n3_source_review_test.py

Energy alignment and ATcT D0 comparisons are diagnostics. Incomplete original
source access, hot thermal/state completeness and physical prediction accuracy
remain explicit in the MCEG contract and English review.
