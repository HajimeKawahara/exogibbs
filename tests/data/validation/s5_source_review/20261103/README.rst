S5 source review, 2026-11-03
===========================

``sources.json`` identifies inspected and missing sources, their precise access
scope and the hashes of previously archived inputs. ``janaf_s5_nodes.json`` is
a factual transcription of eight official HTML rows, retaining the inconsistent
298 K free-energy-function cell. It is not a downloaded raw table.
``nasa-s5.txt`` contains only the public NASA report's eight-line S5 coefficient
block, checked visually against printed page 184. No full paper is committed.

``comparison.json`` reproduces the six-temperature native-source decomposition,
four frozen comparison nodes, NASA header check, unused-column audit and S5
footprint extracted from the existing 75-case ion-inclusive archive. No new
equilibrium solve or curated replacement is performed. ``receipt.json`` pins
the replay dependencies and packet. Literature gaps and physical accuracy
remain explicit in the canonical MCEG contract and investigation document.

Run from the repository root::

   python mceg/reviews/s5_source_review.py
   pytest -q tests/model_evidence/s5_source_review_test.py
