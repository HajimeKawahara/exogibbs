C4 source-review packet
=======================

``sources.json`` records source access, selected facts and unresolved tasks.
``janaf_c4_nodes.json`` transcribes nine official JANAF nodes; null denotes
the infinite unused free-energy function at 0 K. ``nasa-c4.txt`` is the
selected official NASA record. Existing C-atom tables/headers and the frozen
comparison/grid archive are referenced by hash rather than copied.

``comparison.json`` replays energy/thermal decomposition, native 0 K anchors,
a limited electronic-weight diagnostic and an existing-grid footprint.
``receipt.json`` pins the inputs and artifacts. Full papers remain outside
the repository. No candidate adoption, new equilibrium solve or physical
error bound is implied.

From the repository root::

   python mceg/reviews/c4_source_review.py
   pytest -q tests/model_evidence/c4_source_review_test.py
