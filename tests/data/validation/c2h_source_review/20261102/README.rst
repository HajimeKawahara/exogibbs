C2H source investigation, 2026-11-02
===================================

This snapshot supplements baseline-001 without changing it. The three small
JANAF text tables are fresh official downloads. The NASA excerpt preserves
selected printed records; ``sources.json`` distinguishes its extraction hash
from the original PDF hash. Large PDFs and web pages are acquisition references,
not offline replay dependencies. ``literature.json`` records inspected material,
searches, citation ancestry and access gaps.

``comparison.json`` and ``receipt.json`` archive a fresh four-node reaction
decomposition. Replay with ``python mceg/reviews/c2h_source_review.py --output
results/c2h_review/comparison.json``. Numerical source disagreement is separate
from physical accuracy, production adoption and abundance effects. No equilibrium
substitution was run.
