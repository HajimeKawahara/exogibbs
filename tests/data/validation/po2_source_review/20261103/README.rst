PO2 source assessment, 2026-11-03
================================

This packet pins the initial neutral PO2 energy/thermal review for #308/#309.
The four public JANAF text tables are fresh downloads; P-001 is the elemental
reference phase and is not used as a gas atom. Numerical replay uses PO2
O-032, P-008 and O-001, existing baseline NASA9 records and the unchanged
FastChem fit. ``nasa9_excerpt.txt`` is a compact factual transcription of the
official NASA report's neutral PO2/P/O records, independently compared in tests.

``sources.json`` records URLs, download hashes, archival and inspection status.
``literature.json`` records identity, source hierarchy, inspected factual values,
reference conventions and access gaps. Full PDFs and HTML are not vendored.
A hash for a nonarchived document identifies the inspected copy; it is not
claimed to be an offline artifact. ``comparison.json`` and ``receipt.json``
freeze the offline replay and the existing 75-case phosphorus footprint.

Run from the repository root::

   PYTHONPATH=src python mceg/reviews/po2_source_review.py --output results/po2-review.json
   PYTHONPATH=src pytest tests/model_evidence/po2_source_review_test.py

The source differences are model disagreements, not physical error bars.
H298 alignment is diagnostic only. No production change, new equilibrium
solve or candidate substitution is performed. Supported source attribution and
conditional weaker-binding direction are distinct from unknown complete-model
accuracy in ``mceg/contracts/po2_source_review.yaml``. See the accompanying
``documents/mceg/po2_source_review.rst`` for scope and follow-ups.
