H2S source review packet (2026-11-04)
===================================

``states.npz`` preserves all 220630 current AYT2 states: energy in cm-1,
printed total weight, printed J and original calculated energy. It also stores
complete published PF/Cp tables and only the H2(32S) literal TIPS2017/2021/2025
arrays from the pinned official HAPI file. No state or source metadata anomaly
is silently repaired. ``state_assignment_excerpt.txt`` preserves the original
inconsistent assignment and adjacent lines; ``AYT2.def.json`` and the readme
are byte-identical source downloads. PDFs are cited and hashed, not vendored.

Offline replay::

   PYTHONPATH=src python mceg/reviews/h2s_source_review.py
   pytest tests/model_evidence/h2s_source_review_test.py

Optional acquisition replay requires ``AYT2.states.bz2``, ``AYT2.pf`` and
``AYT2.cp`` from the URLs/hashes in ``sources.json`` and the pinned ``hapi.py``::

   python mceg/reviews/h2s_source_review.py --extract RAW --hapi HAPI --packet OUT

The extractor reads only literal HAPI assignments, never imports that module.
Compare extracted arrays and inventory, not NPZ container timestamps. The
normal replay requires no raw download, HAPI, MCEG or network access.
``receipt.json`` pins inputs and output; the MCEG contract separately records
supported numerical claims and unknown physical accuracy. All archived
75-case potentials and sulfur budgets are reused from curated-h2o-v1; no
new substitution or atmospheric validation was performed.
