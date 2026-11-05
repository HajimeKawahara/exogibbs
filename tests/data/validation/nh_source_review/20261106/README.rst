NH source review, 2026-11-06
============================

This archive pins selected NASA/JANAF/ATcT inputs, downloaded finite ExoMol
states, exact partition-table nodes and source-access records. Full papers
and web pages are not committed. ``sources.json`` records full retrieved-file
hashes; ``partition_nodes.json`` retains only the exact reviewed table nodes.

``comparison.json`` is the actual offline output. ``receipt.json`` pins the
replay code, input dependencies, current reaction and archived curated-cos-v1
grid (75 converged cases, 537 species). The finite state sum supplies exact
moments for Q/H/S/Cp and is explicitly not a completeness certification.
No continuous candidate, replacement equilibrium solve or curated change
is included. Source review does not establish full atmospheric accuracy.

Run ``PYTHONPATH=src python mceg/reviews/nh_source_review.py`` from the repository root.
Tests additionally check named production values, source-specific state
normalization and caloric identities.
