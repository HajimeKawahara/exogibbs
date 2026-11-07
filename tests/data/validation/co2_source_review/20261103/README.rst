CO2 source review, 2026-11-03
=============================

This packet preserves the NIST 2021 dataset described by Tashkun and Harvey
(2025), current UCL-4000/Dozen state moments, the existing ATcT/NASA hybrid
and its saved gas-grid footprint. No production or equilibrium data change.

``source_nodes.json`` contains eight selected nodes from all 12 official
NIST isotope tables and the natural table, five-point derivative windows,
complete state-sum moments from both ExoMol states files, independently
published PF/Cp nodes and restricted UCL sums. Published data and derived
summaries are identified separately. Original files are hashed in
``sources.json``; large states and full papers are not committed.
``literature.json`` records source inspection, conventions, uncertainty,
shared inputs and access gaps. ``comparison.json`` is the offline replay;
``receipt.json`` pins actual inputs and output.

Regenerate the compact extraction from originals downloaded using the URLs
in ``sources.json`` (no automatic network access)::

   PYTHONPATH=src python mceg/reviews/co2_source_review.py \
     --extract /path/to/originals --output /tmp/source_nodes.json

Replay using only repository files::

   PYTHONPATH=src python mceg/reviews/co2_source_review.py \
     --output /tmp/co2_comparison.json
   PYTHONPATH=src pytest tests/model_evidence/co2_source_review_test.py

The printed NIST equation (5) sign is checked against independent published
Q/H/S/Cp identities. The distributed UCL PF/Cp matches a 26000 cm^-1 cutoff,
not the full current states file; its release history is not inferred.
NIST natural entropy is not inserted into the NASA atomic convention.
Same-mass internal-Q proxies are not abundance ratios. The 75-case archive
uses curated-tio-v1 and does not represent all CO2-bearing atmospheres.
