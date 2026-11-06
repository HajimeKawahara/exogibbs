SO2 source review, 2026-11-04
=============================

Offline packet for the scoped retention of NASA thermal functions with ATcT
1.172 balanced energy. No production replacement or new equilibrium solve.

``state_samples.json`` records moments and explicit cutoff sums of all
3,270,270 rows in the pinned current ExoAmes distribution. ``exoames.pf``
and ``exoames.def.json`` preserve the small original files byte for byte.
``tips_tables.json`` preserves literal 32S16O2 arrays from the pinned official
HAPI source. ``comparison.json`` contains source construction, normalized
G/H/S/Cp/Q/K diagnostics and the current 75-case sulphur footprint.

The full PDFs, HAPI source and 51 MB compressed state file are not vendored.
``sources.json`` records their acquisition, URLs and SHA-256 values;
``literature.json`` distinguishes inspected methods from unresolved historical
provenance and distribution claims. Extraction commands and all limitations
are documented in ``documents/mceg/so2_source_review.rst``. Compact replay
checks the recorded moments, while re-extraction checks the original states.
Above 2000 K ExoAmes results are completeness diagnostics outside its
published PF range. TIPS calorics are derivatives of rounded Q, not published
caloric measurements. A one-bar, representative-isotope normalization is
shared; the natural-mass proxy is not full isotope balance.
