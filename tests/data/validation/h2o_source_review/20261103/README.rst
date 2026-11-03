H2O source-review packet (2026-11-03)
=====================================

This offline packet retains selected published table values and derived
full-level summaries, not full papers or the large original state lists.
``sources.json`` pins original URLs, dates, byte counts and SHA256 hashes.
``literature.json`` records inspected scope, conventions and unresolved work.
``nasa9_excerpt.txt`` contains the exact official H2O/H/O coefficient blocks.
``state_samples.json`` summarizes every level in the two hashed state files.
``published_samples.json`` preserves original selected numeric rows and column
names, including the erroneous kJ/mol labels on numeric J/mol enthalpies.
``comparison.json`` separates the current balanced ATcT energy, NASA thermal
shape, normalized modern comparisons and the saved current-default footprint.

Replay from the repository root (no network or MCEG needed)::

   PYTHONPATH=src python mceg/reviews/h2o_source_review.py
   PYTHONPATH=src pytest tests/model_evidence/h2o_source_review_test.py

To independently regenerate the summaries, download the six inputs named in
``extract_sources`` using ``sources.json`` into one directory. The extractor
verifies their hashes before reading any level data::

   PYTHONPATH=src python mceg/reviews/h2o_source_review.py --extract /path/to/originals --output /tmp/h2o-extracted

ExoMol columns are energy, total degeneracy and J (columns 2, 3, 4 in the
one-based format); PW2024 columns are energy, J, O/P and energy uncertainty.
Raw weights are (2J+1) times spin 3 or 1. The frozen moments contain Q, mean
energy/k and energy variance/k^2 at ten temperatures from 298.15 to 4000 K.
All supplied discrete levels are summed; a first-H+OH-threshold cut is only
a sensitivity diagnostic. Exact temperature columns select published rows;
298.15 K is not fabricated for integer-node ExoMol PF/Cp tables. The 2025
1 K row additionally diagnoses the enthalpy unit scale.

The 2025 table and direct PW2024 sums are distinct calculations. Their
4000 K Cp difference exceeds the published Cp uncertainty; full scattering
model reconstruction is not claimed. Q/4 and translational normalization,
published entropy's already removed spin factor, original NASA R and the
first versus total atomization thresholds are kept explicit. Natural mass
translation does not construct an exact natural-isotope mixture.

The archived 75-case grid is curated-tio-v1, not an older default. No new
equilibrium substitution or production adoption is performed. The receipt
hashes selected real inputs and comparison output; documentation-index and
default-alias edits are deliberately not evidence dependencies.
