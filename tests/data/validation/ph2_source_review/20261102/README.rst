PH2 source investigation, 2026-11-02
==================================

This packet distinguishes the printed 1998 JANAF PH2 table, the current official
online H-079 record, the unchanged FastChem/curated fit, and frozen NASA records.
It does not adopt new production coefficients. See ``literature.json`` for
identity, source ancestry, inspected material, reference conventions and access
limits. ``sources.json`` pins the selected source extracts by SHA256.

The original scanned PH2 page is extracted from the official JANAF PDF; five
visually inspected rows are transcribed separately. Official online tables
include PH2, H, P(g), H2(ref) and P(ref). NASA PH2/P/H coefficients extracted
from the official report are checked against the frozen baseline coefficients.
Later source extracts are leads with explicitly recorded reference-state and
inspection limits; no unverified white/red phosphorus conversion is applied.

``comparison.json`` reproduces the baseline source comparison and decomposes
reaction differences into energy, thermal and atomic terms. It also records
the online formation-column signature (1.5 H2, no P), correct printed-table
closure against P(ref)+H2, and current-fit agreement with printed JANAF over
26 nodes from 1500 to 4000 K. Numerical agreement does not recover the original
fit-generation procedure or establish physical accuracy. ``receipt.json``
identifies a fresh offline execution with input/output hashes, rather than
relabeling historical baseline output as a new run.

Replay without overwriting the archive::

   PYTHONPATH=src python mceg/reviews/ph2_source_review.py \
     --output results/ph2-source-replay/comparison.json \
     --receipt results/ph2-source-replay/receipt.json

The script performs no downloads. A new receipt records its actual execution
time and paths; compare numerical content instead of expecting identical receipt
bytes. Tests check the frozen source/receipt hashes and numerical reconstruction::

   PYTHONPATH=src pytest tests/model_evidence/ph2_source_review_test.py

The separate ``ph2_equilibrium_impact/20261102`` archive measures a raw-NASA
PH2-only substitution against the current C2H-corrected curated model. Its
solver evidence must not be substituted for source or atmospheric validation.
