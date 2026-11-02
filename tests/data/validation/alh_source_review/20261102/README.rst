AlH source investigation, 2026-11-02
==================================

This packet audits neutral gas-phase AlH under the unchanged current curated
model. ``literature.json`` identifies the original evaluations, inspected
material, quotation chains and access gaps. ``sources.json`` pins archived
source bytes. Printed JANAF nodes, original NASA records, and an ExoMol AloHa
partition-function file are included for offline reconstruction.

``comparison.json`` reproduces the frozen four-node source comparison and
separates energy, thermal and atomic-reference contributions. It checks
printed/online JANAF agreement, elemental-reference closure, the current fit
and its caloric derivatives, NASA interval metadata, and inferred internal
partition functions. NASA polynomial interval evaluation is not extrapolated:
the official record and frozen baseline start at 200 K; the older major-molecule
fixture's 300 K lower bound is recorded as a separate metadata difference.

The reaction comparison uses the frozen baseline gas constant. Inferring
internal Q uses each source's gas constant (JANAF 8.31441, NASA 8.31451 J/mol/K)
and an explicit common mass and pressure convention. ExoMol Q includes a
nuclear-spin factor of 12 for 27AlH, which is removed before comparing it with
thermochemical Q. A five-point derivative diagnostic uses 10, 20 and 40 K steps;
its spread measures numerical stability, not physical uncertainty. The X/A
bound and quasi-bound model does not establish full electronic completeness.

``dissociation_evidence.json`` records De/D0 labels and a conditional zero-point
and fine-structure interpretation. A value near 3.16 eV must not be directly
adopted as D0. The evidence conditionally favours a NASA-like energy over the
old JANAF estimate, without certifying an exact replacement or an entire
NASA thermal model. The original 1979 full-text dissociation limit and the
Gurvich re-evaluation remain access gaps. No production coefficients change.

``receipt.json`` records a fresh offline execution, selected implementation
and source hashes, output hash, command and environment. Replay into a new
directory rather than overwriting this archive::

   PYTHONPATH=src python mceg/reviews/alh_source_review.py \
     --output results/alh-source-replay/comparison.json \
     --receipt results/alh-source-replay/receipt.json

The replay performs no downloads. Its receipt has a new time and actual paths;
numerical results are compared independently of byte-identical frozen receipts.
Run the regression and source-integrity checks with::

   PYTHONPATH=src pytest tests/model_evidence/alh_source_review_test.py

The separate ``alh_equilibrium_impact/20261102`` archive contains a 75-pair
raw-NASA AlH-only substitution and the saved priority screen. Solver agreement
is evidence for numerical sensitivity, not physical atmospheric validation.
