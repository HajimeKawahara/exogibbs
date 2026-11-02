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
and fine-structure interpretation, updated from the supplied 1979 full text. A value near 3.16 eV must not be directly
adopted as D0. The evidence conditionally favours a NASA-like energy over the
old JANAF estimate, without certifying an exact replacement or an entire
NASA thermal model. Both the printed abstract and body give De=3.16 eV,
resolving the registered-abstract value 3.17 eV. The X asymptote already
incorporates the stated Al fine-structure offset; no extra term-center shift
is applied. The earlier rounded two-term zero-point diagnostic is preserved.
The supplied Szajna (2011) Tables II and IV add a five-term check:
ZPE=0.1033973143 eV and diagnostic D0=3.0566026857 eV under an explicit
zero-Y00 convention. The k=3--5 contribution is only 3.83e-6 eV, and printed
v=0--5 spacings are reproduced within 0.0003 cm^-1. Y50 was fixed in the
original fit; covariance, its uncertainty and the absolute potential zero are
not determined here. No calibrated D0 uncertainty follows from the coefficient
errors. Gurvich/Luo source access, reference reconciliation and thermal
completeness remain open. The original +/-0.01 eV estimate has no stated
confidence level. No production coefficients change.

The user-provided ``Baltayan_1979.pdf`` remains local under
``ref/refs_exogibbs_202611``. Its SHA-256, size and inspected page locations
are recorded in ``sources.json`` and ``literature.json``; the full article
and extracted text are not redistributed or needed for offline replay.
The supplied ``2011SzajnaAlHActaPhysPolA.pdf`` and ``Yurchenko_2024.pdf``
remain under ``ref/refs_exogibbs_202611-2`` with the same policy. The latter
is the AloHa publisher article, not the methane paper. Its state limits,
shared 47 Szajna transitions and constrained WYLLoT X-state potential are
recorded separately from the earlier preprint inspection. This update does
not run a new full line-list state sum or equilibrium substitution.

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
