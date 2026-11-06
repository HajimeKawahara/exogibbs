TiO default adoption, 2026-11-03
================================

The installed curated factory now selects curated-tio-v1: the frozen TOTO/Jain
TiO candidate on the unchanged curated-c2h-v1 network and elemental budgets.
The prior curated_c2h and curated_v1 implementations and data remain unchanged.

The receipt records 150 new cold solves using the production entry point:
75 ion-inclusive and 75 neutral cases at 1500--4000 K, 1e-6--100 bar and
three C/O budgets. Both modes match the candidate chemical potentials;
ion-inclusive solutions also reproduce its archived equilibrium endpoints.
All species and constraints are retained without a reporting floor.

Recompute to a new, ignored output directory::

   PYTHONPATH=src JAX_PLATFORMS=cpu python mceg/reviews/curated_tio_adoption.py \
     --output-dir results/tio-production-adoption-new

``ions.npz`` and ``neutral.npz`` contain the complete executions. Their hashes,
selected implementation hashes and runtime versions are in ``receipt.json``.
These are numerical checks, not quantified atmospheric accuracy.

``source/`` preserves exact pre-adoption source bytes referenced by historical
receipts. The receipts and old numerical results have not been rewritten.
Use the live, explicitly versioned review scripts to repeat those studies;
the source snapshots document what originally ran, not a standalone environment.
The new production receipt checks current files directly and never resolves
through this historical archive.
