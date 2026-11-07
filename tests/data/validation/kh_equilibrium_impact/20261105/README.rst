KH finite-substitution evidence (2026-11-05)
===========================================

The actual curated-oh-v1 networks are checked against every saved species,
formula entry, potential and budget before reusing their 150 baseline
endpoints. There are 150 fresh KH-only candidate endpoints, 12 cold tighter
controls, 10 interpolation-path solves and six NASA thermal-reference
controls. Two ion-inclusive conditions have independent FastChem current/
candidate comparisons. No production data change.

``receipt.json`` is copied unchanged from the successful run, including its
actual command and FastChem executable hash. Results were generated with
JAX_PLATFORMS=cpu, OMP_NUM_THREADS=1, OPENBLAS_NUM_THREADS=1, MKL_NUM_THREADS=1,
PYTHONPATH=src, and taskset -c 4-7. CPU JAX reports an unavailable CUDA-plugin
warning on this host; its explicit CPU run completes successfully.

``diagnostics.json`` records the 150 new endpoints, while
``selected_diagnostics.json`` records the additional controls. The NPZ
files preserve complete logarithmic amounts, potentials, inventories and
raw/corrected same-model Gibbs comparisons. Primary/tighter tolerances are
5e-14/2e-14. Numerical limits and the limited thermal-control coverage are
explicit in documents/mceg/kh_candidate.rst; solver agreement is not a
physical uncertainty interval or atmospheric validation.

Reproduce into an unused directory with the command in ``receipt.json``.
The frozen evidence replays offline without FastChem via
``pytest tests/model_evidence/kh_equilibrium_impact_test.py``.
