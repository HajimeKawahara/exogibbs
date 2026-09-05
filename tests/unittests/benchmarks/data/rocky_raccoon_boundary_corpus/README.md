# Rocky Raccoon warm-boundary corpus

These seven named snapshots are copied byte-for-byte from ExoExamples
`tests/rocky_raccoon/data/`. `manifest.json` pins each fixture and its original
accepted-layer snapshot with SHA-256 hashes. The NPZ also retains source-provider
revisions/hashes, species ordering, exact parent gas logs, and both inventories.
All cases use the example's `paper_extrapolated` validity mode; the final case
enables SiO(s) in the original catalog order.

The provider test reuses the packaged example's network builder and calls public
`regauge_gas_only_warm_start` and rainout `solve_profile`, with the true parent
problem supplied as initializer provenance. It requires caller-gauge KKT and
floorless positive-element conservation, not a particular fallback route or
support ordering. No ExoExamples installation or full-column replay is needed.

The normal ExoGibbs CPU CI discovers this test automatically. The identical
seven solve cases can be checked on CUDA from the repository root with:

```console
env JAX_PLATFORMS=cuda JAX_PLATFORM_NAME=cuda JAX_ENABLE_X64=1 python -m pytest tests/unittests/benchmarks/rocky_raccoon_boundary_corpus_test.py
```

Fixture 1805 was extracted from the accepted layer 1804 and failed candidate
record in ExoExamples' `comparison_oxygen_poor_sio_transition_probe_gpu` run
with ExoGibbs `4bc83e5`; extraction preserved the exact snapshot arrays and
checked that the failed candidate inventory equals the parent's outgoing one.
