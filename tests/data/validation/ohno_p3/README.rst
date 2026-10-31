Ohno-style phosphorus comparison receipt
========================================

``summary.json.gz`` contains computed phosphorus number-fraction curves,
P-atom allocations, numerical checks and input hashes for the 2026-10-31
study. Read with Python's gzip and json modules. It does not contain source
thermochemical coefficients or sampled equilibrium-constant tables.

The two native manifests record the pinned public GGChem revision, source
priority, final P records, compilation inputs and diagnostic checks. Full
primitive arrays and source downloads remain generated files under
``results/ohno_p3``. Reproduction instructions are in
``documents/mceg/ohno_p3.rst`` and the issue's artifact-branch report.

All calculations exclude ions, condensation and rainout. The sampled
200--3000 K interval is not a physical-validity guarantee for every source
record. The exact original inputs behind Ohno's supplied slide are unknown.
