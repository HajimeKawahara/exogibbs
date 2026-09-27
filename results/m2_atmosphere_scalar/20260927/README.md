# Independent atmospheric scalar derivatives

The published-liquid 76-gas source attempt at 2173.15 K / 250 bar reported
successful scalar minimization but failed the final independent derivative
audit. Its trace Ti carrier is about 2e-11 of atmospheric H. Finite differences
of the whole atmospheric energy lost trace contributions; the old reported
maximum derivative error was 1.3939034e-4 RT.

This archive **does not accept or rename that source result**. It freshly
solves only the atmospheric parcel at the preserved final atom allocation.
The explicit primitive gas/cloud scalar is differentiated with JAX AD,
then its gas and active-condensate gradients are projected onto the
full-rank elemental formula system. This independent envelope derivative
uses no callback chemical potential or saved elemental dual.

The maximum difference from the callback is **7.46e-14 RT** and the scalar
relative difference is **-4.39e-16**. The existing outer 5e-6 RT derivative
threshold, elemental budgets, reference energies and phase selection remain
unchanged. A fresh whole-source and pressure solve is still required.

At source 89a6522, all 17 atmosphere tests and 81 affected finite-gas,
contact, common-Gibbs and metal-selection tests passed. They include both
trace-potential and scalar-energy fault injection. The documentation build
succeeded. The duplicated whole-suite process was intentionally stopped;
its original partial log and reason are kept, separately from the complete
base-stack suite reported in the finite-condensate archive.

To reproduce this **parcel-only** result, use JAX CPU/x64 and this checkout's
`src` on PYTHONPATH, then run `python recheck_parcel.py --output /path/to/new.json`.
The input preserves the original failed-source SHA256 and all thirteen
atmospheric atom amounts. `parcel_audit.json` contains the first actual fresh
parcel, independently evaluated scalar and gradient, elapsed time and exact
helper hash. No native property evaluator is called.
