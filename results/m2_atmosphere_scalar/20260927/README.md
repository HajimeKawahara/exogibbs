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

## Fresh metal-free source history

The following original JSON files are copied without editing their bytes.
`source_history_manifest.json` records their hashes and original paths. All
three use 76 gases at 2173.15 K / 250 bar. The later source re-solves consume
an unaccepted prior numerical candidate as a starting vector, with fresh
energy, KKT, derivative and atom audits; they do not promote its old result.

| Stage | Result | Derivative error / RT | Interpretation |
|---|---|---:|---|
| `source_00_finite_difference_failed.json` | rejected | 1.3939034e-4 | Original published-liquid source trial; finite-difference trace audit unresolved |
| `source_01_atmosphere_ad_failed.json` | rejected | 2.3078542e-7 | Atmospheric scalar AD passes, but the conservative finite-difference resolution gate remains unresolved in the liquid |
| `source_02_atmosphere_liquid_ad_accepted.json` | accepted | 3.9079850e-14 | Fresh metal-free local source solve with both independent atmospheric and liquid scalar AD; 51.658 seconds |

The final accepted source uses **ExoGibbs
7cf41f1374469239bdcd2255c8e2676d3c9c61c9** and **ExoEOS
ee21fa19bcff94ef6919b494e82e2c2348d23d5c**. It includes the separate liquid
scalar derivative and H2-dilution/amount-scale forwarding implemented by the
coordinator and liquid-provider work; this is **not** an atmosphere-hook-only
result. Its original runner is retained as `replay_source_liquid_audit.py`,
with the same SHA256 as recorded in the accepted result. The runner's original
local paths are intentionally preserved; it is an execution record, not a
portable replacement for the public consumer.

Acceptance here is limited to a newly solved **metal-free local branch**.
No metal selection or global pressure closure is performed by these replays,
and material/global physical certification is not claimed. Those calculations
continue independently in ExoInventory.
