# Extended phase certificates: fixed-pressure pilot and validation

This archive preserves a **250 bar fixed-pressure water-model pilot**, its
unsuccessful predecessor bounds, source-binding controls, and test results.
The pilot is not a planetary pressure root: its original
`pressure_closure_performed` flag is false and its pressure log residual is
`0.05199473610542105`. No finite-response or empirical-material acceptance
is inferred from it.

The new proof code combines outward global phase bounds with the common-plane
primal/dual gap from #240. The phase and gap tolerances remain `1e-8 RT` and
`1e-9 RT` per inventory atom. A tiny negative fixed-plane lower bound is retained;
it is never renamed a strictly nonnegative certificate.

## Reconstructed-water pilot

The same saved source, native dry standards, declared water scalar, H2 standard,
and elemental plane are used in the three independently executed searches.

| Actual search | Evaluated nodes | Global lower bound, RT per dry-component mole | Requested `1e-10` bound |
| --- | ---: | ---: | --- |
| Initial ambient enclosure | 19,999 | `-6.247241744e-4` | Not established |
| Tangent-only enclosure | 19,999 | `-1.2954445e-1` | Not established |
| Verified ambient-or-tangent enclosure | 14,077 | `-5.903781189813045e-14` | Accepted |

The successful bound covers the complete element-supported dry simplex, has no
unresolved boxes, and analytically minimizes the nonnegative water and H2
amounts. Its strict nonnegative flag remains false. The water capacity is
relaxed only when obtaining a lower bound; feasible primal states and negative
witnesses must respect the original capacity. This is a bound on the declared
expression, not a native-build equivalence or an empirical BSE validity claim.

Each `assessment.json.gz` restores the **original bytes** on decompression.
The execution receipts preserve actual commands, code/provider pins, elapsed
times, and exit codes. The successful execution used Gibbs `a34d7b7`; it was
subsequently incorporated without expression changes into the full-test code.
The original pilot trial is retained as `water_pilot/trial_0000.json.gz`.

The additional fresh native control re-evaluates the water host and verifies
its expression, element/oxide basis, G, every chemical potential, and component
amount against the saved state. Its one `alloy-solid` candidate has lower bound
`+0.22607875706625782 RT/formula`; this control is not a twenty-phase result.

The schema preflight uses the real pilot data only in explicitly identified
unit fixtures. It first verifies that the actual false pressure-closure flag
is rejected. It reproduces all saved water proof coefficients and gas standard
arrays, checks all twenty solution-model inputs, and rejects eight intentionally
incomplete one-box bounds. It does not create or certify a pressure root.
Its original uncaught incomplete-budget rejection is retained separately.

## Tests and delivery identity

The full fixed-source collection contains 2,111 tests. Twelve disjoint file
shards cover every collected node exactly once: **2,110 passed and one skipped**.
The only skip is the unavailable setuptools-scm generated version module.
The full source is Gibbs `54cfba8`, with explicit EOS `9cd03ec` and CPU/x64
settings. No provider test was skipped.

One original shard failed because its sparse checkout lacked the tracked
`external_data/Ito_2025.xlsx` fixture. After expanding that file without changing
source or tests, its single test passed. Both original failure and retry
receipts are retained; the archive does not relabel the original exit code.

The subsequent He, historical-recipe, selected-water-host, and normalized-scenario
changes were checked separately. The final delivery run has **175 passed,
zero skipped**, using EOS `813131a`. It is not presented as a second full-suite
execution. All 395 Python files under `src`, `examples`, `tests`, and `benchmarks`
in delivery `b68e122` are byte-identical to the frozen proof runtime `5004fda`.
The delivery additionally retains the earlier #240 result link.

`manifest.json` maps every preserved original path to an archive path and records
both original and stored SHA-256 values. Existing source/root outputs were never
edited. Final pressure-root phase proofs are separate subsequent executions;
this archive neither anticipates their result nor changes their acceptance flags.
