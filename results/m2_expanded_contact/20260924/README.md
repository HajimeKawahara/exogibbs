# Finite BSE with the full retained atmosphere: 2026-09-24

The registered `metal_m2_retained_suppressed` job freshly solved the finite
BSE with the same 35 gases and 26 retained condensates as the upper parcel.
The native source and the independently recounted expanded contact passed.
Pure atmospheric Fe condensate is present; the deep Fe-Si-O-H alloy is
explicitly suppressed in this control.

| Check | Result | Acceptance |
| --- | ---: | ---: |
| Fresh native MELTS evaluations | 380 | Positive count |
| Full 13-element source residual, maximum relative | `8.88e-16` | `< 1e-9` |
| Source constrained KKT, maximum absolute | `1.71e-13` | `< 1e-8` |
| All 35 gas log partial-pressure differences | `1.42e-14` | `< 1e-8` |
| Common gas/cloud standard nongauge residual | `4.26e-14` | `< 1e-8` |
| Source/upper cloud atom difference, relative to atmospheric atoms | `0` | `< 1e-9` |
| Retained `Fe(s,l)` amount | `3.4564276852175936e22 mol` | Included in atmospheric atoms and mass |
| Atmospheric gas mass fraction | `0.5435342407618166` | Diagnostic |

The atmosphere was included in the source scalar Gibbs minimization, rather
than appended to a frozen source gas. Clouds add atmospheric mass but no gas
pressure. The full primitive gas/cloud amounts and internal atom-coordinate
source are preserved separately in [contact.json](contact.json). There is no
rainout or exchange with an unrecorded reservoir.

## Source and model provenance

| Input | Value |
| --- | --- |
| ExoGibbs source | `15a269d2913a306d8b7e63a9f5e8bbd8424eb9e5`, no changed tracked files |
| ExoEOS source | `f380500d1e81b8aa51b31dd56eff186cf1e54256`, no changed tracked files |
| Native runtime | alphaMELTS Python `2.3.2`, Ubuntu 22.04 x86_64 |
| Temperature / pressure | `2173.15 K` / `1 bar` |
| Dry BSE / hydrogen / helium | `1e23 kg` / `1e24 mol H atoms` / `1e23 mol He atoms` |
| Initial core / ferric fraction / C, N, S | Exactly zero |
| Gas/cloud catalog | Packaged M1 FastChem4, 35 neutral gases and 26 condensate entries |
| Lower element-reference anchors | H, He, O2, Mg, SiO, Fe, Na |

The subsequent `f2e0d20fcc68a542fc0724c034274e75f599c2d9` source adds strict
selected-metal completion gates, reuses zero-support callback embedding,
and repairs documentation headings. Those safeguards have separate targeted
regressions. They do not change this all-positive, suppressed-metal physical
model. This archive retains its actual earlier source commit and file hashes;
no later provenance is substituted for the executed source.

The preserved input is [bse_inventory.json](bse_inventory.json).
The unchanged adopted BSE is McDonough & Sun (1995), Table 4, p. 237, first
column. Source values remain unnormalized in the provenance ledger:

| Retained oxide | Source wt% |
| --- | ---: |
| SiO2 | 45.0 |
| MgO | 37.8 |
| FeO | 8.05 |
| Al2O3 | 4.45 |
| CaO | 3.55 |
| Cr2O3 | 0.384 |
| Na2O | 0.36 |
| TiO2 | 0.201 |
| K2O | 0.029 |
| P2O5 | 0.021 |

MnO (`0.135`) and NiO (`0.25`) remain omitted. The complete source total is
`100.230`; the retained total is `99.845`. Retained oxides are normalized by
`99.845`, with retained fraction `99.845/100.230`. No new composition fit is
introduced.

## Reproduce and interpret

Run the documented-example harness with `--platform cpu
--case metal_m2_retained_suppressed`, the explicit ExoEOS checkout, exported
BSE ledger, native runtime, worker Python and Japanese documentation checkout.
The exact command, hashes and worker result are in
[harness_summary.json](harness_summary.json) and [worker.json](worker.json).
[sha256.json](sha256.json) fingerprints the raw evidence. Existing outputs
are never overwritten. The preliminary interrupted run and its lack of
scientific evidence are described in [trial_history.md](trial_history.md).

This is a conditional local-equilibrium result. Common BSE material
calibration, global MELTS/competitive-phase stability, omitted transfer
bounds and planetary pressure/inventory closure remain unestablished.
M2-A/B/C remain pending. The registered selected-metal case is a separate
control; this archive does not infer alloy absence or certify a phase boundary.


## Regression and documentation validation

The complete unit-test collection at
`f2e0d20fcc68a542fc0724c034274e75f599c2d9` contains 1,811 cases. All were
assigned once across eight independent pytest shards: **1,810 passed,
1 skipped, no failures or errors**. The skip concerns the absent generated
setuptools-scm version file in this checkout, not chemistry. See
[full_suite_summary.json](full_suite_summary.json) and the manifest/JUnit
fingerprints in [validation_receipt.json](validation_receipt.json).

`./update_doc.sh` succeeded with 125 existing documentation warnings and
zero warnings in the new retained-atmosphere sections. Initial missing-title
warnings in the two new gallery modules were fixed before this final build.
The archive adds only evidence and prose after the tested source commit.
