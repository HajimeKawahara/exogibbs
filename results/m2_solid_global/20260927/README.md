# Complete declared phase bounds at the newly closed published central root

This archive keeps the complete raw solid and second-liquid proof partitions
for the accepted 35-gas central BSE source at **2173.15 K and
267.2028341602767 bar**, H inventory 1e24 mol, and 16 layers.
The accepted source audit is [Inventory #31](https://github.com/HajimeKawahara/exoinventory/pull/31),
[pinned raw source](https://github.com/HajimeKawahara/exoinventory/blob/1b1fcbd/examples/subneptune_taxonomy/finite_melt/validation/20260927_m2_selected_model_audit/published_m1_physical.json).
The copied input bytes and hashes are preserved in each original run directory.

The frozen fresh-native runs used Gibbs **3ca51168a6a08d293491d61166fb5fa05d088ead**
and EOS **38475d56a7cbf0b00aa1e51ef6d6e61e4c4291fc**. Both exited 0: solids
249.853 s and liquid 149.575 s. The selected published host was retained; native
candidate standards and independent native liquid controls were evaluated
separately. No proof runner solved a new pressure root.

A separate final replay uses the same preserved native receipts, Gibbs
**2f96049** and EOS **05155344**, with strict nonnegative flags, retained
source-expression checks, clamped interval anchors and deterministic entropy
term order. It makes **zero native calls**. Its complete output, actual script,
input hashes and log are in `strict_offline_replay`; the original fresh runs
are never relabeled as this replay. All 20 lower bounds remain strictly
positive, and the complete second-liquid bound remains exactly zero.

| Declared phase | Verified lower bound (RT/formula unit) | Fresh nodes | Final replay nodes |
| --- | ---: | ---: | ---: |
| hornblende | 14.97826319942902 | 1 | 1 |
| biotite | 13.35339662045246 | 1 | 1 |
| garnet | 1.140485730435997 | 3 | 3 |
| alkali-feldspar | 2.429315922798022 | 3 | 3 |
| kalsilite | 8.405848619328227 | 1 | 1 |
| leucite | 1.177473932506 | 1 | 1 |
| alloy-solid | 0.2306027839422462 | 1 | 1 |
| alloy-liquid | 0.04536556585172421 | 1 | 1 |
| cummingtonite | 9.778603466507093 | 1 | 1 |
| olivine | 0.001562348662674569 | 9 | 9 |
| nepheline | 4.667491913622299 | 1 | 1 |
| clinoamphibole | 3.933737635185302 | 1 | 1 |
| orthoamphibole | 3.564008628189676 | 1 | 1 |
| melilite | 1.903911992506224 | 1 | 1 |
| ortho-oxide | 0.9416522933615877 | 3 | 3 |
| rhm-oxide | 0.09796099625621209 | 59 | 59 |
| plagioclase | 1.93193589131545 | 1 | 1 |
| clinopyroxene | 0.007455451042864402 | 517 | 515 |
| orthopyroxene | 0.000370622565599408 | 735 | 731 |
| spinel | 0.01028145072060834 | 313 | 313 |

The lower-bound units are per phase formula unit, not per atom. All physical
site and ordering coordinates of the declaration are covered, including
signed native basis coordinates and continuous empty-site energies.
Conditional absence of Mn/Ni/Co is checked against the actual host inventory.
Positive singular endpoint barriers remain divergent; undefined endpoint
chemical potentials are not invented. Pure ordering reference minima retain
verified intervals over their complete declared domains.

The liquid proof covers all 12 BSE-supported components and every nonnegative
additional ideal-H2 amount; its native controls attempted 14 requests, with
13 finite and one unavailable. The failed 98% vertex request remains saved;
an explicit 50% alternative is an independent request. Maximum finite
potential difference is 3.1850211357209446e-9 RT. Finite native agreement is
separate from the formal interval bound.

`formal_global_insertion_bound_accepted` and the formal liquid flags require
actual lower bounds >= 0. `bound_within_requested_tolerance` is a separate
numerical statement. The final regression suite includes negative examples
inside the requested tolerance and exact singleton interval anchors.

These results establish stability of the fixed declared expressions and
fixed numerical standards at this particular source. They do not certify
empirical BSE calibration, native-binary-wide error, omitted phases, or the
source Fe/Si/O/H alloy. The native Fe/Ni alloy's Fe-only domain is a different
property model. New gas/condensate/standard scenarios require their own
source and proof; no historical root is substituted for them.

## Files and validation

- `fresh_published_m1/solids` and `liquid`: unmodified fresh-run inputs,
  protocols, raw native receipts, complete proof partitions and summaries.
- `strict_offline_replay`: final-code replay from the same fixed raw receipts.
- `independent_math_review.md`: independent read-only interval/dual/entropy
  review at the original frozen code, including the subsequently fixed
  strict-sign and singleton-anchor findings.
- `tests`: original focused pass and all strict-test attempts. The first
  attempt exposed the singleton regression and sparse-checkout omissions;
  after the fix and acquisition of tracked assets, **56 tests passed**.
  Full Gibbs validation runs in the PR's GitHub CI to reserve local resources
  for ongoing planetary closure calculations.
- `execution_receipt.json` and `manifest.json`: actual process outcomes and
  a complete byte inventory.

[Algorithm and fresh-native reproduction](../../../examples/metal_silicate/m2_solid_global.md),
[liquid algorithm](../../../examples/metal_silicate/m2_liquid_global.md),
[provider source/native evidence](https://github.com/HajimeKawahara/exoeos/tree/113ae4d/results/m2_solid_mixing/20260927).

The archived offline script retains its original scratch paths to identify
its actual execution. To rerun it elsewhere, map those paths to checkouts of
the recorded revisions and these copied input directories, and use a new
output directory. Do not overwrite this archive.
