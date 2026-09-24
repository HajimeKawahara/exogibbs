# Supplied-host competing phases, 2026-09-25

These are new native property evaluations of unchanged archived source
compositions and prescribed dry controls. They do not solve the finite
BSE source or close the planetary pressure/mass budget. Scientific material
acceptance and global host stability remain unresolved.

The executed ExoGibbs implementation is `5aec2ef`, with clean ExoEOS
`4e3983a832a15f91fbad6d5e6f874a27a0bc7011` and the pinned alphaMELTS 2.3.2
rhyolite-MELTS 1.0.2 runtime. Each JSON records the original input SHA256,
the full native supplied-liquid receipt, external runtime hashes, evaluator
hash, candidate properties, and assessment hash. Original source archives
remain unchanged. `manifest.json` fixes every raw input, output, and executed
harness. Machine paths in the harnesses describe the executed environment;
change them in a new copy when reproducing elsewhere.

| Supplied host | Temperature (K) | Mg/Si relative to dry BSE | Smallest mineral trial / RT per mol atoms | Unresolved candidates |
| --- | ---: | ---: | ---: | ---: |
| Archived suppressed source | 2173.15 | Original source inventory | +0.0831377253 | 2 |
| Archived selected source | 2173.15 | Original source inventory | +0.0831003019 | 3 |
| Dry composition control | 2173.15 | 0.8 | +0.1136552373 | 13 |
| Dry composition control | 2173.15 | 1.0 | +0.0843704291 | 13 |
| Dry composition control | 2173.15 | 1.2 | +0.0581613907 | 14 |
| Dry lower-temperature control | 1873.15 | 1.0 | -0.0538308431 | 13 |

Every tabulated minimum is olivine. Its finite one-sided energy differences
in `finite_trials.json` agree with the analytic direction to at most
`1.81e-6 RT` per mole of atoms at the smaller recorded step. The 1873.15 K
control actually lowers the full scalar by `9.12538198e-6` in units of
`G/(RT)` (mol). This rejects that *formal dry supplied liquid* within the
native phase models; it does not measure a liquidus or establish the behavior
of a coupled H-rich BSE system.

Nonnegative trials at 2173.15 K do not certify absence. The native catalog
has 30 minerals, water, and two Fe-Ni alloys. Native incipient solution
compositions are not reoptimized after the formal H2 dilution correction.
The water and Fe-Ni models are alternatives to the separate atmosphere and
Ma alloy models. Liquid splitting, complete solution-phase minima, and
empirical model applicability remain unproved.

The suppressed source has an unavailable kalsilite affinity and an invalid
hornblende oxide/mass round trip. The selected source additionally fails the
orthoamphibole round trip. Dry controls also retain unavailable exact-zero
endpoint potentials or infeasible component consumption. None is silently
classified as absent. Raw affinity and energy outputs for failed round trips
remain available in the native property receipt.

`inputs/` contains primitive dry control records, including the full
Inventory-built absolute ledger. `prepare_dry_controls.py` preserves the
executed Inventory-owned construction: keep MgO+SiO2 mass, dry mass, other
oxides, and all independent inputs fixed; redistribute that combined mass to
0.8, 1, or 1.2 times the original atomic Mg/Si; recompute O stoichiometrically.
The atomic ratios are 1.0017925983, 1.2522407479, and 1.5026888975. H/He remain
unchanged in the exported bookkeeping input but are explicitly *not inserted*
into these dry host property controls. No dry-source equilibrium is claimed.

The original suppressed state is
[the archived provider contact](../../m2_expanded_contact/20260924/contact.json).
The selected state is Inventory's
[`20260924_m2/metal_select.json`](https://github.com/HajimeKawahara/exoinventory/blob/b964f1f5b3633153faf450bbc31b95beefe6045d/examples/subneptune_taxonomy/finite_melt/validation/20260924_m2/metal_select.json).
The archived assessment is reproducible with
[`run_m2_host_stability.py`](../../../examples/metal_silicate/run_m2_host_stability.py)
and a source JSON input; see the
[assessment contract](../../../examples/metal_silicate/m2_host_stability.md).

## Complete validation

[validation.json](validation.json) records all 1821 collected unit tests on
`b814b38`: 1820 passed and one existing generated-version-module skip.
The suite used eight disjoint file shards and the pinned ExoEOS `4e3983a`.
The documentation build succeeds with 127 existing API/gallery warnings
(including repeated warnings for the pre-existing M1 gallery title); neither
new host-assessment page has a warning.
