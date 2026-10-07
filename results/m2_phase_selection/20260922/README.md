# Constrained metal selection and native BSE local closure

These receipts exercise the declared Fe-rich composition domain in the
present-metal solve as well as the incipient-metal search. The material
models and absolute BSE input retain their recorded calibration limitations.
**A locally accepted BSE solution is not a globally certified assemblage.**

The final native runs use clean ExoGibbs source commit
`1bfd7918d4a34bcc038d27f1e13a07dbb391da1f` and ExoEOS commit
`9f3f48dbdb7514b78195dd4c0e8c438aefc2de29`. The records retain the actual
commands, source hashes, model choices, full amounts and native runtime
provenance. ExoEOS's extra experimental response helpers are not substituted
for the existing Ma/H2 controls in this calculation.

| Input | Value |
| --- | --- |
| Dry BSE | McDonough and Sun (1995), Table 4, first column; retained oxides normalized from 99.845 wt% |
| Omitted source oxides | MnO 0.135 wt%, NiO 0.25 wt%; their missing elements are not silently introduced |
| Dry-rock mass | `1e23 kg` |
| Atomic H and He | `1e24 mol`, `1e23 mol` |
| Initial ferric fraction / preexisting core | 0 / `0 kg` |
| Element basis | O, Mg, Si, Fe, Al, Ca, Na, K, Ti, Cr, P, H, He; C=N=S=0 |
| Conditions | 2173.15 K, 1 bar |
| Alloy mathematical domain | Fe >= 0.86, Si <= 0.08, O <= 0.02, H <= 0.04, fractions summing to one |
| Gas reactions | Continuous shared M1 gas laws, lower reference anchors retained (`m1_shared`) |

The native [BSE receipt](bse.json) gives:

| Diagnostic | Result |
| --- | --- |
| Local minimum | `accepted=true` |
| Metal amount | `3.6196087923284053e22 mol` |
| Alloy fractions (Fe, Si, O, H) | `(0.9817036258645454, 0.00023557978213760883, 0.01700675697810637, 0.0010540373752107113)` |
| Maximum constrained KKT residual / RT | `1.14e-13` |
| Independent relative atom error | `8.85e-16` |
| Independent scalar derivative error / RT | `4.40e-8` |
| Insertion minimum upper bound / RT | `1.95e-14` |
| Insertion lower/upper gap / RT | `2.85e-14` |
| Native MELTS evaluations | 1063 |
| Global phase status | `unresolved`: `Host global stability is not established.` |
| Scientific gates | M2-A and M2-B pending |

All composition constraints are inactive at this native solution. Boundary
duals are instead exercised by the independent two-element scalar test and
the [four-point Ma control](../reference_20260922.json). The latter retains
the original synthetic standards: offset 4.62 now gives a certified
restricted-domain metal-bearing solution with `0.000828981120906 mol` of
alloy and Si=0.08. At 4.63 the metal amount is exactly zero. These offsets
are mathematical controls, not physical BSE or hydrogen-series boundaries.

The [initial seed failure](initial_seed_failure.json) preserves a rejected
development attempt. Its unrestricted starting-point LP transferred large
amounts into initially trace O2 and SiH4, preventing a finite backtracking
step from lowering energy. It retained the accepted metal-free state as a
reference, while its negative insertion energy rejected stable absence.
The successful solver bounds only the starting-point host/gas changes
relative to their actual amounts; it does not add equilibrium constraints,
atoms, composition clipping, or positive metal floors.

The original [2026-09-20 reference](../reference_20260920.json) remains
unchanged. Its unresolved offset-4.62 result is not relabeled. The earlier
unconstrained BSE failure remains in ExoInventory's separate
`validation/20260921_m2/select.json` archive.

The [harness summary](harness_summary.json) and two worker receipts require
both fresh jobs to pass: `metal_phase_selection` and
`metal_bse_phase_selection`. Native acceptance independently checks the
composition domain, local KKT residual, atom audit, actual MELTS calls, and
preservation of unresolved global host stability. This targeted run does
not claim full acceptance of every documented example.

Reproduce from the ExoGibbs checkout, selecting existing local resources:

```sh
JAX_PLATFORMS=cpu JAX_PLATFORM_NAME=cpu JAX_ENABLE_X64=1 \
PYTHONPATH=.:src:/path/to/exoeos/src \
python -m benchmarks.documented_examples.run_all --platform cpu \
  --case metal_phase_selection --case metal_bse_phase_selection \
  --exoeos-source /path/to/exoeos/src \
  --melts-runtime /path/to/alphamelts-py-2.3.2-ubuntu_22_04-x86_64 \
  --melts-python /path/to/native-worker/python \
  --bse-inventory /path/to/exoeos/examples/m2_material/bse_inventory.json \
  --japanese-docs /path/to/doc_ExoGibbs \
  --output-directory /tmp/new-constrained-metal-run
```
