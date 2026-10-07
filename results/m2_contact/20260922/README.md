# Native common-gas contact validation

`contact.json` was freshly evaluated through the `metal_m2_contact`
documented-example job using clean ExoGibbs source commit
`0f42c8c9a66de5f42ca5ef8a61e1ff02ee0a409d` and clean ExoEOS provider commit
`d32d9c1000bc85c873fdfd20bb83ef418e52558e`. The native runtime and data hashes
are recorded in the artifact. No archived chemical composition was used as
the final state or as an optimization guess. There were 703 native MELTS
calls. The initial absolute BSE ledger is retained in the record.

At 2173.15 K and 1 bar, with metal explicitly suppressed:

| Diagnostic | Result |
| --- | ---: |
| Non-element-gauge standard residual | `1.42108547e-14 RT` |
| Same 11 gases: maximum log partial-pressure residual | `3.90798505e-14` |
| Expanded 35 gases: maximum shared log partial-pressure residual | `0.0129354942` |
| 35 gases + 26-condensate catalog: maximum shared log partial-pressure residual | `1.2927941528` |
| Expanded gas/cloud parcel gas mass fraction | `0.9122590364` |

The matched contact and all three independent local atom/KKT audits pass.
Both expanded catalogs retain contact differences. The cloud-bearing
residual corresponds to a maximum shared partial-pressure ratio of about
3.64. Only `Fe(s,l)` is present, at `3.61483729e21 mol`; nine condensate
entries are temperature-ineligible and remain zero. Adding this upper pure
Fe phase to an explicitly alloy-suppressed lower control changes the phase
catalog. It does not certify the stability of the lower Fe-Si-O-H alloy.
These diagnostics do not establish expanded-atmosphere contact,
metal stability, independent MELTS/alloy calibration, a common material
domain, or pressure closure. M2-A/B/C remain pending.

`harness_summary.json` and `harness_worker.json` preserve the actual
registered run and its artifact acceptance. All 63 public/documented jobs
have coverage, but only `metal_m2_contact` ran here: `full_acceptance` is
false by construction. Runtime was about 209 seconds on the recorded CPU
environment.

`attempt_maxiter100.json` preserves the earlier exploratory failure before
the clean source commit: the source scalar minimizer reached its 100-iteration
limit and failed its KKT gate. Its recorded working-tree provenance is not
the clean validation revision. The completed replay uses the existing
default maximum of 1000 iterations. No acceptance tolerance was relaxed.

See [the common-gas contract](../../../examples/metal_silicate/m2_common_gas.md)
for model conventions, independent checks, and reproduction.
