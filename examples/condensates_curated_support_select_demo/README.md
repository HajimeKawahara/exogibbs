# Curated Support-Selection Condensate Demos

This directory mirrors the curated condensate demo families, but it does not
pass explicit condensate support into `condensate_equilibrium()`.
It therefore exercises the default HEAD route v1.4 support-selection path,
including the support-free retry gates when they are triggered.

Current HEAD route v1.4 fresh API counts over the 10 full profile families are:

| route group | layers |
|---|---:|
| empty gas-only boundary | 17 |
| primary promoted route | 78 |
| native seed fallback route | 4 |
| exception | 0 |

Current public status after the v1.4 full-condensate element-budget gate is:

| status group | layers |
|---|---:|
| converged | 93 |
| not_converged | 6 |
| exception | 0 |

The v1.3 support-free gates include center-gate retry with
`head_route_center_gate_retry_multiplier=1.0e11`, residual-worsening retry,
soft-restoration retry with the same center gate,
Ipopt-style persistent h-type retry with guarded feasibility components,
multi-cap support-cap retry with `support_cap_retry_counts=(34, 48, 80, 128)`,
fallback-only staged support-growth retry with
`support_growth_staging_retry_add_per_rounds=(64, 32, 16, 8)`,
support-growth warm-start amount flooring, and stable L2 residual norm
evaluation for large finite condensate activity residuals. v1.4 additionally
checks the full condensate element budget before reporting a row as converged
and reflects lifecycle continuation final states with their post-growth support
indices in the public result. The current v1.4 HEAD route also applies
per-condensate element capacity caps, final support amount polish, and a
relative joint gas/condensate budget-correction retry that starts from the
lifecycle final state when the public full-budget gate rejects an accepted row.

The demos are scratch-facing examples for auditing native support selection.
They intentionally keep the original pressure/temperature families from
`examples/condensates_curated_demo` so that support-selection behavior can be
compared against the existing explicit-support demo path.

Each script writes:

- a PNG profile next to the script
- a `.support_selection.json` file with layer-by-layer selected support names

Run one demo with:

```bash
python examples/condensates_curated_support_select_demo/demo_solar_water_condensation.py
```

Run all demos with:

```bash
for f in examples/condensates_curated_support_select_demo/demo_*.py; do python "$f"; done
```

These examples use ExoGibbs-native thermochemistry only. They do not call
FastChem4 or `pyfastchem`.
