# Curated Support-Selection Condensate Demos

This directory mirrors the curated condensate demo families, but it does not
pass explicit condensate support into `condensate_equilibrium()`.
It therefore exercises the current default HEAD route support-selection path,
including support-free retry and support-closure repair gates when they are
triggered.

These examples are for human inspection of native support selection. They are
not GPU throughput benchmarks and they do not exercise the profile-level
fixed-support batch path used by `condensate_equilibrium_profile(method="auto")`.

Current HEAD route v1.18 fresh API counts over the 10 full profile families are:

| route group | layers |
|---|---:|
| empty gas-only boundary | 17 |
| primary promoted route | 82 |
| native seed fallback route | 0 |
| exception | 0 |

Current public status after the full-condensate element-budget gate is:

| status group | layers |
|---|---:|
| converged | 99 |
| not_converged | 0 |
| exception | 0 |

The current HEAD route keeps the one-layer solver responsible for robust
active-set orchestration.  It includes lifecycle continuation, scalar
fraction-to-boundary PD-IPM step control, support-cap and staged support-growth
retries, inactive-driving support closure, final support amount polishing,
relative gas/condensate budget correction, and strict gas-only retry for
empty-support rows.  These repairs are applied inside the one-layer public API
before the profile-level code decides whether to carry the result to another
layer.

The demos are example-facing audits for native support selection.
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
