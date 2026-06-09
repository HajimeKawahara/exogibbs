# Curated Support-Selection Condensate Demos

This directory mirrors the curated condensate demo families, but it does not
pass explicit condensate support into `condensate_equilibrium()`.
It therefore exercises the default HEAD route v1.2 support-selection path.

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
