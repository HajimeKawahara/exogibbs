# Curated Condensate HEAD Route Fresh API Regressions

These tests run curated HEAD route cases through the public condensate API from
a fresh FastChem4 preset setup. They do not read saved `results/` artifacts and
do not use FastChem4 trace, public, or runtime values as constructor inputs.

Run all curated fresh API tests:

```bash
pytest -q tests/endtoend/curated_cases
```

There are two curated surfaces:

- Fixed-support rows: the 14-row regression keeps the v1.1-era explicit
  `support_indices` and `support_amounts_init` path. It verifies accepted
  fixed-support fallback behavior, not support-free selection.
- Support-free rows: the v1.4 default regression calls `condensate_equilibrium`
  without explicit support on curated profile midlayers. This is the API path
  mirrored by `examples/condensates_curated_support_select_demo/` and exercises
  native support selection, max-density seeds, support growth, and the
  support-free retry gates. Current v1.3 gates include center-gate retry with
  `head_route_center_gate_retry_multiplier=1.0e11`, residual-worsening retry,
  soft-restoration retry with the same center gate, multi-cap support-cap retry
  with `support_cap_retry_counts=(34, 48, 80, 128)`, Ipopt-style persistent
  h-type retry with guarded feasibility components, fallback-only staged
  support-growth retry with `support_growth_staging_retry_add_per_rounds=(64,
  32, 16, 8)`, and support-growth warm-start amount flooring. The v1.3 solver
  residual reports use stable L2 norm
  evaluation so large finite condensate activity residuals do not become
  non-finite solely through norm overflow. v1.4 adds the full-condensate
  element-budget gate, external condensate budget terms in RGIE/PD-IPM residuals,
  lifecycle final-state public result wiring with post-growth support indices,
  and a fallback-only active condensate budget-correction direction.
