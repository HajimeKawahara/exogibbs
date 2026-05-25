# Guarded Retained-Floor Experimental Diagnostic Surface

## Summary

The guarded retained-floor surface is an explicit opt-in experimental
diagnostic API for condensate callsite experiments.

It compares a no-floor retained-support candidate with a retained-floor
candidate and selects the retained-floor candidate only when it is budget-safe
and has a lower KKT diagnostic residual.

This surface is not a default production solver path and is not an equilibrium
acceptance gate.

## Included Diagnostic Modules

- `exogibbs.diagnostics.condensate_retained_floor_selector`
- `exogibbs.diagnostics.condensate_guarded_retained_floor_policy`
- `exogibbs.diagnostics.condensate_guarded_retained_floor_callsite_adapter`

## Guardrails

- Explicit opt-in only.
- Default-off.
- No production solver behavior change.
- No production return signature change.
- No presets or defaults wiring.
- No FastChem4 public, runtime, or trace values as constructor inputs.
- KKT residuals are diagnostics, not acceptance gates.
- Budget non-worsening is a required guard.

## FC4-M2081 Release Review Status

- Case count: `4`
- Selection match count: `4`
- Shape match count: `4`
- Budget-safe count: `4`
- KKT-improved count: `4`
- Policy gate passed: `true`

## Recommended Commit Scope

Include:

- source diagnostics under `src/exogibbs/diagnostics`
- targeted diagnostics tests under `tests/unittests/diagnostics`
- targeted artifact tests under `tests/unittests/presets`
- comparison scripts under `examples/comparisons`
- diagnostic documentation under `documents/diagnostics`

Do not include:

- `results/`
- `FastChem4/`
- broad generated docs builds
- unrelated worktree changes
