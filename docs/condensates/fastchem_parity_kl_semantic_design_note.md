# FastChem-Parity KL Semantic Design Note

This note defines the production-design semantics implied by the Round 9
FastChem Cond KL parity diagnostics. It is not a production behavior change.

## Scope

The design target is a future KL semantic interface that can name and test the
state basis used by donor, molecule, inventory, and removed-condensate
comparisons. The current evidence is strong enough for design work, but not for
promotion into solver behavior.

## Required State Interfaces

1. Normalized donor state
   - KL raw `gas_only_final` is normalized.
   - It must not be compared directly to FastChem physical donor values.

2. Physical donor state
   - A diagnostic physical conversion is required before FastChem donor
     comparison.
   - The conversion removes the inflated `C1H4` / `H2O1` donor gap.

3. Coherent molecule + inventory state
   - Molecule-only and inventory-only replays are destructive.
   - The design interface must treat molecule and inventory rows as a coherent
     state bundle when parity diagnostics rely on their cancellation.

4. Removed-condensate correction state
   - The remaining `45:-10` PMI residual is fully explained by the emitted
     `Al4C3(s)` removed-condensate analytic correction.
   - This is a provenance boundary, not a production removed-tail patch.

## Candidate Guarded KL Option

A future prototype may add a default-off diagnostic option that emits:

- normalized donor rows,
- physical donor rows,
- molecule-cache rows,
- inventory/atom rows,
- removed-condensate correction rows,
- metric-family labels for focused and broad scorecards.

The option must not change presets, defaults, production solver behavior,
active-selection behavior, row selection, row scaling, lifecycle handling, or
FastChem compatibility behavior.

## Non-Promotion Rules

The following are explicitly not production-promotable from the current evidence:

- CH4 data-validity masking as a KL default.
- MgCO3/SiC FastChem donor snapshot transplant.
- Group-B reduced-solve exclusion as a selected-row rule.
- Legacy KL-reference burden-ratio conversion.
- Full-vector infinity-norm fallback.
- Molecule-only, inventory-only, donor-only, or removed-tail transplants.
- Broad projection as focused regression.
- Repaired alpha/beta production behavior.

## Production Candidate Gate

A future production candidate must pass the invariant checklist in
`results/fastchem_cond_kl_production_readiness_compact.json`, regenerate or
explicitly account for broad cases beyond the current five-case set, and show
that any guarded option improves the relevant metric without dropping rows,
species, or cases.

Current conclusion: semantic levers ready for production design note but not
promotable.
