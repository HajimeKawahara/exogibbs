# Finite metal transfer with chemical associates

`build_expanded_bse_problem(..., gas_model="janaf_condensed",
metal_model="associated", phosphorus_options=...)` extends the finite-P
source with Mg/Ca/Al/Cr/Ti and eight oxygen associates. It preserves the existing
`ma` and `phosphorus` modes and their defaults. All eighteen species share one
EOS-owned scalar. The thirteen-element source ledger conserves atoms across
gas, retained condensates, silicate and metal; associates do not create a new
reservoir.

The registered order is Fe, Si, O, H, P, Mg, Ca, Al, Cr, Ti, MgO, CaO, AlO,
CrO, TiO, Al2O, Cr2O, Ti2O. Source components append `_metal`. Consumers must
use the explicit formulas; a mole of Cr2O carries two Cr atoms and one O atom.
The selection box and Hessian bound use moles of chemical species. Returned
`metal_composition_domain` therefore uses `lower_species_fractions`,
`upper_species_fractions`, `effective_upper_species_fractions`, and explicit
atom counts. Existing all-atomic models keep their previous metadata keys.

`associated_metal_domain(metadata)` returns the actual eighteen-species bounds
and matched curvature lower bound. The `associated_metal` declaration records
standards, formulas, source receipts, interactions, numerical domain and file
hashes. `phosphorus_metal` remains as the parent P-reference receipt; its presence
alone does not identify the selected final model.

The independent derivative audit differentiates the EOS excess scalar and the
ideal entropy on the exact active support. Exact-zero components do not acquire
floors; their undefined boundary derivatives are not passed through `xlogy(0,0)`
into active derivatives. Original scalar, conservation, KKT, extensivity and
contact tolerances remain unchanged.

Named O/H scenarios shift their original free species only. Independently
anchored associate standards retain the Jung oxygen reference and explicitly
report its difference from the Ma oxygen reference. The retained host plus
new primary-data continuation is a declared hybrid, not a calibrated
reproduction of every measured subsystem. Finite-response runs and physical
error assessment remain separate from numerical acceptance. K transfer remains
omitted and no inventory-only coupled-error bound is asserted.

## Explicit finite K sensitivity

`metal_model="associated_k"` requires a finite
`potassium_standard_offset_rt`. Other metal modes reject that argument. The
nineteenth component is `K_metal`, with one K atom per species. Its standard is
the actual source `K1` gas standard plus the explicitly supplied offset; no
empirical value is inferred. `phosphorus_options` continues to configure only
the P reference.

The EOS perspective scalar dilutes the eighteen-species host and mixes finite
K ideally. The selected domain adds `x_K<=0.02`; the returned matched curvature
bound is about 0.3443 at 2173.15 K. `associated_metal` describes all nineteen
species, and `potassium_metal` records the actual gas standard, offset and
uncalibrated sensitivity scope. All eleven alloy elements must be reconstructed
from the species formulas before any mass-percent material comparison.

The existing finite source solver controls K uptake and compensating changes
in the other reservoirs. Standard scans remain conditional finite responses;
they do not establish a universal coupled-error bound or physical acceptance.
