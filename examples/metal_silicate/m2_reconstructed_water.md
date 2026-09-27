# Opt-in reconstructed water free energy

`build_bse_problem` and `build_expanded_bse_problem` accept
`liquid_model="published_water"`. The EOS provider evaluates dry published
MELTS and an integrated Thompson2025 capacity on its reconstructed
H2O-equivalent mass basis. The source uses its actual common-gauge H2O gas
standard, keeping the same conserved components and initial atom ledger.
Both analytical potentials and independent scalar derivatives include all
water-induced host composition shifts. Molecular H2 dilution is then added
once through the existing finite-host construction.

The provider is identified as `dry_melts_thompson2025_water_equivalent_v1`.
Metadata records its source hashes, dry standard receipts, and actual
`water_standard_receipts` (K, Pa, common R, H2O standard in RT,1bar reference).
To reconstruct an independent property callback, use
`load_melts_evaluator(..., liquid_model="published_water", gas_water_standard_rt=callback)`;
the callback receives K/Pa. Missing gas-reference input is an error. A saved
scalar standard is reusable only at its declared temperature and reference
convention, not as a general temperature law.

This is an explicit replacement of native water, not an additive `-2ln2`
shift to native MELTS. The latter conversion applies only between the EOS
provider's earlier literal-OH completion and its reconstructed water basis.
The water pressure-volume term remains an unfitted zero assumption. The
independent basalt pressure comparisons do not establish a total BSE error
bound or calibrate alloy/H2 properties. Fresh finite equilibrium and phase
proofs are required; the earlier published hydrated-liquid bound is not
inherited. Existing `native` and `published` selections retain their behavior.

Validation includes the offline atom-ledger/provider-selection regression,
the EOS scalar-derivative and zero-water tests, and one fixed-composition
evaluation of the saved OH source. This point evaluation is not a new
equilibrium or pressure closure.

For a declared capacity sensitivity of the reconstructed-water model, use
`standard_offsets_rt["h2o_melts"]`. Multiplying capacity by a positive factor
`c` is the linear water-standard shift `-2*log(c)` in units of RT. This uses
the existing extensive-G/chemical-potential offset mechanism; it does not
change the Hessian or the separate `H2_dissolved` standard. The capacity
interpretation applies to `published_water`, not to native hydrated MELTS.
A chosen finite range is a declared model sensitivity, not a calibrated
universal error bound. The default offset is zero.
