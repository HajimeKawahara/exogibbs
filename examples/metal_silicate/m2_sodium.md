# Conditional finite sodium in the associated alloy

`build_expanded_bse_problem(..., metal_model="associated_k_na",
sodium_options={"projection": "remove_potassium_0.25",
"temperature_policy": "published_exchange_slope"},
potassium_standard_offset_rt=..., liquid_model="published_water")`
adds one conserved `Na_metal` component to the 19-species K-associated alloy.
Both sodium options and the K standard remain explicit. The published dry or
reconstructed-water host is required; other host models are rejected.

ExoEOS owns the scalar and projected reference law. Its 20-species excess
uses the parent expression at `x_parent_Fe = x_Fe + x_Na`; full 20-species
ideal entropy still distinguishes Fe and Na. The physical atom ledger keeps
separate Fe1 and Na1 columns. The pseudo-Fe excess coordinate does not transfer
or rename any atoms. Unmeasured Na interactions are assumptions of this
conditional scalar, not zero-valued measurements.

At every source pressure the selected host property callback populates its
native standard receipt. The adapter supplies those exact T/P properties and
the actual parent Fe-metal standard to `sodium_standard_rt`; it appends the
returned Na standard to the unchanged 19 parent standards. It preserves the
complete input properties under `sodium_metal.native_liquid_properties`, their
hash, the selected projection/temperature law, source calibration hashes and
the full 20-component standard vector. No previous-pressure properties are
reused. The existing O/H scenario offsets are applied separately afterward.

The first domain extends the parent numerical species box by `0 <= x_Na <=
0.02`, with `x_Fe >= 0.75`. This is a numerical model restriction. Its contact
must be reported and cannot be described as a calibrated physical boundary.
The Na global-insertion factory is required even when the optional H-O term
is omitted; positive local curvature or local minimizer success does not
replace a global insertion certificate.

The reference is a one-sided conditional fit to projected high-pressure
S-free nondetections. Choosing its lower permitted standard is an explicit
finite-transfer scenario, not a measured low-pressure standard or a universal
omitted-path bound. Numerical conservation, source contact and pressure closure
remain separate from empirical BSE applicability.
