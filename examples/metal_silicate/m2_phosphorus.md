# Finite phosphorus exchange

`build_expanded_bse_problem(..., gas_model="janaf_condensed",
metal_model="phosphorus")` adds a conserved `P_metal` component to the
Fe-Si-O-H source alloy. The existing default `metal_model="ma"` is unchanged.
The finite thirteen-element source budget supplies P; no external P reservoir
or frozen-potential addition is used. All existing gas and cloud candidates
remain available.

The scalar and phosphorus standards belong to ExoEOS
(`examples/m2_material/phosphorus_reference.py`). The source's actual P/P2 gas
standard anchors the alloy standard. A single extensive scalar supplies both
chemical potentials and an independent scalar AD audit. Atom conservation,
source acceptance, metal selection and contact criteria remain unchanged.

`phosphorus_options` accepts `gas_reference` (`P1`, default, or `P2`),
`temperature_policy` (`constant`, default, or `enthalpic`), and
`standard_shift_kcal_mol` (default zero). Each choice is a separate model
continuation, not a confidence interval. Existing O/H linear standard scenarios
apply after construction. Four-component custom bounds cannot silently define
the new five-component domain and are rejected.

`phosphorus_metal_domain(metadata)` returns the exact five-component lower and
upper fractions and the scalar-matched outward curvature bound for selection.
Metadata retains source standards, conversions, coefficients, options, bounds
and hashes of the provider recipe. The initial appended P amount is zero;
canonical or saved numerical seeds must still conserve the complete finite
budget. Initial guesses do not constitute accepted equilibria.

Primary standard measurements span 1863.15–1923.15 K. Their use at the M2
temperature 2173.15 K is explicitly recorded as continuation. P/P2 reference
disagreement, temperature continuation and missing calibrated pressure response
remain separate from numerical acceptance. Adding P resolves this particular
omitted component in the declared model; Mg/Al/Ca/K/Ti/Cr alloy transfer and
empirical coupled error bounds remain independent work.
