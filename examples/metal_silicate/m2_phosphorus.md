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
`temperature_policy` (`constant`, default, or `enthalpic`),
`standard_shift_kcal_mol` (default zero), and `upper_mole_fraction` (default
0.02). The last option changes only the numerical P constraint; it does not
change the Gibbs scalar, standards, or coefficients. The other choices select a model
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

## Numerical P-domain sensitivity

The original 0.02 upper bound was an example-level numerical restriction,
not a measured validity limit. `upper_mole_fraction` accepts a finite positive
number no greater than one; construction and the matched global proof must
also support the resulting full box. A syntactically valid number alone does
not guarantee a usable curvature bound or accepted equilibrium.

For the five-component model this number bounds the atomic fraction of P.
For `associated`, `associated_k`, and `associated_k_na`, the same option bounds
the P chemical-species mole fraction in the selected final mixture. The
parent `phosphorus_metal.upper_atomic_fractions` and final
`associated_metal.upper_species_fractions` are separate declarations. Equal
numeric limits in these bases do not imply equal elemental P concentrations.
Use the final species formulas to reconstruct atom fractions or mass percent.

For example, keep every other saved input fixed and use the following three
separate `phosphorus_options` inputs, preserving any existing P options:

```json
{"upper_mole_fraction": 0.01}
{"upper_mole_fraction": 0.02}
{"upper_mole_fraction": 0.021}
```

The middle case reproduces the default domain and scalar. The smaller and
larger boxes test dependence on this constraint. Recompute the finite source,
pressure root, contact audit, and full common-plane proof for every case;
reusing the baseline proof would certify the wrong domain. Saved-alloy replay
checks both parent and final P bounds against the source option. Selection
and exact atom-repaired primal checks retain the actual selected box.

Report selected and incipient P fractions, their active bounds and KKT
multipliers, elemental P mass percent, root pressure, and the complete proof
status. An active positive upper-bound multiplier diagnoses a constrained
minimum and the local incentive to increase the upper limit. Its units and
normalization belong to that local constrained problem; it is not an
experimental uncertainty or a derivative of the coupled pressure root.
An interior result after expansion can show reduced sensitivity over the
tested boxes, but cannot establish empirical admission. Failure of a proof
or solve remains a failed case, not permission to change its tolerances.

At 2173.15 K with ExoEOS commit
`6f3bfdc94336e6bad3106c819281944178e7f015`, the nineteen-species H–O-free
reference curvature lower bound is approximately 0.34428 at a P upper bound
of 0.02, 0.22020 at 0.021, and 0.09501 at 0.022. At 0.023 and 0.025 the
existing curvature certificate is nonpositive; at 0.03 the independent box
also violates the Fe constraint required by this certificate. The twenty-
species Na proof rejects 0.03 because its parent rectangle lacks a feasible
Fe/Na split at every point. These are limitations of the current proof,
not demonstrations of physical instability or a new empirical upper limit.
The recommended expansion to 0.021 retains the existing proof method and
all other bounds. This preflight does not replace the complete proof at
the new selected state.

## Primary evidence and unresolved physical domain

The ExoEOS source ledger `examples/m2_material/phosphorus_sources.json` pins
the primary PDFs and their hashes. The published evidence is:

| Contribution | Measured conditions | Source |
| --- | --- | --- |
| P/P2 dissolution standard in Fe | 1–3 wt% P; 1863.15–1923.15 K | [Yamamoto et al. (1980)](https://doi.org/10.2355/tetsutohagane1955.66.14_2032) |
| P self interaction | 0.7–3.2 wt% P, reported as 0.013–0.057 atomic fraction; 1873.15 K | [Yamada and Kato (1979)](https://doi.org/10.2355/tetsutohagane1955.65.2_264) |
| P–Si dilute interaction | Fe–P–Si at 1873.15 K | [Yamada and Kato (1983)](https://doi.org/10.2355/isijinternational1966.23.51) |
| O response to P | Fe–P equilibrated with H2/H2O at 1813.15, 1858.15, and 1898.15 K | [Sanbongi and Koizumi (1962)](https://doi.org/10.2355/tetsutohagane1955.48.14_1729) |
| H response to P | Less than 6 wt% P; study temperature range 1723.15–1943.15 K; atmospheric H2 pressure | [Nozaki et al. (1966)](https://doi.org/10.2355/tetsutohagane1955.52.13_1823) |

These subsystem ranges do not define a common composition rectangle. Even a
P concentration inside a reported binary interval does not validate the
2173.15 K Fe–Si–O–H–P continuation, its pressure response, or the larger
associated alloy with K and Na. The scalar combines dilute coefficients in
an integrable finite extension; the measurements do not uniquely determine
that extension. P self-interaction data above 0.02 atomic fraction therefore
do not authorize a larger calibrated species-fraction limit.

The numerical scan can measure how the declared model's root and phase
selection depend on the chosen P box. Resolving physical admission requires
independent thermodynamic data or a validated material model for finite P,
H, O, Si and the retained alloy components near the candidate T/P, including
their cross interactions and pressure response, on compatible standard
states. More accurate optimization of the current scalar cannot supply
those missing measurements. No new empirical composition limit is assigned.
