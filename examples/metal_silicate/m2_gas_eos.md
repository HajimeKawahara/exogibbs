# Conditional full-catalog major-gas nonideality

The retained M2 source can consume the ExoEOS major-gas second-virial recipe
through `build_expanded_bse_problem(..., gas_eos_options=options)`. The default
`None` preserves the ideal model. Gas catalog selection remains the separate
`gas_model` argument; the production comparison uses `janaf_condensed` with
76 gases and 67 retained condensate candidates.

```python
options = {
    "h2_he_cm3_mol": 10.0,
    "water_cross_temperature_policy": "extrapolate",
    "trace_pair_policy": "zero",
}
```

All three fields are explicit. The water-cross alternatives are `extrapolate`
and `hold_2000`; the latter holds only the H2-H2O and He-H2O cross coefficients
at their 2000 K values above that temperature. The H2-He coefficient is an
explicit constant scenario, in cm3/mol. These choices do not define measured
uncertainty bounds. ExoEOS records the coefficient sources, temperature
assumptions, and the unbounded higher-virial and unmodelled-pair limitations.

`m2_gas_eos.build_gas_eos(checkout, full_species, options)` loads the requested
ExoEOS example provider. Its complete species order is passed to
`make_atmosphere_phase(setup, gauge, gas_eos=model)`. Only the H2/He/H2O block
has assigned pair coefficients; every other pair is explicitly zero. The
denominator remains the total number of gas moles. In particular, a zero-pair
trace species has `ln(phi) = -ln(Z)` in this density-virial model; it does not
silently become a separate ideal species.

ExoEOS owns the residual scalar, chemical-potential corrections, density and
temperature-dependent coefficients. ExoGibbs adds the extensive residual
`G_res/(RT)` to the primitive ideal gas and pure-cloud scalar. The local parcel
solver repeatedly freezes the current full-mixture fugacity coefficients for
the existing gas/cloud solver, then reevaluates them at the resulting mixture.
The maximum change in `ln(phi)` must be at most `1e-11` within 64 iterations.
This is a solver tolerance, not a physical uncertainty. A failure returns no
accepted parcel. Every final parcel passes fresh elemental and chemical KKT
audits at its actual composition. Its independent primitive AD derivative
includes the same residual scalar. Standard potentials and elemental gauges
remain separate from composition-dependent fugacity.

Before solving each new T/P, the interval gas verifier requires a positive
global entropy-curvature bound over the complete nonnegative simplex.
The [gas certificate](m2_common_plane.md) includes the nonideal gas lower bound
and the residual energy of the exactly repaired primitive primal. The published
source gas callback, independent contact audit and reference-gauge audit all
use the same EOS. Exact-zero elemental budgets remove unsupported species from
the local solve and restore their zero amounts in the full catalog.

The callback exposes `gas_eos` so that the inventory consumer can reuse the
same model for every column parcel and for `model.mass_density(T_K, P_Pa, x, w)`.
The consumer must pass physical pressure in Pa to that density method and use
gas molar masses in kg/mol in the complete species order. Retained cloud mass
is accounted for separately by the existing column support-density closure.
Changing density alone does not constitute this coupled comparison.

The source metadata and basal parcel both preserve `gas_eos = model.parameters`
at the actual source T/P, including options, full pair matrix, species order,
provider file hashes and source provenance. Each parcel also records its
global convexity receipt, full `gas_lnphi`, `gas_residual_gibbs_rt`, and fugacity
iteration convergence. The common-plane verifier replays the recipe and binds
these records to the same source and elemental plane. Historical ideal results
and their physical-adoption status are unchanged by this implementation.

Focused CPU checks cover zero-coefficient recovery, nonideal cloud saturation,
trace dilution, exact-zero support, independent scalar derivatives, contact
consistency, and rejection of unavailable convexity or failed fugacity iteration:

```sh
JAX_PLATFORMS=cpu JAX_ENABLE_X64=1 python -m pytest -q \
  tests/unittests/examples/m2_gas_eos_test.py \
  tests/unittests/examples/m2_gas_global_test.py
```

These checks establish software behavior on small test systems. They do not
reclose the BSE atmosphere, generate fresh condensed-phase stability evidence,
or decide the physical adequacy of the ideal-gas approximation. Those results
require separately preserved source, column, pressure-root and phase runs.
