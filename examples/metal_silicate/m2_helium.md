# Finite dissolved helium

`build_expanded_bse_problem(..., helium_solubility_model=...)` accepts `gas_only`
(the default), `guillot2012_olivine`, `guillot2012_morb`, and
`guillot2012_rhyolite`. The latter three append atomic `He_dissolved` to the
silicate. The canonical inventory remains dry host plus H2/He gas; He only
moves through the existing finite elemental constraints.

The EOS provider supplies one extensive trace-He Gibbs scalar on a declared
dry simulated-host capacity. The callback adds its host derivatives as well
as its He potential. Native water and dissolved H2 have zero dry mass weight;
He is excluded from the existing H2 mixing denominator. Gas pressure enters
through the complete retained gas, not through an imposed extra He reservoir.
The dissolved standard uses the actual retained `He1` gas standard plus the
same source element gauge, with a one-bar pressure standard.

Source metadata records `helium_solubility_model` and `helium_dissolution`,
including the complete augmented basis, actual T/P, raw/gauged gas anchor,
capacity, host mass weights, interpolation interval and provider hashes.
`host_ledger` still describes the underlying MELTS/water/H2 provider;
`helium_dissolution.component_order` describes the augmented phase.

`m2_helium.wrap_helium_phase(host_callback, eos_checkout, receipt)` reconstructs
the saved scalar for an independent host callback. It checks the provider
recipe and exposes independent scalar derivatives when the original host
does. `evaluate_host_stability` accepts `helium_dissolution` and
`exoeos_checkout`; its native candidates remain He-free. Candidate insertion
includes the He-derived host work, saved separately from the H2 dilution.
The bare `provider_properties` and its model identity remain unchanged.

Zero He preserves a water-only original host even at zero dry mass. Positive
He at zero dry mass is infeasible; the fixed-zero-dry-mass insertion limit is
positive infinity, not a smooth joint derivative. The fresh audit labels
nonfinite He endpoints explicitly while serializing their numeric value as null.

The three capacities are named constitutive choices, not an empirical BSE
error bracket. Temperatures beyond the chosen simulation interval are
rejected. No low-pressure He-metal or Na-metal transfer law is inferred.
`atmosphere.missing_paths` always lists these two paths and lists He-silicate
dissolution when the gas-only option is selected. A finite source and global
pressure response must be evaluated separately; this implementation alone
does not complete the physical material-domain assessment.
