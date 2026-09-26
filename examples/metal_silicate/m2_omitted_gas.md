# Omitted-gas reference screen

The M2 atmosphere currently omits Al, Ca, K, Ti, Cr and P. This screen selects
all neutral packaged FastChem4 gases containing these elements and no element
outside the saved thirteen-element source. It evaluates a local trace-gas
condition at the **saved** temperature, pressure and elemental potentials.
It performs no native equilibrium or planetary pressure solve.

For an ideal gas at the common 1 bar standard,

```text
log(x_i) = A_i^T lambda - h_i(raw) - A_i^T gauge - log(P/bar).
```

The seven existing atmospheric reference anchors are preserved. The six new
atomic references are unavailable by default. Their zero in the FastChem
atomic reference is not an absolute energy aligned with MELTS. Instead of
inventing that alignment, each affected gas records the linear inequality on
the missing references required to keep its mole fraction below the explicit
screening target. This identifies the quantitative thermochemical information
needed before expanding the coupled model.

If temperature-matched additional atomic references are supplied, the same
equation returns a conditional equilibrium trace estimate. It does not
normalize hypothetical traces into an apparent new equilibrium. Logarithms
remain available when exponentiation would overflow; no gas is declared
exactly absent from the sign of a pure-phase insertion energy.

```sh
python examples/metal_silicate/run_m2_omitted_gas.py \
  --source /path/to/closure.json --layers 16 --root-index 0 \
  --mole-fraction-target 1e-8 --output /path/to/new-screen.json
```

The optional `--atomic-references` JSON uses the following schema. Values must
be obtained independently; this is a schema illustration, not a calibration:

```text
{
  "temperature_K": <saved temperature>,
  "pressure_standard_bar": 1.0,
  "elements": {
    "Al": {"value_rt": <absolute atomic chemical potential / RT>,
           "source": "<reference, convention conversion and uncertainty>"}
  }
}
```

The target is a declared numerical screening threshold, not an empirical
error tolerance. Fixed-reservoir estimates omit depletion, accompanying O/H,
and re-equilibration of the source, incipient alloy and planetary column.
They cannot establish a one-percent atmospheric error or a boundary shift.
The Mg/Al/Ca/K/Ti/Cr/P metal paths retain their missing alloy standard/activity
requirements; gas thermochemistry does not supply those quantities.

The thermochemical reference convention must be checked against independent
gas/melt reactions before interpreting an expanded calculation physically.
[NIST-JANAF](https://janaf.nist.gov/) supplies potential atomic reference data;
[VapoRock](https://doi.org/10.3847/1538-4357/acbcc7) supplies relevant silicate
vapor comparisons. Neither validates the current H-bearing finite system by
itself. This screen always retains `accepted_omission_bound=false`.
