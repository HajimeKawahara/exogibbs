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

## Pinned JANAF assessment

`--janaf-atomic-references` opts into the independently sourced six atomic
references in [data/janaf_atomic.json](data/janaf_atomic.json). The input
contains exact numeric excerpts of the linked NIST-JANAF tables for Al, Ca,
K, Ti, Cr and P, and independent Mg/Fe/Na checks on three existing anchors.
The original seven atmospheric anchors remain unchanged. The data are
transcribed numeric excerpts, not byte copies of downloaded pages; the
helper verifies their pinned SHA-256 before evaluation.

```sh
python examples/metal_silicate/run_m2_omitted_gas.py \
  --source /path/to/closure.json --layers 16 --root-index 0 \
  --mole-fraction-target 1e-8 --janaf-atomic-references \
  --output /path/to/new-janaf-screen.json
```

The imported convention is

```text
G_atom(T, 1 bar) = Hf_atom(298.15 K) + [H_atom(T)-H_atom(298.15 K)] - T*S_atom(T).
```

It uses absolute entropy and the fixed elemental enthalpy references at
298.15 K, matching the convention of the existing source Shomate construction.
It does **not** use the JANAF `delta-f G(T)` column, which subtracts the
chemical potential of the reference element at the evaluation temperature.
For example, the latter is zero for K(g) above its boiling point, while the
required absolute chemical potential is not zero. Phosphorus retains JANAF's
white-phosphorus enthalpy reference at 298.15 K.

The evaluator interpolates G with cubic Hermite polynomials and tabulated
`-S` slopes between 2100 and 2400 K, with no extrapolation or interpolation
from the separate 298.15 K enthalpy datum. It reports the difference from
linear interpolation of H and S. This is numerical sensitivity, not an
uncertainty bound. The redundant JANAF Gibbs-function column independently
checks the transcribed H and S to table rounding. Source Mg/Fe/Na differences
are reported in RT without fitting away any offset. Agreement on these three
atoms does not validate every gas reaction, calibrate MELTS/alloy exchange,
or establish the coupled model's material domain.

`fixed_reservoir_demand` calculates `n_i = n_gas(saved) * exp(log(x_i))` on
the saved gas amount scale without renormalization. Its element demands are
compared with the **same local source's** complete public component ledger,
including host, alloy, gas and retained condensates. This is not the
planetary inventory; source and planetary normalization must not be mixed.
Saved gas species, formulas and individual amounts must match the public
ledger before it supplies the diagnostic amount scale.

The report separately identifies a fraction sum at least one, finite-source
budget violations, and species exceeding the declared trace target. The
first two reject a frozen-reservoir trace interpretation; the target remains
a user-declared screening threshold. Even when none fails, no depleted,
re-equilibrated source or pressure closure has been calculated and
`accepted_omission_bound` stays false. In particular, trace demands must not
be clipped to available inventories and then described as an equilibrium
solution or a coupled error bound.

The gas data do not determine the missing Mg/Al/Ca/K/Ti/Cr/P alloy standards
or activities. A failed screen requires an expanded finite-source model,
new contact checks and independently closed planetary responses before an
omitted-transfer error can be accepted.
