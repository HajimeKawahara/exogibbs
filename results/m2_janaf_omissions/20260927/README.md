# JANAF omitted-gas demands at three preserved BSE roots

These fresh diagnostic evaluations use ExoGibbs
`c861b469a44561529d6974edce1dea9e4a94cc3c` and the unchanged accepted
16-layer source records preserved in ExoInventory
`564e73ec97224b66285bbe63c9234a9564fe4561`. Each JSON identifies the immutable
source URL and SHA-256. This assessment performs no native chemistry solve,
planetary pressure closure, or replacement of the historical source result.
The declared per-species trace target is `x <= 1e-8`.

| Total H input (mol atoms) | Saved bottom pressure (bar) | Sum of 41 omitted fixed-reservoir fractions | P demand / same local source's total P | Trace screen |
| --- | ---: | ---: | ---: | --- |
| `1e22` | 59.8248011676 | 0.00617771580 | 3.92232504 | rejected |
| `1e24` | 267.202834162 | 0.00932865335 | 28.93075997 | rejected |
| `2e24` | 432.298471238 | 0.02156249385 | 125.41384406 | rejected |

At the central point the leading fractions are P2 `0.00514339`, PH2
`0.00210994`, PH3 `0.00155842` and PO `0.000346500`. On the unchanged saved
gas amount scale, their combined P-bearing gas demand is
`8.573618767e21 mol` P, whereas the complete local source contains only
`2.963495870e20 mol` P. All three points exceed their available P.
These unnormalized trace demands therefore cannot be interpreted as a
finite equilibrium. They are not additions to the historical atmosphere.
A fraction sum below one does not resolve this elemental depletion failure.

The JANAF atomic reference is reconstructed from formation enthalpy at
298.15 K plus the atomic enthalpy increment minus absolute `T*S`, at
`p0 = 1 bar`. The six new atomic inputs preserve all seven old anchors.
Independent JANAF-minus-saved Mg/Fe/Na controls are respectively
`-5.93389e-6`, `6.25818e-5` and `7.99120e-4 RT` at 2173.15 K; no correction
is fitted. The largest Hermite-G versus linear-H/S difference is
`0.000646371 RT`. These differences check reference conventions and numerical
interpolation. They are not empirical uncertainty estimates, validation of
P-bearing melt standards, or bounds on changes in metal stability.

All reports retain `accepted_omission_bound=false`. The quantified failure
requires a depleted finite-source gas/melt model, new contact checks and
independent planetary closures before accepting an omitted-gas error.
Mg/Al/Ca/K/Ti/Cr/P transfer into metal additionally requires alloy activities
and standards. Neither the existing finite inventory nor the trace fraction
is a bound on the re-equilibrated composition or metal boundary.

The raw JSON files, compact summary and stdout are preserved here. The
manifest checks their bytes and the reproduction script. The script checks
each closure against its source git blob before assessment. Replay code is
pinned at `09453dd87e0aab9d7c0b76f91a7d9d024db7c848`, whose sole change
from the executed code is a gallery-title docstring. An AST comparison
excluding that docstring and a fresh replay verify unchanged numerical
results; the new evaluator provenance hash is retained as a new execution:

```sh
JAX_PLATFORMS=cpu JAX_ENABLE_X64=1 PYTHONPATH=src \
  python results/m2_janaf_omissions/20260927/reproduce.py \
  --inventory-checkout /path/to/exoinventory \
  --output /path/to/new-assessment-directory
```

The primary tables and implementation conventions are linked in the
[diagnostic documentation](../../../examples/metal_silicate/m2_omitted_gas.md).

## Validation

The full unit suite completed with 1872 passed, 6 skipped and one failure
(1770.41 s). The failure was an existing test requiring `exoeos.ma_interval`,
which the installed optional provider lacked; five skips also required the
explicit EOS checkout. All 21 affected provider-dependent tests passed
with ExoEOS `44b977847ea7ed1e16377907722277a3e75bc9d7` selected explicitly
(33.12 s, no skips). The remaining skip requires the SCM-generated version
module. There is no unresolved test failure; this is a full run plus a
provider-specific recheck, not a claimed single all-green full run.

After the title-only documentation change, 33 targeted checks passed.
The final source AST agrees with the executed implementation after removing
only the module docstring, and a fresh three-case replay agrees in every
non-provenance field. The HTML build succeeds with existing documentation
warnings, none in the new JANAF module. `validation.json` and preserved logs
record the individual checks.
