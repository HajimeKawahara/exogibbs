# Finite background condensates and reference contrasts

The optional `janaf_condensed` catalog retains 76 gases and adds 41 neutral
background-element condensates to the original 26. Existing temperature
eligibility and solid/liquid segments are kept. It uses the same finite
thirteen-element atmospheric ledger; it does not add an external reservoir.

`native_standard_comparison.json` evaluates FastChem standards against
preserved native unit-composition trials from the linked, hashed archive.
No new native evaluation or phase-boundary calculation is performed. Native
oxide masses independently determine the number of formula moles; the saved
chemical trial supplies a second check of native energy normalization.
Only eligible FastChem phases are reported, together with selected solid or
liquid branches and segment limits. Native unit endpoints may include
internal ordering, and FastChem pure-phase standards omit pressure terms.
Matching formulas therefore do not establish identical phases or models.

At 2173.15 K and 267.202834 bar, representative FastChem minus native values
are shown below in RT per formula mole. These contrasts are not measured
errors or calibrated uncertainty intervals.

| FastChem formula | Native endpoint | Difference / RT | Phase qualification |
|---|---|---:|---|
| Al2O3 | corundum | -0.063846 | FastChem solid, upper solid segment 2327 K |
| TiO2 | rutile | +0.444505 | FastChem liquid above 2130 K; native solid |
| MgAl2O4 | spinel | -0.668262 | FastChem solid below 2408 K; native endpoint ordering may differ |
| CaAl2Si2O8 | alkali-feldspar | +7.012535 | Formula scale independently 1; distinct endpoint model |
| CaAl2Si2O8 | plagioclase | +8.091038 | Formula scale independently 1; distinct endpoint model |
| Al2SiO5 | sillimanite | +0.533528 | FastChem kyanite; different polymorph |

Mg3P2O8(s,l), which switches to liquid at 1626 K with a 4500 K fit upper
limit, has no same-formula native comparator in these records. Native
whitlockite is Ca3P2O8. An absent match is not evidence that the P transfer
path can be omitted.

To reproduce, run from this checkout with its `src` on PYTHONPATH:

```sh
python results/m2_finite_condensates/20260927/compare_saved_standards.py \
  --native-archive /path/to/exogibbs/results/m2_stability_search/20260927/explicit_coordinates \
  --exoeos-evaluator /path/to/exoeos/examples/melts_liquid_evaluator.py \
  --output /path/to/new-standard-comparison.json
```

Use the native archive commit and evaluator hash pinned in the JSON, with
JAX CPU and x64 enabled. The script reads the saved host T/P and rejects
existing outputs. Original focused-test logs, including an initially harder
control where the old 26-condensate model itself failed, are retained.
The first full suite was intentionally stopped under memory pressure; its
partial log and termination receipt are also retained, without claiming a
completed suite or fresh source result.
