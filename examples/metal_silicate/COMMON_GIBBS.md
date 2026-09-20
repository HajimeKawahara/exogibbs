# Common-energy and BSE mechanism checks

`common_gibbs.minimize_gibbs` minimizes the scalar extensive `G/(R*T)`
provided by the existing full-potential callbacks. It shares `LocalProblem`,
`PhaseState` and the absolute elemental ledger with `solve_full_potentials`;
the original local-root path remains available unchanged. No model equations
are moved out of ExoEOS and no reaction-specific offsets are permitted.

The numerical variables are nonnegative component amounts divided by the
sum of the specified atom amounts. Linear equality constraints preserve
those atom amounts. A feasible LP supplies an initial composition without
adding material. Explicit initial amounts must satisfy the same inventory.
Exact-zero element budgets retain the original static reduced support, and
an equality-forced zero component is not replaced by a positive floor.
Rank-deficient elemental potentials are reported as unresolved.

SLSQP first minimizes the supplied scalar. An explicit `PhaseEvaluationError`
from a provider makes an unavailable line-search trial cost infinity, so
another trial can be attempted without changing any amount or energy law.
Initial/final states and independent audits still require actual properties;
other provider contract violations remain errors. For a converged interior case,
stationarity refinement is allowed only without an energy increase. The
returned state is evaluated again, with independent atom and KKT audits,
scalar-energy derivatives and an extensivity check. Ideal phases expose an
independent JAX scalar through the optional callback attribute
`energy_value_and_grad_rt(T, P, n)`; its value and derivative must agree with
the full-potential callback. BSE alloy callbacks use the ExoEOS total scalar
in the same way. Other providers use phase-local, adaptive-step Richardson
finite differences. The independent AD path remains usable when a trace
component's energy increment is smaller than the total energy's precision. Acceptance
requires relative atom error below `1e-9`, dimensionless KKT error below
`1e-8`, derivative agreement below `5e-6`, and extensivity error below `5e-9`.
A trace-component derivative that cannot be resolved in float64 is recorded
as unresolved, rather than accepted using a stale or untested potential.

Ideal mixing uses the continuous `n*log(n)=0` energy limit. A present ideal
phase can contain an exactly zero component with insertion potential
`-inf`. A wholly absent ideal phase has zero extensive energy and undefined
chemical potentials. A phase that disappears requires a separate phase
candidate; local KKT acceptance alone cannot certify its stability. The
provider must define finite continuous energy on any endpoint it accepts.

## Independent reference and provider checks

The synthetic H-Si-O test uses `K_R=0.010`, `eta=0.020`, `P=1 bar`, and atom
amounts `(H,Si,O)=(2,20,40) mol`. It uses the silica/OH-equivalent scalar
from the M2 reference design: gas H2/H2O/SiO, positive silica host `s`, and
water-equivalent dissolved amount `u`. The host derivative includes
`-2*u/s`; the capacity is not imposed independently of its host response.
The input reference is `planet_atmosphere_liquid_surfaces_ja_v5.pdf`, SHA-256
`99fa20d89583a2f79a83122d7a8ec3c2fab800a8257cd971cc6b8342f8f880a8`,
as recorded in the ExoInventory M2 reference design. These synthetic
coefficients do not calibrate BSE or the multicomponent alloy.

Tests independently minimize the two-variable reduced energy and check
absolute atom reconstruction, dry and fixed-host controls, automatic and
step-refined numerical derivatives, host cross response, convex reference
curvature, inventory scaling, elemental gauge invariance, and fresh final
provider calls. The fixed-host control removes SiO exactly; the dry control
removes dissolved water exactly. Releasing either constraint cannot raise
the independently verified minimum in this convex reference.

Additional tests minimize through the actual `make_melts_h2_phase` adapter
using a synthetic offline property evaluator with Al/Ca/background atoms,
and through the actual ExoEOS `MaFeSiOHLiquid` and `total_solution_state`
provider in a finite hydrogen-exchange control. Those interface checks do
not establish physical standard alignment or liquid stability.

Run the targeted checks with the explicitly selected provider checkout:

```sh
JAX_PLATFORMS=cpu JAX_ENABLE_X64=1 PYTHONPATH=src:/path/to/exoeos/src pytest -q \
  tests/unittests/examples/metal_silicate_common_gibbs_test.py \
  tests/unittests/examples/metal_silicate_melts_coupled_test.py \
  tests/unittests/examples/metal_silicate_full_potential_test.py
```

## BSE native-property connection

`run_bse_common_gibbs.py` consumes the exported stage-1 thirteen-element
absolute inventory JSON directly; it does not import ExoInventory. The
ledger's oxide amounts are converted to the provider's MELTS endmembers and
independently checked against its dry-rock atoms. The canonical initial
ledger adds only the independently specified H2/He gas. It has no existing
core, dissolved hydrogen, initial water, or initial metal. Initially zero
native MELTS water and ferric iron remain available as equilibrium unknowns.
Al, Ca, K, Ti, Cr and P remain active host components.

`build_bse_problem(...)` returns `(record, budget, callbacks, initial,
metadata)` for the scalar solver or a separate phase-selection workflow.
Native MELTS calls use the amount scale corresponding to 100 g of the dry
rock; their energy is scaled back to the full physical mol basis. This is
an extensive property evaluation, not a rescaling of the specified H/He
or planetary inventory.

```sh
JAX_PLATFORMS=cpu JAX_ENABLE_X64=1 PYTHONPATH=src:/path/to/exoeos/src \
python examples/metal_silicate/run_bse_common_gibbs.py \
  --inventory /path/to/baseline_inventory.json \
  --exoeos-checkout /path/to/exoeos \
  --runtime /path/to/alphamelts-py-2.3.2-ubuntu_22_04-x86_64 \
  --python /path/to/native-worker/python \
  --temperature 2173.15 --pressure 1 --metal-absent \
  --output /tmp/bse_scalar_absent.json
```

Omitting `--metal-absent` requests the metal-bearing candidate. A suppressed
metal phase is a constrained control, not evidence of stable absence.
Reports preserve the input digest, source receipts, provider/code digests,
absolute amounts, settings, diagnostics and unavailable-state reasons.
Existing files are not overwritten. The archived MORB runner and results
retain their original definition.

The source thermochemistry uses its explicitly recorded 2350 K branch at
the requested temperature, while MELTS supplies its own full potentials.
The current BSE temperature is a formal test condition; common standards,
host-specific H2 calibration, alloy H interactions and pressure response,
liquid stability, omitted transfer bounds and upper-atmosphere alignment
remain missing. A numerical result cannot certify M2-A or M2-B. Both gates
remain pending even if a conditional local energy calculation converges.

## Recorded native BSE trial

The [2026-09-20 native report](../../results/m2_common_gibbs/20260920_bse_conditional/attempt_02_trace_derivatives.json)
uses the full absolute stage-1 BSE inventory, `T=2173.15 K`, `P=1 bar`, and
an explicitly constrained metal-free branch. It reached scalar convergence
and stationarity refinement after 524 native melt evaluations:

| Check | Recorded value |
| --- | ---: |
| Initial `G/(RT)` | `-9.51053011563905e25 mol` |
| Final `G/(RT)` | `-1.153430723340075e26 mol` |
| Maximum relative element residual | `8.88e-16` |
| Maximum reaction residual | `7.11e-15` |
| Maximum component KKT residual | `3.41e-13` |
| Extensivity error | `8.83e-15` |
| Numerical acceptance | **Unresolved** |

The unresolved audit is the scalar finite-difference derivative of trace
components: their tiny energy increments cannot be resolved against the
full extensive energy in float64. The largest reported derivative mismatch
is `1.09` in RT units. This numerical limitation is separate from all the
physical M2-A/M2-B gaps above; it is not evidence of phase absence or a
validated BSE result.

The [initial failed attempt](../../results/m2_common_gibbs/20260920_bse_conditional/attempt_01_native_endpoint.json)
is retained with its thirtieth trial composition. Native MELTS rejected
near-zero SiO2/Fe2SiO4/Na2SiO3 components even though the supplied amounts
were nonnegative. The [endpoint property probes](../../results/m2_common_gibbs/20260920_bse_conditional/endpoint_property_probe.json)
record the response to tiny silica increments on that sample: the original
native failure changed to a returned-endmember consistency failure. Those
probes change only diagnostic samples; no such increment is inserted into
the optimizer's fixed atomic inventory. The successful retry rejects
unavailable trial states and preserves all final component amounts and
failed-state metadata in the report.
