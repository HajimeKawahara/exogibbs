# Gas-only fixed-support warmup control

The previous lifecycle compiled and ran a fixed-support condensate solve
when every layer had empty initial support. Its output contributed timing
and a `fixed_shape_warmup` flag, but did not provide the accepted gas state.
Removing that unused solve preserves gas audits, exact refinement and
chemical fallback. Active condensate calculations retain their existing path.
`return_diagnostics=True` and every acceptance threshold are unchanged.

| Saved finite parcel at 250 bar | Cold time, old → new | Peak RSS, old → new | Physical output |
| --- | --- | --- | --- |
| 76 gas / 26 cloud, 2173.15 K | 46.20 → 2.05 s | 1001.75 → 391.38 MiB | Exactly equal |
| 76 gas / 67 cloud, 2173.15 K | 47.97 → 1.99 s | 999.95 → 391.00 MiB | Exactly equal |
| 76 gas / 67 cloud, 1000 K, 8 active clouds | 48.97 → 48.86 s | 1020.57 → 1020.19 MiB | Exactly equal |

Equality covers every parcel field, primitive gas/cloud amounts, Gibbs energy,
atom potentials, independent scalar AD value/gradient, conservation, solver
status/tier/route and independent numerical acceptance. The low-temperature
control uses the same saved atom budget/reference gauge at 250 bar; it is not
a new closed column. Each new measurement used a fresh process on CPU 36
with persistent compilation caching disabled. Timings are single samples;
no full-source or planetary speedup is established by this experiment.

`comparison.json` retains old/new hashes. Original pre-change raw measurements
are reused from [Inventory's preserved diagnostics control](https://github.com/HajimeKawahara/exoinventory/tree/e9806e9dd1e61c640d0c12b1d65b0c9b7732b720/examples/subneptune_taxonomy/finite_melt/validation/20260927_m2_finite_response/parcel_diagnostics_benchmark).
The 250 bar inputs retain their original reconnection hashes, distinct from
pressure-closed roots. `benchmark.py` is the exact executed script, including
its original checkout/input paths. The prototype was `f2a87ce` on source
runtime `7cf41f1`; delivery rebases the same core change onto the selected-host
audit and existing O-standard support. The manifest records this distinction.

Validation: 83 lifecycle/route/initialization tests and 17 M2 atmosphere tests
passed, including wrong-scalar/wrong-potential rejection. The dependency
integration passed 54 audit/scenario tests; actual frozen O and O+H scenario
JSONs normalize successfully without a source solve. The diagnostic key
`fixed_shape_warmup` remains present and false; profile fixed-support timings
are zero when that solver is not used. Full-suite CI is separate from these
focused local checks. No archived scientific run was modified.
