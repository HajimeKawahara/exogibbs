# Milestone 1 local chemistry diagnostic, 2026-09-14

`central.json` is a fresh ExoGibbs calculation of the finite source at 2350 K,
100 bar, C=N=S=0 and He/H=0.1, followed by five local gas/condensate controls.
The upper point at 1400 K, 0.1 bar uses the same gas atom inventory. These are
independent parcels; no global inventory, pressure root, column, opacity or
spectral error gate is closed here.

| Local control | T (K) | P (bar) | H2O/H2 | Retained cloud mass fraction |
| --- | ---: | ---: | ---: | ---: |
| Frozen source gas | 2350 | 100 | 0.00837657194 | — |
| Shared nine gases | 2350 | 100 | 0.00837656408 | 0 |
| Expanded gases | 2350 | 100 | 0.00831731937 | 0 |
| Expanded gases + pure condensates | 2350 | 100 | 0.00831704180 | 0.003114584 |
| Expanded gases | 1400 | 0.1 | 0.00835618044 | 0 |
| Expanded gases + pure condensates | 1400 | 0.1 | 0.00592593153 | 0.044662327 |

All five local controls pass the independent element/mass and KKT audits.
The maximum per-element relative residual is `4.00e-11`. At 2350 K the upper
model's gas-only source-reaction residuals are approximately `-0.001333` (R9)
and `-0.02679` (R18), distinct from its near-zero own-model stationarity error.
This difference is retained, without adjusting the source standards.

SiO(s) appears at the boundary; the cooler condensed parcel contains Fe(s,l),
MgSiO3(s,l) and Na2Si2O5(s,l). A small total cloud mass fraction does not imply
small depletion of each rock-forming element. These results do not accept the
boundary for a retained atmosphere. The next physical gate is to evaluate this
recondensation and model difference over Inventory's actual trial conditions,
then assess their impact against the intended spectral signal.

On this CPU run, source evaluation took about 1.3 s, the first condensed solve
about 45 s (including compilation), and the subsequent cooler condensed solve
about 1 s. See the JSON for exact times and actual code/data provenance. The
record was generated before committing the implementation; its Python-file
hash identifies the source stored in this change. Timings are local to this
process and are not an estimate of the full milestone cost.

Reproduce from the repository root with a new output path:

```bash
JAX_PLATFORMS=cpu JAX_PLATFORM_NAME=cpu JAX_ENABLE_X64=1 PYTHONPATH=src \
python examples/metal_silicate/m1_chemistry.py \
    --pressure-bar 100 --point 1400 0.1 \
    --output results/m1_chemistry/reproduction.json
```

The molecular and condensate coefficients are packaged FastChem4 data; the
source retains its separately identified frozen GCE reaction model. The
original temperature upper bounds are enforced, with no new joint calibration
claim. See `examples/metal_silicate/m1_chemistry.md` for the full amount contract.
