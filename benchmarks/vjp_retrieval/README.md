# ExoJAX NUTS VJP GPU runs

`run_exojax_nuts_gpu.csh` is a scheduler-independent tcsh launcher for the
four retrieval demonstrations:

- `gas_no_grid`: gas equilibrium with the default initialization;
- `gas_grid`: the same model with `GridEquilibriumInitializer`;
- `condensate_fixed_support`: a locally differentiable, fixed-support model
  with the full FastChem4 gas catalog but an explicit `C(s)`-only reduced
  condensate catalog;
- `condensate_grid`: the same fixed-support model with a shared gas-only grid
  and one local fixed-graphite grid per active layer supplying runtime initial
  values at every NUTS evaluation.

ExoJAX, NumPyro, and a CUDA-enabled JAX installation are optional runtime
dependencies and are not installed by ExoGibbs itself. The demos were
validated with ExoJAX 1.6, NumPyro 0.16.1, and JAX 0.4.30. Match the JAX build
to the CUDA installation on the execution host.

The production launcher requires a CUDA-enabled JAX installation and the
exact local ExoMol CO database directory, for example
`/data/exojax/CO/12C-16O/Li2015`. It checks both `nvidia-smi` and the JAX
backend, performs the demo's read-only preflight, and then requests 500 NUTS
warmup steps and 1000 posterior samples. It does not submit a PBS or Slurm job.

Run one case directly with:

```tcsh
benchmarks/vjp_retrieval/run_exojax_nuts_gpu.csh \
  gas_no_grid /data/exojax/CO/12C-16O/Li2015
```

The database path can instead be supplied through the environment:

```tcsh
setenv EXOJAX_CO_DATABASE /data/exojax/CO/12C-16O/Li2015
benchmarks/vjp_retrieval/run_exojax_nuts_gpu.csh gas_grid
```

Additional demo options are forwarded to both the preflight and retrieval
commands after the database argument. In particular, use `--quick` to exercise
the complete pipeline with the demo's short configuration before submitting a
production job:

```tcsh
benchmarks/vjp_retrieval/run_exojax_nuts_gpu.csh \
  condensate_grid /data/exojax/CO/12C-16O/Li2015 --quick
```

The gas quick profile uses at most 100 warmup steps, 100 samples, and tree
depth 8; the less expensive condensate smoke profile retains caps of 5 warmup
steps, 10 samples, and tree depth 4. Neither replaces the production
configuration.
Every sampling run records acceptance and trajectory diagnostics and exits
nonzero if any transition diverges, a parameter is completely stuck, or the
saved samples are incomplete or non-finite. ESS and R-hat are intentionally
not reported for these single-chain demo runs.
`run_status.json` lists the artifacts belonging to the current invocation and
uses `started`, `preflight_complete`, `complete`, or `failed` state; known
artifacts from an earlier invocation are invalidated at startup.

The condensate cases demonstrate the fixed-support VJP mechanics only. They
do not claim equilibrium or support closure against the other FastChem4
condensates. Each active layer has a local singleton-pressure grid that stores
converged fixed-support gas states and graphite amounts for the retrieval's
exact C/O composition rule and interpolates them together at runtime. Inactive
layers use a shared canonical uniform-metals physical-metallicity grid, which
is an approximate initialization source for the sampled C/O-only inventory.
Only the nominal support and layer partition remain frozen. At all eight prior
corners, the frozen baseline must converge, the grid and baseline CO VMR
profiles must agree, and the grid must not increase the maximum fixed-support
iteration count. The grid-side reverse-mode gradient is checked for
finiteness at the truth point; no second baseline gradient is computed. Grid
construction happens before NUTS and its time is recorded separately. No
speedup should be inferred without comparing both cases on the same hardware.
Full-catalog support discovery and its production-scale NUTS practicality
require a separate study.

Outputs are written under `results/vjp_retrieval/<case>/`, which is ignored by
Git. Set `EXOGIBBS_VJP_OUTPUT_ROOT` to choose a different output root. The
launcher enables JAX 64-bit mode, requires the CUDA platform, disables JAX GPU
memory preallocation, and selects a non-interactive Matplotlib backend. It
also sets `NUMBA_DISABLE_JIT=1` to avoid the read-only RADIS cache issue in the
current environment; the retrieval calculation itself is compiled by JAX.

The corresponding narrative tutorials are generated from the notebooks under
`documents/ipynb/`. Notebook-to-RST conversion never runs the expensive NUTS
calculation. The converter additionally requires `nbformat` and `nbconvert`;
the committed output was checked with versions 5.1.3 and 6.1.0, respectively:

```bash
python documents/ipynb/convert_vjp_retrieval_notebooks.py
python documents/ipynb/convert_vjp_retrieval_notebooks.py --check
```
