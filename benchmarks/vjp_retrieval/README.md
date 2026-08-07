# ExoJAX NUTS VJP GPU runs

`run_exojax_nuts_gpu.csh` is a scheduler-independent tcsh launcher for the
three retrieval demonstrations:

- `gas_no_grid`: gas equilibrium with the default initialization;
- `gas_grid`: the same model with `GridEquilibriumInitializer`;
- `condensate_fixed_support`: a locally differentiable, fixed-support model
  with the full FastChem4 gas catalog but an explicit `C(s)`-only reduced
  condensate catalog.

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
  condensate_fixed_support /data/exojax/CO/12C-16O/Li2015 --quick
```

The quick configuration is only an end-to-end smoke test. Its five warmup
steps are too few for scientific inference; in particular, short gas-case
runs can consist entirely of divergent transitions.

The condensate case demonstrates the fixed-support VJP mechanics only. It
does not claim equilibrium or support closure against the other FastChem4
condensates. Full-catalog support discovery and its production-scale NUTS
practicality require a separate A100 study.

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
