# Comparison examples

The comparison examples have two distinct roles.

The current FastChem 4 scripts are readable companions to the formal
production comparison in `benchmarks/fastchem4`:

- `comparison_with_fastchem4_gas.py` compares two gas-only calculations over
  a pressure profile.
- `comparison_with_fastchem4_condensates.py` compares the production
  gas-plus-condensate solver at the four v0.4 validation points.

The historical entry points `comparison_with_fastchem.py`,
`comparison_with_fastchem_extended.py`, and
`comparison_with_fastchem_cond.py` are retained as small compatibility
wrappers around these current implementations.

They replace the historical PyFastChem 3 examples with the current ExoGibbs
public APIs and an independently built FastChem 4 standalone executable.
FastChem results are aligned and plotted only after ExoGibbs has solved; they
are never used as ExoGibbs constructor, initialization, support-selection,
retry, or route inputs.

Four restored examples preserve older validation lineages:

- `comparison_with_fastchem_initializer.py` repeats the historical
  grid-initializer experiment with the packaged FastChem-v3-compatible
  thermochemical data and equilibrium grid. It runs the current standalone
  FastChem adapter only as an independent comparison and also shows that grid
  and uniform initial guesses converge to the same final ExoGibbs state.
- `comparison_with_hsystem.py` compares H/H2 equilibrium and its temperature
  and pressure derivatives with an analytical solution.
- `comparison_with_hcosystem.py` compares the H/C/O reaction system and the
  CO elemental-abundance Jacobian with analytical references.
- `comparison_with_ykcode.py` checks the current solver against the archived
  500 K, 10 bar YK B4 composition snapshot.

These four files are traceability demonstrations, not additional FastChem 4
production acceptance points. In particular, the YK snapshot has no pinned,
independently runnable reference executable in this repository.

Build the FastChem v4.0.3 standalone executable as described in
`benchmarks/fastchem4/README.md`. Matplotlib is also required for these
visual examples; install it in the active environment if necessary:

```bash
python -m pip install matplotlib
```

Then run from the repository root:

```bash
python examples/comparisons/comparison_with_fastchem4_gas.py \
  --fastchem-executable /path/to/fastchem

python examples/comparisons/comparison_with_fastchem4_condensates.py \
  --fastchem-executable /path/to/fastchem

python examples/comparisons/comparison_with_fastchem_initializer.py \
  --fastchem-executable /path/to/fastchem

python examples/comparisons/comparison_with_hsystem.py
python examples/comparisons/comparison_with_hcosystem.py
python examples/comparisons/comparison_with_ykcode.py
```

PNG files are written under `results/` by default. Add `--show` to display the
Matplotlib window.

On a typical CPU, the gas-only example finishes in a few seconds. The
gas-plus-condensate example runs the real four-layer production lifecycle and
can take roughly 4--5 minutes; it may not print anything until the solve is
complete.

The examples are for inspection and visualization. Use
`benchmarks/fastchem4/run_production_comparison.py` when provenance,
preflight checks, and machine-readable metrics are required.
