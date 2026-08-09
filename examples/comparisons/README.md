# Comparison examples

The comparison examples have two distinct roles.

The current scripts include readable companions to the formal FastChem 4
production comparison in `benchmarks/fastchem4` and focused literature
demonstrations:

- `comparison_with_fastchem4_gas.py` compares two gas-only calculations over
  a pressure profile.
- `comparison_with_fastchem4_condensates.py` compares the production
  gas-plus-condensate solver at the four v0.4 validation points. Its optional
  `l-dwarf` mode makes a 2-by-2 atmospheric-profile plot with gas and
  condensate rows and FastChem and ExoGibbs columns. Each gas panel overlays
  gas-only equilibrium and the gas phase in equilibrium with condensates.
- `comparison_with_visscher_2006_na2s_morley_2012_kcl.py` scans independent
  reduced H/He/K/Cl and H/He/Na/S systems at 1 bar. It checks the Visscher et
  al. (2006) Na2S and Morley et al. (2012) KCl analytic fits and optionally
  compares the same local-equilibrium catalogs with FastChem 4.
- `comparison_with_visscher_2010_forsterite_enstatite.py` compares two
  one-bar reduced protosolar systems that differ only by whether enstatite is
  an allowed condensate. It demonstrates forsterite formation, enstatite
  takeover, silica suppression, and the alternative silica phase when
  enstatite is excluded.
- `comparison_with_ito_2025.py` compares the supplied Ito et al. (2025)
  H/O/Si rainout profile with ExoGibbs and FastChem 4. For each target above
  the ground, both solvers receive the gas-derived elemental inventory from
  the one-grid-higher-pressure Ito layer.
- `comparison_with_ito_2025_rainout.py` uses the Ito Layer 1 gas as the single
  lower boundary, then propagates each solver's own gas-phase H/O/Si inventory
  through Layers 2 and above. ExoGibbs uses its bottom-scan rainout option and
  FastChem uses its native `cr` profile mode.

The historical entry points `comparison_with_fastchem.py`,
`comparison_with_fastchem_extended.py`, and
`comparison_with_fastchem_cond.py` are retained as small compatibility
wrappers around these current implementations.

The scripts that invoke FastChem 4 replace the historical PyFastChem 3
examples with the current ExoGibbs public APIs and an independently built
FastChem 4 standalone executable. FastChem results are aligned and plotted
only after ExoGibbs has solved; they are never used as ExoGibbs constructor,
initialization, support-selection, retry, or route inputs.

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

python examples/comparisons/comparison_with_fastchem4_condensates.py \
  --fastchem-executable /path/to/fastchem \
  --profile l-dwarf

python examples/comparisons/comparison_with_visscher_2006_na2s_morley_2012_kcl.py \
  --fastchem-executable /path/to/fastchem

python examples/comparisons/comparison_with_visscher_2010_forsterite_enstatite.py

python examples/comparisons/comparison_with_ito_2025.py \
  --fastchem-executable /path/to/fastchem \
  --input external_data/Ito_2025.xlsx

python examples/comparisons/comparison_with_ito_2025_rainout.py \
  --fastchem-executable /path/to/fastchem \
  --input external_data/Ito_2025.xlsx

python examples/comparisons/comparison_with_fastchem_initializer.py \
  --fastchem-executable /path/to/fastchem

python examples/comparisons/comparison_with_hsystem.py
python examples/comparisons/comparison_with_hcosystem.py
python examples/comparisons/comparison_with_ykcode.py
```

PNG files are written under `results/` by default. Add `--show` to display the
Matplotlib window.

The Ito comparison excludes Layer 1 because the ground calculation uses a
different magma-interface and water-solubility system. Target Layer `i >= 2`
uses the H/O/Si inventory reconstructed from Ito Layer `i - 1`; solver output
is not recursively propagated. ExoGibbs uses Ito's exact five gas molecules
and two condensates. FastChem uses the same molecules and condensates but
necessarily adds the H/O/Si elemental reference gases. Helium remains outside
equilibrium chemistry and is
restored with the fixed Ito EOS ratio `He/H2 = 0.1896551724`, including its
effect on reactive partial pressure. The restartable NPZ, layer CSV, JSON
summary, and comparison PNG are written under `results/ito_2025/`.
The ExoGibbs input is uniformly rescaled when rainout leaves a trace element
below the production solver's absolute numerical floor. This numerical gauge
does not change H/O/Si ratios or gas mole fractions, and reported condensate
amounts are scaled back to the original input convention. Minimum element
gauges are tried from `1e-3` through `1e-7`, using the largest convergent gauge
for the best trace-species conditioning. Resume checkpoints
include hashes for the script, workbook, FastChem executable, and both
thermochemical tables; use `--retry-failed` to repeat only unsuccessful layers.
The figure uses species-specific panels and clips plotted mole fractions below
`1e-45`; raw values remain in the CSV and NPZ outputs.

The propagated-rainout comparison also excludes Layer 1 from the shared
equilibrium solve, but uses its gas composition once as the lower boundary.
The workbook's ground-to-top Layers 2+ are reversed for the ExoGibbs
top-to-bottom profile API; the returned arrays are restored to workbook order.
FastChem receives workbook order directly because native `cr` advances from
the first row upward. The He pressure correction is a whole-profile fixed
point for both solvers, so every iteration restarts from the same Layer 1
boundary. Its CSV, JSON, NPZ, and eight-panel comparison PNG are written under
`results/ito_2025_rainout/`. The production calculation requires an accepted
zero-barrier physical refinement for every positive condensate and does not
accept the legacy trace-capacity terminal tier. Rainout propagation uses
`b_current - A_cond @ m_cond`; `A_gas @ n_gas` remains an independent
cross-check. Once an ExoGibbs element target reaches exact zero, reported gas
species requiring that element are masked and renormalized; the raw gas-element
inventory remains in the NPZ audit arrays. The CSV, NPZ, JSON, and figure also
report SiO saturation and support state, conservative-inventory mismatch,
numerical-depletion events, and gas reintroduction after exact depletion. The
outputs retain FastChem's per-layer element-conservation mask.

On a typical CPU, the gas-only example finishes in a few seconds. The default
gas-plus-condensate example runs the real four-layer production lifecycle and
can take roughly 4--5 minutes. The 13-layer `l-dwarf` mode takes roughly
10 minutes on the same class of CPU. Either condensate run may not print
anything until the solve is complete.

The `l-dwarf` mode follows an explicit analytic trajectory from 1100 K at
`1e-4` bar to 2600 K at `1e2` bar. It is an illustrative local-equilibrium
profile with the same elemental budget at every layer, not a self-consistent
radiative-convective atmosphere or cloud model. It uses equilibrium
condensation without rainout or vertical transport. In the gas row, dashed
curves show a separate gas-only solve and solid curves with markers show the
gas phase when condensates are allowed. Both use the same elemental budget,
and neither result is used as input to the other solve. Their separation shows
the local gas-phase response to equilibrium sequestration rather than rainout
or cloud transport.

The examples are for inspection and visualization. Use
`benchmarks/fastchem4/run_production_comparison.py` when provenance,
preflight checks, and machine-readable metrics are required.
