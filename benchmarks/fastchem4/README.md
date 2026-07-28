# ExoGibbs production comparison with FastChem 4

This benchmark compares the ExoGibbs gas-plus-condensate production default
with an independent FastChem 4 standalone process at the same temperature,
pressure, elemental composition, and thermochemical data.

The comparison has a strict trust boundary:

- ExoGibbs runs through its public production API and therefore uses the
  fixed-support v2 condensate lifecycle.
- FastChem runs as a separate executable. Its public output is a comparison
  target only.
- FastChem values are never passed to an ExoGibbs solver constructor and are
  never used for initialization, support selection, retries, or route
  selection.
- The retired fixed-support v1 runtime is not imported or replayed.

The formal reference supported by this benchmark is a standalone FastChem
v4.0.3 build. The `pyfastchem` dependency currently used elsewhere in this
repository is version 3.1.3 and is **not** a FastChem 4 oracle for this
comparison.

The release-facing interpretation of the recorded four-point result is in
`documents/v0_4_fastchem4_validation_demo.rst`. This README remains the
benchmark contract and reproduction guide.

Readable visual companions are available in
`examples/comparisons/comparison_with_fastchem4_gas.py` and
`examples/comparisons/comparison_with_fastchem4_condensates.py`. The former
uses the adapter's `chemistry_mode="gas"` path; the production runner retains
the default equilibrium-condensation mode.

## Shared input contract

Both engines use the following three packaged files:

| Role | Packaged path |
| --- | --- |
| gas equilibrium constants | `src/exogibbs/data/FastChem4/logK/logK_wo_ions.dat` |
| condensate equilibrium constants | `src/exogibbs/data/FastChem4/logK/logK_condensates.dat` |
| elemental abundances | `src/exogibbs/data/FastChem4/element_abundances/asplund_2021.dat` |

The runner copies these exact bytes into its temporary FastChem working
directory under whitespace-safe names. It records the packaged-file hashes and
requires them to be byte-identical to the corresponding files in the audited
FastChem source checkout.

The abundance file uses
`A(X) = log10(n_X / n_H) + 12`. The runner explicitly parses this file, maps
the values into the ExoGibbs element order, sets the electron budget to zero,
and normalizes the linear elemental vector to sum to one. It does not use the
FastChem4 preset's built-in reference vector. In particular, the built-in
default and the packaged `asplund_2021.dat` file have different Ge values;
explicit parsing avoids silently comparing different compositions.

Each repeated `--point` is interpreted as
`TEMPERATURE_K,PRESSURE_BAR`. Point order is preserved and is also the profile
order used for adjacent-layer phase-transition reporting.

## Build the standalone FastChem oracle

Use a clean FastChem source checkout at tag `v4.0.3` (commit `ae67cbd`). The
following command compiles the official standalone model directly and was
verified to produce `/tmp/exogibbs_fastchem4`:

```bash
cd /path/to/FastChem
git describe --tags --always --dirty
g++ -std=c++17 -O3 -DNDEBUG -fopenmp \
  model_src/model_main.cpp \
  fastchem_src/*.cpp \
  fastchem_src/elements/*.cpp \
  fastchem_src/gas_phase/*.cpp \
  fastchem_src/condensed_phase/*.cpp \
  -o /tmp/exogibbs_fastchem4
```

The first command should identify `v4.0.3` without a dirty suffix. A CMake
build of the same tag is also valid, but pass the resulting official
`fastchem` standalone executable to the runner, not a FastChem 3
`pyfastchem` extension.

The adapter creates a temporary pressure-temperature file and FastChem
configuration, invokes:

```text
/tmp/exogibbs_fastchem4 config.input
```

from that temporary working directory and parses the standalone chemistry,
condensate, and monitor tables. The configuration selects `ce` (equilibrium
condensation without rainout) and `ND` (particle number-density output).
The JSON records the fixed chemistry and element-conservation accuracies,
iteration limits, and selected verbosity.

## Run the comparison

Run from the repository root:

```bash
PYTHONPATH=src python -m benchmarks.fastchem4.run_production_comparison \
  --fastchem-executable /tmp/exogibbs_fastchem4 \
  --fastchem-version-label "4.0.3 (ae67cbd)" \
  --fastchem-source-root /path/to/FastChem \
  --point 1400,0.1 \
  --jax-platform cpu \
  --output results/fastchem4_production_comparison/summary.json
```

The default point is `1400 K, 0.1 bar`, the default JAX platform is `cpu`, and
the default output is:

```text
results/fastchem4_production_comparison/summary.json
```

Specify more than one point to inspect phase-boundary behavior:

```bash
PYTHONPATH=src python -m benchmarks.fastchem4.run_production_comparison \
  --fastchem-executable /tmp/exogibbs_fastchem4 \
  --fastchem-version-label "4.0.3 (ae67cbd)" \
  --fastchem-source-root /path/to/FastChem \
  --point 1800,0.1 \
  --point 1600,0.1 \
  --point 1400,0.1 \
  --point 1200,0.1 \
  --jax-platform cpu \
  --output results/fastchem4_production_comparison/temperature_scan.json
```

Use an ordered, physically meaningful sequence. The runner does not sort or
interpolate points.

`--fastchem-source-root` defaults to `FastChem` under the repository root.
Preflight requires the clean `v4.0.3` checkout at full commit
`ae67cbd559bc64a3233a1cee6030b8e6b50520de`, an explicit label containing
`4.0.3` and `ae67cbd`, and three input files byte-identical to the packaged
copies. It records the source commit and executable SHA256 separately. The
version label is the operator's assertion that the executable was built from
that source; the runner cannot prove this correspondence from an opaque
binary. Use `--preflight-only` to write
`<output-stem>.preflight.json` and stop before either solver runs.

## Output and metrics

The JSON schema identifier is:

```text
exogibbs_fastchem4_production_comparison_v1
```

The runner writes `<output-stem>.preflight.json`, the requested JSON file, and
a Markdown report with the same stem as the requested JSON. Output-specific
preflight names prevent one scan from overwriting another scan's provenance.
Top-level JSON sections contain `provenance`, `input_contract`, `fastchem`,
`exogibbs`, `layers`,
`profile_phase_transitions`, and `summary`.

If preflight or comparison execution fails, the requested JSON and Markdown
paths are overwritten with a fail-closed report, so a previous successful
summary cannot be mistaken for the current run.

Per-layer comparison sections include:

- `status`: ExoGibbs public status and FastChem convergence and element
  conservation status. A completed comparison is not by itself a numerical
  agreement claim. ExoGibbs lifecycle KKT values describe the accepted
  fixed-support state before an optional full-budget gas polish; species
  comparisons use the final public state, and the polish report is retained.
- `element_budget`: reconstructed element totals and absolute and relative
  closure residuals on the shared normalized abundance basis. FastChem's
  monitor conservation flags are retained as independent runtime diagnostics.
- `total_gas`: total gas number-density comparison and consistency checks
  against the common pressure-temperature scale.
- `gas_major_species`: slot-aligned gas mixing ratios, the union of species
  above the configurable major threshold (default `1e-8`), set overlap, and
  abundance differences in dex. Trace species should not dominate an
  interpretation intended for major gases.
- `condensates`: active-species and amount comparisons. Species are aligned
  with slot-aware keys because the packaged condensate table intentionally
  contains duplicate names. The report evaluates normalized amount floors
  `1e-20`, `1e-12`, and `1e-8`; each report uses the larger of its amount
  floor and `--ratio-floor` when clipping log-ratios for phases absent from
  one solver. These finite, floor-clipped values are not literal ratios to
  zero, so active counts, Jaccard overlap, and absolute amounts are the
  primary comparison for such phases.
- `gibbs_over_rt`: the ExoGibbs state and the converted FastChem state
  evaluated on one explicitly recorded ExoGibbs `G/RT` basis.

`profile_phase_transitions` reports active-condensate entries and exits between
adjacent input points for each engine at the same three amount floors. It is
empty or non-informative for a single point. A transition located between two
coarse points is interval evidence, not a precise phase-boundary temperature
or pressure.

Summary status `complete` means that preflight, catalog matching, both
engines' convergence, FastChem's element-conservation flags, and finite metric
generation completed. No scientific agreement threshold is applied; the JSON
explicitly records `scientific_acceptance_thresholds_applied: false`.

The packaged condensate file contains duplicate `Zn(s,l)` entries at zero-based
slots 167 and 202. These are independent thermochemical slots and must not be
collapsed by name before amount, active-set, phase-transition, or Gibbs
comparisons. The JSON retains slot identity even when display names match.

## Interpretation limits

The common-basis `G/RT` metric is a convergence diagnostic, not a direct
comparison of undocumented internal objectives. The FastChem number-density
state must first be aligned to the packaged species slots and rescaled to the
shared ExoGibbs elemental-budget gauge. Interpret `G/RT` only when species
alignment, finite-state checks, and budget closure are satisfactory.

The two solvers can differ in numerical floors, active-phase thresholds,
duplicate-slot allocation, and treatment of values written at finite text
precision. Small trace-species or near-boundary differences are therefore not
automatically failures. Conversely, agreement in total gas density alone is a
weak check because both calculations share the same ideal-gas pressure and
temperature.

This runner records evidence; it does not use FastChem to repair, warm-start,
or otherwise influence the ExoGibbs production result.
