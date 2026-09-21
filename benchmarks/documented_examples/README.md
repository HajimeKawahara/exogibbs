# Documented example benchmarks

## Execute the documented examples

The acceptance runner executes **61 fresh jobs** from the current English
manual and an explicitly selected Japanese manual checkout. It is separate
from `run_all_gpu.csh`, which measures six solver workloads and compiler modes.

```console
csh benchmarks/documented_examples/run_documented_acceptance_gpu.csh \
  --fastchem-executable /path/to/fastchem \
  --fastchem-source-root /path/to/FastChem-v4.0.3 \
  --co-database /path/to/CO/12C-16O/Li2015 \
  --gce-checkout /path/to/GlobalChemicalEquilibrium_Release \
  --exoeos-source /path/to/exoeos/src \
  --melts-runtime /path/to/alphamelts-py-2.3.2-ubuntu_22_04-x86_64 \
  --melts-python /path/to/native-worker/python \
  --bse-inventory /path/to/baseline_inventory.json \
  --japanese-docs /path/to/doc_ExoGibbs
```

The csh wrapper preserves all arguments, supports paths containing spaces, and
returns the Python runner's exit status. Scheduler copies can set
`EXOGIBBS_REPOSITORY_ROOT`; `EXOGIBBS_PYTHON` selects the Python executable.
The wrapper requires GPU execution. For an explicit CPU run use
`python -m benchmarks.documented_examples.run_all --platform cpu` with the
same resource options.

| Category | Jobs | Execution and acceptance |
| --- | ---: | --- |
| Comparisons | 14 | All comparison programs, both FastChem4 profiles, both full 855-layer Ito comparisons, and the optional Visscher/Morley FastChem comparison. The programs' scientific gates must pass. The pinned FastChem production CLI additionally runs all four documented temperature points, requires its clean reference checkout and byte-identical thermochemical inputs, and must report `comparison_completed=true`. |
| Curated condensate demos | 10 | Run each actual plotting program and independently audit every layer, inventory, and generated figure. Internally caught numerical failures still fail the job. |
| ExoEOS pure fugacity | 1 | Execute the gallery example with independent element and chemical stationarity checks. Missing optional ExoEOS is a failure. |
| Metal/silicate examples | 22 | Reference audit/extraction/generation, all four native cases, both gas-exchange cases, hydrogen and dry control, sulfur source/extraction, sulfide, CNS inventory scans, M1 chemistry/boundary, both native MELTS branches, native BSE common Gibbs, the four-point mixed-metal selection control, and archive revalidation with fresh MELTS potentials. |
| Retrieval tutorials | 4 | Actual NUTS, including preflight, **500 warmup steps and 1000 samples** by default. Require completed sampling, posterior diagnostics, and finite posterior arrays with the requested sample count. |
| Inline examples | 10 | Eleven complete code blocks from the actual RST/TeX sources: gas presets, condensate/profile/rainout, solubility, stable magma-gas, ideal activity adapter, and the Japanese legacy magma-gas example. Each source block is hashed. |

Three historical FastChem script aliases share canonical jobs. The archived
`results/subneptune_taxonomy/20260911/verify.py` wrapper shares the explicit
`revalidate_archive.py` job. Archive revalidation recomputes properties at
saved amounts; the two MELTS jobs independently rerun the equilibrium roots.
Intentional rejected candidate phases in the sulfide example remain scientific
outputs; the example's accepted-branch gate must pass.
The mixed-metal control requires its numerical acceptance flags and the documented
present/present/unresolved/absent sequence, including exactly zero absent metal.
It uses synthetic standards and needs ExoEOS but no MELTS runtime; it does not
establish a calibrated BSE phase boundary.

The scope is executable scientific examples, rather than development commands
such as pytest, documentation builds, or performance measurements. The
scientific `benchmarks.fastchem4.run_production_comparison` CLI is included;
its Python module command maps to the same executable source file. Helper modules and API fragments with undefined inputs are not
standalone examples. The four published retrieval notebooks use their canonical
Python programs because their guarded notebook cells alone do not run NUTS.
Two unlinked exploratory notebooks are outside the manual. Use `./update_doc.sh`
separately for documentation generation.

The coverage report discovers public example entry points and script commands
and `python -m benchmarks...` commands in `documents/`, the metal/silicate
guides, and `doc_ExoGibbs/ja/`. An uncovered
or missing documented example prevents full acceptance. The default Japanese
path is `doc_ExoGibbs`; initialize the submodule or use `--japanese-docs` to
select an existing local checkout. Its actual commit and contents are recorded,
so an external/manual checkout is not confused with the superproject gitlink.
The CNS example and its regression tests are restored unchanged from local
commit `8338be3`, matching the Japanese manual's finite-inventory commands.

Every job starts in a new process with float64 enabled. GPU requests disable CPU
fallback and verify the actual JAX backend. NumPy/SciPy reference calculations
and native MELTS workers still use their own CPU implementations. Imported
ExoGibbs/ExoEOS paths must match the selected source trees. Inputs must already
exist: the runner downloads nothing. Native MELTS needs the shared library and
a working separate Python environment; BSE needs the exported absolute inventory.
The Ito workbook defaults to `external_data/Ito_2025.xlsx` (`--ito-input`).

Resource options also accept `EXOGIBBS_FASTCHEM_EXECUTABLE`, `EXOGIBBS_FASTCHEM_SOURCE_ROOT`,
`EXOJAX_CO_DATABASE`, `EXOGIBBS_GCE_CHECKOUT`, `EXOGIBBS_ITO_INPUT`,
`EXOGIBBS_EXOEOS_SOURCE`, `EXOGIBBS_MELTS_RUNTIME`, `EXOGIBBS_MELTS_PYTHON`,
`EXOGIBBS_BSE_INVENTORY`, and `EXOGIBBS_JAPANESE_DOCS`.

Outputs use a new directory under `results/all_documented_examples/`; an
explicit `--output-directory` must not exist. Each executed job retains
`run.log`, `worker.json`, and its scientific artifacts. `summary.json` records
commands, coverage, actual source/provider/manual commits and content hashes,
input hashes, timestamps, and every failure. Missing inputs are `UNAVAILABLE`;
missing/skipped native chemistry, invalid artifacts, numerical failure, and a
missing worker report all fail acceptance. Remaining jobs continue and the
runner exits nonzero if any job or coverage check fails. Previous PASS records
and Ito checkpoints are never reused.

Inspect coverage and commands without importing JAX or executing examples:

```console
python -m benchmarks.documented_examples.run_all --dry-run \
  --japanese-docs /path/to/doc_ExoGibbs
```

Repeat `--case CASE_ID` for targeted fresh reruns. Such a run reports
`full_acceptance=false`. The explicit development option `--retrieval-quick`
still runs NUTS, but can only produce `SMOKE_PASS`, never full acceptance.
A final acceptance record requires all jobs, full NUTS, and complete coverage.

## Measure the six documented solver workloads

This suite measures the ExoGibbs solver workloads behind every entry in the
`EXAMPLES` toctree in `documents/index.rst`. It is a performance regression
surface, not a replacement for the scientific comparisons in those examples.

The registered workloads are:

- Visscher (2006) / Morley (2012) KCl and Na2S scans;
- Visscher (2010) silicate phase competition;
- Ito (2025) propagated rainout;
- reduced Fe-FeS local equilibrium and rainout;
- the full-catalog L-dwarf condensate and gas profiles; and
- a Rocky Raccoon-like positive trace-Mg rainout layer.

`manifest.py` is deliberately explicit. The unit tests compare it with the
documentation toctree, so adding a documentation example requires adding one
benchmark adapter rather than silently leaving the new workload unmeasured.

## Quick check

Run one layer of a small case on CPU:

```console
PYTHONPATH=src:. python -m benchmarks.documented_examples.run \
  --case fe_fes_rainout_demo \
  --platform cpu \
  --optimization default \
  --smoke-layers 1 \
  --output-directory results/documented_example_benchmarks/smoke
```

Smoke results are marked `scope.kind = "smoke"` and must not be used as
release baselines.

## Repeated L-dwarf forward benchmark

The optional repeated benchmark separates one cold nominal full-catalog
L-dwarf profile from ten synchronized evaluations with deterministic, smooth,
small T-P perturbations:

```console
PYTHONPATH=src:. python -m benchmarks.documented_examples.ldwarf_repeated \
  --platform gpu \
  --optimization default \
  --output results/documented_example_benchmarks/ldwarf_repeated_gpu.json
```

Setup and the cold call are excluded from the reported steady mean, median,
and p95. The ten raw evaluation records are written beside the JSON as
`*.evaluations.csv`. Later executable shapes and compilation are retained and
flagged; they are never silently subtracted from the warm measurements.
The recorded `.lower(...).compile()` boundary can remain slightly nonzero on
an in-memory cache hit; a genuinely new warm executable is identified from a
shape signature not observed by the cold call.
Use `--optimization disable_most_optimizations` with a different output path
to measure the same ten inputs under the compile-light JAX mode.

This measures the public full-catalog production forward path, including
support discovery, host lifecycle work, and zero-barrier refinement. It omits
the comparison figure's separate gas-only profile. It is useful as a
NUTS-like forward-throughput probe, but it is not a JIT-compiled
value-and-gradient, spectral likelihood, or NUTS-transition benchmark.

## Full benchmark matrix

Run the complete GPU sequence (the warm-boundary correctness gate and six
documented workloads under both compiler modes, followed by the two L-dwarf
cold-plus-ten-warm runs) with:

```console
csh benchmarks/documented_examples/run_all_gpu.csh
```

An optional output directory may be supplied as the only argument.
The boundary gate replays seven saved difficult states, including SiO-on, via
the public API. It stops the sequence before the longer benchmarks if either
compiler mode fails. Its logs, JUnit reports, and source provenance are saved
under the output root. The fixture manifest pins each snapshot's source and
digest; the same test module runs in CPU CI. This does not run a full Rocky
atmospheric column or every script in the repository.

The default command runs all six cases in separate CPU/GPU processes with
both compiler modes:

```console
PYTHONPATH=src:. python -m benchmarks.documented_examples.run \
  --output-directory results/documented_example_benchmarks/baseline
```

For a stable baseline, run the workstation otherwise idle and use at least
three repetitions:

```console
PYTHONPATH=src:. python -m benchmarks.documented_examples.run \
  --repeat 3 \
  --output-directory results/documented_example_benchmarks/baseline
```

The two compiler modes are:

- `default`: `jax_disable_most_optimizations = false`;
- `disable_most_optimizations`: the JAX option is explicitly enabled before
  JAX is imported.

The second mode sets XLA's backend optimization level to zero in JAX versions
that support it. It can shorten compilation while producing a slower
executable, so compare compile and execution time together. A worker records
`unsupported` instead of silently falling back when the installed JAX does
not expose this option.

CPU, GPU, and compiler modes are separate processes. The suite disables JAX's
persistent compilation cache so that each worker measures cold compilation.
Warm reuse inside one documented workload remains visible: JSON records the
first and repeated compilation time for each PD-IPM solve executable shape.
The suite also pins each worker to this checkout's `src/exogibbs` tree and
records both its import path and source-tree fingerprint.

## Timing definitions

Each worker writes a detailed JSON result. The suite also writes
`summary.json`, `summary.csv`, `summary.md`, and a long-form
`phase_budgets.csv`. The additive budget is worker/suite schema v2; v1
artifacts predate this breakdown and must be regenerated for budget analysis.

- `workload_wall_seconds`: example/workload imports, setup, and all measured
  solver phases. It starts after worker startup, JAX import, device
  initialization, and environment preflight.
- `solver_phase_wall_seconds`: named ExoGibbs solve workflows, including the
  caller's synchronization, fixed-point orchestration, and result conversion.
  Ito's propagated-rainout phase also constructs its solver setup inside the
  documented helper, so compare its finer PD-IPM/zero-barrier leaves when
  separating solver-core work from orchestration.
- `setup_phase_wall_seconds`: explicitly named catalog, data-loading, and
  reduced-setup phases.
- `unphased_workload_wall_seconds`: imports, validation, and other workload
  work outside named setup and solver phases.
- `pdipm.compilation_seconds`: sum of explicit JAX
  `.lower(...).compile()` calls in the fixed-support solver.
- `pdipm.execution_seconds`: compiled PD-IPM calls, synchronized before the
  timer stops. Unsynchronized input preparation on which the executable
  depends can be charged here, so this is a synchronized execution boundary,
  not a pure kernel profile.
- `pdipm.diagnostic_compilation_seconds` and
  `pdipm.diagnostic_execution_seconds`: explicit terminal-diagnostic
  compilation and synchronized execution. Their sum is retained as
  `pdipm.diagnostic_seconds`.
- `zero_barrier.host_wall_seconds`: host-observed wall time around the
  SciPy/NumPy zero-barrier refinement. The optimization runs on the CPU even
  when PD-IPM uses a GPU, but this boundary can also include input transfer
  and device synchronization needed by the host call.
- `zero_barrier.calls[].selected_normalized_initializers` and the adjacent
  attempt, retry, and function-evaluation fields expose initializer routing.
  Capacity regularization is a primary route only when the smallest nonzero
  element inventory is at most binary64 machine epsilon times the largest;
  other eligible states retain the unregularized route.
- `solver_budget.pdipm_internal_orchestration_wall_seconds`: time inside the
  fixed-support batch wrapper outside its explicit compile, execute, and
  diagnostic timers, including batch preparation and result assembly.
- `solver_budget.outside_pdipm_and_zero_barrier_wall_seconds`: time in solver
  phases outside the fixed-support and zero-barrier wrappers. It includes
  support selection and closure, gas solves, result construction, transfers,
  and other orchestration.
- `other_solver_and_orchestration_wall_seconds`: compatibility total equal to
  the preceding two residuals.

The solver budget is an additive partition:

```text
solver phase wall
= PD-IPM compile
+ PD-IPM execute
+ diagnostic compile
+ diagnostic execute
+ PD-IPM internal orchestration
+ zero-barrier host work
+ work outside PD-IPM and zero-barrier
```

`pdipm_wall_seconds` is a parent subtotal, so do not add it again to its
compile, execute, diagnostic, and internal-orchestration children. A PD-IPM
call is one lifecycle invocation; a bucket is one fixed-shape executable
invocation within that call. Physical supports may differ between rows and
are recorded separately from the executable capacities. Both counts are
recorded.

Every named setup or solver phase has the same partition in `phase_budgets`.
This exposes, for example, the L-dwarf gas-only phase and the separate KCl,
Na2S, silicate, local-equilibrium, and rainout solves without case-specific
instrumentation. `phase_budgets.csv` puts those records in long form for
cross-version comparisons.

`timing_attribution_consistent` must be true for a valid result; it checks the
solver and every phase partition and detects a future instrumentation change
that double-counts named components. The residual categories deliberately do
not add synchronization points. Because JAX dispatch is asynchronous, treat
them as wall-time attribution rather than a device-kernel ownership profile.

`output_layer_count` counts returned rows across the final documented
profiles. Component call and bucket counts are cumulative, so they also expose
repeated full-profile work such as the Ito pressure fixed-point iterations.
`--smoke-layers N` limits each returned profile to at most N conditions; cases
with paired profiles therefore return `2 * N` rows, while Ito returns `N`.

The PD-IPM timing comes from the production timing boundary. The benchmark
adds temporary wrappers around the production batch and zero-barrier entry
points, and restores them after each workload. It does not change solver
mathematics or public APIs.

The measured scope excludes standalone FastChem execution, plotting, and file
generation. Those tasks can obscure the solver regression this suite is
intended to detect. The workload adapters import the example scripts' setup
helpers, constants, conditions, and production API calls so they stay aligned
with the documented calculations.

## Reading results

A performance number is valid only when `status` is `pass` and every expected
output layer converged. Compare like for like:

- same case and full/smoke scope;
- same CPU or GPU backend and device;
- same JAX, jaxlib, Python, revision, and diagnostic setting;
- same compiler mode for regression baselines.

To evaluate the compiler option itself, compare the paired `default` and
`disable_most_optimizations` rows while holding every other item above fixed.
Both rows must independently pass their convergence and physical audits.

Do not add hard wall-time gates from a single run. Establish a versioned
baseline with repeated runs first, then choose thresholds from the observed
variance. Use `pdipm_repeated_shape_compilation_seconds` to identify accidental
recompilation, and use the zero-barrier call and function-evaluation counts to
distinguish host optimization regressions from JAX compile regressions.
