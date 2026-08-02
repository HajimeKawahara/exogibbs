FastChem 4 production comparison
================================

Purpose
-------

For the concise release-facing interpretation of the recorded result, see
:doc:`v0_4_fastchem4_validation_demo`.

The production comparison runner evaluates the public ExoGibbs
gas-plus-condensate default and an independent FastChem 4 standalone process
at identical temperature, pressure, elemental composition, and
thermochemical-data inputs.

ExoGibbs uses the production fixed-support v2 lifecycle.  The retired
fixed-support v1 runtime is not part of the comparison.  FastChem output is a
reference result only: no FastChem public, runtime, or trace value is used as
an ExoGibbs constructor input, initializer, support-selection input, retry
input, or solver-route input.

Formal reference boundary
-------------------------

The formal external reference is a standalone FastChem v4.0.3 executable.
The ``pyfastchem`` dependency used by other ExoGibbs workflows is currently
version 3.1.3.  It must not be treated as the FastChem 4 oracle for this
comparison.

Build a clean FastChem checkout at tag ``v4.0.3`` (commit ``ae67cbd``).  For
example:

.. code-block:: bash

   cd /path/to/FastChem
   git describe --tags --always --dirty
   g++ -std=c++17 -O3 -DNDEBUG -fopenmp \
     model_src/model_main.cpp \
     fastchem_src/*.cpp \
     fastchem_src/elements/*.cpp \
     fastchem_src/gas_phase/*.cpp \
     fastchem_src/condensed_phase/*.cpp \
     -o /tmp/exogibbs_fastchem4

The version command should identify ``v4.0.3`` without a dirty suffix.  A
CMake build of the same source is equivalent; supply its official standalone
``fastchem`` executable to the runner.

Shared input contract
---------------------

Both engines read the three packaged files below:

.. list-table::
   :header-rows: 1
   :widths: 28 72

   * - Role
     - Packaged path
   * - Gas equilibrium constants
     - ``src/exogibbs/data/FastChem4/logK/logK_wo_ions.dat``
   * - Condensate equilibrium constants
     - ``src/exogibbs/data/FastChem4/logK/logK_condensates.dat``
   * - Elemental abundances
     - ``src/exogibbs/data/FastChem4/element_abundances/asplund_2021.dat``

The runner copies those same bytes into a temporary FastChem working directory
under whitespace-safe names.  It records the packaged-file hashes and requires
them to be byte-identical to the corresponding files in the audited FastChem
source checkout.

The Asplund 2021 file is parsed explicitly from the astronomical
``A(X) = log10(n_X/n_H) + 12`` scale.  The runner maps it to the ExoGibbs
element order, assigns a zero electron budget, and normalizes the linear
element vector to sum to one.  It deliberately does not use the preset's
built-in reference vector: the built-in default and packaged file use
different Ge values, which would otherwise make the comparison
compositionally inconsistent.

Standalone execution
--------------------

The adapter writes a temporary pressure-temperature profile and FastChem
configuration, then invokes the executable with that configuration path as
its only argument.  The configuration selects ``ce`` for equilibrium
condensation without rainout and ``ND`` for particle number-density output.
It parses FastChem's chemistry, condensate, and monitor tables after the
subprocess completes.  The result records the fixed chemistry and
element-conservation accuracies, iteration limits, and selected verbosity.

Run the default comparison from the repository root:

.. code-block:: bash

   PYTHONPATH=src python -m benchmarks.fastchem4.run_production_comparison \
     --fastchem-executable /tmp/exogibbs_fastchem4 \
     --fastchem-version-label "4.0.3 (ae67cbd)" \
     --fastchem-source-root /path/to/FastChem \
     --point 1400,0.1 \
     --jax-platform cpu \
     --output results/fastchem4_production_comparison/summary.json

Each ``--point`` value is
``TEMPERATURE_K,PRESSURE_BAR``.  The option may be repeated:

.. code-block:: bash

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

The default point is ``1400 K, 0.1 bar``.  The default JAX platform is
``cpu``, and the default output path is
``results/fastchem4_production_comparison/summary.json``.  Results belong
under ``results/`` and should not be committed as source.

``--fastchem-source-root`` defaults to ``FastChem`` under the repository root.
Preflight requires the clean ``v4.0.3`` checkout at full commit
``ae67cbd559bc64a3233a1cee6030b8e6b50520de``, an explicit label containing
``4.0.3`` and ``ae67cbd``, and input bytes identical to the packaged data.  It
records the source commit and executable SHA256 separately.  The version
label is the operator's assertion that the opaque executable was built from
that source; the runner cannot prove this correspondence from the binary
alone.  ``--preflight-only`` writes ``<output-stem>.preflight.json`` and stops
before either solver runs.

Recorded evidence
-----------------

The runner writes ``<output-stem>.preflight.json``, the requested JSON result,
and a Markdown report with the same stem as the JSON result.  Output-specific
preflight names prevent one scan from overwriting another scan's provenance.
The JSON schema is
``exogibbs_fastchem4_production_comparison_v1``.  Its top-level sections are
``provenance``, ``input_contract``, ``fastchem``, ``exogibbs``, ``layers``,
``profile_phase_transitions``, and ``summary``.

If preflight or comparison execution fails, the requested JSON and Markdown
paths are overwritten with a fail-closed report.  A previous successful
summary therefore cannot be mistaken for the current run.

Each layer records the following comparison groups:

``status``
   ExoGibbs public convergence/status and FastChem convergence and elemental
   conservation status.  A successful subprocess alone is not treated as
   chemical agreement.  ExoGibbs lifecycle KKT values describe the accepted
   fixed-support state before an optional full-budget gas polish; species
   comparisons use the final public state, and the polish report is retained.

``element_budget``
   Reconstructed elemental totals and closure residuals on the common
   normalized abundance basis, together with FastChem's monitor flags.

``total_gas``
   Total gas number densities and common pressure-temperature consistency
   checks.  This is useful for scale errors but weak as a standalone
   agreement test.

``gas_major_species``
   Slot-aligned mixing ratios, the union of gases above the configurable
   major threshold (``1e-8`` by default), set overlap, and abundance
   differences in dex.

``condensates``
   Active condensate species and amounts with slot-aware alignment.  The
   packaged table contains duplicate ``Zn(s,l)`` entries at zero-based slots
   167 and 202; they remain distinct numerical slots.  Amount-floor reports
   are generated at ``1e-20``, ``1e-12``, and ``1e-8``.  Each report uses the
   larger of its amount floor and ``--ratio-floor`` when clipping log-ratios
   for a phase absent from one solver.  A finite floor-clipped value is not a
   literal ratio to zero; active counts, Jaccard overlap, and absolute amounts
   are the primary comparison for such phases.

``gibbs_over_rt``
   Both states evaluated with the recorded ExoGibbs ``G/RT`` convention after
   the FastChem state has been aligned and rescaled to the ExoGibbs elemental
   budget gauge.

For repeated points, ``profile_phase_transitions`` reports condensate entries
and exits between adjacent points for both engines at the same three amount
floors.  Input order is preserved; the runner does not sort or interpolate
the profile.  A transition reported between two coarse points brackets
behavior but does not locate a precise phase boundary.

Summary status ``complete`` requires successful preflight, occurrence-aware
catalog matching, convergence of both engines, FastChem element-conservation
flags, and finite comparison metrics.  It is not a scientific agreement gate:
the output records ``scientific_acceptance_thresholds_applied: false``.

Interpretation limits
---------------------

The common-basis ``G/RT`` value is a convergence diagnostic, not a claim that
the two programs expose the same internal objective.  It is meaningful only
after species alignment, finite-state checks, and elemental-budget closure
have passed inspection.  Duplicate-slot allocation, numerical floors,
active-phase thresholds, and finite text-output precision can produce small
differences, especially for trace gases and near a phase boundary.

FastChem remains independent throughout the run.  Its result is never used to
warm-start, repair, or choose the ExoGibbs production result.  See
``benchmarks/fastchem4/README.md`` for the benchmark-facing contract and
command details.
