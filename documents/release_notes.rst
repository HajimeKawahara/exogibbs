Release notes
=============

v0.8.0
------

**This release updates the curated gas thermochemistry and its source evidence,
and adds reusable fixed-support multiphase sensitivities.**

Curated gas data and upgrading
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

``exogibbs.presets.curated.chemsetup`` now selects ``curated-gas-v2``. The
cumulative updates include reviewed molecular formation energies and thermal
functions; the final :doc:`gas-v2 adoption <mceg/curated_gas_v2_adoption>`
adds 28 molecular energy corrections and the evaluated O2 thermal function
to the preceding CH2 dataset. The ion-inclusive and neutral networks retain
537 and 421 species, respectively, on **1500--4000 K** with a **1 bar** standard
pressure. See :doc:`presets/curated` for source choices and their limitations.

.. important::

   The default thermochemistry changes calculated abundances. To reproduce
   v0.7.0's curated model, import ``chemsetup`` from
   ``exogibbs.presets.curated_v1``. Use ``exogibbs.presets.curated_v2`` to pin
   the new model explicitly; named intermediate presets also remain available.
   Historical evidence and figures retain their original dataset identities.

The :doc:`FastChem 4.0.4 comparison <mceg/curated_fastchem4_comparison>` covers
108 molecules at five temperatures. It compares formation constants, not
mixture abundances or physical accuracy. The packaged ``fastchem4`` preset
continues to use its pinned v4.0.3 data.

Fixed-support multiphase sensitivities
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

``exogibbs.api.multiphase.linearize_fixed_support`` re-audits a supplied
equilibrium and returns a reusable local derivative operator. It supports
forward and transpose products for temperature, pressure, elemental budgets
and model parameters. Derivatives hold phase support fixed and do not
differentiate phase selection; see :doc:`multiphase`.

Evidence and repository maintenance
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Optional provenance inspection now exposes versioned model records and
species-level evidence without requiring MCEG for ordinary calculations.
Source investigations, candidate studies and adopted models remain distinct.
Offline review checks run separately in ``tests/model_evidence``; tests that
require MCEG live in ``tests/model_contracts``. Retired condensate v1 design
notes and trial scorecards have been removed from the repository root;
historical versions remain in Git.

Source distributions now contain the runtime package and required data.
Tests, examples, documentation and archived scientific evidence remain in Git,
keeping installation archives small without changing the packaged models.

Scientific milestone: Gas curation v0.1
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

:doc:`Gas curation v0.1 <mceg/gas_curation_release_0_1>` (canonical ``0.1.0``)
names the existing ``curated-gas-v2`` scientific snapshot at
``f580e66157edbe6f604e2748d8e1a25d46d08b86``. Its manifest and aggregate MCEG
record preserve the source selections, numerical evidence and unresolved
physical limitations. The ``curated_v2`` factory, dataset, numerical functions
and model identifier are unchanged. This is separate from the ExoGibbs
software release number and creates no Git tag or runtime API.

Record-only amendments use ``0.1.1``; numerical, adoption or domain changes
require ``0.2.0``. Earlier release records remain immutable.

v0.7.0
------

**This release brings solution phases into the public equilibrium API and
makes thermochemical source choices and their evidence easier to inspect.**

Multiphase equilibrium
^^^^^^^^^^^^^^^^^^^^^^

The new :mod:`exogibbs.api.multiphase` minimizes full phase Gibbs energies
with elemental conservation, support selection and insertion audits. Implicit
derivatives are available on a fixed support; phase search does not establish
a global minimum for arbitrary nonconvex models. The illustrated
:doc:`plagioclase_exsolution` tutorial explains two-phase coexistence through
an illustrative regular solution and its common tangent. See :doc:`multiphase`
for the numerical contract and ExoEOS connections.

Traceable gas thermochemistry
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The :doc:`curated gas preset <presets/curated>` combines reviewed atomic-ion
and molecular source choices with per-species provenance. It provides
ion-inclusive and neutral networks on **1500--4000 K**, continuous temperature
derivatives, and stricter charge closure. Independent FastChem comparisons
check numerical agreement; inherited or provisional data remain identified.

Models and evidence
^^^^^^^^^^^^^^^^^^^

Volatile solubility laws, pure-gas fugacity adapters and the magma--gas
interface connect atmospheric chemistry to interior models.
:doc:`MCEG documentation <mceg/index>` now links model assumptions,
applicability, source reviews and numerical evidence, with unresolved physical
support kept explicit. MCEG remains optional at runtime.

Reliability and upgrading
^^^^^^^^^^^^^^^^^^^^^^^^^

Condensate solves reject non-finite or negative candidate amounts and apply
stricter closure checks. Scientific regressions cover solver changes, and
example acceptance checks report numerical and sampling failures.

.. important::

   * **Regenerate FastChem-backed grids created before v0.7.** Their stored
     amounts use the old physical-density gauge; there is no automatic migration.
   * **Legacy example-local M1/M2 workflows are retired.** Use
     ``exogibbs.api.multiphase`` for current work, or the historical checkout
     identified in :doc:`multiphase` to reproduce old runs.
   * The development-only ``fastchem4_nasa9_ions`` preset is replaced by
     ``curated.chemsetup(ions=True)``. This changes the model and temperature
     range; see the :doc:`migration guide <presets/curated>`.

``pyfastchem`` is now optional: install ``exogibbs[fastchem]`` only for its
backend or comparisons. Existing gas and condensate compatibility imports
remain available; new code should use the three :doc:`public APIs <equilibrium_solvers>`.

Earlier releases
----------------

v0.6.0
^^^^^^

Condensate equilibrium gained consistent amount normalization, stricter
zero-barrier and conservation checks, and stable profile shapes. Rainout and
budget diagnostics advanced to schema v2. New :doc:`validation examples <comparison_example_lineage>`
and :doc:`profile tools <condensate_profile>` cover condensate competition and
sequential rainout.

v0.5.0
^^^^^^

Fixed-support condensate derivatives, opt-in rainout, grid initialization and
ExoJAX/NumPyro retrieval examples became available. Derivatives apply locally
on a fixed support and do not include support changes or rainout propagation;
see :doc:`equilibrium_solvers`.

v0.4.0
^^^^^^

A :doc:`four-temperature FastChem 4 comparison <v0_4_fastchem4_validation_demo>`
confirmed major-gas species agreement and elemental closure at 0.1 bar and
1200, 1400, 1600 and 1800 K. Detailed condensate phase selection was outside
that milestone.
