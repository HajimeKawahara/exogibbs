Release notes
=============

Scientific milestone: Gas curation v0.1
---------------------------------------

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
