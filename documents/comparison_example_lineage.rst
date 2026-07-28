Comparison example lineage
==========================

The files under ``examples/comparisons`` preserve several different kinds of
validation evidence.  They are kept as ordinary Python programs so that the
thermochemical setup, solver call, numerical comparison, and plot can be read
in one place.

Current FastChem 4 release demonstration
----------------------------------------

The v0.4 release-facing comparison uses the current FastChem 4 data and an
independently built FastChem 4.0.3 standalone executable:

* :download:`gas-only profile
  <../examples/comparisons/comparison_with_fastchem4_gas.py>`
* :download:`production gas-plus-condensate points
  <../examples/comparisons/comparison_with_fastchem4_condensates.py>`

These plots are human-readable companions to the provenance-bearing runner in
``benchmarks/fastchem4``.  The measured scope and limitations are recorded in
:doc:`v0_4_fastchem4_validation_demo` and
:doc:`fastchem4_production_comparison`.

Restored historical validation traces
-------------------------------------

The following entry points were removed during repository cleanup and are
restored because they record independent validation ideas that remain useful.
They now use the current ExoGibbs APIs and write figures under ``results``.

.. list-table::
   :header-rows: 1
   :widths: 32 25 43

   * - Example
     - Reference
     - Meaning
   * - :download:`comparison_with_fastchem_initializer.py
       <../examples/comparisons/comparison_with_fastchem_initializer.py>`
     - Standalone FastChem and the packaged legacy FastChem-v3-compatible grid
     - At 2870 K over ``1e-8``--``1e2`` bar, compare the final gas solution
       with FastChem and compare grid-backed and uniform ExoGibbs
       initialization.  The standalone output is never an initializer input.
   * - :download:`comparison_with_hsystem.py
       <../examples/comparisons/comparison_with_hsystem.py>`
     - Analytical ``2 H <-> H2`` solution
     - Check composition and the automatic derivatives with respect to
       temperature and log pressure over profiles.
   * - :download:`comparison_with_hcosystem.py
       <../examples/comparisons/comparison_with_hcosystem.py>`
     - Analytical ``CO + 3 H2 <-> CH4 + H2O`` reduction
     - Check composition, elemental-budget closure, and the CO abundance
       Jacobian with respect to H, C, and O budgets.
   * - :download:`comparison_with_ykcode.py
       <../examples/comparisons/comparison_with_ykcode.py>`
     - Archived YK B4 composition snapshot
     - Reproduce the historical 500 K, 10 bar regression with its original
       elemental budget and species ordering.

The initializer example intentionally uses the packaged
``fastchem/logK/logK.dat`` dataset because that is the dataset represented by
the saved grid.  The expected external executable is FastChem 4.0.3, but the
lightweight example does not verify its version or hash.  This does not turn
the saved grid into a FastChem 4-data grid, and the example is separate from
the v0.4 production comparison above.  Use the formal benchmark runner when
the executable/source preflight and machine-readable provenance are required.

The legacy grid predates the newer per-record coefficient trace fields.
Compatibility therefore requires its exact element and species catalogs and
every setup-metadata field stored in that grid, while permitting newer
runtime-only FastChem source-record trace fields from an explicit allowlist.
Unknown additional metadata is rejected.  Newly generated grids retain those
detailed trace fields and compare them too.

The YK reference is a numerical snapshot rather than a currently rerunnable
external oracle.  Its source-code revision and executable were not pinned in
this repository, and its thermochemical table has changed since the snapshot
was produced.  It therefore supports regression lineage only; it is not
evidence of present-day independent solver agreement.
The exact :download:`160-value snapshot <../examples/data/p10.txt>` and its
:download:`provenance note <../examples/data/README.md>` are retained beside
the examples.

Representative restored-run checks
----------------------------------

A CPU/x64 rerun of the restored examples gave:

.. list-table::
   :header-rows: 1
   :widths: 25 75

   * - Example
     - Result
   * - Legacy grid initializer
     - All three calculations converged in 100/100 layers and all 523 gas
       species aligned.  Grid and uniform initialization had a worst
       major-gas difference of ``1.08e-5`` dex; grid-initialized ExoGibbs and
       FastChem differed by at most ``9.4e-7`` dex.  Grid initialization used
       5 iterations at the median and 7 at maximum, versus 186 and 233 for the
       uniform initializer.
   * - Analytical H/H2
     - Across the temperature and pressure sweeps, the maximum mixing-ratio
       error was ``7.16e-13``, the maximum derivative error was ``1.20e-11``,
       and the maximum absolute elemental-budget error was ``9.35e-12``.
   * - Analytical H/C/O
     - The scalar reaction residual was ``6.93e-15``.  The maximum composition
       relative error was ``1.14e-11`` and the maximum CO abundance-Jacobian
       relative error was ``7.42e-10``.
   * - Historical YK B4 snapshot
     - ExoGibbs converged in 317 iterations.  Of the 16 reference entries
       above ``1e-14``, the largest difference was ``5.070%`` for ``Na1``,
       inside the historical ``5.1%`` regression limit.

The near-equality in the initializer example means that the saved grid changes
the route to convergence, not the accepted final state.  The approximately
5% YK value has a different interpretation: it compares the unrenormalized
species amounts used by the historical script and demonstrates continuity
with that frozen result and its declared tolerance.  It is not the precision
of the current FastChem comparison.

Run the examples
----------------

From the repository root:

.. code-block:: bash

   python examples/comparisons/comparison_with_fastchem_initializer.py \
     --fastchem-executable /path/to/fastchem
   python examples/comparisons/comparison_with_hsystem.py
   python examples/comparisons/comparison_with_hcosystem.py
   python examples/comparisons/comparison_with_ykcode.py

Use ``--show`` to open a Matplotlib window.  The initializer example requires
the same independently built standalone executable as the current FastChem
examples.  The analytical and YK examples are offline.

All comparisons fail closed on their declared convergence or numerical
checks.  The scripts deliberately have ``comparison_`` rather than ``plot_``
filenames, so Sphinx builds expose them as downloads without executing them.
