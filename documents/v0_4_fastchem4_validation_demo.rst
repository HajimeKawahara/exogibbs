ExoGibbs v0.4 validation demo
=============================

Release takeaway
----------------

This comparison provides scoped independent evidence for treating the current
ExoGibbs production solver as the v0.4 release candidate.  At four
solar-composition, 0.1-bar points (1200, 1400, 1600, and 1800 K), ExoGibbs
and an independent FastChem 4.0.3 process:

* converged at every tested temperature;
* selected exactly the same set of major gas species;
* closely matched the total gas amount;
* closed their elemental budgets; and
* produced nearly identical values when both states were evaluated with one
  recorded ``G/RT`` convention.

The result supports the documented v0.4 milestone: major-gas agreement for the
gas phase of the production gas-plus-condensate solver.  It is a release
demonstration, not a formal scientific acceptance gate.  A clean rerun on the
merged release revision remains part of the normal release checks.

.. important::

   The v0.4 claim is deliberately scoped.  This demo does not claim universal
   agreement outside the tested window or phase-by-phase equivalence of the
   condensates.  Detailed condensate selection and phase-boundary agreement
   remain v0.5 validation targets.

Comparison scope
----------------

The external process was the official standalone FastChem 4.0.3 model at
commit ``ae67cbd559bc64a3233a1cee6030b8e6b50520de``.  ExoGibbs used the public
``head_v2`` production route with the ``validated_2026_07`` fixed-support-v2
preset.

.. list-table::
   :header-rows: 1
   :widths: 32 68

   * - Item
     - Recorded value
   * - Temperature
     - 1800, 1600, 1400, and 1200 K
   * - Pressure
     - 0.1 bar
   * - Element abundances
     - Asplund 2021, read from the shared FastChem-format file
   * - Gas thermochemistry
     - Shared ``logK_wo_ions.dat`` bytes
   * - Condensate thermochemistry
     - Shared ``logK_condensates.dat`` bytes
   * - Catalog sizes
     - 422 gas species and 219 condensate slots
   * - Condensation mode
     - Equilibrium condensation without rainout
   * - Major-gas threshold
     - Mixing ratio ``1e-8`` in either solver

The preflight gate verified the FastChem tag and full commit, a clean FastChem
source tree, and byte identity of the three shared input files.  It also
recorded the executable SHA256 alongside the audited source checkout.  The
correspondence between that opaque executable and the source is an operator
assertion, not something the runner can prove.  FastChem results were
comparison outputs only: no FastChem runtime value was supplied to an
ExoGibbs constructor, initializer, support selector, retry, or route decision.

Major-gas result
----------------

At every temperature the two programs identified the same major-gas set.
``Jaccard = 1`` means that the sets were identical.

.. list-table::
   :header-rows: 1
   :widths: 12 17 13 16 16 13 13

   * - T [K]
     - Major gases Exo/FC
     - Jaccard
     - Mean absolute difference [dex]
     - Maximum absolute difference [dex]
     - Maximum species
     - Absolute relative normalized total-gas difference
   * - 1800
     - 43 / 43
     - 1.000
     - 0.0164
     - 0.653
     - ``O1Ti1``
     - ``8.07e-8``
   * - 1600
     - 38 / 38
     - 1.000
     - ``9.33e-4``
     - 0.0123
     - ``O1Si1``
     - ``1.50e-6``
   * - 1400
     - 34 / 34
     - 1.000
     - 0.0123
     - 0.273
     - ``Co``
     - ``1.99e-7``
   * - 1200
     - 26 / 26
     - 1.000
     - 0.00758
     - 0.0950
     - ``H1Mn1``
     - ``7.53e-8``

The maximum absolute difference is intentionally reported, but it is not the
typical gas error.  At 1800 K the 0.653-dex outlier is ``O1Ti1`` close to the
``1e-8`` major-species threshold: its mixing ratios are ``1.75e-8`` and
``7.87e-8``.  The other 42 major gases differ by at most 0.0176 dex.  At
1400 K the largest differences are concentrated in ``Co`` and ``Cr``; the
other 32 major gases differ by at most 0.0126 dex.

Supporting numerical checks
----------------------------

The gas result is supported by independent consistency checks:

* all four ExoGibbs layers were accepted by fixed-support v2, closed their
  supports, and passed the independent KKT checks;
* all four FastChem layers reported convergence and elemental conservation;
* the maximum reconstructed relative elemental-budget residual was
  ``7.18e-12`` for ExoGibbs and ``2.81e-6`` for FastChem;
* the maximum absolute relative difference in total gas amount was
  ``1.50e-6``; and
* the maximum common-basis ``G/RT`` difference was ``5.08e-7``.

The FastChem-state ``G/RT`` value is a post-evaluation with the recorded
ExoGibbs thermodynamic objective, not a FastChem-reported internal diagnostic.
These checks are evidence against a gross input, unit, normalization, or
element-accounting mismatch.

Known condensate boundary
-------------------------

The condensate result is useful precisely because it defines what v0.4 does
not claim.  With an active-amount floor of ``1e-8``, ExoGibbs retained more
phases than FastChem:

.. list-table::
   :header-rows: 1
   :widths: 18 28 27 27

   * - T [K]
     - Active condensates Exo/FC
     - Common / union
     - Jaccard
   * - 1800
     - 4 / 2
     - 2 / 4
     - 0.500
   * - 1600
     - 9 / 4
     - 4 / 9
     - 0.444
   * - 1400
     - 15 / 9
     - 8 / 16
     - 0.500
   * - 1200
     - 17 / 10
     - 10 / 17
     - 0.588

Both programs changed their active phase sets in every adjacent 200-K
interval, but they did not always assign the same entry or exit interval.
For example, ``SiO(s)`` and ``CaMgSi2O6(s)`` entered earlier in ExoGibbs,
while ``MgSiO3(s,l)`` entered one interval later in FastChem.  At the
``1e-8`` floor, ``Ti3O5(s,l)`` entered the ExoGibbs set but exited the
FastChem set between 1400 and 1200 K.

The total budgets and common ``G/RT`` values remain close while these phase
memberships differ.  This is consistent with competing phase allocations
near a thermodynamic boundary, but it does not by itself identify which
allocation should be preferred.

Version boundary
----------------

For v0.4, this demo records the following release boundary:

* release the public fixed-support-v2 production route;
* claim major-gas agreement only for the demonstrated input domain;
* retain convergence, KKT, budget, and provenance diagnostics; and
* document condensate phase-level differences as known validation scope,
  rather than hiding them or treating them as a release failure.

The natural v0.5 work is:

* bracket phase boundaries with 25--50 K spacing and multiple pressures;
* add composition axes such as C/O and metallicity;
* evaluate both states on a common saturation and chemical-potential basis,
  then inspect ExoGibbs KKT and complementarity for the differing
  condensates;
* follow threshold-adjacent gas outliers such as TiO, Co/Cr, and Mn/MnH; and
* define separate scientific acceptance thresholds for gases and
  condensates.

Python comparison examples
--------------------------

The following scripts expose the comparison as ordinary, readable Python.
They show the shared inputs, independent FastChem process, current ExoGibbs
public API call, species alignment, printed metrics, and plots:

* :download:`gas-only comparison
  <../examples/comparisons/comparison_with_fastchem4_gas.py>`
* :download:`production gas-plus-condensate comparison
  <../examples/comparisons/comparison_with_fastchem4_condensates.py>`

Run them from the repository root with an independently built FastChem 4.0.3
standalone executable:

.. code-block:: bash

   python examples/comparisons/comparison_with_fastchem4_gas.py \
     --fastchem-executable /path/to/fastchem

   python examples/comparisons/comparison_with_fastchem4_condensates.py \
     --fastchem-executable /path/to/fastchem

   python examples/comparisons/comparison_with_fastchem4_condensates.py \
     --fastchem-executable /path/to/fastchem \
     --profile l-dwarf

The downloaded files are source views of repository examples, not standalone
single-file programs: keep them at the paths above inside an ExoGibbs source
checkout.  The gas-only example normally finishes in a few seconds.  The
gas-plus-condensate example runs the full four-layer production lifecycle and
can take roughly 4--5 minutes on CPU without intermediate output.  Its
13-layer ``l-dwarf`` mode can take roughly 10 minutes.

The first script restores the historical gas-only visual comparison on the
current FastChem 4 data and gas API.  The second uses the production
fixed-support-v2 condensate route at the four v0.4 demo points.  Both write
their figures under ``results/fastchem4_examples`` by default.
The optional ``l-dwarf`` mode uses the same gas-plus-condensate route to make
a 2-by-2 pressure-profile figure: gas and condensate rows, with FastChem and
ExoGibbs columns.  It is an illustrative local-equilibrium trajectory without
rainout or vertical transport, not a self-consistent atmosphere model and not
an additional v0.4 acceptance case.  Each gas panel overlays a separate
gas-only calculation (dashed) and the gas phase in local equilibrium with
condensates (solid with markers), using the same total elemental budget.  The
separation directly displays the local gas-phase response to condensation.  No result
from either comparison calculation is used to initialize or configure the
other.
The historical ``comparison_with_fastchem.py``,
``comparison_with_fastchem_extended.py``, and
``comparison_with_fastchem_cond.py`` paths remain as compatibility entry
points.

Older analytical, grid-initializer, and YK snapshot comparisons are also
retained for lineage.  They are supplementary traces rather than additional
v0.4 acceptance points; see :doc:`comparison_example_lineage`.

These files deliberately do not match the Sphinx-Gallery ``plot_`` filename
pattern.  Documentation builds therefore remain deterministic and do not
require an external FastChem executable.  The examples are visual companions;
the runner below remains the provenance-bearing comparison.

Reproduction and provenance
---------------------------

The executable runner, exact input contract, metrics, and limitations are
documented in :doc:`fastchem4_production_comparison`.  The four-point command
is:

.. code-block:: bash

   PYTHONPATH=src python -m benchmarks.fastchem4.run_production_comparison \
     --fastchem-executable /path/to/fastchem \
     --fastchem-version-label "4.0.3 (ae67cbd)" \
     --fastchem-source-root /path/to/FastChem \
     --point 1800,0.1 \
     --point 1600,0.1 \
     --point 1400,0.1 \
     --point 1200,0.1 \
     --jax-platform cpu \
     --output results/fastchem4_production_comparison/temperature_scan.json

The recorded numerical run used ExoGibbs solver revision ``a9856c8``.  It was
performed while the comparison harness and documentation were still
uncommitted, so its Git provenance correctly reports a dirty worktree.  Before
publishing v0.4, rerun the same command on the clean merged release commit or
tag and retain its output-specific preflight sidecar.  This is an archival
provenance step; it does not change the interpretation of the comparison
above.
