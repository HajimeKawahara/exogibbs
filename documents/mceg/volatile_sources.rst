Volatile source review (2026-10-21)
===================================

This review adds source inspections to the six volatile-solubility contracts
from the October 20 survey. The equations in ExoGibbs are unchanged. Agreement
with an inspected formula does not establish accuracy against experiments:
physical accuracy remains ``unknown`` in all six contracts.

.. list-table:: Inspected material and remaining work
   :header-rows: 1
   :widths: 20 45 35

   * - Contract
     - Completed comparison
     - Still needed
   * - ``h2o_lichtenberg2021``
     - Published Eq. 9/Table 1: partial pressure in Pa; ppmw converted to mass
       fraction. Four numerical samples agree.
     - Composition-specific experimental data and uncertainty.
   * - ``co2_lichtenberg2021``
     - Same published equation and table, using the CO2 coefficients. Four
       numerical samples agree.
     - Composition-specific experimental data and uncertainty.
   * - ``ch4_ardia2013``
     - Seo arXiv v1 Eq. 17 uses GPa and CH4 mole fraction. Three samples agree.
       Literal MELTYQ A4 uses bar and gives values 10000 times larger.
     - Ardia full text/run data and the derivation of the simplified law.
   * - ``co_yoshioka2019``
     - The published abstract confirms coefficients (-5.20, 0.80) in wt%
       elemental C. Three samples agree after dividing by 100, with the
       provider's bar convention retained.
     - Published Eq. 6 and run-table units. The abstract does not specify the
       fugacity unit, and the earlier thesis has different coefficients.
   * - ``n2_dasgupta2022``
     - MELTYQ A6 uses total P/T; ExoGibbs uses sqrt(P)/T. Two of three samples
       differ; equality at 1 GPa is incidental.
     - Original Dasgupta Eq. 10 and regression data to resolve that difference.
   * - ``h2o_sun_yao2026``
     - The pinned MAGEMin H2O branch: ten coefficients, bar/K, dry oxide mole
       fractions, wt% divided by 100. Four positive-interior samples agree.
     - Original Eq. 7/Table 1, MATLAB implementation and calibration data.

For CO, the earlier thesis Eq. 2 uses (-5.83, 0.98), not the published-abstract
coefficients. It must not silently replace the final paper. MELTYQ A5 also
labels fugacity in GPa and output as CO mass fraction. Its literal numeric
expression gives values smaller than ExoGibbs by ``10**3.2``; that ratio does
not reconcile CO mass with elemental-C mass.

The Sun--Yao comparison covers the positive-interior formula only. The Julia
branch returns NaN at zero H2O pressure, whereas ExoGibbs returns zero.
ExoGibbs also rejects partial pressure above total pressure and water mass
fractions at least one. No Julia or MATLAB runtime was executed.

Sources and access
------------------

The contracts record inspection locations and SHA-256 hashes of retrieved
files. Downloaded papers, extracted text, images and upstream source files
are not included in the repository.

* `Lichtenberg et al. (2021), published PDF
  <https://ora.ox.ac.uk/objects/uuid:fe730f0a-22f5-4b41-b387-4b21aaea6fd8/files/r1544bp515>`_:
  pp. 6-7, Eq. 9, Table 1 and Fig. 2.
* `Seo et al. (2024), arXiv v1 <https://arxiv.org/pdf/2408.17056v1>`_:
  p. 9, Eq. 17 and adjacent experimental attribution.
* `Ito and Changeat (2026), arXiv v1 <https://arxiv.org/pdf/2605.08752v1>`_:
  p. 17, Appendix A, Eqs. A4-A6.
* `Yoshioka et al. (2019), published abstract at Macquarie University
  <https://researchers.mq.edu.au/en/publications/carbon-solubility-in-silicate-melts-in-equilibrium-with-a-co-cosu/>`_:
  MORB regression and wt% carbon definition only.
* `Yoshioka dissertation, revision 20181120
  <https://epub.uni-bayreuth.de/id/eprint/4056/1/thesis_Yoshioka_revision20181120.pdf>`_:
  printed pp. 81 and 87 (PDF pp. 85 and 91), Table 6.3 and Eq. 2.
* `MAGEMin_C.jl, pinned commit 47e2d12
  <https://github.com/ComputationalThermodynamics/MAGEMin_C.jl/blob/47e2d12bfec0f799acf030b285fa3b1e8eb32098/julia/TE_saturation_models.jl>`_:
  H2O coefficients, units, dry normalization and H2O branch (lines 655-811).

Publisher full-text requests returned HTTP 403 for
`Ardia (2013) <https://doi.org/10.1016/j.gca.2013.03.028>`_,
`Yoshioka (2019) <https://doi.org/10.1016/j.gca.2019.06.007>`_,
`Dasgupta (2022) <https://doi.org/10.1016/j.gca.2022.09.012>`_, and
`Sun and Yao (2026) <https://doi.org/10.1016/j.epsl.2026.120115>`_.
These are access limitations of this review, not claims that no public copy
exists. The available sources above do not close those full-text gaps.

Inspect and replay
------------------

From the repository root, with MCEG installed:

.. code-block:: console

   mceg explain mceg/contracts/ch4_ardia2013.yaml --assessment physical_source_review_20261021 --lang ja
   mceg explain mceg/contracts/ch4_ardia2013.yaml --evidence source_comparison_20261021 --lang ja
   mceg check mceg/contracts/ch4_ardia2013.yaml --assessment source_formula_review_20261021 --root .
   PYTHONPATH=src JAX_PLATFORMS=cpu JAX_ENABLE_X64=1 python mceg/reviews/volatile_source_review_20261021.py --output /tmp/volatile-source-comparison.json

Use any of the six contract names with these assessment IDs. The N2 formula
assessment intentionally remains unknown and its comparison evidence is
``failed``: two samples disagree with the inspected alternative expression.
MCEG reports ``recorded_support: UNKNOWN`` for that assessment; the check
command itself can still exit successfully when the records are well formed.

The saved ``tests/data/validation/mceg/20261021_volatile_sources/comparison.json`` contains
29 diagnostics: 18 adopted-expression comparisons agree; one additional N2
control agrees at 1 GPa; ten alternative-expression comparisons differ.
The replay exits unsuccessfully if an adopted-expression comparison changes.
These are formula samples, not 29 experiments or an accuracy estimate.

The old ``numerical_review`` and ``physical_review`` IDs retain the October 20
snapshot. Use the dated IDs above for this review. Available-material review
tasks have explicit completion records; ``empirical_scope`` and the new
``remaining_source_work_20261021`` tasks remain open. Reported T/P bounds are
still marginal metadata and do not define a validated joint application domain.
