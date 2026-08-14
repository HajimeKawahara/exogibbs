Magma--atmosphere solubility
============================

``exogibbs.solubility`` is an auxiliary package.  It is independent of the
Gibbs-energy minimization solvers and provides empirical volatile-solubility
laws for magma--atmosphere boundary calculations.

The initial implementation contains the six laws selected by MELTYQ (Ito &
Changeat 2026):

.. list-table:: Native inputs and outputs
   :header-rows: 1

   * - Function
     - Pressure input
     - Output
   * - ``h2_hirschmann2012``
     - H2 fugacity in bar; melt pressure in GPa
     - H2 mole fraction
   * - ``h2o_lichtenberg2021``
     - H2O partial pressure in Pa
     - H2O mass fraction
   * - ``co2_lichtenberg2021``
     - CO2 partial pressure in Pa
     - CO2 mass fraction
   * - ``co_yoshioka2019``
     - CO fugacity in bar
     - Elemental-C mass fraction dissolved as CO
   * - ``ch4_ardia2013``
     - CH4 fugacity in GPa; melt pressure in GPa
     - CH4 mole fraction
   * - ``n2_dasgupta2022``
     - N2 partial pressure and total melt pressure in GPa
     - Total elemental-N mass fraction

For example:

.. code-block:: python

   from exogibbs.solubility import h2_hirschmann2012

   x_h2 = h2_hirschmann2012(
       hydrogen_fugacity_bar=1000.0,
       melt_pressure_gpa=1.0,
   )

All functions accept broadcast-compatible JAX arrays and support JIT
compilation and automatic differentiation.  A zero driving partial pressure
or fugacity returns zero; a nonphysical input returns ``nan``.  Values outside
the experimental calibration ranges are evaluated without clipping.  Inspect
``MELTYQ_SOLUBILITY_METADATA`` before extrapolating a law.  Automatic
derivatives are finite at positive interior pressures.  At zero pressure, the
fractional-power H2O, CO, and N2 laws retain their singular derivatives.

Source-consistent corrections
-----------------------------

The implementation follows the cited experimental or formulation source when
the MELTYQ appendix is inconsistent with it:

* CO uses fugacity in bar with the corrected ``-7.2`` elemental-carbon
  mass-fraction intercept from Yoshioka et al. (2019).
* CH4 uses fugacity in GPa, as in Seo et al. (2024), Equation 17.
* The first N term uses the square root of total melt pressure, as in Dasgupta
  et al. (2022), Equation 10.

These differences are also recorded in ``MELTYQ_SOLUBILITY_METADATA``.  The
N2 law takes :math:`\Delta\mathrm{IW}` directly; calculation of the IW buffer
from absolute oxygen fugacity is deliberately outside this package.

References
----------

* Ito, Y. & Changeat, Q. (2026), arXiv:2605.08752.
* Seo, C., Ito, Y. & Fujii, Y. (2024), doi:10.3847/1538-4357/ad7461.
* Hirschmann, M. M. et al. (2012), doi:10.1016/j.epsl.2012.06.031.
* Lichtenberg, T. et al. (2021), doi:10.1029/2020JE006711.
* Yoshioka, T. et al. (2019), doi:10.1016/j.gca.2019.06.007.
* Ardia, P. et al. (2013), doi:10.1016/j.gca.2013.03.028.
* Dasgupta, R. et al. (2022), doi:10.1016/j.gca.2022.09.012.
