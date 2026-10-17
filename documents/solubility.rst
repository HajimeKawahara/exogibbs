Magma--atmosphere solubility
============================

``exogibbs.solubility`` is an auxiliary package.  It is independent of the
Gibbs-energy minimization solvers and provides empirical volatile-solubility
laws for magma--atmosphere boundary calculations.

The implementation lives in ``exogibbs.solubility.volatile`` so that the law
collection can grow beyond MELTYQ.  The former
``exogibbs.solubility.meltyq`` import remains as a compatibility facade.

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

Shared conventions
------------------

Each law preserves the driving quantity defined by its source: partial
pressure and fugacity are not interchangeable.  See
:ref:`fugacity-conventions` for the gas fugacity convention.  The
generic dilute conversion in ``exogibbs.thermo.composition`` evaluates
:math:`Y M_\mathrm{matrix}/M_\mathrm{species}`; the result is a
solute-to-matrix mole ratio, not an exact normalized mole fraction.  The
linear helper requires :math:`0\leq Y\leq1` and returns ``nan`` outside that
mass-fraction domain.

``ln_co_yoshioka2019`` and ``ln_n2_dasgupta2022`` in
``exogibbs.solubility.volatile`` accept natural-log driving pressures and
return natural-log mass fractions.  Their log-domain forms are numerically
preferable when trace gas abundances would underflow in linear pressure units.

``convert_pressure`` in ``exogibbs.utils.units`` performs unit scaling only,
so callers retain responsibility for physical-domain validation.

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
N2 law takes :math:`\Delta\mathrm{IW}` directly.  In
``exogibbs.thermo.oxygen_fugacity``,
``log10_oxygen_fugacity_iw_hirschmann2021`` returns the IW reference, while
``delta_iw_hirschmann2021`` converts absolute oxygen fugacity in bar to
:math:`\Delta\mathrm{IW}`.  See :ref:`magma-gas-iw` for its use by the
stable MELTYQ magma--gas preset.

References
----------

* Ito, Y. & Changeat, Q. (2026), arXiv:2605.08752.
* Seo, C., Ito, Y. & Fujii, Y. (2024), doi:10.3847/1538-4357/ad7461.
* Hirschmann, M. M. et al. (2012), doi:10.1016/j.epsl.2012.06.031.
* Hirschmann, M. M. (2021), doi:10.1016/j.gca.2021.08.039.
* Lichtenberg, T. et al. (2021), doi:10.1029/2020JE006711.
* Yoshioka, T. et al. (2019), doi:10.1016/j.gca.2019.06.007.
* Ardia, P. et al. (2013), doi:10.1016/j.gca.2013.03.028.
* Dasgupta, R. et al. (2022), doi:10.1016/j.gca.2022.09.012.

Sun--Yao water comparison
-------------------------

``h2o_sun_yao2026`` adds the water branch of Sun & Yao (2026),
`doi:10.1016/j.epsl.2026.120115 <https://doi.org/10.1016/j.epsl.2026.120115>`_.
The implementation follows Eq. 7 and Table 1 as transcribed by
`MAGEMin_C.jl at 47e2d12 <https://github.com/ComputationalThermodynamics/MAGEMin_C.jl/blob/47e2d12bfec0f799acf030b285fa3b1e8eb32098/julia/TE_saturation_models.jl>`_
(GPL-3.0, ComputationalThermodynamics contributors). The original MATLAB
program and experiment-level calibration table have not been independently
replayed. This is a conditional comparison model, with that provenance
limitation preserved; it does not replace a default equilibrium model.

.. code-block:: python

   from exogibbs.solubility import h2o_sun_yao2026

   w = h2o_sun_yao2026(
       water_partial_pressure_bar=20., temperature_k=2173.15,
       melt_pressure_bar=440., x_na2o=.002, x_al2o3=.025,
   )

The return value is a mass fraction, after dividing source wt% by 100.
Both molecular and hydroxyl terms count as H2O equivalent; hydroxyl is not
an extra elemental-H or molecular-H2 inventory. T uses K; both pressures
use bar. The total pressure in the coefficients must not be replaced by
water partial pressure. Na2O and Al2O3 are fractions of the entire
volatile-free oxide mole inventory, not fractions normalized to Na+Al.
Callers must declare their iron-redox normalization. The C-free binary
mass basis is water mass divided by dry-plus-water mass; a multisolute
extension needs an explicit consumer convention.

``h2o_sun_yao2026_terms`` returns two capacities C and pressure exponents a,
ordered molecular H2O then hydroxyl, with ``w=sum(C*p_water**a)``.
The capacities retain T, total-pressure and composition derivatives.
An extensive thermodynamic continuation belongs in ExoEOS; the auxiliary
law alone supplies neither host chemical potentials nor a Gibbs scalar.

``SUN_YAO2026_WATER_METADATA`` records the joint H2O--CO2 dataset envelope:
933.15--2197.15 K and 1 bar--6 GPa over 2090 experiments. These are marginal
ranges, not a validated rectangle for every melt, partial pressure or
water-only experiment. No H2/He-rich-fluid or 2173 K BSE validation follows
from them. The CO2 branch is not implemented. Nonphysical inputs and
predicted mass fractions at or above one return NaN; extrapolations are
never clipped into an accepted result. The pre-existing MELTYQ metadata
and compatibility imports retain their original meanings.
