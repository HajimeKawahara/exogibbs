Magma--gas API and MELTYQ preset
================================

``exogibbs.api.magma_gas`` is the stable, model-neutral interface for coupling
a magma boundary model to the gas equilibrium solver.  It is an application
layer over ``exogibbs.equilibrium.gas``, not a third thermochemical-equilibrium
engine.  ``exogibbs.presets.magma_gas`` provides MELTYQ as one built-in model;
the generic API does not import or name that preset.

The model contract has three operations: construct an initial root vector,
map a root vector to gas elemental abundances, and evaluate a square scaled
residual plus a model-specific JAX PyTree state.  The engine owns the damped
outer root, inner gas solve, convergence diagnostics, gas pressure and
fugacity state, and implicit reverse-mode derivative.  A model owns its root
coordinates, boundary equations, melt physics, and model-specific outputs.

The former ``exogibbs.experimental.magma_gas`` path remains a compatibility
adapter.  New workflows should use the stable API and an explicit preset.

The reusable pressure conversion, dilute composition conversion, and oxygen-
fugacity buffer helpers live in ``exogibbs.utils.units``,
``exogibbs.thermo.composition``, and
``exogibbs.thermo.oxygen_fugacity``, respectively.  The fixed 60 g/mol basis,
species reduction, and volatile-law combination belong to the MELTYQ preset,
not to the generic engine.

Chemistry preparation
---------------------

MELTYQ-equivalent mode reduces a source ``ChemicalSetup`` to five elements
(``H``, ``C``, ``O``, ``N``, and ``He``) and nine gas species (``H2``, ``He``,
``O2``, ``H2O``, ``CO``, ``CO2``, ``CH4``, ``N2``, and ``NH3``).  These nine
species provide exactly four independent gas reactions.  Source catalogs with
different species labels require an explicit canonical-to-source mapping.
An optional source fugacity callback passed to ``prepare_meltyq_problem``
must implement ``lnphi_func(T, P_bar, None)`` and return natural-log pure-
component coefficients in the full source-species order.  The preparation
step selects and reorders the nine required entries.

The complete YKB4 example below includes this mapping and the interface solve.

.. _magma-gas-iw:

Thermodynamic and redox conventions
-----------------------------------

The general non-ideal gas convention is described in
:ref:`fugacity-conventions`.  For gas species :math:`s`, this interface uses

.. math::

   p_s = x_s P, \qquad
   f_s = \phi_s p_s = \phi_s x_s P,

where :math:`p_s` is partial pressure, :math:`f_s` is fugacity, and the
fugacity coefficient :math:`\phi_s` is dimensionless.  Fugacity is the
effective pressure that enters the chemical potential,

.. math::

   \mu_s = \mu_s^\circ(T)
   + RT\ln\!\left(\frac{f_s}{f^\circ}\right),
   \qquad f^\circ = 1\ {\rm bar}.

Consequently, the implementation evaluates

.. math::

   \ln\!\left(\frac{f_s}{1\ {\rm bar}}\right)
   = \ln x_s
   + \ln\!\left(\frac{P}{1\ {\rm bar}}\right)
   + \ln\phi_s.

If ``lnphi_func`` is omitted, :math:`\phi_s=1` and fugacity equals partial
pressure.  Otherwise the current interface uses the pure-component
correction returned by ``lnphi_func(T, P_bar, None)``.  A
composition-dependent mixture fugacity model is not supported.  The pressure
closure is :math:`\sum_s p_s=P`; in general there is no corresponding
:math:`\sum_s f_s=P` identity.  In particular,
:math:`x_{\mathrm{O_2}}=f_{\mathrm{O_2}}/(\phi_{\mathrm{O_2}}P)`; the simpler
:math:`x_{\mathrm{O_2}}=f_{\mathrm{O_2}}/P` holds only in ideal-gas mode.

Oxygen fugacity and the IW scale
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Oxygen fugacity is the fugacity of molecular :math:`\mathrm{O_2}` and is a
redox variable, not an atomic-oxygen abundance.  The iron--wüstite
(IW) reference is defined by coexistence along

.. math::

   (1-y)\,\mathrm{Fe} + \frac{1}{2}\,\mathrm{O_2}
   \rightleftharpoons \mathrm{Fe}_{1-y}\mathrm{O}.

At a given temperature and total pressure, define

.. math::

   \begin{aligned}
   L_{\rm IW}(T,P)
      &= \log_{10}\!\left(
         \frac{f_{\mathrm{O_2}}^{\rm IW}}{1\ {\rm bar}}
         \right), \\
   \Delta{\rm IW}
      &= \log_{10}\!\left(
         \frac{f_{\mathrm{O_2}}}{1\ {\rm bar}}
         \right) - L_{\rm IW}(T,P) \\
      &= \log_{10}\!\left(
         \frac{f_{\mathrm{O_2}}}{f_{\mathrm{O_2}}^{\rm IW}}
         \right).
   \end{aligned}

Both quantities are dimensionless decimal logarithms.  Thus
:math:`\Delta{\rm IW}=-2` means an oxygen fugacity one hundredth of IW, while
:math:`\Delta{\rm IW}=+1` means ten times IW.  The pressure passed to the IW
model is the system total pressure, not the oxygen partial pressure.

``log10_oxygen_fugacity_iw_hirschmann2021(T_K, P_GPa)`` returns
:math:`L_{\rm IW}`.  ``delta_iw_hirschmann2021(f_O2_bar, T_K, P_GPa)`` converts
an absolute oxygen fugacity to :math:`\Delta{\rm IW}`.  The implementation
uses the fcc/bcc or hcp iron branch selected by the phase boundary in
Hirschmann (2021), Table 1 (doi:10.1016/j.gca.2021.08.039).  It is calibrated
over 1000--3000 K and 0.0001--100 GPa and does not clip extrapolated inputs.
The underlying model also omits electronic and spin transitions in FeO that
become an additional concern at roughly 30--50 GPa and above.

IW is used here only as a redox reference.  The reduced gas chemistry does
not contain Fe or wüstite, and evaluation of :math:`\Delta{\rm IW}`
does not assert that these phases are present or stable.  In particular, the
crystalline buffer may serve as a metastable reference above melting.

Four-variable interface closure
--------------------------------

With :math:`b_{\rm H}=1` as an arbitrary amount gauge, the outer unknown is

.. math::

   \boldsymbol{u} = \left(
      \ln\frac{b_{\rm C}}{b_{\rm H}},
      \ln\frac{b_{\rm O}}{b_{\rm H}},
      \ln\frac{b_{\rm N}}{b_{\rm H}},
      \ln\frac{b_{\rm He}}{b_{\rm H}}
   \right).

For each trial :math:`\boldsymbol{u}`, the inner ExoGibbs solve predicts the
equilibrium gas at fixed :math:`T` and :math:`P`.  The outer damped-Newton
solve drives four natural-log residuals to zero,

.. math::

   \begin{aligned}
   r_1 &= \ln\frac{f_{\mathrm{O_2}}^{\rm gas}}
                        {f_{\mathrm{O_2}}^{\rm input}}, \\
   r_2 &= \ln\frac{R_{\rm CO}^{\rm solubility}}
                        {R_{\rm CO}^{\rm input}}, \\
   r_3 &= \ln\frac{R_{\rm N}^{\rm solubility}}
                        {R_{\rm N}^{\rm input}}, \\
   r_4 &= \ln\frac{x_{\mathrm{H_2}}}{x_{\rm He}}
           - \ln\frac{q_{\rm H_2}}{1-q_{\rm H_2}}.
   \end{aligned}

Here :math:`R_{\rm CO}` is the dilute mole ratio derived from the CO law's
elemental-C mass basis, and :math:`R_{\rm N}` is an atomic-N dilute mole
ratio.  The latter is twice the corresponding :math:`\mathrm{N_2}`-equivalent
molecular ratio.  The CO and N solubility constraints are evaluated directly
in log space.  Here :math:`q_{\rm H_2}` is
``model_inputs.h2_fraction_in_h_he``, whose default is 0.84.

The input oxygen fugacity, CO ratio, N ratio, and H2 fraction within H2+He are
constraints.  The element ratios are inferred.  Gas reaction equilibrium and
:math:`\sum_s p_s=P` are already enforced by the inner fixed-pressure solve,
so they are not duplicated as outer equations.  All nine gas abundances and
the dissolved H2, H2O, CO2, and CH4 ratios are predictions.  The returned CO
and N melt ratios reproduce their input constraints only for a converged
root, and :math:`\Delta{\rm IW}` is derived from the specified
:math:`f_{\mathrm{O_2}}`, :math:`T`, and :math:`P`.

Interface solve
---------------

The public pressure convention is bar.  All five physical inputs must be
strictly positive because the outer residuals are evaluated in log space.
Inputs are not clipped or host-validated; invalid traced values yield a
non-converged state with non-finite values.

The following complete YKB4 example is within the simple temperature and
pressure overlap of the six solubility-law metadata ranges:

.. code-block:: python

   from exogibbs.api import magma_gas
   from exogibbs.presets.magma_gas import (
       MELTYQ_SPECIES,
       MELTYQ_MELT_QUANTITIES,
       MeltyqMagmaGasInputs,
       prepare_meltyq_problem,
   )
   from exogibbs.presets.ykb4 import chemsetup

   problem = prepare_meltyq_problem(
       chemsetup(),
       species_map={
           "He": "He1",
           "H2O": "H2O1",
           "CO": "C1O1",
           "CO2": "C1O2",
           "CH4": "C1H4",
           "NH3": "H3N1",
       },
   )
   result = magma_gas.solve(
       problem,
       temperature_k=1700.0,
       pressure_bar=7000.0,
       model_inputs=MeltyqMagmaGasInputs(
           oxygen_fugacity_bar=1.0e-10,
           co_melt_mole_ratio=5.0e-5,
           n_melt_mole_ratio=1.0e-4,
       ),
   )

   if not bool(result.diagnostics.converged):
       raise RuntimeError("magma--gas interface did not converge")

   gas_by_species = dict(
       zip(MELTYQ_SPECIES, map(float, result.gas.equilibrium.x))
   )
   melt_by_quantity = dict(
       zip(
           MELTYQ_MELT_QUANTITIES,
           map(float, result.model_state.melt_volatile_mole_ratios),
       )
   )
   co_index = MELTYQ_SPECIES.index("CO")

   print("converged:", bool(result.diagnostics.converged))
   print("delta IW:", f"{float(result.model_state.delta_iw):.3f}")
   print("gas H2O mole fraction:", f"{gas_by_species['H2O']:.3f}")
   print(
       "gas CO p/f (bar):",
       f"{float(result.gas.partial_pressures_bar[co_index]):.3g}",
       f"{float(result.gas.fugacities_bar[co_index]):.3g}",
   )
   print("melt CO dilute ratio:", f"{melt_by_quantity['CO']:.3g}")
   print("melt N dilute ratio:", f"{melt_by_quantity['N']:.3g}")

Typical output (minor final digits depend on JAX precision) is:

.. code-block:: text

   converged: True
   delta IW: -0.774
   gas H2O mole fraction: 0.550
   gas CO p/f (bar): 563 563
   melt CO dilute ratio: 5e-05
   melt N dilute ratio: 0.0001

The CO partial pressure and fugacity coincide here because no ``lnphi_func``
was supplied.  A non-ideal pure-component correction generally separates
them.

Units and logarithm bases
-------------------------

.. list-table:: Interface conventions
   :header-rows: 1
   :widths: 32 68

   * - Quantity
     - Convention
   * - ``temperature_k``
     - Kelvin.
   * - Public total pressure, partial pressure, and fugacity
     - Numerical values in bar.  ``pressure_bar`` is also the fixed gas
       total pressure at the interface.
   * - ``oxygen_fugacity_bar``
     - Absolute molecular-O2 fugacity in bar, supplied as a positive linear
       value rather than :math:`\Delta{\rm IW}` or an O2 mole fraction.
   * - ``co_melt_mole_ratio``
     - Dilute elemental-C ratio derived from the CO law's native mass basis.
   * - ``n_melt_mole_ratio``
     - Dilute atomic-N ratio; twice the N2-equivalent molecular ratio.
   * - ``lnphi_func``
     - Natural logarithm of a dimensionless fugacity coefficient.
   * - ``result.gas.log_mole_fractions`` and ``result.root_variables``
     - Natural logarithms.
   * - ``result.gas.equilibrium.ln_n`` and ``ntot``
     - Raw gas solver amounts in the :math:`b_{\rm H}=1` gauge.  ``ln_n``
       contains natural logarithms; ``ntot`` is linear.
   * - IW helper output and ``delta_iw``
     - Dimensionless base-10 logarithms; one unit is one decade.
   * - Empirical solubility-law pressures
     - Source-defined mixture of Pa, bar, and GPa, converted internally as
       listed in :doc:`solubility`.
   * - ``melt_volatile_mole_ratios``
     - H2 and CH4 retain their laws' native mole-fraction outputs.  H2O, CO2,
       elemental C, and elemental N use MELTYQ's dilute 60 g/mol matrix
       conversion.  The combined vector is not renormalized.

``convert_pressure`` performs pure unit scaling among Pa, bar, and GPa; it
does not validate the physical domain.  A raw value in Pa must never be passed
to an argument ending in ``_bar``.  For example, doing so in the oxygen-
fugacity helper would shift :math:`\Delta{\rm IW}` by five decimal-log units.

State and diagnostics
---------------------

Gas arrays use ``MELTYQ_SPECIES`` order and melt arrays use
``MELTYQ_MELT_QUANTITIES`` order.

.. list-table:: ``MagmaGasResult`` with the MELTYQ preset
   :header-rows: 1
   :widths: 32 68

   * - Field
     - Meaning
   * - ``result.element_abundances``
     - Inferred ``(H, C, O, N, He)`` amounts in the :math:`b_{\rm H}=1`
       gauge.
   * - ``result.gas.equilibrium``
     - Generic ``EquilibriumResult`` containing raw ``ln_n``, ``n``, ``x``,
       and ``ntot`` values.  Amounts use the :math:`b_{\rm H}=1` gauge and are
       not a physical number density.
   * - ``result.gas.log_mole_fractions``
     - Natural logarithms of the nine gas mole fractions.
   * - ``result.gas.partial_pressures_bar``
     - :math:`p_s=x_sP`, in bar; these sum to the total pressure.
   * - ``result.gas.fugacities_bar``
     - :math:`f_s=\phi_s p_s`, in bar; these need not sum to the total
       pressure.
   * - ``result.model_state.melt_volatile_mole_ratios``
     - ``(H2, H2O, CO, CO2, CH4, N)`` outputs in the mixed native/converted
       bases described above; the vector is not a normalized composition.
   * - ``result.model_state.delta_iw``
     - Base-10 oxygen-fugacity offset from the Hirschmann IW reference.
   * - ``result.root_variables``
     - Natural-log ``(C/H, O/H, N/H, He/H)`` ratios.
   * - ``result.diagnostics``
     - Nested outer-root and inner-equilibrium convergence information.

.. list-table:: ``MagmaGasDiagnostics``
   :header-rows: 1
   :widths: 32 68

   * - Field
     - Meaning
   * - ``converged``
     - True only if both the outer root and final inner gas audit converged.
   * - ``outer_converged``, ``inner_converged``
     - Separate convergence decisions for the two solves.
   * - ``iterations``, ``inner_iterations``
     - Iteration counts for the outer solve and final inner audit.
   * - ``residual``
     - Four outer residuals in ``MELTYQ_ROOT_RESIDUALS`` order:
       ``(O2, CO_melt, N_melt, H2_He)``.
   * - ``residual_norm``, ``root_tolerance``
     - Outer infinity norm and its dtype-adjusted acceptance threshold.
   * - ``inner_residual_norm``, ``inner_tolerance``
     - Final gas-equilibrium audit norm and threshold.
   * - ``step_accepted``
     - Whether the most recently attempted outer iteration accepted its
       line-search step; this is not an overall success flag.

Differentiation and batching
----------------------------

The MELTYQ four-variable root uses a JAX damped Newton iteration. Reverse-mode
derivatives use an implicit 4-by-4 adjoint solve, so ``jax.grad`` and
JIT-wrapped reverse-mode differentiation are supported.  Forward-mode JVP is
not supported by the underlying gas-equilibrium solver.  If either the outer
root or final inner audit fails, the implicit-root backward path returns
``nan`` parameter cotangents rather than silently differentiating an invalid
state.  Diagnostic fields themselves are not part of the differentiation
contract.  Always check ``result.diagnostics.converged`` before using either a
result or its gradient.

For a converged root :math:`\boldsymbol{r}(\boldsymbol{u},\theta)=0`, the local
derivative is

.. math::

   \frac{\partial\boldsymbol{u}^*}{\partial\theta}
   = -\left(\frac{\partial\boldsymbol{r}}{\partial\boldsymbol{u}}\right)^{-1}
     \frac{\partial\boldsymbol{r}}{\partial\theta}.

This differentiates the converged equations rather than the Newton and
backtracking iteration path.  It requires a nonsingular local root Jacobian;
the initial guess is not a differentiation target.

Requested root and gas tolerances are bounded below by dtype-aware roundoff
floors, allowing the documented solve to run with JAX's default float32
precision as well as float64.  One call solves one scalar thermodynamic state.
For a batch, keep the prepared problem and numerical options static and map
a scalar closure with ``jax.vmap``.

Scientific scope and calibration
--------------------------------

The generic engine can host other smooth square-residual magma models. The
MELTYQ preset documented here is one boundary model, not a general magma
thermodynamics model. Its equilibrium prediction is conditional on the
reduced five-element, nine-gas-species network; omitted gases and all
condensed phases cannot appear.  The fixed 60 g/mol melt matrix and default
basaltic oxide composition in the N law are also model assumptions.

The six solubility laws are a compilation of separate experiments and
formulations.  Their metadata ranges have only a simple common overlap of
approximately 1673--1723 K and 0.7--0.8 GPa, and even this overlap is not a
joint calibration because the source melt compositions and datasets differ.
The laws and the IW helper evaluate outside their calibration ranges without
clipping or automatic warnings.  Numerical convergence therefore does not
establish scientific validity.  Inspect ``MELTYQ_SOLUBILITY_METADATA`` in
:doc:`solubility` for each law's range, provenance, basis, and specific
caveats before interpreting or differentiating a result.

The Hirschmann iron-polymorph selection is piecewise.  Values and derivatives
within either branch are JAX-compatible, but smooth differentiation exactly
at the branch boundary is not guaranteed.
