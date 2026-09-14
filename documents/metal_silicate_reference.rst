Metal--silicate reference thermochemistry
=========================================

The metal--silicate examples fix the component basis, standard thermochemistry,
and local amount convention, then solve a declared local phase assemblage.
``examples/metal_silicate/reference.py`` audits the frozen thermochemistry
using NumPy only. Source-model examples and tests need neither ExoEOS nor GCE;
the completed ternary-metal example uses the optional ExoEOS provider.

Reference selection
-------------------

The numerical source is ``Young_2023_Version`` in
`GCE commit 31558873d8da460c3cd11986b574cb347621b43d
<https://github.com/ExoInteriors/GlobalChemicalEquilibrium_Release/tree/31558873d8da460c3cd11986b574cb347621b43d/Young_2023_Version>`_,
associated with Young et al. (2023),
`Earth shaped by primordial H2 atmospheres
<https://doi.org/10.1038/s41586-023-05823-0>`_. The JSON record stores source
paths and SHA256 hashes, coefficient provenance, and the original elemental
amounts. Coefficients are audited against this pinned implementation; their
experimental calibrations are not independently established here.

All 25 components and 18 independent reactions of that version are retained.
The component suffix identifies its phase, so ``Fe_metal`` and ``Fe_gas``
are separate species.

.. list-table:: Component order within each phase
   :header-rows: 1
   :widths: 15 85

   * - Phase
     - Components
   * - Silicate
     - MgO, SiO2, MgSiO3, FeO, FeSiO3, Na2O, Na2SiO3, H2, H2O, CO, CO2
   * - Metal
     - Fe, Si, O, H
   * - Gas
     - H2, CO, CO2, CH4, O2, H2O, Fe, Mg, SiO, Na

The element order is ``(Si, Mg, O, Fe, H, Na, C)``. The Mg-bearing host
explicitly includes MgO, SiO2, and MgSiO3; the first record is not reduced
to Fe/Si oxides. The more general GCE network and an unused ``initial.dat``
entry include SiH4, which is absent from this version's actual equations.

The formula matrix :math:`A` has shape :math:`(7,25)` and rank 7. Rows of
:math:`\nu`, shape :math:`(18,25)`, use negative reactants and positive products.
They satisfy :math:`A\nu^T=0` and have rank 18. Reaction R3 is
:math:`\tfrac12\mathrm{Si}_{metal}+\mathrm{O}_{metal}
\rightarrow\tfrac12\mathrm{SiO}_{2,silicate}`: the reference follows the
actual source residual, whose direction differs from a comment in the source
Gibbs script.

Two frozen cases evaluate all reactions at one temperature, respectively
2350 K and 3000 K, with 1 bar standards. They are **thermochemistry checks**,
not equilibrium compositions or measured activities. The original source
uses 3000 K for R1, R3, R4, and R6, and 2350 K for the other reactions.
That two-temperature prescription is recorded but is not interpreted as
minimization of one isothermal Gibbs potential.

Consistent standard states
--------------------------

The source uses :math:`R=8.314462618153` J/(mol K), the rounded
``log10_to_ln=2.302585093``, and :math:`P^\circ=1` bar. Its ``GRT`` output
is :math:`\Delta_r G^\circ/(RT)`, despite an output-file header saying J/mol.
Thus

.. math::

   \ln K_r=-\frac{\Delta_r G^\circ}{RT}
          =-\sum_i\nu_{ri}\frac{\mu_i^\circ}{RT}.

For 16 components, the record freezes the source's selected Shomate branch
and independently captured standard potential. The audit evaluates the
enthalpy in kJ/mol, converts it to J/mol, and subtracts :math:`T S^\circ`
with entropy in J/(mol K). This convention uses formation enthalpy at the
reference temperature and absolute entropy. It must not be mixed directly
with FastChem's atomic-reference ``h=-ln(K)`` or formation-Gibbs tables.

The other nine standards are reconstructed from the chosen reaction fits:
Na2SiO3 and FeSiO3 from R0/R5; dissolved H2, H2O, CO, and CO2 from R14--R17;
metal Si/O from the selected exchange fits; and metal H from the Okuchi fit.
In particular, the source's unused standalone Na2SiO3/FeSiO3 potentials
cannot replace these reaction-consistent standards. No missing potential is
filled with an assumed zero. The full reconstruction reproduces all 18
source reaction free energies at both temperatures.

A common elemental reference change
:math:`\boldsymbol\mu^\circ\mapsto\boldsymbol\mu^\circ+A^T\boldsymbol c`
cancels from every balanced reaction. A phase-specific standard-state change
instead requires a compensating activity conversion.

ExoEOS standard-state mapping
-----------------------------

The record also pins the two ternary conversion fixtures from
`ExoEOS commit 39176c423e9e7783dd956e1abe75c9b610a81864
<https://github.com/HajimeKawahara/exoeos/blob/39176c423e9e7783dd956e1abe75c9b610a81864/documents/fe_si_o_reference.rst>`_,
model ``ma2001_fe_si_o_young2023_printed_v1``. Its component order is
``(Fe, Si, O)``. For the completed model, define

.. math::

   \boldsymbol h(T)&=\left(0,5.76\frac{1873}{T},4.29-\frac{33000}{T}\right),\\
   \boldsymbol\mu^{\circ,formal}&=\boldsymbol\mu^{\circ,source}+RT\boldsymbol h,\\
   \ln\boldsymbol\gamma^{formal}&=\ln\boldsymbol\gamma^{source}-\boldsymbol h.

The chemical potential
:math:`\mu_i=\mu_i^\circ+RT(\ln x_i+\ln\gamma_i)` is unchanged when both
transformations are applied. These formal symmetric endmembers are not
calibrated pure Si/O liquids. The fixture checks this conversion only; it
does not supply a runtime activity model or adapter.

The ExoEOS model restores the Fe solvent contribution and selects Young's
printed oxygen coefficient, whereas the pinned GCE implementation uses
``gamma_Fe=1`` and the author-code oxygen variant. They are distinct physical
models. Also, a ternary ExoEOS model cannot provide the four-component GCE
alloy's H activity. Original-model reproduction and completed-model
verification are separately identified below.

Optional solution activity adapter
----------------------------------

``exogibbs.interop.exoeos.make_solution_lngamma_func`` now connects a
homogeneous ExoEOS solution to a callback ``lngamma(T_K, P_bar, x_phase)``.
ExoEOS is imported only when the factory is called and is not a required
ExoGibbs dependency. For example, with a current ExoEOS checkout:

.. code-block:: python

   from exoeos import IdealSolution
   from exogibbs.interop.exoeos import make_solution_lngamma_func

   lngamma = make_solution_lngamma_func(
       source_components=("Fe", "Si", "O"),
       model=IdealSolution(),
       model_components=("Fe", "Si", "O"),
   )
   # T in K, P in bar, normalized fractions in consumer order.
   values = lngamma(3000.0, 1.0, [0.9, 0.08, 0.02])

Models with ``components`` supply their own static component order; an
explicit ``model_components`` must agree with that declaration. Input
fractions and returned coefficients are permuted between that order and
``source_components``. Component sets must match exactly. The adapter
requires ``activity_basis="mole_fraction"`` and
``standard_state_convention="symmetric"``; it rejects missing components
instead of assigning them ideal activities.

The callback converts bar to Pa once and returns only
``solution_state(...).lngamma``. It adds neither ideal mixing nor standard
potentials; the caller must apply the documented standard-state conversion.
JIT, batching with ``jax.vmap``, and derivatives through T, P, and composition
are preserved. Shape checks apply while tracing. Normalization, positivity,
and the provider's physical domain remain caller obligations; compositions
are not clipped or renormalized.

Local amount and pressure contract
----------------------------------

For each phase :math:`\alpha`, :math:`N_\alpha` is mol of its declared
components, and :math:`n_{\alpha i}=N_\alpha x_{\alpha i}`. Silicate and gas
components count formula units; metal components count atoms. Elemental
amounts are mol of atoms:

.. math::

   \boldsymbol b_\alpha=A_\alpha\boldsymbol n_\alpha,
   \qquad \boldsymbol b=\sum_\alpha\boldsymbol b_\alpha.

``phase_element_amounts`` exposes these contributions independently, including
trace elements and exact zeros. Normalized compositions alone do not determine
the physical phase amounts. ``original_element_amounts_mol`` can be supplied
as a local parcel budget; the stored planet mass is provenance only.
The declared local closure conserves finite oxygen and does not also impose
an external oxygen buffer. Existing buffered-H/MELTYQ calculations retain
their separate meaning.

Temperature and pressure are supplied independently in K and bar.
ExoInventory owns planetary pressure and global reservoir exchange.
The optional ExoEOS adapter converts bar to Pa once and obtains only
``ln(gamma)`` from ``solution_state(...).lngamma``.

For an ordinary ideal gas, the pressure contribution to reaction R is
:math:`\Delta\nu_{gas}\ln(P/P^\circ)`. The source H2 dissolution residual R14
instead uses :math:`-\ln(10^4/P^\circ)` with pressures in bar. This additional
prescription is recorded explicitly and is not silently included in the
standard potentials or implemented as a general pressure law in this audit.

Reproduction and scope
----------------------

Fixed-phase numerical solve
~~~~~~~~~~~~~~~~~~~~~~~~~~~

``examples/metal_silicate/local.py`` supplies a small example-level
``build_problem`` / ``solve`` interface. It reuses the existing magma--gas
implicit root solver for a direct square mass-action system; there is no
nested gas solve in this path. Standard potentials, phase activities, and
any empirical reaction offsets are explicit callbacks in K/bar.

The unknowns are ``log(n_i / sum(b))`` for active components. The residual
contains logarithmic elemental balances and an independent reaction basis.
Mole fractions are formed within each phase with ``logsumexp``. The factory
removes components containing exact-zero-budget elements and checks the
rank of the remaining element matrix. Surviving source reactions are kept;
balanced null-space reactions complete a reduced basis when necessary.
No artificial trace material is introduced.

``LocalResult`` returns component and phase amounts in mol, mole fractions,
phase-by-element atomic amounts, relative elemental residuals, independent
reaction residuals, and the root solver's diagnostics. Excluded components
and elements remain exact zeros in the original record's output basis.
The positive element support and selected phases are fixed before tracing.
Uniformly scaling all budgets scales phase amounts without changing the
equilibrium compositions.

Supply positive initial component amounts near the intended phase branch;
the default seed is a numerical convenience and does not select a basin.
The solver does not compare phases or test stability against absent phases.
JIT and implicit reverse derivatives apply on a smooth converged branch;
failed solves retain their residuals and return NaN implicit derivatives.
Convergence alone is insufficient: inspect elemental and reaction residuals
independently. Targeted tests include analytic ideal/nonideal controls,
temperature and composition derivatives, elemental reference changes,
exact zeros, trace carbon, amount scaling, and failure diagnostics.

Reference audit commands
~~~~~~~~~~~~~~~~~~~~~~~~

Run from the ExoGibbs repository root:

.. code-block:: console

   python examples/metal_silicate/reference.py
   python -m pytest tests/unittests/examples/metal_silicate_reference_test.py

To check the frozen values against an already available source checkout:

.. code-block:: console

   python examples/metal_silicate/extract_reference.py --gce-checkout /path/to/GCE --check

The optional extractor verifies the three source hashes before evaluating the
Gibbs script in a temporary directory. It does not download anything or run
GCE's equilibrium solver. Ordinary tests use the committed JSON only.
The source comparison tolerance is ``rtol=atol=5e-12``; comparisons using
the exact ``ln(10)`` allow ``atol=1e-9`` for the source's rounded conversion.
These tolerances describe numerical reproduction, not experimental error.

Independent local equilibrium references
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

``equilibrium_reference.json`` adds independent NumPy/SciPy solutions from
``generate_equilibrium_reference.py``. It first preserves all 25 source
components at 2350 K and 3000 K, then records the smaller dry model at the
same temperatures. All cases use a locally supplied pressure of 1 bar and
the recorded elemental budgets. They replace the source's two temperatures
with one isothermal condition and omit its planetary pressure law.

The full-source model retains its author-code Si/O metal coefficients,
``gamma_Fe=gamma_H=1``, and ideal silicate/gas mixing. ``source.py`` evaluates
the chosen Shomate coefficient branches in JAX and reconstructs all standards.
Temperature differentiation is local to those fixed branches. R14 alone
receives the explicit residual offset ``ln(P_bar / 1e4)``; this empirical
mass-action prescription is not presented as a common Gibbs minimum.

The dry model contains silicate MgO, SiO2, MgSiO3, FeO, FeSiO3 and metal
Fe, Si, O. Its H, Na, C budgets are exactly zero, and gas is excluded by
the declared phase assemblage. Its four reactions are R1, R2, R3, R5.
``native.py`` connects ExoEOS ``MaFeSiOLiquid`` from
`commit 9c62197d3a7c1aa882e246c9de3fbc4102410d81
<https://github.com/HajimeKawahara/exoeos/tree/9c62197d3a7c1aa882e246c9de3fbc4102410d81>`_.
It obtains activities and ``standard_state_shift_RT`` from the same model
instance. Adding that shift to the source standards preserves the completed
model's chemical potentials. Independent analytic partial derivatives of
the excess energy generate the reference without importing ExoEOS or JAX.

.. list-table:: Independent reference phase amounts (mol of declared components)
   :header-rows: 1

   * - Model
     - T (K)
     - Silicate
     - Metal
     - Gas
   * - Full source
     - 2350
     - 226.80445923
     - 201.29911591
     - 26.94491093
   * - Full source
     - 3000
     - 204.54046576
     - 296.31023037
     - 119.77285916
   * - Completed dry
     - 2350
     - 236.13126613
     - 179.90174448
     - Excluded
   * - Completed dry
     - 3000
     - 253.84091261
     - 180.51961433
     - Excluded

Run with float64 enabled. A local ExoEOS checkout may be supplied through
``PYTHONPATH=/path/to/exoeos/src`` for the dry case:

.. code-block:: console

   JAX_PLATFORMS=cpu JAX_ENABLE_X64=1 python examples/metal_silicate/native.py --case source_full_2350
   JAX_PLATFORMS=cpu JAX_ENABLE_X64=1 python examples/metal_silicate/native.py --case completed_dry_2350
   python examples/metal_silicate/generate_equilibrium_reference.py --output /tmp/equilibrium_reference.json

Acceptance requires root convergence, per-element relative residuals at
most ``1e-9`` and independent reaction residuals at most ``1e-8``. The
amount comparison tolerance is ``rtol=1e-8``. Native-provider tests are
optional when ExoEOS is unavailable; the frozen reference and source tests
remain offline. Verification also covers JIT, implicit reverse derivatives,
ideal controls, amount scaling, and common elemental reference shifts.

The silicate approximation in the dry example remains ideal. ExoEOS's
MELTS reference data are external reference states with a different
endmember basis, not a callable JAX activity model; they are not substituted
for these source standards. Likewise, its hydrogen-bearing silicate EOS
does not supply this network's solution chemical potentials.

Finite hydrogen exchange through the magma--gas service
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

``gas_exchange.py`` solves the same full 25-component source model through
the existing ``MagmaGasProblem`` service. Its inner gas equilibrium uses the
ten gas species, seven gas element abundances, and the compatible source
``mu0/(RT)`` standards. It handles gas-only reactions R7, R8, R9.

The outer root has 22 coordinates: 15 logarithmic condensed component
amounts, six gas elemental ratios relative to oxygen, and one logarithmic
physical gas amount. Its 22 residuals comprise seven finite elemental
balances and 15 remaining reactions. Oxygen is a normalization anchor for
the inner gas input and a conserved finite inventory in the outer problem;
it is not an external oxygen buffer. The physical gas amount multiplies
the gas mole fractions before computing the parcel's elemental totals.

The returned ``model_state`` includes full component amounts, per-phase
amounts and atomic contributions, all 18 reaction residuals, and seven
relative elemental residuals. Use these physical amounts for reservoir
accounting; the gas service's normalized ``element_abundances`` are numerical
inputs and do not specify parcel size. Inner and outer convergence flags
remain available in ``diagnostics``. The command-line acceptance gate checks
both convergence and the independent elemental/reaction tolerances.

All seven budgets must remain positive in this full-source example; exact
zero support uses a separately constructed reduced problem such as the dry
example. Temperature and H inventory may vary on a nearby smooth branch.
Tests compare perturbed initial roots against both independent reference
cases, vary finite H, scale all budgets, check implicit T/H derivatives,
and preserve NaN gradients when either nested solve fails.

.. code-block:: console

   JAX_PLATFORMS=cpu JAX_ENABLE_X64=1 python examples/metal_silicate/gas_exchange.py --case source_full_2350
   JAX_PLATFORMS=cpu JAX_ENABLE_X64=1 python examples/metal_silicate/gas_exchange.py --case source_full_3000

This path retains the source's four-component Fe-Si-O-H activities, including
its explicit unit H coefficient, and its R14 pressure prescription. It does
not extend the ternary ExoEOS model to H. Gas and silicate mixing remain
ideal, phases are declared, and finite local oxygen conservation replaces
no existing buffered-H/MELTYQ behavior. Mixture gas EOS, nonideal hydrous
silicate activities, and phase selection require separate physical models.

The current cases use formal liquid standards: for example, the source lists
3105--5000 K for its MgO liquid coefficients, so both cases extrapolate that
endmember fit. Neither a stable liquid assemblage nor a calibrated joint
T/P/composition domain has been established. A jointly calibrated liquid
equilibrium benchmark and phase selection remain outside these fixed-phase
numerical references.

Milestone 1 source-to-upper chemistry
------------------------------------

``examples/metal_silicate/m1_chemistry.py`` compares the existing 2350 K
S/N source with ideal gas-only and gas/pure-condensate equilibrium from
packaged FastChem4 data. It retains H/He/O/Mg/Si/Fe/Na with exact-zero C/N/S,
and separates a shared nine-gas control, 35 neutral gases, and 26 condensate
candidates with their original temperature upper bounds.

.. code-block:: console

   JAX_PLATFORMS=cpu JAX_ENABLE_X64=1 PYTHONPATH=src python examples/metal_silicate/m1_chemistry.py --pressure-bar 100 --point 1400 0.1 --output results/m1_chemistry/central.json

The JSON report records source/upper model differences, independent atom/mass
and phase-condition audits, gas/cloud amounts on the caller's mol basis,
actual imports, source/data hashes, and failed points. These are local
diagnostics: ExoInventory still owns the retained column and global pressure
closure. See the :download:`amount contract and scope
<../examples/metal_silicate/m1_chemistry.md>` for usage and acceptance details.
