Local multiphase Gibbs minimization
===================================

``exogibbs.api.multiphase`` implements the R1/R2 public contract and a
fixed-support JAX primal. It minimizes a sum of full extensive phase Gibbs
energies with an absolute elemental ledger. It does not change the existing
``gas``, ``condensate`` or ``magma_gas`` APIs.

ExoEOS owns thermodynamic functions and their physical domains. ExoGibbs owns
this local constrained minimization. ExoInventory owns absolute planetary
reservoirs, atmospheric columns and pressure closure. No sibling package or
provider example is needed for a synthetic problem.

Minimal use
-----------

.. code-block:: python

   import jax
   import jax.numpy as jnp
   from exogibbs.api import multiphase

   jax.config.update("jax_enable_x64", True)

   def gibbs_j(t, p, n, parameters):
       # An illustrative ideal binary solution, not an M2 material recipe.
       return 8.31446261815324 * t * (
           n @ (jnp.log(n / n.sum()) + parameters["standard_rt"])
       )

   problem = multiphase.Problem(
       elements=("X",),
       phases=(multiphase.Phase(
           name="solution", components=("A", "B"),
           atom_matrix=((1., 1.),), gibbs_j=gibbs_j,
       ),),
   )
   result = multiphase.solve(
       problem, temperature_k=1200., pressure_pa=1e5,
       element_amounts_mol=jnp.array([2.]), init=jnp.array([.6, 1.4]),
       parameters={"standard_rt": jnp.array([0., 1.])},
   )
   assert result.converged and result.minimum_valid and result.support_valid

Contract and numerical method
-----------------------------

* ``gibbs_j(T_K, P_Pa, n_mol, parameters)`` returns a scalar in J, homogeneous
  of degree one in component amounts. It includes standard, mixing, excess
  and pressure terms exactly once. It must be differentiable on the selected
  component face. The caller supplies an explicit dynamic parameter PyTree;
  differentiable coefficients must not be hidden in a static closure.
* Atom-matrix rows follow ``Problem.elements``; columns follow each phase's
  components. ``problem.component_slices`` identifies the concatenated amount
  vector. Phase-local names may repeat in another phase. Charge and signed
  formal endmembers are not represented by this nonnegative atom ledger.
* Optional ``Phase.domain_matrix`` rows impose homogeneous physical constraints
  :math:`C_p n_p\geq0`. Declare their meaning in ``Phase.domain`` and the
  standard-state convention in ``Phase.standard_state``. All phases share a
  compatible energy reference. Numerical tolerances are not physical caps.
* ``Support.components`` lists strictly positive component indices; all others
  are exactly zero. ``Support.active_constraints`` lists equality-active rows
  in the concatenated domain matrix. A supplied initial amount vector is
  projected onto elemental and active-domain equalities. This projection must
  remain inside the chosen face; no general feasible initializer is promised.
* Internally :math:`B=\sum_e b_e`, :math:`q=n/B` and
  :math:`G/(R_{scale}TB)` remove the absolute amount scale. Static SVD handles
  redundant elemental rows and constructs a constraint tangent basis. Active
  domain rows must add independent constraints. Full AD Hessians, regularized
  reduced Newton directions and a positivity-preserving Armijo line search
  minimize the scalar; line-search comparisons allow floating-point roundoff.
  Negative-curvature directions allow departure from stationary maxima.
  Final curvature and KKT audits use the unmodified Gibbs Hessian.

``solve`` is currently a host convenience wrapper: it removes components
forbidden by exact zero element budgets and otherwise uses all reachable
components unless given a support. It does **not** search phase supports or
activate bounds. Failure at a boundary requires a new declared support, not
an amount floor. Missing phases are not evaluated and have no composition or
chemical potential. A present phase callback must support derivatives on its
selected face; the core cannot repair a provider's singular face derivative.

For JIT, capture the static problem/support/options in a closure and call
``solve_fixed_support`` with dynamic T, P, budgets, initialization and parameters.
The primal uses JAX, with no SciPy optimizer or host callback. This R2 entry
explicitly raises ``NotImplementedError`` for JVP/VJP: local implicit
derivatives are R4, not differentiation of the iteration history. There is no
high-level phase-search ``jit/vmap`` contract.

Results and limits
------------------

The result returns absolute component and phase amounts, phase elemental
amounts, G in J, present-component chemical potentials, elemental potentials,
active-domain multipliers, conservation residuals in mol atoms and normalized
KKT/domain diagnostics. Redundant elemental equations use a row-scaled
minimum-norm dual gauge, with ``dual_unique=False``. Unsupported elemental
potentials are NaN; compare duals only in compatible gauges.

``converged`` describes numerical stopping, ``local_kkt_valid`` independently
checks conservation, stationarity, domain, multiplier sign and complementarity.
These are checks on the declared face; insertion into omitted components or
phases is unchecked.
``minimum_valid`` additionally requires strictly positive tangent curvature
(or no free direction); flat/degenerate stationary states are not accepted as
isolated minima. ``support_valid`` also checks inactive domain margins.
Inspect all applicable flags, not just an optimizer stopping flag.

``stability_status`` is ``unchecked``; insertion lower/upper bounds and
incipient compositions are NaN. Neither a local minimum nor an excluded phase
is a phase-stability certificate. ``derivative_valid`` is false in R1/R2.
Phase search, insertion bounds and proof connection are R3; implicit derivatives
are R4. These functions do not establish a natural BSE metal-free endpoint,
adopt a new M2 material model, or close a planetary pressure root.

Optional native ExoEOS connection
---------------------------------

``exogibbs.interop.exoeos.make_magma_liquid_phase`` accepts a complete native
liquid basis, optionally reordered, and maps elemental rows by name. Missing
elements with nonzero atom columns are errors; extra ledger elements (e.g. He)
receive explicit zero rows. It forwards K/Pa/mol directly to
``exoeos.magma.liquid_gibbs``. Unlike ``make_solution_lngamma_func``, this adapter
already supplies **full G**; do not add mixing or water terms again.
``make_magma_pure_phase`` converts a selected native molar function to
:math:`G=n g(T,P)`.

Native two- through twelve-component liquid bases retain their provider
standards, pressure terms and gas constant. Native hydrous12 does not replace
the old 14-component M2 H2/He dissolution model or its 20-component alloy.
The adapter does not extend native derivative domains (positive composition,
temperature branches and provider-specific site limits). It makes no claim
about signed-coordinate solid solutions or omitted competing phases.

Validation
----------

``tests/unittests/equilibrium/multiphase`` checks same-scalar SLSQP parity,
gas-only and gas-plus-pure-condensate API reductions, absolute scaling,
elemental gauge, zero budgets, active constraints, invalid supports, nonconvex
descent and JIT. ``tests/provider_integration/multiphase_test.py`` compares
native liquid/forsterite equilibria with an independent analytic-potential
root, and checks full-G extensivity and step-refined thermodynamic derivatives.
The native integration job uses provider wheels at a separately pinned ExoEOS
revision; the historical M2 provider pin and evidence are unchanged.
