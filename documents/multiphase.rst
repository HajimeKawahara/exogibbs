Local multiphase Gibbs minimization
===================================

``exogibbs.api.multiphase`` implements the R1--R4 public contract, a
fixed-support JAX primal with implicit derivatives and host phase selection.
It minimizes a sum of full extensive phase Gibbs
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
           n @ (jnp.log(jnp.where(n > 0, n / n.sum(), 1.)) + parameters["standard_rt"])
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

``solve`` removes components forbidden by exact zero element budgets and
searches component/phase supports and active linear-domain faces on the host.
It tries the supplied positive support and the full reachable support, then
adds boundary faces when a constrained solve fails. A linear program supplies
feasible starts; it does not minimize Gibbs energy. All equilibrium and
insertion minimizations use the JAX kernel. ``Options.max_supports`` bounds
each search, which is not an exhaustive enumeration of a nonconvex catalog.
An explicit ``support=`` retains fixed-face behavior without phase search.

A small amount only proposes a reduced face after a failed solve. Its fresh
constrained solution and insertion audit determine acceptance; no accepted
positive amount is rounded to zero. An absent phase has exactly zero amount
and no chemical potential. Its separate incipient composition describes a
one-mole insertion trial. A present callback must support derivatives on its
selected face; the core cannot repair singular provider derivatives.

For JIT, capture the static problem/support/options in a closure and call
``solve_fixed_support`` with dynamic T, P, budgets, initialization and parameters.
The primal uses JAX, with no SciPy optimizer or host callback. This entry
provides local implicit JVP/VJP without differentiating the iteration history.
There is no high-level phase-search ``jit/vmap`` or AD contract.

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
These are checks on the declared face, separate from insertion stability.
``minimum_valid`` additionally requires strictly positive tangent curvature
(or no free direction); flat/degenerate stationary states are not accepted as
isolated minima. ``support_valid`` also checks inactive domain margins.
Inspect all applicable flags, not just an optimizer stopping flag.

``solve_fixed_support`` leaves stability ``unchecked`` and insertion diagnostics
NaN. ``solve`` additionally audits every declared phase, including alternative
compositions of present phases. ``audit_stability`` repolishes a supplied face
at the supplied T/P/b/parameters before repeating that audit, so a previous
state's dual or certificate is not reused at changed conditions.

Insertion costs are dimensionless:

.. math::

   \tau_p(x) = [G_p(T,P,x)-\lambda^T A_p x]/(R_{scale}T),\qquad \sum_i x_i=1.

``insertion_upper`` is a freshly evaluated feasible trial. Multi-start local
minima and feasible vertices can reveal a negative trial, but do not prove a
lower bound. Pure phases have exact insertion bounds. A solution phase can
supply ``insertion_lower_bound_j_mol(T, P, element_dual, parameters,
allowed_components)``. It must bound the same full G over the **whole** declared
simplex/domain restricted to the static tuple of allowed component indices.
Return J/mol components, not G/RT or a phase's total joules. NaN denotes an
unavailable proof. The adapter can call an existing independently pinned
interval/common-plane proof using these actual inputs; old receipts are not
proofs for new states or different material models. The forward package does
not import a provider's example files. Bounds above an evaluated trial are
rejected; the mathematical validity of a supplied bound remains its owner's
contract. Components forbidden by zero budgets are excluded exactly; an
entirely unreachable phase has +inf bounds and NaN incipient composition.

``stability_status`` is ``unstable`` for a negative trial, ``certified`` when
all declared phase lower bounds are at least ``-kkt_tolerance``, and
``unresolved`` when convergence or proof is insufficient. Certification is
limited to the supplied catalog, domains, full G and bound contracts.
In particular, positive local trials without global bounds remain unresolved.
These functions do not establish a natural BSE metal-free endpoint,
adopt a new M2 material model, or close a planetary pressure root.

Local implicit derivatives
--------------------------

After ``solve`` selects a support, capture that support and call the public
``solve_fixed_support`` entry. Its ``custom_jvp`` solves the differentiated
active KKT equations, including the constraint multipliers through elimination:

.. math::

   M\,dq = d\widehat b,\qquad
   Z^T H Z\,dy = -Z^T(H\,dq_{particular}+d_\theta\nabla_q E),\qquad
   dq = dq_{particular}+Z\,dy.

Here :math:`M` is the row-scaled element/active-domain matrix, :math:`Z` spans
its null space, and :math:`E=G/(R_{scale}TB)`. Reverse mode transposes this
linear map. The derivatives include :math:`dB` in normalized constraints and
in restoring :math:`n=Bq`; temperature normalization and full thermodynamic
T/P/material dependence are included. Initialization has zero derivative.

.. code-block:: python

   selected = result
   support = selected.support

   @jax.jit
   def fixed(b, parameters):
       return multiphase.solve_fixed_support(
           problem, temperature_k=1200., pressure_pa=1e5,
           element_amounts_mol=b, parameters=parameters,
           init=selected.amounts_mol, support=support,
       )

   parameters = {"standard_rt": jnp.array([0., 1.])}
   base = fixed(jnp.array([2.]), parameters)
   assert base.derivative_valid
   dn_db = jax.jacfwd(lambda b: fixed(b, parameters).amounts_mol)(jnp.array([2.]))

The differentiable outputs are amounts, phase amounts, phase elemental
amounts, G, defined chemical/element potentials, active-domain multipliers
and conservation residuals. Numerical diagnostics are nondifferentiable.
``solve`` and ``audit_stability`` perform host discovery/proof work and leave
``derivative_valid=False``; evaluate the selected face with
``solve_fixed_support`` to obtain the derivative audit.

``derivative_valid`` requires a converged KKT minimum, positive component and
inactive-domain margins, strictly positive active multipliers, and well
conditioned constraint and tangent-Hessian systems. ``derivative_rcond`` sets
the relative conditioning threshold. These numerical margins do not round
small phases to zero. Pure absent phases require positive insertion cost;
absent solution phases require a positive JAX-compatible global lower bound.
Missing such bounds leaves derivatives invalid. Host-only proof callbacks
remain usable by ``solve``/``audit_stability``. Present phases with omitted
reachable components lack a smooth one-sided contract and are also invalid.

``Phase.derivative_domain(T, P, n, parameters)`` can declare a provider's
smooth-branch predicate. Without this optional predicate, the caller asserts
that the callback is smooth on the selected face. Active linear inequalities
with strict complementarity admit a constrained-branch derivative; a zero
multiplier or a vanishing inactive slack marks an active-set transition.
Failed/degenerate roots, detected phase switches and material-branch boundaries
return ``derivative_valid=False`` and NaN JVPs/VJPs, including otherwise zero
physical Jacobian columns. Malformed inputs still raise normal validation
errors. Restart host phase selection when support validity is lost.

The contract is first-order and local to the selected isolated branch, not a
proof of global uniqueness among nonconvex solutions. With redundant element
rows, budget variations must remain in the selected atom matrix's column
space; parameterize compatible budgets rather than varying redundant rows
independently. Zero-budget rows must remain zero. The row-scaled minimum-norm
extension outside that feasible budget manifold is not a physical derivative.
For compatible variations, :math:`A\,dn=db`. Material applicability, global
stability and ExoInventory's outer pressure-root derivative remain separate.

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
The liquid adapter's derivative predicate requires positive components and
excludes the SiO2 standard-state branch point at 1480 K. A caller-selected
pure-phase function remains responsible for declaring any additional branch
limits through its Phase metadata.

Validation
----------

``tests/unittests/equilibrium/multiphase`` checks same-scalar SLSQP parity,
gas-only and gas-plus-pure-condensate API reductions, absolute scaling,
elemental gauge, zero budgets, active constraints, invalid supports, nonconvex
descent, phase switches, global-bound contracts, JIT, step-refined equilibrium
derivatives, JVP/VJP transpose consistency, differentiated conservation and
invalid-gradient propagation. ``tests/provider_integration/multiphase_test.py`` compares
native liquid/forsterite equilibria with an independent analytic-potential
root, and checks full-G extensivity and step-refined thermodynamic derivatives.
The native integration job uses provider wheels at a separately pinned ExoEOS
revision; the historical M2 provider pin and evidence are unchanged.

Trace species and retained atmospheres
-------------------------------------

The primal uses a positive amount-weighted feasibility correction and a
square-root amount scaling of its equality tangent. These operations retain
positive trace species without composition floors. ``minimum_curvature`` is
measured in this scaled tangent; congruence preserves the inertia of the
physical Hessian. Conservation and stationarity tolerances are unchanged.
The implicit derivative audit still rejects ill-conditioned physical systems.

A valid global insertion lower bound can establish stability without repeated
local insertion minimizations. An evaluated feasible upper trial is retained
and checked against that lower bound. Missing bounds remain unresolved.

``exogibbs.api.condensate.audit`` independently reconstructs elemental
conservation, gas chemical potentials and pure-condensate insertion from
primitive mol amounts. It assumes ideal gas and retained pure condensates.
It accepts an explicit common elemental gauge for cross-phase G comparisons,
handles exact-zero element budgets, and enforces temperature eligibility.
It does not accept a saved solver flag or saved dual as evidence.

The packaged ``thermo.young2023`` and ``thermo.janaf_atomic`` functions expose
pinned standards needed by the M2 native recipe. The former retains its frozen
2350 K Shomate branches; the latter retains its 2100--2400 K interpolation
range. These functions do not define phase assemblages or improve empirical
cross-phase calibration. ``thermo.ideal_gibbs`` supplies the full ideal-gas
scalar in K/Pa/mol and its analytic normalized insertion lower bound.

``examples/metal_silicate/run_m2_native_proof.py`` remains a separate proof
workflow. It consumes a newly serialized primitive source, repairs feasibility
with exact arithmetic, and recomputes liquid/alloy, gas/cloud and all 33 native
mineral competitor bounds. It may use a pinned provider checkout and native
binary; neither is needed by the Inventory forward calculation.
