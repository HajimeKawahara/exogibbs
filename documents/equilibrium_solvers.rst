Gas and Condensate Solver APIs
==============================

ExoGibbs exposes two parallel public solver modules:

* :mod:`exogibbs.api.gas` for gas-only equilibrium;
* :mod:`exogibbs.api.condensate` for gas-plus-condensate equilibrium.

Both accept temperature ``T``, pressure ``P``, an elemental abundance vector
``b``, and an optional reference pressure ``Pref``.  They deliberately have
different result and execution contracts because condensate support discovery
is a host-side lifecycle rather than one fixed numerical solve.

Gas Quick Start
---------------

.. code-block:: python

   from exogibbs.api.gas import EquilibriumOptions, solve_profile
   from exogibbs.presets.fastchem4 import chemsetup

   setup = chemsetup(silent=True)
   result = solve_profile(
       setup,
       T=temperature_profile,
       P=pressure_profile,
       b=setup.element_vector_reference,
       options=EquilibriumOptions(method="scan_hot_from_bottom"),
   )

For ``N`` layers and ``K`` gas species, ``result.ln_n``, ``result.n``, and
``result.x`` have shape ``(N, K)`` and ``result.ntot`` has shape ``(N,)``.
One-layer :func:`exogibbs.api.gas.solve` returns the corresponding ``(K,)``
arrays and scalar total.

The optional ``init=EquilibriumInit(...)`` keyword supplies an explicit gas
state.  With the default initializer, scan methods use it for the first
scheduled layer and warm-start later layers from the preceding solution;
``vmap_cold`` uses the same state for every independent layer.  A custom
initializer receives the value as ``request.user_init`` and controls its own
precedence.  The effective convergence tolerance is the larger of the
requested ``epsilon_crit`` and eight machine epsilons in the solver dtype.

The ordinary gas route provides a first-order implicit JVP and an
automatically transposed VJP.  Initial values, the formula matrix, and solver
configuration are held fixed.  The derivative assumes a converged root and a
nonsingular bordered equilibrium system.  Use the diagnostics route to certify
convergence before differentiation; for compatibility, an uncertified gas
iterate is not automatically replaced by non-finite derivatives.

Condensate Quick Start
----------------------

.. code-block:: python

   from exogibbs.api.condensate import (
       CondensateEquilibriumOptions,
       solve_profile,
   )
   from exogibbs.presets.fastchem4_cond import condensate_chemical_setup

   setup = condensate_chemical_setup(silent=True)
   result = solve_profile(
       setup,
       T=temperature_profile,
       P=pressure_profile,
       b=setup.gas_setup.element_vector_reference,
       options=CondensateEquilibriumOptions(profile_method="auto"),
   )

``result.layers`` contains one
``CondensateEquilibriumResult`` per layer.  Each layer contains the full gas
vector, the full condensate vector, the accepted active-support indices and
names, status, route, and optional diagnostics.  Common dense arrays are also
available in ``result.batched_arrays``.

.. _fugacity-conventions:

Pure-component non-ideal gas correction
---------------------------------------

For gas species :math:`i`, partial pressure, fugacity, and chemical potential
are related by

.. math::

   p_i = x_i P, \qquad
   f_i = \phi_i p_i, \qquad
   \mu_i = \mu_i^\circ(T) + RT \ln\!\left(\frac{f_i}{f^\circ}\right),
   \quad f^\circ = 1\ \mathrm{bar}.

The mechanical pressure closure is :math:`\sum_i p_i=P`; fugacities do not in
general sum to :math:`P`.  The fugacity coefficient :math:`\phi_i` is
dimensionless.  The API accepts its natural logarithm, :math:`\ln\phi_i`, so
an ideal gas has :math:`\phi_i=1` and :math:`\ln\phi_i=0`.  The current
non-ideal correction is limited to pure-component
:math:`\phi_i^{\mathrm{pure}}(T,P)` values rather than mixture-dependent
coefficients.

Both solver modules accept an optional ``lnphi_func`` keyword:

.. code-block:: python

   def lnphi_func(temperature, pressure_bar, mole_fractions):
       # Return one natural-log fugacity coefficient per gas species.
       assert mole_fractions is None  # pure-component mode
       return lnphi_pure

   result = solve(
       setup,
       T=temperature,
       P=pressure_bar,
       b=element_vector,
       lnphi_func=lnphi_func,
   )

A current ExoEOS checkout can supply these pure-component values through the
optional consumer-side adapter. For example, a setup containing the nine
MELTYQ gas species can assign one-component Zhang--Duan models to the six
supported species; He, N2, and NH3 omitted under the explicit ``"ideal"``
policy receive zero correction.

.. code-block:: python

   from exoeos import ZhangDuanEOS
   from exogibbs.interop.exoeos import make_pure_lnphi_func

   nonideal_species = ("H2", "O2", "H2O", "CO", "CO2", "CH4")
   lnphi_func = make_pure_lnphi_func(
       source_species=setup.species,
       eos_by_species={
           name: ZhangDuanEOS.from_species((name,))
           for name in nonideal_species
       },
       unspecified_species="ideal",
       phase="vapor",
   )

The returned vector must have shape ``(K,)`` and follow ``setup.species``
order.  ``pressure_bar`` is the physical pressure in bar, independent of
``Pref``.  ExoGibbs adds the correction once to the standard gas source,

.. math::

   h_k^{\mathrm{eff}}(T,P)
   = h_k^{\mathrm{ideal}}(T) + \ln \phi_k^{\mathrm{pure}}(T,P).

Omitting ``lnphi_func`` preserves ideal-gas behavior.  The third callable
argument reserves the composition-dependent interface; the current
pure-component implementation always passes ``None``.  Mixture fugacity
coefficients are not yet supported because their composition derivatives must
be included in the equilibrium Jacobian rather than evaluated once and held
fixed.  The gas solver's implicit JVP and its automatically transposed VJP
include the temperature and pressure dependence of a JAX-differentiable
callback.

Condensate Rainout
------------------

Rainout is an opt-in dependent profile calculation:

.. code-block:: python

   result = solve_profile(
       setup,
       # Package profile order is top -> bottom; the last entry is deepest.
       T=temperature_profile,
       P=pressure_profile,
       b=bottom_element_budget,
       options=CondensateEquilibriumOptions(
           rainout=True,
           profile_method="scan_hot_from_bottom",
       ),
   )

Input and output arrays remain in top-to-bottom order. Internally, rainout
solves the final (bottom) entry first and proceeds upward. No sorting is
performed, so callers with bottom-to-top source data must reverse those inputs
before the call.

After a layer is accepted, ExoGibbs subtracts the accepted condensate inventory
from the current target,
``b_remaining = b_current - setup.formula_matrix_cond @ condensate_amounts``.
It normalizes the initially positive element entries back to the input
abundance sum and uses that conservative inventory for the next upper layer.
``setup.formula_matrix @ layer.gas_n`` is retained as an independent numerical
cross-check and cannot drift a gas-only boundary. Condensates are reported at
their formation layer and removed from the propagated inventory. Initially
zero entries remain zero.

Positive element rows must also pass a floorless relative budget check. A
remainder within the measured reconstruction error and a floating-point
roundoff bound may be snapped to exact depletion; such rows use a reduced
propagation state so trace gas values in the raw full-network result cannot
resurrect the element in a later layer.

Rainout stops with ``RuntimeError`` if a layer cannot be accepted at its single
scheduled abundance scale after all available initialization attempts;
dependent upper layers are not evaluated.
The one-layer condensate solver rejects ``rainout=True`` because there is no
adjacent layer to receive the depleted inventory.

The production preset disables the legacy
``rainout_trace_capacity_accepted`` terminal tier. A ``NORMAL_MAX_ITER`` state
is therefore never used as an upper-layer boundary. A positive-condensate
state must pass a joint zero-barrier refinement of gas amounts, condensate
amounts, total gas, and element potentials, including active stationarity and
inactive-support closure. Gas-only budget repair is also not applied as a
post-solve transformation; a state outside the budget gate is rejected and a
cold initialization is tried once at the same scheduler abundance scale when
a resolved warm or custom initializer was available.

When a finite-barrier round converges to a finite state and passes the
independent KKT gate but its support remains open, the lifecycle makes one
catalog-wide zero-barrier closure attempt before adding phases to the finite-
barrier support. Passing both the unchanged internal audit and the caller-
gauge audit skips further finite-barrier expansion. A rejected exact candidate
is discarded, and expansion resumes from the unmodified finite-barrier state.

The exact refinement uses a deterministic, bounded add/drop active-set search.
Before its first joint root, a guarded zero-barrier dual oracle considers all
temperature-valid, structurally possible phases on the positive-gas boundary.
It maximizes the target-weighted element potential subject to gas
normalization and nonnegative phase driving. Tight dual constraints select an
initializer support; only the subsequent exact root and full physical audit
can accept it.

If the oracle is ineligible, infeasible, or its selected support does not
produce a local exact root, a bounded gas-eliminated homotopy may follow the
original finite-barrier central state toward zero. It retains the deepest
certified half-decade state when a later step loses its residual certificate,
and selects a support only across a clear capacity-relative amount gap. A
nonnegative linear program may then replace an eligible rank-deficient active
support by a basic support while preserving ``A_cond @ m``. If its single LP
vertex remains rank deficient, fails validation, or does not produce a local
exact root, a deterministic portfolio visits the untried full-rank bases
connected by one-phase exchanges, up to 32 bases in total. Feasible bases
preserve the same condensed inventory and use canonical catalog order.
Each candidate uses the existing exact root and physical audit under the
shared function-evaluation budget. The first local root advances the outer
active-set closure, while only the complete audit can terminate it. Failed
initial LP reduction also retains the builder state of the first canonical
feasible alternative as a possible support-release source. After an applied LP
basis fails, a successfully terminated alternative-basis solve with mixed-sign
active amounts may direct release toward its original feasible basis. Only the
terminal sign pattern is used: the rejected amounts are never reused, and a
suggested face that repeats the failed LP basis is ignored. Failed initializer
selection otherwise retains or retries the original support. These support
selectors and reductions are initializers only; none can bypass the joint
physical audit.

If all burden-preserving bases miss a local root, a support-release
initializer portfolio explores proper faces :math:`F\subset B` of either the
selected full-rank basis or, after an initial LP failure, the retained first
canonical feasible alternative. The burden-preserving search fixes

.. math::

   d_c = A_{c,B}m_B,

whereas a released initializer copies amounts on :math:`F` and zeros amounts
on :math:`B\setminus F`. Thus :math:`A_{c,F}m_F\ne d_c` in general. This does
not relax conservation: the reused exact solver must still satisfy

.. math::

   A_g n_g + A_{c,F}m_F = b.

The bounded breadth-first one-phase-removal graph includes the gas-only face.
Edges are ordered by conservative maximum phase amount and then catalog
index, but that scale is not a physical gate. Both the applied-LP and
initial-LP-failure routes reserve up to two complete call allowances from the
alternative-basis search: one for support release and one for ordinary outer
closure. When less work remains, all remaining work is preserved downstream
and the search still fails closed. Released faces try the existing mixed
positive-log/signed-linear formulation before the normalized-linear
formulation. Only a local KKT root may seed outer inactive-support closure,
whose full-catalog audit remains the acceptance authority.

It adds the most negative temperature-valid inactive phase only when every
other zero-barrier acceptance block passes. Locally valid states are cached.
When the addition creates exactly one rank dependency, a guarded simplex pivot
may exchange the unique limiting phase while preserving ``A_cond @ m``.
If an addition returns to any visited state, that edge is rejected; exhausted
child searches are unwound and the next unblacklisted candidate from the
nearest cached ancestor is tried within the search bounds.
Acceptance requires one visited support to pass the full audit over every
temperature-valid phase, and search exhaustion fails closed.
An optimizer that reaches only its function-evaluation limit (SciPy status
zero) may return an accepted state when that state independently passes every
unchanged physical audit block. The diagnostic acceptance source is
``physical_kkt_after_optimizer_limit``. Other failed terminations fail closed,
and only optimizer success can authorize deleting a phase from the support.

Before any condensate lifecycle solve, the positive non-charge element
inventory is normalized to unit total. The continuation values
``(-11, -13, -15, -17)`` are log barriers in this canonical amount gauge;
seeds, restoration floors, support closure, and zero-barrier refinement use
the same gauge. Accepted positive-condensate states are restored to the
caller's scale and pass a final caller-scale zero-barrier KKT audit. Public
result construction also applies the configured caller-scale element-budget
gate. The v2 budget denominator for a non-charge element row ``i`` is
``max(abs(b[i]), relative_floor * B)``, with ``B`` equal to the sum of positive
non-charge targets; non-finite scales fail closed.

The experimental prepared fixed-support adapter remains a low-level interface;
its caller owns both the input amount gauge and barrier schedule.

Trace-gas initial amounts are regularized only for the zero-barrier optimizer,
using their elemental capacities; the accepted equations remain floorless.
Only rows with non-negative coefficients across the joint gas and condensate
catalogs contribute to those capacities. A signed row such as charge balance
is not a capacity ceiling, although it remains an exact linear budget equation.
With a full-rank potential fit, the normalized-linear reduced solver uses this
regularized gas--potential state only when the smallest nonzero element
inventory is no greater than binary64 machine epsilon times the largest. This
formulation admits exact-zero targets and signed conservation rows such as
charge balance because it keeps the element-budget equations linear. The
initializer need not satisfy those budgets: the solve and unchanged full
physical audit remain authoritative.

The capacity-regularized attempt starts with initializer-relative variable
scaling. A dimensionless-unit-scaled restart is eligible only when that solve
returns a finite candidate with positive active condensate amounts and
optimizer status zero, indicating its function-evaluation limit, and the
candidate has not passed the local KKT gate. The restart is seeded from that
terminal candidate's gas logs, condensate amounts, total gas, element
potentials, and support; it does not reset the condensates or select an
alternative gas--condensate inventory partition. It shares the bounded
regularized allowance while preserving the unregularized retry reserve. Other
unsuccessful regularized outcomes fall through to the unregularized support-
selected initializer with initializer-relative scaling before any compatibility
fallback.
In the exact refinement, gas stationarity analytically eliminates the
per-species gas log amounts from the preferred nonlinear systems. Their size
therefore depends on the number of elements and active condensates rather than
the full gas catalog. Every reduced candidate is reconstructed in the full
catalog and must pass the unchanged physical audit and outer active-set
closure. Closure-round diagnostics expose the selected initializer and variable
scaling together with the guarded unit-restart eligibility and attempt. For
non-negative stoichiometry with exact-zero element rows,
structurally impossible gases and phases are omitted and the absent-element
potentials are reconstructed to close both gas and inactive-phase inequalities.
The dense all-gas formulation is retained only as a compatibility fallback
when reduced formulations do not pass a local KKT block. The normalized
log-domain fallback partitions conservation rows by formula geometry. Rows
that are non-negative across the joint gas and condensate catalogs and have
positive targets use logarithmic budget residuals; signed rows such as zero
charge balance remain scaled linear residuals. Exact-zero monotone rows remain
the responsibility of the structural-zero route. The fallback may explore a
bounded set of leave-one-out supports.
Its starting support and condensate amounts are the basic representation when
the reduction was applied, or the original closed support otherwise; its gas
variables come from the capacity-regularized initializer. If every normalized-
linear candidate is rejected, the same ordered alternative-basic-support
portfolio can apply this mixed log/linear formulation while sharing its work
budget; it starts only when one complete solver-call allowance remains. A
phase at the fallback amount bound cannot be accepted as active. Every branch
must pass the ordinary physical KKT and budget audit, so exhausting the search
is a hard failure rather than an approximate trace-phase acceptance. The
generic leave-one-out log-domain branch graph remains distinct from the
support-release face graph. The initial-LP-failure route may nevertheless
reuse the same mixed formulation on released faces, before the normalized-
linear formulation, within its protected release allowance.

A closed and finite barrier terminal state may also serve as an initializer
for the zero-barrier refinement when its gas, budget, complementarity, and
total-density residuals already pass. This handles the trace-phase case where
the finite barrier biases only condensate stationarity. This initializer-only
path uses a ``1e-5`` gas-stationarity bound while retaining the ordinary
``1e-8`` bounds for budget, complementarity, and total density. It does not
relax the final zero-barrier or caller-gauge KKT audit. The raw failure remains
in diagnostics, and acceptance still depends exclusively on the complete
zero-barrier physical gate.

That terminal-state route retains precedence. If it is unavailable after one
of the eligible finite-barrier failure classifications
``NORMAL_DUAL_STEP_FAILED``, ``RESTORATION_LINE_SEARCH_FAILED``,
``RESTORATION_MAX_ITER``, or ``RESTORATION_LOCALLY_INFEASIBLE``, the lifecycle
compares each support phase's elemental capacity with the first finite-barrier
amount, using only
conservation rows whose gas and condensate coefficients are all non-negative.
Signed rows such as charge balance do not define an amount ceiling. A support
with capacity at or below that barrier may instead initialize one bounded
exact closure from the preserved pre-PDIPM state. Linear-solve, representation,
and non-finite failures remain ineligible.

If the initial support was rank-deficient, PDIPM first uses one full-rank basis.
For an initial-round terminal or pre-PDIPM exact initializer, the lifecycle may
expand its candidate support back to the original envelope. It changes only
the support tuple; all numerical initializer fields remain unchanged. Phases
added to the support therefore have exactly zero amount, so the current
condensate burden is preserved identically. All envelope phases must remain
temperature-valid. The existing exact support machinery, including its bounded
basic-support portfolio, then selects a physical basis. The failed terminal
state remains the reported finite-barrier result. These initializer routes do
not relax the internal zero-barrier or caller-gauge KKT audit, and no exact
candidate is accepted without both audits.

``result.element_inventory_target``, ``result.gas_element_inventory``,
``result.rainout_element_inventory_out``, and
``result.rainout_abundance_scale`` provide a dense audit trail in the original
top-to-bottom order. The same arrays are available from
``result.batched_arrays``. That mapping also distinguishes the raw public
``raw_condensate_element_inventory`` from the exact-zero-compatible
``rainout_propagation_condensate_element_inventory`` used by subtraction. The
older ``condensate_element_inventory`` key is a compatibility alias for the
latter.

Rainout is distinct from solver warm starting. The
``"scan_hot_from_bottom"`` method is reserved for ``rainout=True``; the retired
fixed-``b`` condensate hot scan is not available. Fixed-budget local-equilibrium
profiles continue to use ``"auto"`` or ``"vmap_cold"``.
The preceding exact-zero-compatible gas state is offered only as a warm start;
raw species requiring a depleted element are not carried into it. The warm
initializer may also identify the accepted source problem through
``inventory_bridge_origin``. If the exact target warm solve fails, the rainout
scheduler can solve one log-interpolated inventory midpoint at the target
temperature and pressure, use only an accepted midpoint gas state to retry the
exact target, and then fall back cold. The midpoint and target retry share the
ordinary lifecycle and floorless budget gates. Neither intermediate
condensates nor intermediate rainout inventories are propagated, and the
bridge adds at most two lifecycle calls. No condensate support is carried
between layers.

Fixed-support implicit autodiff
-------------------------------

The accepted zero-barrier equations have a JAX-compatible custom JVP, with
VJPs generated by automatic transposition, once the positive condensate
support is fixed.  The numerical kernel is available as
``exogibbs.equilibrium.condensate.fixed_support.minimize_gibbs_fixed_support``.
It returns gas log amounts and physical amounts for the supplied active
condensate columns.  Temperature, normalized log pressure, and the elemental
target are differentiable; formula matrices, support, and initial values are
held fixed.  Before differentiating, run
``minimize_gibbs_fixed_support_with_diagnostics`` once and check
``diagnostics.converged``, ``diagnostics.residual_norm``, and
``diagnostics.iterations``.  This audit route is separate from the custom-JVP
result so that diagnostic booleans and counters do not become differentiation
targets; its returned physical state is for certification, not an AD target.
Differentiate the ordinary ``minimize_gibbs_fixed_support`` result.  The
requested residual tolerance is floored at ten machine epsilons, so the default
is also usable when JAX runs in float32 mode.
The condensate formula matrix, initial amounts, and thermochemical source
callable must all be restricted to the same support and use the same order;
slice a full-catalog ``condensate_setup.hvector_func(T)`` before passing it to
the kernel.

This is a local, piecewise-smooth contract.  Active amounts must remain
positive, inactive condensate driving forces must remain strictly positive,
and the reduced zero-barrier KKT matrix must remain nonsingular.  A support or
temperature-validity change is generally nondifferentiable.  The host-side
support discovery, support expansion, acceptance gates, and rainout inventory
propagation are therefore not included in this local implicit differentiation
contract.  A failed zero-barrier
solve reports ``diagnostics.converged=False`` on the audit route, while the
custom JVP returns non-finite output tangents and its generated VJP returns
non-finite source cotangents rather than differentiating an uncertified
iterate.  Primal convergence alone does not certify a derivative when the
reduced KKT matrix is singular.  This is a first-order forward- and
reverse-mode contract; forward-mode ``jvp`` and the automatically generated
VJP are supported.  Higher-order derivatives are not part of the supported
contract.

Execution and JAX Contracts
---------------------------

.. list-table::
   :header-rows: 1
   :widths: 22 38 40

   * - Contract
     - Gas
     - Gas plus condensates
   * - Profile methods
     - ``vmap_cold``, ``scan_hot_from_top``,
       ``scan_hot_from_bottom``
     - ``auto`` and ``vmap_cold`` for a fixed budget;
       ``scan_hot_from_bottom`` for rainout
   * - Support
     - Fixed gas species set
     - Host-side discovery around fixed-support solves, followed by bounded
       exact add/drop closure
   * - JIT and batching
     - One-layer and profile numerical routes are JAX-compatible
     - The complete support lifecycle and dependent rainout scan are Python
       host-side
   * - Differentiation
     - An implicit custom JVP supports forward mode and generates the
       reverse-mode VJP by transposition
     - The zero-barrier fixed-support kernel has an implicit custom JVP with a
       generated reverse-mode VJP; the complete lifecycle is not a
       differentiable or JIT-compatible public contract
   * - Diagnostics
     - Optional numerical diagnostics use a distinct solver route
     - Optional lifecycle, KKT, support-closure, and timing diagnostics

Compatibility
-------------

The historical :mod:`exogibbs.api.equilibrium` and
:mod:`exogibbs.api.condensate_equilibrium` modules remain supported
compatibility facades.  New code should use :mod:`exogibbs.api.gas` and
:mod:`exogibbs.api.condensate`.

API migration note
------------------

The umbrella imports ``from exogibbs.api import equilibrium`` and
``from exogibbs.api import condensate_equilibrium`` now deterministically
return their child modules, independent of import order.  They no longer
sometimes resolve to solver functions.  Import ``solve`` from
:mod:`exogibbs.api.gas` or :mod:`exogibbs.api.condensate` when a callable is
required.  Non-colliding historical exports remain available from
``exogibbs.api``.
