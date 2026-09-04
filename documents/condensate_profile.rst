Condensate Equilibrium Profiles
===============================

The condensate API uses the fixed-support v2 solver and an external support
lifecycle. The only runtime route is ``"head_v2"`` and the immutable production
preset is ``"validated_2026_07"``.

Production Contract
-------------------

The one-layer and profile APIs share the same lifecycle:

.. code-block:: text

   select support
       -> solve fixed support with v2
       -> check independent KKT conditions
       -> try catalog-wide exact closure from a valid open state
       -> check inactive-condensate closure
       -> expand support when required
       -> refine an active support at zero barrier

Support changes never occur inside a fixed-support solve. When a finite-
barrier round converges to a finite state and passes the independent KKT gate
but its support remains open, the lifecycle makes one catalog-wide
zero-barrier closure attempt for that layer before adding phases to the
finite-barrier support. A candidate accepted by both the unchanged internal
physical audit and the caller-gauge audit skips further finite-barrier
expansion. If either audit rejects it, the candidate is discarded and support
expansion resumes from the unmodified finite-barrier state.

For this early closure and the final exact refinement, a guarded zero-barrier
dual oracle considers every temperature-valid, structurally possible
condensate. On the positive-gas boundary it maximizes the
target-weighted element potential subject to gas normalization and
nonnegative phase driving. Tight dual constraints define an initializer
support; they never constitute acceptance. The preferred exact root then
eliminates the per-species gas variables and reconstructs the full gas catalog
for the unchanged physical audit.

If the dual oracle is ineligible, fails its feasibility checks, or selects a
support that does not produce a local exact root, a bounded gas-eliminated
homotopy may follow the original finite-barrier central state toward zero. It
retains the deepest certified half-decade continuation state when a later
step loses its residual certificate, and selects a support only across a clear
capacity-relative amount gap. A further fallback may replace an eligible
rank-deficient active support by a
nonnegative basic support. Its bounded linear program preserves
``A_cond @ m`` and minimizes the condensate Gibbs term. If the LP vertex does
not produce a valid full-rank basis or does not reach a local exact root, a
deterministic one-phase-exchange portfolio visits the untried feasible bases,
up to 32 bases in canonical catalog order. Every candidate reuses the exact
solver and unchanged physical audit under the shared function-evaluation
budget; the first local root continues outer active-set closure, but only the
full audit can terminate it. If the initial LP fails, the builder state of the
first canonical feasible alternative is retained as a possible support-release
source. After an applied LP basis fails, the sign pattern of the first
successfully terminated alternative-basis solve with mixed-sign active amounts
may instead direct support release toward that basis. The nonphysical terminal
amounts are never reused: release starts from the original nonnegative,
builder-produced basis amounts, and a suggested face that repeats the failed LP
basis is ignored. Failed initializer selection otherwise retains or retries the
original support. All support selectors and reductions are initializers only,
and the subsequent physical audit remains authoritative.

If every burden-preserving basis misses a local root, a support-release
initializer portfolio may visit proper faces :math:`F\subset B` of either the
selected full-rank basis or, after an initial LP failure, the retained first
canonical feasible alternative. The basic-support search fixes the initial
condensed element inventory

.. math::

   d_c = A_{c,B}m_B.

For a released face, retained amounts are copied and removed amounts are set
to zero, so in general :math:`A_{c,F}m_F\ne d_c`. This relaxes only the
initializer's condensed inventory partition. The exact solve still enforces

.. math::

   A_g n_g + A_{c,F}m_F = b.

The portfolio traverses a bounded breadth-first one-phase-removal graph,
including the gas-only face. Removal edges are ordered by conservative phase
capacity and then catalog index; capacity is never an eligibility or
acceptance threshold. Both the applied-LP and initial-LP-failure routes reserve
up to two complete call allowances from a preceding alternative-basis
portfolio: one for support release and one for the ordinary outer closure.
When less work remains, all remaining work is preserved downstream and the
search still fails closed. Released faces try the existing mixed
positive-log/signed-linear formulation before the normalized-linear
formulation. A local KKT root only initializes the unchanged outer inactive-
support closure, and the full-catalog physical audit remains authoritative.

The host-side refinement uses a deterministic, bounded exact add/drop
active-set search. After each converged joint root, phases with non-positive
exact amounts may be removed.
If all checks except inactive-support closure pass, the most negative
temperature-valid inactive phase is added, with species index breaking ties,
and the exact solve is repeated. A rank-one dependent addition may first pivot
at fixed ``A_cond @ m`` to exchange the unique limiting phase. Locally valid
states are cached. If an
addition returns to any visited state, that edge is rejected; exhausted child
searches are unwound and the next unblacklisted candidate from the nearest
cached ancestor is tried within the search bounds.

A state is accepted only when one support passes finiteness, active-amount
positivity, gas and active-phase stationarity, element budget, total density,
and inactive closure over every temperature-valid phase. The optimizer must
either report success or stop only because its function-evaluation limit was
reached. In the latter case, acceptance requires the same independent full
physical certificate and is labeled
``physical_kkt_after_optimizer_limit``. An optimizer exception, any other
failed termination, or a failed physical block fails closed. A
function-limit candidate never authorizes a phase deletion; that transition
still requires optimizer success. Exhausting the bounded search fails closed. A
capacity-aware initializer keeps trace gas amounts away from exponential
underflow without changing the final equations or acceptance tolerances. The
preferred exact formulations eliminate the per-species gas log amounts using
gas stationarity, so their nonlinear dimension depends on the element and
active-phase counts rather than the full gas catalog. They reconstruct every
gas amount before applying the unchanged full physical audit. For a
non-negative-stoichiometry inventory with exact-zero element rows,
structurally impossible gases and phases are removed from the reduced solve
and the absent-element potentials are reconstructed to satisfy both the gas
floor and inactive-phase inequalities. The dense all-gas formulation remains
a compatibility fallback when the reduced formulations do not pass a local
KKT block. A normalized log-domain fallback may also explore bounded leave-
one-out support branches. It uses log residuals for positive monotone element
budgets and scaled linear residuals for signed rows such as zero charge
balance. Exact-zero monotone rows remain on the structural-zero route. Each
branch
starts with the basic support and amounts when reduction was applied, or with
the original closed support and amounts otherwise; its gas variables come
from the capacity-regularized initializer. Phases pinned to the numerical
amount floor are not accepted as active, and every candidate passes through
the same physical audit. After normalized-linear alternative bases are
rejected, their existing ordered portfolio can reuse this mixed formulation
when one full solver-call budget remains. The generic leave-one-out log-domain
branch graph remains separate from the support-release face graph. The initial-
LP-failure route may nevertheless reuse the same mixed formulation on released
faces, before the normalized-linear formulation, within its protected release
allowance. An ineligible or exhausted fallback fails closed.

Each production lifecycle fixes one extensive amount scale from the sum of its
positive non-charge element targets. It expresses the element target, initial
gas and condensate amounts, generated seeds, and fixed amount floors in this
normalized gauge. The fixed ``epsilon`` continuation schedule therefore
denotes the normalized barrier ``log(nu / B)``. Element potentials and phase
driving are unchanged. Accepted positive-condensate states are restored to the
caller's amount gauge and pass a second zero-barrier KKT audit there. Public
result construction also applies the configured caller-gauge element-budget
gate. Its v2 denominator for a non-charge element row ``i`` is
``max(abs(b[i]), relative_floor * B)``, where ``B`` is the sum of positive
non-charge targets. A non-finite amount scale or floor fails closed.

When the support-selected initializer contains capacity-underflow trace gases,
the zero-barrier reduced solve first tries the existing capacity-regularized
gas and element-potential initializer with a full-rank potential fit only when
its smallest nonzero element inventory is no greater than binary64 machine
epsilon times its largest. Its gas capacities use only formula rows whose
coefficients are non-negative across the joint gas and condensate catalogs.
A signed row such as charge balance is therefore not a capacity ceiling,
although it remains in the normalized-linear budget equations.

The first capacity-regularized solve uses initializer-relative variable
scaling. Only when it ends at a finite, optimizer-unsuccessful status-0
function-evaluation-limit state with positive active condensate amounts and
without a local KKT certificate may one dimensionless-unit-scaled restart
follow. That restart is seeded from the terminal gas logs, condensate amounts,
total gas, element potentials, and support; it does not introduce a condensate
reset or an alternative inventory partition. It remains
inside the bounded regularized allowance and does not consume the protected
unregularized reserve. An ineligible or unsuccessful regularized route can
still fall through to the unregularized, initializer-relative attempt. Every
candidate faces the unchanged full physical audit and outer active-set closure;
the separate log-domain fallback keeps positive monotone budgets logarithmic
while retaining signed conservation rows in scaled linear form. Closure-round
diagnostics record the selected initializer and variable scaling together with
the guarded unit-restart eligibility and attempt.

A closed, finite terminal barrier state whose gas, budget, complementarity,
and total-density residuals pass may also initialize this exact refinement
even when finite-barrier condensate stationarity prevents the barrier solver
from declaring convergence. Its initializer-only gas-stationarity bound is
``1e-5``; the other initializer blocks retain their ordinary ``1e-8`` bounds.
This gate only permits a bounded exact solve. The final zero-barrier and
caller-gauge physical audits continue to use the ordinary ``1e-8`` KKT
tolerances. Neither an open converged state nor a closed failed state is
accepted directly: the lifecycle preserves its finite-barrier status, labels
the exact path, and accepts only an audited zero-barrier result. Other failed
v2 states are reported to the caller; none is retried with a retired solver.
Operational rollback uses a previous release artifact.

A narrow initializer-only fallback covers the eligible finite-barrier failure
classifications ``NORMAL_DUAL_STEP_FAILED``,
``RESTORATION_LINE_SEARCH_FAILED``, ``RESTORATION_MAX_ITER``, and
``RESTORATION_LOCALLY_INFEASIBLE`` when the established terminal-state route is
unavailable. If a phase's capacity from monotone non-negative conservation
rows is at or below the first barrier amount, the preserved pre-PDIPM state may
initialize the same bounded exact closure. The monotone mask is computed over
the joint gas and condensate formula matrices; signed rows such as charge
balance do not impose an amount ceiling. Linear-solve, representation, and
non-finite failures remain ineligible.

PDIPM requires a full-rank initial support, so the lifecycle may first reduce a
rank-deficient support to one basis. On the initial lifecycle round only, exact
polishing can expand the selected terminal or pre-PDIPM initializer back to
that candidate support envelope. The gas log amounts, condensate amounts,
total-gas log amount, and element potentials remain unchanged; reintroduced
phases therefore have exactly zero amount and the condensate burden is
unchanged. Every envelope phase must still be valid. Existing zero-barrier
support selection, including its bounded basic-support portfolio, then chooses
the physical basis. The
``zero_barrier_initializer["initial_support_envelope"]`` diagnostic records
the source, envelope, and added supports together with the state-preservation
guards. The failed finite-barrier state remains in diagnostics, and both the
ordinary internal and caller-gauge zero-barrier audits are still required.

The public defaults are:

.. code-block:: python

   from exogibbs.api.condensate import (
       CondensateEquilibriumOptions,
   )

   options = CondensateEquilibriumOptions()
   assert options.route == "head_v2"
   assert options.fixed_support_v2_preset == "validated_2026_07"

Profile Method
--------------

For local equilibrium with one fixed element budget, ``solve_profile`` accepts
``method="auto"`` and ``method="vmap_cold"``. Both select the production
batched v2 lifecycle.

Rainout is enabled separately with
``CondensateEquilibriumOptions(rainout=True)``. In this mode, ``None`` and
``"auto"`` resolve to ``"scan_hot_from_bottom"``; callers may also request
``"scan_hot_from_bottom"`` explicitly. ``"vmap_cold"`` is incompatible with
rainout because the element budget of an upper layer depends on the accepted
gas state below it.

``"scan_hot_from_bottom"`` is reserved for rainout. The old sequential hot
scan with one fixed ``b`` at every layer remains retired and is not restored by
this option.

Profile Ordering
----------------

Condensate profile inputs and outputs follow the package-wide top-to-bottom
ordering: the first array entry is the top of the atmosphere and the final
entry is the bottom boundary. No pressure sorting is performed. A rainout run
visits the final entry first, scans internally from bottom to top, and restores
all returned layers and dense arrays to the caller's original top-to-bottom
order.

For example, a pressure array increasing with depth is already in the expected
order:

.. code-block:: python

   pressure = jnp.asarray([1.0e-3, 1.0e-2, 1.0e-1])  # top -> bottom

If source data are stored from the ground upward, reverse the temperature and
pressure arrays before calling ``solve_profile`` and reverse the returned
arrays when aligning them back to that source order.

Rainout Propagation
-------------------

At each layer, the production fixed-support lifecycle first computes and
accepts a complete gas-plus-condensate equilibrium state. Condensates are then
removed before the next upper layer. If ``A_cond`` is the condensate
stoichiometric matrix, ``m_cond`` is the accepted condensate amount vector,
and ``b_i`` is the current element target, the retained elemental inventory is

.. math::

   g_i = b_i - A_{\mathrm{cond}}m_{\mathrm{cond},i}.

The next upper layer receives the normalized gas-only inventory

.. math::

   b_{i+1} = B_0\frac{g_i}{\sum_{j\in\mathcal A}g_{i,j}},

where :math:`\mathcal A` contains the entries that were positive in the input
``b`` and :math:`B_0=\sum_{j\in\mathcal A}b_{0,j}`. Initially zero entries,
including the usual electron budget, remain zero and do not enter the
normalization. Thus rainout changes elemental ratios but preserves the input
abundance gauge. The condensed inventory is reported for the layer but is not
carried upward. ``A_gas n_gas`` is recorded separately as a conservation
cross-check; its finite solver residual never changes a gas-only layer's
propagated inventory.

Every positive target row is certified using a floorless relative budget
residual before it may be propagated. An exactly zero target is handled in the
reduced propagation state: species that require that element remain visible in
the raw solver result for audit, but cannot reintroduce the element into an
upper-layer boundary. Both condensate subtraction and its reconstruction-error
bound use this same reduced state. When subtraction leaves a non-positive
remainder within the reduced reconstruction error plus a floating-point
roundoff bound, that element is snapped to exact numerical depletion. The layer
diagnostics record the snap mask, amount, error bound, and error source.

The rainout scheduler passes each layer to the lifecycle in the caller's
amount gauge. The lifecycle is the sole owner of conversion to the unit-total
internal amount gauge, so ordinary layers do not undergo a scale-up and
normalization round trip. Only an inventory whose total exceeds the configured
finite transport cap is uniformly downscaled before the lifecycle call.
Element ratios and gas mole fractions are unchanged, and extensive gas and
condensate amounts are returned in the caller's original abundance gauge.
Layer diagnostics record the transport scale actually applied separately from
the canonical solver gauge.

Within a rainout layer, the nested lifecycle ``amount_gauge`` and
``caller_gauge_zero_barrier_kkt`` refer to the lifecycle caller. This is the
original profile caller gauge unless the overflow cap required a downscale.
The enclosing rainout ``budget_audit_gauge`` names that same lifecycle-caller
audit. Its ``floorless_budget_certification`` and the returned arrays refer to
the original profile caller gauge after any transport rescaling.

The accepted finite gas log amounts are also used as a gas-only warm start for
the adjacent upper layer. The public
``regauge_gas_only_warm_start(setup, gas_ln_n, element_inventory)`` helper
applies one uniform log-space shift to every compatible finite species, so its
finite log ratios are preserved in the new element-inventory amount gauge.
Species that are absent or require an exactly depleted element receive a
finite numerical floor. The helper uses the stoichiometric matrix directly and
therefore requires neither atomic gas species nor an electron row. It returns
no condensate amounts or active support. The rainout initializer separately
records the accepted source problem as
``CondensateEquilibriumPoint(temperature, pressure, element_inventory)``.
This provenance is not a physical state carried into the next layer.

If the direct warm solve fails and source provenance is available, the
scheduler may try one bounded inventory bridge at the exact target temperature
and pressure. For bridge fraction :math:`f=1/2`, rows that are positive at both
endpoints use

.. math::

   b_j(f)=\exp\!\left[(1-f)\log b_{j,0}+f\log b_{j,1}\right],

while a row with a zero endpoint is interpolated linearly. A bridge result may
seed the exact target only after the ordinary lifecycle acceptance and the
floorless budget certification pass for the bridge inventory itself. Only its
gas state is used; its condensates, support, and proposed rainout inventory are
discarded. The exact target is then solved and audited normally. This route is
limited to two additional lifecycle calls and is skipped after a successful
direct solve. If either stage fails, the exact target is retried once from the
cold initializer at the same abundance scale.

The bridge is a numerical preconditioner, not an interpolated atmospheric
layer. Rainout subtraction and inventory propagation occur exactly once, after
the exact target has passed every production gate. Diagnostics record the
source and target inventories, both trial outcomes, and the termination reason
under ``attempts[*].inventory_bridge``. Its ``inventory_gauge`` is the rainout
lifecycle-caller gauge; the enclosing attempt records any overflow transport
scale needed to recover the original profile-caller gauge.

Only an accepted layer may supply the next inventory. If all available
initialization attempts fail at a layer, ``solve_profile`` raises
``RuntimeError`` and does not evaluate any dependent upper layers. This
fail-closed behavior prevents an unaccepted numerical state from becoming a
physical rainout boundary.

The legacy ``"rainout_trace_capacity_accepted"`` escape hatch remains an
internal policy field for diagnostic compatibility but is disabled in the
production preset. In particular, a ``NORMAL_MAX_ITER`` state cannot become an
irreversible rainout boundary merely because its condensate capacity is small.
Every production layer must instead pass the ordinary solver gates, the
zero-barrier physical refinement when a condensate is positive, and the
floorless propagation-budget certification.

Initial Support
---------------

Callers may provide one fixed support for every layer:

.. code-block:: python

   result = solve_profile(
       setup,
       temperatures,
       pressures,
       element_budget,
       support_indices=(0, 3),
       support_amounts_init=(1.0e-8, 1.0e-8),
       method="vmap_cold",
   )

Alternatively, provide one ``CondensateEquilibriumInit`` per layer. An init may
contain gas amounts, support indices and amounts, and an element-potential warm
start. The production policy may expand the supplied support after checking
inactive-condensate driving.

A custom initializer receives the caller-gauge element target and must return
all gauge-dependent fields in that same gauge. The lifecycle normalizes those
fields before support expansion and solution. Built-in initializers preserve
this contract under a uniform rescaling of the target. Log gas amounts and
``barrier_epsilon`` shift with the logarithm of that scale; linear amounts
scale directly, while ``element_potential`` and ``rho`` remain unchanged.

Minimal Example
---------------

.. code-block:: python

   import jax.numpy as jnp

   from exogibbs.api.condensate import (
       CondensateEquilibriumOptions,
       solve_profile,
   )
   from exogibbs.presets.fastchem4_cond import (
       condensate_chemical_setup,
   )

   setup = condensate_chemical_setup(silent=True)

   profile = solve_profile(
       setup,
       T=jnp.asarray([1800.0, 1400.0, 900.0]),
       P=jnp.asarray([1.0e-3, 1.0e-2, 1.0e-1]),
       b=jnp.asarray(setup.gas_setup.element_vector_reference),
       options=CondensateEquilibriumOptions(return_diagnostics=True),
       method="vmap_cold",
       return_diagnostics=True,
   )

   print(profile.method)
   print([layer.status for layer in profile.layers])

Rainout Example
---------------

.. code-block:: python

   profile = solve_profile(
       setup,
       # Input remains top -> bottom; the final layer is solved first.
       T=jnp.asarray([900.0, 1200.0, 1600.0]),
       P=jnp.asarray([1.0e-3, 1.0e-2, 1.0e-1]),
       b=jnp.asarray(setup.gas_setup.element_vector_reference),
       options=CondensateEquilibriumOptions(
           rainout=True,
           profile_method="scan_hot_from_bottom",
           return_diagnostics=True,
       ),
   )

   assert profile.rainout
   assert profile.method == "scan_hot_from_bottom"

Rainout is a dependent profile operation and is therefore rejected by the
one-layer ``solve`` function.

Curated Examples
----------------

The repository includes ten production-route examples covering gas-only,
phase-boundary, silicate, sulfide, graphite, water, and budget-stress cases in
`examples/condensates_curated_demo <https://github.com/HajimeKawahara/exogibbs/tree/main/examples/condensates_curated_demo>`_.
These examples run from a source checkout because they reuse the benchmark case
definitions.

Diagnostics
-----------

When diagnostics are enabled, the profile report includes:

``result.method``
   ``"vmap_cold"`` for fixed-budget local equilibrium, or
   ``"scan_hot_from_bottom"`` for rainout.

``result.rainout``
   Whether condensate-depleted element inventories were propagated between
   layers.

``result.layers[i].selected_route``
   ``"head_v2_fixed_support_lifecycle"`` for an accepted active-support
   result, or ``"head_v2_gas_only_no_candidate"`` when no condensate candidate
   is selected.

``result.layers[i].diagnostics["fixed_support_v2"]``
   Fixed-support terminal status, independent KKT result, support-closure
   result, lifecycle rounds, canonical ``amount_gauge``, zero-barrier dual,
   homotopy, ``alternative_basic_support_portfolio`` and
   ``support_release_portfolio`` initializer reports, exact active-set closure
   traces (including simplex pivots, rejected addition edges, and search
   limits), and the final lifecycle-caller-gauge zero-barrier KKT audit when
   applicable. The saved initial support-release report records its source,
   formulation order, protected work, and initializer-only outcome.

``result.diagnostics``
   Profile-level route, preset, backend, and separated compilation, execution,
   and diagnostic timings for fixed-budget profiles. Rainout diagnostics also
   record ``input_order="top_to_bottom"``,
   ``scan_direction="bottom_to_top"``, the original processing indices, and
   each layer's single scheduler abundance scale and available initialization
   attempts.

Rainout results also expose dense arrays in both named fields and
``result.batched_arrays``:

``result.element_inventory_target``
   The element inventory used to solve each layer, shape ``(N, E)``.

``result.gas_element_inventory``
   The raw ``A_gas @ gas_n`` conservation cross-check for each accepted layer,
   shape ``(N, E)``. It is not the authoritative propagation source.

``result.rainout_element_inventory_out``
   The normalized conservative inventory derived from
   ``b_current - rainout_propagation_condensate_element_inventory`` and passed
   toward the next upper layer, shape ``(N, E)``.

``result.batched_arrays["raw_condensate_element_inventory"]``
   ``A_cond @ condensate_amounts`` from the raw public full-network state,
   shape ``(N, E)``.

``result.batched_arrays["rainout_propagation_condensate_element_inventory"]``
   The exact-zero-compatible condensate inventory used by conservative
   subtraction, shape ``(N, E)``. The original
   ``"condensate_element_inventory"`` key remains as an explicit compatibility
   alias for this array.

``result.rainout_abundance_scale``
   The scheduler transport scale actually applied at each layer, shape
   ``(N,)``. It is normally exactly one and is less than one only when the
   caller inventory exceeds the finite transport cap. It is not the canonical
   internal solver gauge.

All four arrays use the original top-to-bottom profile order. Consequently,
for adjacent entries ``i - 1`` (upper) and ``i`` (lower), the accepted rainout
profile satisfies
``element_inventory_target[i - 1] == rainout_element_inventory_out[i]`` up to
floating-point roundoff.

Experimental Prepared v2 Plans
------------------------------

The submodule API retains an opt-in prepared-plan adapter for research with a
caller-supplied ``FixedSupportV2Config``. Its buckets and layer-state carriers
are owned by ``equilibrium.condensate.fixed_support.batch`` and do not import a
historical v1 solver. This adapter is not a production preset and does not
change the public default.
