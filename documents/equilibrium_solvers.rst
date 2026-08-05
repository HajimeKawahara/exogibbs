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

Rainout stops with ``RuntimeError`` if a layer cannot be accepted after the
production numerical-gauge attempts; dependent upper layers are not evaluated.
The one-layer condensate solver rejects ``rainout=True`` because there is no
adjacent layer to receive the depleted inventory.

The production preset disables the legacy
``rainout_trace_capacity_accepted`` terminal tier. A ``NORMAL_MAX_ITER`` state
is therefore never used as an upper-layer boundary. A positive-condensate
state must pass a joint zero-barrier refinement of gas amounts, condensate
amounts, total gas, and element potentials, including active stationarity and
inactive-support closure. Gas-only budget repair is also not applied as a
post-solve transformation; a state outside the budget gate is rejected and a
different production gauge or cold initialization is tried.

Trace-gas initial amounts are regularized only for the zero-barrier optimizer,
using their elemental capacities; the accepted equations remain floorless. If
the primary capacity-scaled linear-amount refinement fails for an eligible
positive budget, a normalized log-domain fallback explores a bounded set of
leave-one-out supports from fresh copies of the original initializer. A phase
at the fallback amount bound cannot be accepted as active. Every branch must
pass the ordinary physical KKT and budget audit, so exhausting the search is a
hard failure rather than an approximate trace-phase acceptance.

A closed and finite barrier terminal state may serve as an initializer for the
zero-barrier refinement when its gas, budget, complementarity, and
total-density residuals already pass. This handles the trace-phase case where
the finite barrier biases only condensate stationarity. The raw failure remains
in diagnostics, and acceptance still depends exclusively on the complete
zero-barrier physical gate.

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
raw species requiring a depleted element are not carried into it. A failed
warm solve is retried cold at the same abundance scale, and no condensate
support is carried between layers.

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
     - Monotone active-condensate discovery and expansion around fixed-support
       solves
   * - JIT and batching
     - One-layer and profile numerical routes are JAX-compatible
     - The complete support lifecycle and dependent rainout scan are Python
       host-side
   * - Differentiation
     - Custom VJP supports reverse-mode derivatives; forward-mode ``jvp`` is
       intentionally unsupported
     - The complete lifecycle is not a differentiable or JIT-compatible
       public contract
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
