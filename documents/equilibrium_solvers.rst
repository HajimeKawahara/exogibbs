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
     - ``auto`` and ``vmap_cold``
   * - Support
     - Fixed gas species set
     - Monotone active-condensate discovery and expansion around fixed-support
       solves
   * - JIT and batching
     - One-layer and profile numerical routes are JAX-compatible
     - The complete support lifecycle is Python host-side
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
