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
       -> check inactive-condensate closure
       -> expand support when required

Support changes never occur inside a fixed-support solve. A failed v2 solve is
reported to the caller; it is not retried with a retired solver. Operational
rollback uses a previous release artifact.

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

``solve_profile`` accepts ``method="auto"`` and
``method="vmap_cold"``. Both select the production batched v2 lifecycle.
Sequential condensate hot-scan methods belonged to the retired v1 route and
are no longer accepted.

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

Diagnostics
-----------

When diagnostics are enabled, the profile report includes:

``result.method``
   ``"vmap_cold"`` for the production route.

``result.layers[i].selected_route``
   ``"head_v2_fixed_support_lifecycle"`` for an accepted active-support
   result, or ``"head_v2_gas_only_no_candidate"`` when no condensate candidate
   is selected.

``result.layers[i].diagnostics["fixed_support_v2"]``
   Fixed-support terminal status, independent KKT result, support-closure
   result, and lifecycle rounds.

``result.diagnostics``
   Profile-level route, preset, backend, and separated compilation, execution,
   and diagnostic timings.

Experimental Prepared v2 Plans
------------------------------

The submodule API retains an opt-in prepared-plan adapter for research with a
caller-supplied ``FixedSupportV2Config``. Its buckets and layer-state carriers
are owned by ``equilibrium.condensate.fixed_support.batch`` and do not import a
historical v1 solver. This adapter is not a production preset and does not
change the public default.
