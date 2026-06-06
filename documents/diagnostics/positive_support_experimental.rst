Positive-Support Experimental API
=================================

Overview
--------
The positive-support experimental API exposes a diagnostic-only, explicit
opt-in boundary for testing condensate support initialization against the
restricted support condensate solver callsite.

The API lives under::

   exogibbs.diagnostics.condensate_positive_support_experimental

It is intentionally not imported by ``exogibbs``, ``exogibbs.presets``, or any
default equilibrium path. Users must import it explicitly.

Status
------
This surface is release-ready as an experimental diagnostics API. It is not a
production initialization path.

Allowed behavior:

* explicit opt-in through ``enable_experimental_positive_support=True``;
* construction from ExoGibbs-native arrays and thermochemistry functions;
* restricted solver input adapter use;
* injection of ``support_indices`` and ``support_amounts_init`` into the
  restricted solver callsite;
* top1 positive support as an experimental candidate;
* ``seed_fraction <= 1.0e-3`` and ``max_seed_amount <= 1.0e-3``;
* KKT residual reporting as solver-stage diagnostics.

Forbidden behavior:

* default-on production initialization;
* production solver behavior changes;
* production return signature changes;
* presets or defaults wiring;
* FastChem4 public, runtime, trace, branch replay, or reference-fit values as
  constructor inputs;
* treating top1 positive support as a production standard;
* using an initial KKT residual as an acceptance gate;
* broad p-T grid claims from this diagnostic surface.

Quick Start
-----------
The example below uses small native arrays. Real applications should supply
arrays from an ExoGibbs-native setup, not from FastChem4 outputs or traces.

.. code-block:: python

   import jax.numpy as jnp

   from exogibbs.api.chemistry import ThermoState
   from exogibbs.diagnostics.condensate_positive_support_experimental import (
       PositiveSupportExperimentalConfig,
       run_positive_support_experimental_callsite,
   )

   state = ThermoState(
       temperature=jnp.asarray(1000.0),
       ln_normalized_pressure=jnp.asarray(0.0),
       element_vector=jnp.asarray([2.0, 1.0, 5.0]),
   )

   formula_matrix = (
       (1.0, 0.0, 0.0),
       (0.0, 1.0, 0.0),
       (0.0, 0.0, 1.0),
   )
   formula_matrix_cond = (
       (1.0, 1.0),
       (1.0, 0.0),
       (3.0, 1.0),
   )

   def hvector(_temperature):
       return jnp.zeros((3,))

   def hvector_cond(_temperature):
       return jnp.asarray([-0.5, -0.1])

   config = PositiveSupportExperimentalConfig(
       enable_experimental_positive_support=True,
       seed_fraction=1.0e-3,
       max_seed_amount=1.0e-3,
   )

   report = run_positive_support_experimental_callsite(
       config=config,
       state=state,
       formula_matrix=formula_matrix,
       formula_matrix_cond=formula_matrix_cond,
       hvector_func=hvector,
       hvector_cond_func=hvector_cond,
       condensate_species_order=("MgSiO3_s", "MgO_s"),
       element_order=("Mg", "Si", "O"),
       field_provenance={
           "formula_matrix_cond": "exogibbs_native_curated_case",
           "element_inventory_target": "exogibbs_native_curated_case",
           "hvector_cond": "exogibbs_native_thermochemistry",
       },
   )

   print(report.solver_called)
   print(report.post_solver_budget_residual)
   print(report.post_solver_kkt_residual_diagnostic)

Contract
--------
``PositiveSupportExperimentalConfig`` is the explicit opt-in contract. The
default constructor is safe because ``enable_experimental_positive_support`` is
``False``. A call without the opt-in flag raises ``ValueError`` before the
restricted solver callsite can run.

The safe seed envelope is enforced at the public boundary:

* ``seed_fraction`` must be nonnegative and no larger than ``1.0e-3``;
* ``max_seed_amount`` must be nonnegative and no larger than ``1.0e-3``.

The provenance firewall rejects FastChem4-derived constructor inputs. The
following provenance labels are not valid for fields used to build solver
inputs:

* ``fastchem4_trace``
* ``fastchem4_public``
* ``fastchem4_runtime``
* ``branch_replay``
* ``reference_fit``
* ``unknown_reference``

Return Report
-------------
The return value is a ``PositiveSupportCallsiteExperimentResult``. The most
important fields are:

``explicit_opt_in``
   Confirms the call reached the experimental path.

``support_indices`` and ``support_amounts_init``
   Available through ``report.initializer.solver_inputs``. These are the
   restricted solver callsite inputs.

``budget_closure_before_solver``
   Fractional element budget consumed by the seed before the solver call.

``solver_called`` and ``solver_success``
   Solver-stage status. Empty positive support boundaries may skip the solver.

``post_solver_budget_residual``
   Solver-stage budget residual after the restricted support solve.

``post_solver_kkt_residual_diagnostic``
   Solver-stage diagnostic only. It is not an acceptance gate.

Autodoc
-------
The API reference is included in
:doc:`../exogibbs/exogibbs.diagnostics`.
