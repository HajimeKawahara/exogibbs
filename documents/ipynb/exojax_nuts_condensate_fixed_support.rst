.. This file is generated from the sibling .ipynb by convert_vjp_retrieval_notebooks.py.
.. Do not edit this RST file directly.

ExoJAX NUTS with a graphite-only fixed-support VJP
==================================================

This tutorial demonstrates the local condensate generated VJP used by
``examples/retrievals/exojax_nuts_condensate_fixed_support.py``. The
atmosphere is carbon rich with C/O = 2. The declared chemistry model
uses the full FastChem4 gas catalog but an explicit reduced condensate
catalog containing only graphite, ``C(s)``. A nominal calculation
discovers graphite within that one-phase catalog and freezes the
graphite-active layer mask and all numerical initial values before
inference. Active layers use the condensate fixed-support VJP; the
remaining layers use the gas VJP.

This is deliberately not a full FastChem4 condensate-equilibrium,
differentiable phase-discovery, production-lifecycle, or rainout
example. The other 218 phases in the current FastChem4 condensate
catalog are outside the model, so their phase stability and full-catalog
support closure are not claimed. A support or temperature-validity
change is generally nondifferentiable. The narrow priors are
:math:`T_0\in[1155,1165]` K, :math:`\alpha\in[0.029,0.031]`, and
``log_co_scale`` :math:`\in[-0.005,0.005]`; all eight corners must
retain the reduced-model support and pass the primal diagnostics using
exactly the frozen initialization used by NUTS. The validated corners
had a minimum active graphite amount of :math:`2.498\times10^{-7}`, a
minimum inactive driving margin of :math:`0.714`, a maximum active
fixed-support residual of :math:`2.274\times10^{-13}`, and a maximum
scaled element-budget residual of :math:`1.159\times10^{-13}`. The
completed reference GPU run below exercises the normal 500-warmup,
1000-sample configuration; ``--quick`` remains an end-to-end smoke mode
only.

Discover, freeze, and certify support
-------------------------------------

The chemistry-only preflight needs no ExoJAX database.
``prepare_graphite_profile`` constructs the explicit graphite-only
setup, discovers its nominal support, and freezes the numerical initial
values. ``preflight_graphite_plan`` checks the complete prior box with
those same initial values and checks a reverse-mode chemistry gradient.
The report, rather than an assumed layer list, is the authority for the
fixed mask within this reduced model.

.. code:: python

    from jax import config
    config.update("jax_enable_x64", True)

    import jax
    import jax.numpy as jnp

    from examples.retrievals.exojax_nuts_condensate_fixed_support import (
        TRUTH_ALPHA,
        TRUTH_LOG_CO_SCALE,
        TRUTH_T0_K,
        co_vmr_profile,
        powerlaw_temperature,
        preflight_graphite_plan,
        prepare_graphite_profile,
        pressure_profile,
        scale_carbon_and_oxygen,
    )

    pressures_bar = pressure_profile(8)
    plan = prepare_graphite_profile(pressures_bar)
    preflight = preflight_graphite_plan(plan)
    (
        preflight["condensate_catalog_species"],
        preflight["active_indices"],
        preflight["gradient_finite"],
        preflight["full_catalog_equilibrium_claimed"],
    )


The active-layer fixed-support call
-----------------------------------

The following is the essential active-layer call used by the hybrid
profile. The condensate formula matrix, thermochemical source, amount
seed, and ordering all contain exactly the one frozen graphite support.
Initialization and support are nondifferentiable; temperature,
normalized log pressure, and the elemental inventory in ``ThermoState``
are differentiable.

.. code:: python

    from exogibbs.equilibrium.condensate.fixed_support import (
        minimize_gibbs_fixed_support,
    )
    from exogibbs.equilibrium.gas.types import ThermoState

    support = jnp.asarray([plan.graphite_species_index], dtype=jnp.int32)
    formula_matrix_cond = plan.setup.formula_matrix_cond[:, support]

    def graphite_hvector(temperature):
        return plan.setup.condensate_setup.hvector_func(temperature)[support]

    temperature = powerlaw_temperature(
        plan.pressures_bar, TRUTH_T0_K, TRUTH_ALPHA
    )
    inventory = scale_carbon_and_oxygen(
        plan.reference_element_vector,
        plan.carbon_index,
        plan.oxygen_index,
        TRUTH_LOG_CO_SCALE,
    )
    layer = plan.active_indices[0]
    fixed_result = minimize_gibbs_fixed_support(
        ThermoState(
            temperature[layer],
            jnp.log(plan.pressures_bar[layer]),
            inventory,
        ),
        plan.hybrid_log_amounts_init[layer],
        plan.graphite_amounts_init[layer].reshape((1,)),
        plan.hybrid_total_log_amounts_init[layer],
        plan.setup.formula_matrix,
        formula_matrix_cond,
        plan.setup.gas_setup.hvector_func,
        graphite_hvector,
        residual_crit=1.0e-10,
        max_iter=100,
    )
    fixed_result.gas_log_amounts.shape, fixed_result.condensate_amounts


Reverse-mode hybrid-profile check
---------------------------------

``co_vmr_profile`` batches the frozen graphite-active layers through
``minimize_gibbs_fixed_support`` and batches the inactive layers through
the gas generated VJP. This scalar check traverses both groups. The full
demo additionally checks an ExoJAX spectral loss when a CO database is
supplied.

.. code:: python

    def chemistry_summary(t0_kelvin, alpha, log_co_scale):
        temperatures = powerlaw_temperature(
            plan.pressures_bar, t0_kelvin, alpha
        )
        co_vmr = co_vmr_profile(plan, temperatures, log_co_scale)
        return jnp.sum(jnp.log(jnp.clip(co_vmr, 1.0e-300)))

    summary, gradient = jax.value_and_grad(
        chemistry_summary, argnums=(0, 1, 2)
    )(TRUTH_T0_K, TRUTH_ALPHA, TRUTH_LOG_CO_SCALE)
    summary, gradient


Reverse-mode NUTS
-----------------

The fixed-support and gas equilibrium solvers expose first-order custom
JVPs and generated VJPs. Forward-mode JVPs are supported; the shared
runner selects reverse mode because NUTS differentiates a scalar log
density. Higher-order derivatives remain outside the supported contract.

.. code:: python

    from numpyro.infer import MCMC, NUTS

    def build_reverse_mode_mcmc(model):
        kernel = NUTS(
            model,
            forward_mode_differentiation=False,
            max_tree_depth=10,
        )
        return MCMC(
            kernel, num_warmup=500, num_samples=1000, num_chains=1
        )


Completed reference GPU run
---------------------------

A reference CUDA GPU run completed with 8 atmospheric layers, 1024
spectral points, 500 warmup steps, 1000 posterior samples, and seed 0.
The NUTS call took 170.96 s, including JIT compilation, warmup, and
sampling, and reported zero divergences. All eight prior corners passed
the frozen-support preflight. Layers 0–2 used the graphite fixed-support
VJP and layers 3–7 used the gas VJP.

============ ========== =============================================
parameter    mock truth posterior mean :math:`\pm` standard deviation
============ ========== =============================================
T0 [K]       1160       1159.9514 :math:`\pm` 0.0299
alpha        0.03       0.0300438 :math:`\pm` 0.0000313
log_co_scale 0          -0.0003831 :math:`\pm` 0.0003866
============ ========== =============================================

All three mock truth values lie inside their central 90% posterior
intervals. This is a compact, one-chain demonstration that the hybrid
fixed-support/gas VJP can be used end to end by ExoJAX and NUTS.

The timing is not a like-for-like comparison with the 24-layer gas
no-grid baseline. This reduced case has only eight layers, retains only
graphite from the condensate catalog, reuses frozen nominal solutions as
warm starts, and performs no support discovery inside NUTS. The result
therefore does not imply that general or full-catalog condensation
retrieval is faster than gas-only equilibrium. Runtime is also hardware
and software dependent.

Run the plain demo
------------------

A chemistry-only preflight can run without ExoJAX. The guarded command
below writes the support and corner audit to ``results/``.

.. code:: python

    from pathlib import Path
    import subprocess
    import sys

    RUN_CHEMISTRY_PREFLIGHT = False
    preflight_command = [
        sys.executable,
        "examples/retrievals/exojax_nuts_condensate_fixed_support.py",
        "--preflight-only",
        "--nlayer", "8",
        "--output-dir",
        "results/vjp_retrieval/condensate_chemistry_preflight",
    ]
    if RUN_CHEMISTRY_PREFLIGHT:
        subprocess.run(preflight_command, check=True)
    preflight_command


For a short ExoJAX + NUTS exercise, set ``EXOJAX_CO_DATABASE`` to the
exact local ``CO/12C-16O/Li2015`` directory, add ``--quick``, and supply
``--co-database`` as in the two gas tutorials. The demo never downloads
database files.

The CUDA-only wrapper first runs the complete spectral preflight for
this reduced model, then requests 500 warmup steps and 1000 samples:

.. code:: tcsh

   benchmarks/vjp_retrieval/run_exojax_nuts_gpu.csh \
     condensate_fixed_support /path/to/CO/12C-16O/Li2015

The completed reference run establishes practicality for this local
reduced-model demonstration only. It does not justify widening the
priors or interpreting the posterior as full-catalog condensate
equilibrium. Full FastChem4 condensate-catalog support discovery,
closure certification, phase changes, and rainout remain separate
problems.
