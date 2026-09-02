.. This file is generated from the sibling .ipynb by convert_vjp_retrieval_notebooks.py.
.. Do not edit this RST file directly.

ExoJAX NUTS with the condensate VJP: grid initialization
========================================================

This tutorial is the grid-initialized counterpart to
``exojax_nuts_condensate_fixed_support``. The executable source is
``examples/retrievals/exojax_nuts_condensate_grid.py``; it shares the
carbon-rich C/O = 2 chemistry, narrow priors, deterministic mock
spectrum, frozen graphite support, and reverse-mode NUTS settings with
``examples/retrievals/exojax_nuts_condensate_fixed_support.py``. The
runtime solver uses one local fixed-support graphite grid per active
layer and one shared gas-equilibrium grid for inactive layers.

For each active layer, its ``FixedSupportCondensateEquilibriumGrid``
stores converged fixed-support states: gas log amounts, the gas total,
and the graphite amount. The corresponding
``GridCondensateEquilibriumInitializer`` interpolates all three values
at the same coordinate. Only the graphite support and active/inactive
layer partition are frozen from the nominal calculation; the graphite
amount used to initialize each runtime solve is interpolated
dynamically. The inactive layers use the separate shared gas grid.
Before sampling, all eight prior corners must retain the nominal
support, the frozen baseline must converge, the grid and baseline CO VMR
profiles must agree, and the grid must not increase the maximum
fixed-support iteration count. The grid-side reverse-mode gradient must
be finite at the truth point.

This remains a local fixed-support demonstration with the full FastChem4
gas catalog and an explicit reduced condensate catalog containing only
``C(s)``. It does not differentiate support discovery, phase
transitions, the production condensate lifecycle, or rainout, and it
makes no full-catalog phase-stability or support-closure claim.

Precompute the gas and fixed-support grids
------------------------------------------

First, ``prepare_graphite_profile`` discovers and certifies the nominal
graphite support without a grid. The grid helper then constructs one
``FixedSupportCondensateEquilibriumGrid`` per active layer. Each local
grid uses that layer’s four prior-corner temperatures, a singleton
pressure axis at the fixed layer pressure, and three composition points.
Keeping the active grids local avoids evaluating a rectangular
cross-product of temperatures and pressures whose off-layer points need
not retain graphite support. Each local grid is wrapped in
``GridCondensateEquilibriumInitializer``. A small example-level
composite keeps those initializers together with the shared canonical
gas-grid initializer used by inactive layers. Grid construction is
completed before inference, and its wall time is returned separately
from NUTS timing.

.. code:: python

    from dataclasses import replace

    from jax import config
    config.update("jax_enable_x64", True)

    import jax
    import jax.numpy as jnp

    from examples.retrievals.exojax_nuts_condensate_fixed_support import (
        TRUTH_ALPHA,
        TRUTH_LOG_CO_SCALE,
        TRUTH_T0_K,
        build_graphite_grid_initializer,
        co_vmr_profile,
        graphite_only_chemical_setup,
        interpolate_graphite_grid_initial_values,
        powerlaw_temperature,
        preflight_graphite_plan,
        prepare_graphite_profile,
        pressure_profile,
        scale_carbon_and_oxygen,
    )

    pressures_bar = pressure_profile(8)
    chemistry = graphite_only_chemical_setup()
    base_plan = prepare_graphite_profile(pressures_bar, setup=chemistry)
    grid_initializer, grid_build_seconds = build_graphite_grid_initializer(
        base_plan
    )
    plan = replace(
        base_plan,
        grid_initializer=grid_initializer,
        grid_build_seconds=grid_build_seconds,
    )
    gas_grid = grid_initializer.gas_initializer.grid
    fixed_grids = tuple(
        initializer.grid
        for initializer in grid_initializer.fixed_initializers
    )
    (
        gas_grid.outputs.ln_n.shape,
        tuple(grid.gas_grid.outputs.ln_n.shape for grid in fixed_grids),
        tuple(grid.condensate_amounts.shape for grid in fixed_grids),
        grid_build_seconds,
    )

The two grid paths use different composition constructions. Each
active-layer local fixed-support grid is built with
``scale_carbon_and_oxygen``, exactly the elemental-inventory map used
inside NUTS: carbon and oxygen are scaled together while C/O remains
two. Its stored gas log amounts, gas total, and graphite amount are
therefore converged states for the retrieval’s composition family. The
shared gas grid reuses ``build_equilibrium_grid`` and its canonical
physical-``log10(Z/Zsun)`` coordinate, which uniformly scales metals.
For the retrieval’s C/O-only sampled inventory, that shared grid
supplies an approximate numerical seed. The gas-only equilibrium kernel
still receives the exact sampled elemental vector and determines the
converged physical state from it.

Freeze and certify graphite support
-----------------------------------

The base plan freezes only the nominal graphite support and the
resulting active/inactive layer partition. It does not freeze the
graphite amount used by the grid-enabled runtime solver.
``preflight_graphite_plan`` checks all eight corners of the
three-parameter prior box with the same lookup path used during NUTS:
the support must match the nominal mask, the frozen-baseline solves must
converge, and the grid and baseline CO VMR profiles must agree. At every
corner, the grid-initialized maximum fixed-support iteration count must
also be no greater than the frozen-baseline count. Separately, the
preflight checks that the grid-side reverse-mode gradient is finite at
the truth point; it does not compute a second baseline gradient. The
corner iteration assertion checks the intended initialization behavior
but is not a wall-time speedup claim.

.. code:: python

    preflight = preflight_graphite_plan(
        plan, case_name="condensate_grid"
    )
    (
        preflight["passed"],
        preflight["active_indices"],
        preflight["inactive_indices"],
        preflight["gradient_finite"],
        preflight["grid_no_grid_equivalence"],
        preflight["grid_bounds"],
        preflight["grid_build_seconds"],
    )

Interpolate at every NUTS evaluation
------------------------------------

For each sampled temperature profile and elemental vector, every active
layer interpolates the converged gas log amounts, gas total, and
graphite amount together from its corresponding local fixed-support
grid. Those three dynamically selected values initialize the
fixed-support condensate kernel. Inactive layers interpolate gas log
amounts and the gas total from the shared gas grid and pass them to the
gas-only kernel. The nominal support mask only selects which lookup and
solver each layer uses.

.. code:: python

    temperature = powerlaw_temperature(
        plan.pressures_bar, TRUTH_T0_K, TRUTH_ALPHA
    )
    inventory = scale_carbon_and_oxygen(
        plan.reference_element_vector,
        plan.carbon_index,
        plan.oxygen_index,
        TRUTH_LOG_CO_SCALE,
    )
    initial_values = interpolate_graphite_grid_initial_values(
        plan, temperature, inventory
    )
    (
        initial_values.gas_log_amounts.shape,
        initial_values.fixed_gas_log_amounts.shape,
        initial_values.graphite_amounts.shape,
    )

Reverse-mode check
------------------

The gas and fixed-support implicit derivatives stop gradients through
all numerical initialization values, including the interpolated graphite
amount. Grid values can depend on the sampled parameters in the forward
pass, but posterior derivatives are implicit derivatives of the
converged equilibrium state, not derivatives through the initializer.
Support remains a static, nondifferentiable contract. The preflight
checks the grid-side gradient for finiteness at the truth point. It does
not repeat the gradient calculation for the frozen baseline; the eight
corners are used for baseline convergence, CO VMR primal equivalence,
support, and iteration checks.

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

The equilibrium kernels support forward-mode JVPs and generated VJPs.
This sampler selects reverse mode because it differentiates a scalar log
density. The shared gas grid and all local fixed-support grids are
constructed outside the sampler, and ``grid_build_seconds`` is reported
separately from sampling time. No speedup is assumed: a performance
claim requires completed grid and non-grid runs with equivalent
converged results on the same hardware.

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

Run the complete demo
---------------------

A chemistry-only preflight builds the shared gas grid and all local
fixed-support grids. At all prior corners it checks frozen-baseline
convergence, support, grid/baseline CO VMR primal equivalence, and
maximum fixed-support iteration counts. It then checks the grid-side
reverse-mode gradient for finiteness at the truth point, without opening
an ExoJAX database. The command is guarded so opening this notebook does
not start the calculation.

.. code:: python

    from pathlib import Path
    import subprocess
    import sys

    RUN_CHEMISTRY_PREFLIGHT = False
    preflight_command = [
        sys.executable,
        "examples/retrievals/exojax_nuts_condensate_grid.py",
        "--preflight-only",
        "--nlayer", "8",
        "--output-dir",
        "results/vjp_retrieval/condensate_grid_preflight",
    ]
    if RUN_CHEMISTRY_PREFLIGHT:
        subprocess.run(preflight_command, check=True)
    preflight_command

For a guarded end-to-end smoke run, point ``EXOJAX_CO_DATABASE`` at the
exact existing ``CO/12C-16O/Li2015`` directory. The demo never downloads
database files.

.. code:: python

    import os

    RUN_QUICK = False
    co_database = Path(
        os.environ.get(
            "EXOJAX_CO_DATABASE", "/path/to/CO/12C-16O/Li2015"
        )
    )
    quick_command = [
        sys.executable,
        "examples/retrievals/exojax_nuts_condensate_grid.py",
        "--co-database", str(co_database),
        "--output-dir",
        "results/vjp_retrieval/condensate_grid_quick",
        "--quick",
        "--no-progress-bar",
    ]
    if RUN_QUICK:
        subprocess.run(quick_command, check=True)
    quick_command

The CUDA-only launcher first runs the spectral preflight, then requests
500 warmup steps and 1000 samples with seed 0:

.. code:: tcsh

   benchmarks/vjp_retrieval/run_exojax_nuts_gpu.csh \
     condensate_grid /path/to/CO/12C-16O/Li2015

The five-warmup ``--quick`` mode is only an end-to-end smoke test. No
performance gain is claimed without completed grid and non-grid runs on
the same hardware. Compare solver iterations and NUTS time separately,
keep ``grid_build_seconds`` outside the sampling time, require agreement
of converged spectra, and confirm that the grid-side reverse-mode
gradient is finite before interpreting a timing comparison.
