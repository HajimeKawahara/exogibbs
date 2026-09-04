.. This file is generated from the sibling .ipynb by convert_vjp_retrieval_notebooks.py.
.. Do not edit this RST file directly.

ExoJAX NUTS with the gas VJP: grid initialization
=================================================

This tutorial repeats the no-grid retrieval with ExoGibbs’ packaged
FastChem grid supplying the initial gas amounts. The likelihood, priors,
deterministic mock data, profile method, and reverse-mode NUTS settings
are shared with ``examples/retrievals/exojax_nuts_gas_no_grid.py``. The
only algorithmic change in
``examples/retrievals/exojax_nuts_gas_grid.py`` is
``GridEquilibriumInitializer``.

Initialization values have stopped gradients in the implicit derivative.
Once both primal solves converge, grid and no-grid spectra and posterior
derivatives should agree; the grid is intended to reduce primal
iterations. The plain demo checks all eight prior corners and records
grid bounds and the spectral-loss gradient in ``run_summary.json``.

Load the packaged grid
----------------------

The grid metadata must match the runtime FastChem setup exactly. Its
interpolation is used only to initialize each scalar layer solve.

.. code:: python

    from jax import config
    config.update("jax_enable_x64", True)

    import jax
    import jax.numpy as jnp

    from exogibbs.api import (
        get_default_equilibrium_grid_path,
        load_equilibrium_grid_netcdf,
    )
    from exogibbs.api.gas import (
        EquilibriumOptions,
        GridEquilibriumInitializer,
        solve_profile,
    )
    from exogibbs.presets.fastchem import chemsetup

    chemistry = chemsetup(silent=True)
    grid_path = get_default_equilibrium_grid_path("fastchem")
    grid = load_equilibrium_grid_netcdf(str(grid_path))
    initializer = GridEquilibriumInitializer(
        grid=grid, preset_name="fastchem"
    )


The grid-initialized equilibrium call
-------------------------------------

As in the plain demo, carbon and oxygen are scaled together and
``vmap_cold`` solves all layers in parallel.
``GridEquilibriumInitializer`` infers the physical metallicity
coordinate from the traced elemental vector.

.. code:: python

    pressure_bar = jnp.logspace(-3.0, 1.0, 8)
    reference = jnp.asarray(chemistry.element_vector_reference)
    carbon_index = chemistry.elements.index("C")
    oxygen_index = chemistry.elements.index("O")
    co_species_index = chemistry.species.index("C1O1")
    co_element_indices = jnp.asarray([carbon_index, oxygen_index])
    options = EquilibriumOptions(
        epsilon_crit=1.0e-10, max_iter=1000, method="vmap_cold"
    )

    def co_vmr_with_grid(t0_kelvin, alpha, log_co_scale):
        temperature = t0_kelvin * pressure_bar**alpha
        scale = jnp.power(10.0, log_co_scale)
        inventory = reference.at[co_element_indices].set(
            reference[co_element_indices] * scale
        )
        result = solve_profile(
            chemistry,
            temperature,
            pressure_bar,
            inventory,
            Pref=1.0,
            initializer=initializer,
            options=options,
        )
        return result.x[:, co_species_index]


Reverse-mode check
------------------

The grid affects the forward initialization, but the implicit
equilibrium VJP differentiates the converged physical state with respect
to temperature, pressure, and elemental inventory. It does not
differentiate the initial guess.

.. code:: python

    def chemistry_summary(t0_kelvin, alpha, log_co_scale):
        co_vmr = co_vmr_with_grid(t0_kelvin, alpha, log_co_scale)
        return jnp.sum(jnp.log(jnp.clip(co_vmr, 1.0e-300)))

    summary, gradient = jax.value_and_grad(
        chemistry_summary, argnums=(0, 1, 2)
    )(1160.0, 0.03, 0.0)
    summary, gradient


Completed reference GPU runs
----------------------------

Reference CUDA GPU runs completed for both gas cases with the same
normal demo settings: 24 atmospheric layers, 1024 spectral points, 500
warmup steps, 1000 posterior samples, and seed 0.

=========================================== ========= ================
quantity                                    no grid   grid initialized
=========================================== ========= ================
NUTS time                                   6067.69 s 331.95 s
mean prior-corner Newton iterations         240.75    10.0
largest prior-corner Newton iteration count 252       11
divergences                                 0         0
=========================================== ========= ================

The grid initializer made this run 18.28 times faster. Runtime is
hardware and software dependent, but the reduction in Newton iterations
shows the intended role of the initializer directly. All eight prior
corners converged in both cases, and all grid-case corners remained
inside the initializer grid.

+-----------------+------------+-----------------+-----------------+
| parameter       | mock truth | no-grid         | grid posterior  |
|                 |            | posterior       |                 |
+=================+============+=================+=================+
| :math:`T_0` [K] | 1160       | 1159.9783       | 1159.9768       |
|                 |            | :math:`\pm`     | :math:`\pm`     |
|                 |            | 0.0335          | 0.0325          |
+-----------------+------------+-----------------+-----------------+
| :math:`\alpha`  | 0.03       | 0.0300155       | 0.0300121       |
|                 |            | :math:`\pm`     | :math:`\pm`     |
|                 |            | 0.0000341       | 0.0000334       |
+-----------------+------------+-----------------+-----------------+
| log_co_scale    | 0          | -0.0000416      | 0.0000027       |
|                 |            | :math:`\pm`     | :math:`\pm`     |
|                 |            | 0.0005270       | 0.0005110       |
+-----------------+------------+-----------------+-----------------+

The mock truth lies inside every central 90% posterior interval, and the
grid and no-grid posteriors agree at the scale resolved by this demo. At
the mock truth, the complete spectral loss agrees to about
:math:`2 \times 10^{-15}` relatively and the three reverse-mode
gradients agree to better than :math:`6 \times 10^{-9}` relatively. Thus
the initializer accelerates the primal equilibrium solves without
materially changing the converged spectrum or its VJP. These are
compact, one-chain demonstrations rather than precision convergence
benchmarks.

Reverse-mode NUTS
-----------------

ExoGibbs supports both forward-mode JVPs and generated VJPs for these
equilibrium solves. This NUTS example selects reverse mode because it
differentiates a scalar log density.

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

Launch Jupyter from the repository root and point ``EXOJAX_CO_DATABASE``
at the exact existing ``CO/12C-16O/Li2015`` directory. No database is
downloaded.

.. code:: python

    import os
    from pathlib import Path
    import subprocess
    import sys

    RUN_QUICK = False
    co_database = Path(
        os.environ.get(
            "EXOJAX_CO_DATABASE", "/path/to/CO/12C-16O/Li2015"
        )
    )
    command = [
        sys.executable,
        "examples/retrievals/exojax_nuts_gas_grid.py",
        "--co-database", str(co_database),
        "--output-dir", "results/vjp_retrieval/gas_grid_quick",
        "--quick",
        "--no-progress-bar",
    ]
    if RUN_QUICK:
        subprocess.run(command, check=True)
    command


The CUDA-only production wrapper uses 500 warmup steps, 1000 samples,
seed 0, and writes to ``results/vjp_retrieval/gas_grid/``. The gas
``--quick`` profile uses at most 100 warmup steps, 100 samples, and tree
depth 8; it checks sampler adaptation but is not an inference-quality
chain. Sampling runs record the effective configuration and sampler
diagnostics and fail if a transition diverges, a parameter is completely
stuck, or samples are incomplete or non-finite.

.. code:: tcsh

   benchmarks/vjp_retrieval/run_exojax_nuts_gpu.csh \
     gas_grid /path/to/CO/12C-16O/Li2015
