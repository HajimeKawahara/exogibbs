.. This file is generated from the sibling .ipynb by convert_vjp_retrieval_notebooks.py.
.. Do not edit this RST file directly.

ExoJAX NUTS with the gas VJP: no grid initializer
=================================================

This tutorial demonstrates the gas-equilibrium generated VJP inside an
ExoJAX emission-spectrum retrieval. It is the no-grid baseline: every
atmospheric layer uses the same uniform cold initialization. The full,
executable source is ``examples/retrievals/exojax_nuts_gas_no_grid.py``;
this notebook keeps the key equilibrium and reverse-mode calls visible
and delegates the complete spectrum and output workflow to that script.

The deterministic mock truth is :math:`T_0=1160` K, :math:`\alpha=0.03`,
and ``log_co_scale=0``. Carbon and oxygen are scaled together, so C/O
remains fixed. The normal defaults use 24 layers, 1024 spectral points,
500 warmup steps, and 1000 samples. Use ``--quick`` before submitting
the production GPU job. The gas quick profile uses at most 100 warmup
steps, 100 samples, and tree depth 8; it checks sampler adaptation but
is not an inference-quality chain. Sampling runs record the effective
configuration and sampler diagnostics and fail if a transition diverges,
a parameter is completely stuck, or samples are incomplete or
non-finite.

The equilibrium call used by the forward model
----------------------------------------------

The following chemistry-only function is a compact version of the call
inside the shared spectral model. It can be run without an ExoJAX
database. Both gas demonstrations explicitly use ``vmap_cold``; their
only intended difference is the initializer.

.. code:: python

    from jax import config
    config.update("jax_enable_x64", True)

    import jax
    import jax.numpy as jnp

    from exogibbs.api.gas import EquilibriumOptions, solve_profile
    from exogibbs.presets.fastchem import chemsetup

    chemistry = chemsetup(silent=True)
    pressure_bar = jnp.logspace(-3.0, 1.0, 8)
    reference = jnp.asarray(chemistry.element_vector_reference)
    carbon_index = chemistry.elements.index("C")
    oxygen_index = chemistry.elements.index("O")
    co_species_index = chemistry.species.index("C1O1")
    co_element_indices = jnp.asarray([carbon_index, oxygen_index])
    options = EquilibriumOptions(
        epsilon_crit=1.0e-10, max_iter=1000, method="vmap_cold"
    )

    def co_vmr_no_grid(t0_kelvin, alpha, log_co_scale):
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
            options=options,
        )
        return result.x[:, co_species_index]


Exercise the reverse-mode VJP
-----------------------------

NUTS needs gradients of a scalar log density. This small scalar summary
checks the same temperature and elemental-inventory VJP paths before the
spectral calculation is opened. The plain demo performs the stronger
check on the complete ExoJAX spectral loss and checks all eight prior
corners for primal convergence.

.. code:: python

    def chemistry_summary(t0_kelvin, alpha, log_co_scale):
        co_vmr = co_vmr_no_grid(t0_kelvin, alpha, log_co_scale)
        return jnp.sum(jnp.log(jnp.clip(co_vmr, 1.0e-300)))

    summary, gradient = jax.value_and_grad(
        chemistry_summary, argnums=(0, 1, 2)
    )(1160.0, 0.03, 0.0)
    summary, gradient


Reverse-mode NUTS
-----------------

The ExoGibbs gas solver provides an implicit custom JVP and an
automatically transposed VJP. The shared runner selects reverse mode
explicitly because NUTS differentiates a scalar log density. Its
essential NumPyro construction is:

.. code:: python

    from numpyro.infer import MCMC, NUTS

    def build_reverse_mode_mcmc(model):
        # `model` closes over the deterministic observation and ExoJAX context.
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

A reference CUDA GPU run completed with the normal demo settings: 24
atmospheric layers, 1024 spectral points, 500 warmup steps, 1000
posterior samples, and seed 0. The NUTS call took 6067.69 s (1 h 41 min
8 s), including JIT compilation, warmup, and sampling, and reported zero
divergences. Runtime is hardware and software dependent.

=============== ========== =============================================
parameter       mock truth posterior mean :math:`\pm` standard deviation
=============== ========== =============================================
:math:`T_0` [K] 1160       1159.9783 :math:`\pm` 0.0335
:math:`\alpha`  0.03       0.0300155 :math:`\pm` 0.0000341
log_co_scale    0          -0.0000416 :math:`\pm` 0.0005270
=============== ========== =============================================

All three mock truth values lie inside their central 90% posterior
intervals. This is deliberately a compact, one-chain demonstration of an
end-to-end ExoJAX retrieval through the ExoGibbs generated VJP, rather
than a precision convergence benchmark.

Run the complete demo
---------------------

Launch Jupyter from the repository root. Set ``EXOJAX_CO_DATABASE`` to
the exact existing ``CO/12C-16O/Li2015`` directory; the demo validates
local files and never downloads them. The guarded cell below runs the
small end-to-end configuration and writes only under ``results/``.

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
        "examples/retrievals/exojax_nuts_gas_no_grid.py",
        "--co-database", str(co_database),
        "--output-dir", "results/vjp_retrieval/gas_no_grid_quick",
        "--quick",
        "--no-progress-bar",
    ]
    if RUN_QUICK:
        subprocess.run(command, check=True)
    command


For the CUDA-only production configuration, submit or run the
scheduler-independent tcsh wrapper:

.. code:: tcsh

   benchmarks/vjp_retrieval/run_exojax_nuts_gpu.csh \
     gas_no_grid /path/to/CO/12C-16O/Li2015

The wrapper checks ``nvidia-smi``, requires the JAX GPU backend, runs a
preflight, and writes artifacts to
``results/vjp_retrieval/gas_no_grid/``.
