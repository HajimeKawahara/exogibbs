.. exojax documentation master file, created by
   sphinx-quickstart on Mon Jan 11 14:38:51 2021.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

ExoGibbs
==================================

Version 0.6

|:frog:| `DeepWiki for ExoGibbs <https://deepwiki.com/HajimeKawahara/exogibbs>`_


`ExoGibbs <https://github.com/HajimeKawahara/exogibbs>`_ provides an auto-differentiable thermochemical equilibrium solver
based on the Gibbs free energy minimization method, powered by `JAX <https://github.com/google/jax>`_.


Contents
==================================

.. toctree::
   :maxdepth: 1
   :caption: PRESETS:

   presets/ykb4.rst
   presets/fastchem.rst
   presets/fastchem4.rst
   equilibrium_solvers.rst
   condensate_profile.rst

.. toctree::
   :maxdepth: 1
   :caption: EXAMPLES:

   visscher_2006_na2s_morley_2012_kcl
   visscher_2010_forsterite_enstatite_competition
   ito_2025_rainout_comparison
   fe_fes_rainout_demo
   comparison_example_lineage

.. toctree::
   :maxdepth: 1
   :caption: VJP RETRIEVALS:

   ipynb/exojax_nuts_gas_no_grid
   ipynb/exojax_nuts_gas_grid
   ipynb/exojax_nuts_condensate_fixed_support
   ipynb/exojax_nuts_condensate_grid

.. toctree::
   :maxdepth: 1
   :caption: API:

   exogibbs/index.rst

.. toctree::
   :maxdepth: 1
   :caption: VALIDATION:

   v0_4_fastchem4_validation_demo
   fastchem4_production_comparison

License & Attribution
---------------------

Copyright 2025-, Contributors

- `Hajime Kawahara <http://secondearths.sakura.ne.jp/en/index.html>`_ (@HajimeKawahara, maintainer)
- `Yui Kawashima <https://sites.google.com/view/yuikawashima/home>`_ (@ykawashima, the reference code author we inherit, the presets ykb4 originally comes from this code.)

ExoGibbs is free software made available under the GPLv3 License.
