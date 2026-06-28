FastChem4 Preset
================

Overview
--------
The FastChem4 preset bundles native ExoGibbs ``ChemicalSetup`` objects based on
packaged FastChem4 thermochemistry data.  It does not call the FastChem4 runtime
and does not use FastChem4 public output, runtime state, or trace values as
constructor inputs.

The preset is split in the same way as the FastChem preset:

* ``exogibbs.presets.fastchem4`` provides the gas-phase ``ChemicalSetup``.
* ``exogibbs.presets.fastchem4_cond`` provides the condensate ``ChemicalSetup``
  and the combined gas-condensate setup bundle used by the condensate API.

Quick Start: Gas Equilibrium
----------------------------
.. code-block:: python

   from jax import config
   config.update("jax_enable_x64", True)

   from exogibbs.api.equilibrium import equilibrium
   from exogibbs.presets.fastchem4 import chemsetup

   setup = chemsetup()
   T, P = 1500.0, 1.0  # K, bar
   b = setup.element_vector_reference
   result = equilibrium(setup, T=T, P=P, b=b)
   print(result.x)

Quick Start: Condensate Equilibrium
-----------------------------------
.. code-block:: python

   from jax import config
   import jax.numpy as jnp
   config.update("jax_enable_x64", True)

   from exogibbs.api.condensate_equilibrium import (
       CondensateEquilibriumOptions,
       condensate_equilibrium,
   )
   from exogibbs.presets.fastchem4_cond import condensate_chemical_setup

   setup = condensate_chemical_setup()
   b = jnp.asarray(setup.gas_setup.element_vector_reference, dtype=jnp.float64)
   species_index = {name: index for index, name in enumerate(setup.condensate_species)}
   support = tuple(species_index[name] for name in ("MgSiO3(s,l)", "Mg2SiO4(s,l)", "SiO2(s,l)"))

   result = condensate_equilibrium(
       setup,
       T=1400.0,
       P=0.1,
       b=b,
       support_indices=support,
       support_amounts_init=tuple(1.0e-12 for _ in support),
       options=CondensateEquilibriumOptions(return_diagnostics=True),
   )
   print(result.status, result.condensate_support_names)

Quick Start: Condensate Profile
-------------------------------
Use ``condensate_equilibrium_profile`` for a one-dimensional pressure and
temperature profile.  The recommended profile method is ``"auto"``.  With a
complete fixed-support initializer, ``"auto"`` can use the fixed-support batch
path with fallback rescue; otherwise it falls back to a conservative hot scan.

.. code-block:: python

   from jax import config
   import jax.numpy as jnp
   config.update("jax_enable_x64", True)

   from exogibbs.api.condensate_equilibrium import (
       CondensateEquilibriumOptions,
       condensate_equilibrium_profile,
   )
   from exogibbs.presets.fastchem4_cond import condensate_chemical_setup

   setup = condensate_chemical_setup()
   b = jnp.asarray(setup.gas_setup.element_vector_reference, dtype=jnp.float64)
   T = jnp.asarray([1700.0, 1600.0, 1500.0, 1400.0], dtype=jnp.float64)
   P = jnp.asarray([1.0e-3, 3.0e-3, 1.0e-2, 3.0e-2], dtype=jnp.float64)

   profile = condensate_equilibrium_profile(
       setup,
       T=T,
       P=P,
       b=b,
       options=CondensateEquilibriumOptions(profile_method="auto"),
       return_diagnostics=True,
   )
   print(profile.method)
   print([layer.status for layer in profile.layers])

See :doc:`../condensate_profile` for the profile method contract and the
fixed-support batch/rescue diagnostics.

Packaged Data
-------------
The default gas setup reads:

``src/exogibbs/data/FastChem4/logK/logK_wo_ions.dat``

The default condensate setup reads:

``src/exogibbs/data/FastChem4/logK/logK_condensates.dat``

The reference elemental abundances are the same ExoGibbs default element vector
used by the FastChem preset, based on Asplund, Amarsi & Grevesse (2021).

Elements
--------
The following elements are included in the default setup, including electrons as
``e-``:

``Al, Ar, C, Ca, Cl, Co, Cr, Cu, F, Fe, Ge, H, He, K, Mg, Mn, N, Na, Ne, Ni, O, P, S, Si, Ti, V, Zn, e-``

Species Counts
--------------
The default native FastChem4 gas setup contains 422 gas species, including the
28 elemental species.  The default native FastChem4 condensate setup contains
219 condensate species.

The condensate species order follows the packaged FastChem4
``logK_condensates.dat`` order.  This order is part of the API contract for
support indices.  In particular:

.. code-block:: text

   condensate_species[186:191]
   = ("Ca(VO3)2(s)", "Ca2V2O7(s)", "Ca3(VO4)2(s)", "CaZn(s)", "CaZn2(s)")

   condensate_species[188] == "Ca3(VO4)2(s)"

The duplicate ``Zn(s,l)`` entries are preserved at slots 167 and 202.  The
condensate formula matrix therefore keeps one column per data-file entry, not
one column per unique species name.

Preset Functions
----------------
Gas setup:

.. code-block:: python

   from exogibbs.presets.fastchem4 import chemsetup

   gas_setup = chemsetup(
       path="FastChem4/logK/logK_wo_ions.dat",
       species_defalt_elements=True,
       element_file=None,
       silent=False,
   )

Condensate-only setup:

.. code-block:: python

   from exogibbs.presets.fastchem4_cond import chemsetup

   condensate_setup = chemsetup(
       path="FastChem4/logK/logK_condensates.dat",
       gas_setup=gas_setup,
       silent=False,
   )

Combined gas-condensate setup:

.. code-block:: python

   from exogibbs.presets.fastchem4_cond import condensate_chemical_setup

   setup = condensate_chemical_setup(
       gas_path="FastChem4/logK/logK_wo_ions.dat",
       condensate_path="FastChem4/logK/logK_condensates.dat",
       species_defalt_elements=True,
       element_file=None,
       silent=False,
   )

Notes
-----
The FastChem4 preset is a native ExoGibbs data preset.  FastChem4 may be used as
a reference implementation during audits, but the preset itself is independent
of a FastChem4 installation.
