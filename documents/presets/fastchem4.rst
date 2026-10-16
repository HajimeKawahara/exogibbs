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

   from exogibbs.api.gas import solve
   from exogibbs.presets.fastchem4 import chemsetup

   setup = chemsetup()
   T, P = 1500.0, 1.0  # K, bar
   b = setup.element_vector_reference
   result = solve(setup, T=T, P=P, b=b)
   print(result.x)

Quick Start: Condensate Equilibrium
-----------------------------------
.. code-block:: python

   from jax import config
   import jax.numpy as jnp
   config.update("jax_enable_x64", True)

   from exogibbs.api.condensate import (
       CondensateEquilibriumOptions,
       solve,
   )
   from exogibbs.presets.fastchem4_cond import condensate_chemical_setup

   setup = condensate_chemical_setup()
   b = jnp.asarray(setup.gas_setup.element_vector_reference, dtype=jnp.float64)
   species_index = {name: index for index, name in enumerate(setup.condensate_species)}
   support = tuple(species_index[name] for name in ("MgSiO3(s,l)", "Mg2SiO4(s,l)", "SiO2(s,l)"))

   result = solve(
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
Use ``solve_profile`` for a one-dimensional pressure and
temperature profile. ``"auto"`` and ``"vmap_cold"`` both select the production
fixed-support v2 lifecycle. Support expansion is owned by the profile layer
outside the fixed-support solver, and failures are reported without retrying a
retired solver.

.. code-block:: python

   from jax import config
   import jax.numpy as jnp
   config.update("jax_enable_x64", True)

   from exogibbs.api.condensate import (
       CondensateEquilibriumOptions,
       solve_profile,
   )
   from exogibbs.presets.fastchem4_cond import condensate_chemical_setup

   setup = condensate_chemical_setup()
   b = jnp.asarray(setup.gas_setup.element_vector_reference, dtype=jnp.float64)
   T = jnp.asarray([1700.0, 1600.0, 1500.0, 1400.0], dtype=jnp.float64)
   P = jnp.asarray([1.0e-3, 3.0e-3, 1.0e-2, 3.0e-2], dtype=jnp.float64)

   profile = solve_profile(
       setup,
       T=T,
       P=P,
       b=b,
       options=CondensateEquilibriumOptions(profile_method="auto"),
       return_diagnostics=True,
   )
   print(profile.method)
   print([layer.status for layer in profile.layers])

See :doc:`../condensate_profile` for the profile method contract and v2
lifecycle diagnostics.

NASA Ni/Co Ionization
---------------------

To correct the JANAF-derived Ni+ and Co+ ionization data, explicitly select
``fastchem4_nasa9_ions``:

.. code-block:: python

   from exogibbs.api.gas import solve
   from exogibbs.presets.fastchem4_nasa9_ions import chemsetup

   setup = chemsetup(silent=True)
   result = solve(setup, T=3000.0, P=0.01, b=setup.element_vector_reference)

This preset starts from the **ion-inclusive** ``FastChem4/logK/logK.dat`` and
retains all 539 default species. Only Ni1+ and Co1+ receive NASA CEA reaction
Gibbs energies, including the neutral atom and electron reference terms.
The other species, elemental abundances and stoichiometry retain their
FastChem4 values. The original ``fastchem4.chemsetup()`` remains available
for comparisons; its default is still the table without ions.

The replacement interval is **1000--6000 K**. Scalar and array temperatures
are supported, and the resulting functions are JAX differentiable. Values
outside this interval return NaN, with no fallback or extrapolation. This
domain describes the NASA replacement data, not an independent validation
of every retained FastChem species.

The optional ``path``, ``species_default_elements`` and ``element_file``
arguments select the underlying FastChem network and abundances as in the
original preset. A network missing Ni, Ni+, Co, Co+ or the electron is rejected.
Metadata records the NASA source, replacement identities and temperature
interval, and keeps the replaced FastChem records separately from the active
FastChem records.

This two-ion preset is distinct from
:doc:`../standard_thermodynamics`: that model changes 35 species, omits one,
provides consistent absolute entropy and heat capacity, and has a narrower
1500--4500 K domain. Both now use NASA data for Ni+ and Co+.
See :doc:`../ni_co_ionization_audit` for the discrepancy and independent checks.

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
