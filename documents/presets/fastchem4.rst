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

Reviewed Gas Thermochemistry
----------------------------

For gas-only calculations with reviewed atomic-ion and major-molecule source
choices, use :doc:`curated` on **1500--4000 K**, with ``ions=True`` or
``ions=False``. That page also describes migration from the retired Ni/Co-only
preset. The :doc:`../ni_co_ionization_audit` retains the original discrepancy
and independent checks. The separate :doc:`../standard_thermodynamics` model
continues to provide absolute entropy and heat capacity on **1500--4500 K**.

Water saturation and the scope of a cold audit
----------------------------------------------

A finite thermochemical callback or an accepted equilibrium audit does not
establish low-temperature empirical accuracy. For an ideal gas and a
unit-activity pure condensate with the same water formula and elemental gauge,

.. math::

   \ln\frac{p_{\rm sat}}{1\,\mathrm{bar}}
   =\frac{g_c^\circ-g_g^\circ}{RT},\qquad
   D_w=\frac{g_c^\circ-\mu_w}{RT}
   =\ln\frac{p_{\rm sat}}{p_w}.

The common atomic gauge cancels in this difference. Positive ``D_w`` excludes
insertion of this pure-water phase at the given gas state. It does not certify
other phases or the separate gas and condensed standard potentials. The
equivalent native saturation prescription can be inspected without solving:

.. code-block:: python

   import jax
   jax.config.update("jax_enable_x64", True)
   import numpy as np
   from exogibbs.presets.fastchem4_cond import condensate_chemical_setup

   setup = condensate_chemical_setup(silent=True)
   gas, cloud = setup.gas_setup, setup.condensate_setup
   temperature = 260.0
   gas_water = gas.species.index("H2O1")
   condensed_water = cloud.species.index("H2O(s,l)")
   gas_rt = float(gas.hvector_func(temperature)[gas_water])
   condensed_rt = float(cloud.hvector_func(temperature)[condensed_water])
   saturation_bar = np.exp(condensed_rt - gas_rt)
   print(saturation_bar)

The native water entry switches its ice/liquid segments at 273.16 K and
becomes ineligible above 647.10 K. Neither rule supplies lower-temperature
accuracy for all catalog species. The
`S2 cold-water comparison <https://github.com/HajimeKawahara/exoinventory/pull/129>`_
measured maximum native/reference saturation differences of 3.54% for ice
over 251.19--273.16 K and 4.02% for liquid over 273.16--630.96 K. Its native
segment switch has a 1.91% pressure jump. The references are
`Murphy and Koop (2005), Eq. (7) <https://doi.org/10.1256/qj.04.94>`_
and `IAPWS SR1-86(1992), Eq. (1) <https://iapws.org/public/documents/6dGkr/Supp-sat.pdf>`_.
The IAPWS release is a related reference prescription, not the exact 2008
steam-table book in the native header. These differences are not a certified
error bound on mixture fugacity or on each standard potential.

In that declared application, all 302 reference-covered layers have zero
water-condensate amount. Replacing only its standard by
``gas_rt + log(reference_saturation_bar)`` leaves all those amount vectors
accepted by the independent :doc:`../condensate_audit`, with minimum water
insertion cost 7.447526 in RT units. This supports a specific absent-phase
robustness statement while preserving the failed 1% reference-comparison
screen. It is not a new equilibrium solve or a catalog-wide physical
validation. Native coefficients, solver behavior and defaults are unchanged.

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
