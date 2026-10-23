Public API reference
====================

Start with :doc:`equilibrium_solvers` to choose an equilibrium mode and check
its units and derivative contract. This reference documents the supported
entry points once, using their public import paths. Compatibility aliases and
internal solver modules remain available in Python but are not indexed here.

Gas equilibrium
---------------

.. automodule:: exogibbs.api.gas
   :members:
   :exclude-members: LogFugacityCoefficientFunction

Pure-condensate equilibrium
---------------------------

.. automodule:: exogibbs.api.condensate
   :members:
   :exclude-members: LogFugacityCoefficientFunction

Multiphase equilibrium
----------------------

.. automodule:: exogibbs.api.multiphase
   :members:

Gas equilibrium grids
---------------------

.. automodule:: exogibbs.api.equilibrium_grid
   :members:

Magma--gas application
----------------------

.. automodule:: exogibbs.api.magma_gas
   :members:

Thermochemical inputs
---------------------

.. automodule:: exogibbs.thermo.models
   :members: ChemicalSetup

.. automodule:: exogibbs.thermo.fugacity
   :members: LogFugacityCoefficientFunction

For packaged inputs, see :doc:`presets/curated`, :doc:`presets/ykb4`,
:doc:`presets/fastchem` and :doc:`presets/fastchem4`. Standard-state conventions
are described in :doc:`standard_thermodynamics`.

Solubility and provider adapters
--------------------------------

.. automodule:: exogibbs.solubility
   :members:

.. automodule:: exogibbs.interop.exoeos
   :members:

Compatibility imports
---------------------

.. py:module:: exogibbs.api.equilibrium

.. py:module:: exogibbs.api.condensate_equilibrium

These older equilibrium modules remain supported in v0.7. New workflows use
:mod:`exogibbs.api.gas` and :mod:`exogibbs.api.condensate`; see the compatibility
notes in :doc:`equilibrium_solvers`. The ``exogibbs.optimize`` and former
``exogibbs.experimental.magma_gas`` paths also retain their existing aliases.
