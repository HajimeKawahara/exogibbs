"""Common M2 gas reactions with an explicit, conserved elemental reference.
============================================================================

FastChem4 supplies the temperature-dependent gas reactions. Its atomic-zero
reference cannot replace absolute MELTS/source energies without a gauge.
Seven independent existing source standards anchor that gauge; this preserves
the declared lower reference and does not calibrate cross-phase reactions.
"""

from __future__ import annotations

import numpy as np

from m1_chemistry import build_setups, subset_setup


SHARED_SPECIES = ("H2", "O2", "H2O", "Fe", "Mg", "SiO", "Na", "H", "He", "OH", "SiH4")
UPPER_SPECIES = ("H2", "O2", "H2O1", "Fe1", "Mg1", "O1Si1", "Na1", "H1", "He1", "H1O1", "H4Si1")
REFERENCE_ANCHORS = ("H", "He", "O2", "Mg", "SiO", "Fe", "Na")


def build_common_gas_setup(*, expanded=False):
    """Return the eleven-species control or all 35 gases on the M1 reference."""
    _, catalog = build_setups()
    return catalog.gas_setup if expanded else subset_setup(catalog.gas_setup, UPPER_SPECIES)


def source_gas_names(setup):
    """Preserve historical source names and name additional FastChem species."""
    aliases = dict(zip(UPPER_SPECIES, SHARED_SPECIES))
    return tuple(aliases.get(name, name) + "_gas" for name in setup.species)


def anchored_standards_rt(setup, temperature_k, reference_standards):
    """Adopt FastChem reactions while retaining seven lower reference anchors.

    The output follows ``setup.species``. The same element shift must apply to
    every upper gas and condensate if its absolute energy is compared with the
    lower system. Equilibrium in an isolated conserved parcel is invariant to
    this shift. No reaction-specific adjustment or temperature fit is made.
    """
    if not np.isfinite(temperature_k) or temperature_k <= 0:
        raise ValueError("Temperature must be positive and finite.")
    anchors = dict(zip(SHARED_SPECIES, UPPER_SPECIES))
    if len(set(setup.species)) != len(setup.species) or not all(
            anchors[name] in setup.species for name in REFERENCE_ANCHORS):
        raise ValueError("The common gas setup must contain all seven reference anchors.")
    matrix = np.asarray(setup.formula_matrix)
    standard = np.asarray(setup.hvector_func(temperature_k))
    columns = [setup.species.index(anchors[name]) for name in REFERENCE_ANCHORS]
    targets = np.array([reference_standards[name + "_gas"] for name in REFERENCE_ANCHORS])
    gauge = np.linalg.solve(matrix[:, columns].T, targets - standard[columns])
    anchored = standard + matrix.T @ gauge
    if not np.all(np.isfinite(anchored)):
        raise ValueError("The selected common gas standards are unavailable.")
    return anchored, gauge
