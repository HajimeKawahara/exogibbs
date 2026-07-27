"""Validated thermochemical setup for condensate equilibrium."""

from dataclasses import dataclass

import jax
import jax.numpy as jnp
import numpy as np

from exogibbs.thermo.models import ChemicalSetup


Array = jax.Array


@dataclass(frozen=True)
class CondensateChemicalSetup:
    """Gas and condensate thermochemistry bundle for condensate equilibrium."""

    gas_setup: ChemicalSetup
    condensate_setup: ChemicalSetup
    formula_matrix: Array
    formula_matrix_cond: Array
    gas_species: tuple[str, ...]
    condensate_species: tuple[str, ...]
    elements: tuple[str, ...]


def validate_condensate_chemical_setup(setup: CondensateChemicalSetup) -> None:
    """Validate gas-condensate setup compatibility."""

    if not isinstance(setup.gas_setup, ChemicalSetup):
        raise TypeError("gas_setup must be a ChemicalSetup.")
    if not isinstance(setup.condensate_setup, ChemicalSetup):
        raise TypeError("condensate_setup must be a ChemicalSetup.")
    if setup.gas_setup.elements is None:
        raise ValueError("gas_setup.elements is required for condensate equilibrium.")
    if setup.condensate_setup.elements is None:
        raise ValueError(
            "condensate_setup.elements is required for condensate equilibrium."
        )
    if tuple(setup.gas_setup.elements) != tuple(setup.condensate_setup.elements):
        raise ValueError("gas and condensate element orders must match.")
    formula_matrix = jnp.asarray(setup.formula_matrix)
    formula_matrix_cond = jnp.asarray(setup.formula_matrix_cond)
    if formula_matrix.ndim != 2:
        raise ValueError("formula_matrix must be a two-dimensional array.")
    if formula_matrix_cond.ndim != 2:
        raise ValueError("formula_matrix_cond must be a two-dimensional array.")
    if formula_matrix.shape[0] != formula_matrix_cond.shape[0]:
        raise ValueError(
            "gas and condensate formula matrices must have the same element count."
        )
    if formula_matrix.shape[0] != len(setup.elements):
        raise ValueError("elements length must match formula matrix rows.")
    if formula_matrix.shape[1] != len(setup.gas_species):
        raise ValueError("gas_species length must match formula_matrix columns.")
    if formula_matrix_cond.shape[1] != len(setup.condensate_species):
        raise ValueError(
            "condensate_species length must match formula_matrix_cond columns."
        )
    if tuple(setup.elements) != tuple(setup.gas_setup.elements):
        raise ValueError("elements must match gas_setup.elements exactly.")
    if setup.gas_setup.species is None:
        raise ValueError("gas_setup.species is required for condensate equilibrium.")
    if tuple(setup.gas_species) != tuple(setup.gas_setup.species):
        raise ValueError("gas_species must match gas_setup.species exactly.")
    if setup.condensate_setup.species is None:
        raise ValueError(
            "condensate_setup.species is required for condensate equilibrium."
        )
    if tuple(setup.condensate_species) != tuple(
        setup.condensate_setup.species
    ):
        raise ValueError(
            "condensate_species must match condensate_setup.species exactly."
        )
    if not np.array_equal(
        np.asarray(formula_matrix),
        np.asarray(setup.gas_setup.formula_matrix),
    ):
        raise ValueError(
            "formula_matrix must match gas_setup.formula_matrix exactly."
        )
    if not np.array_equal(
        np.asarray(formula_matrix_cond),
        np.asarray(setup.condensate_setup.formula_matrix),
    ):
        raise ValueError(
            "formula_matrix_cond must match "
            "condensate_setup.formula_matrix exactly."
        )
    validity_upper = condensate_temperature_validity_upper(setup)
    if validity_upper is not None:
        validity = np.asarray(validity_upper, dtype=float)
        if validity.ndim != 1 or validity.shape[0] != len(
            setup.condensate_species
        ):
            raise ValueError(
                "temperature_validity_upper must have one value per "
                "condensate species."
            )
        if not np.all(np.isfinite(validity)):
            raise ValueError(
                "temperature_validity_upper values must be finite."
            )


def build_condensate_chemical_setup(
    *,
    gas_setup: ChemicalSetup,
    condensate_setup: ChemicalSetup,
) -> CondensateChemicalSetup:
    """Build and validate a gas-condensate thermochemical bundle."""

    if gas_setup.elements is None:
        raise ValueError("gas_setup.elements is required for condensate equilibrium.")
    if gas_setup.species is None:
        raise ValueError("gas_setup.species is required for condensate equilibrium.")
    if condensate_setup.elements is None:
        raise ValueError(
            "condensate_setup.elements is required for condensate equilibrium."
        )
    if condensate_setup.species is None:
        raise ValueError(
            "condensate_setup.species is required for condensate equilibrium."
        )
    setup = CondensateChemicalSetup(
        gas_setup=gas_setup,
        condensate_setup=condensate_setup,
        formula_matrix=jnp.asarray(gas_setup.formula_matrix),
        formula_matrix_cond=jnp.asarray(condensate_setup.formula_matrix),
        gas_species=tuple(gas_setup.species),
        condensate_species=tuple(condensate_setup.species),
        elements=tuple(gas_setup.elements),
    )
    validate_condensate_chemical_setup(setup)
    return setup


def condensate_temperature_validity_upper(
    setup: CondensateChemicalSetup,
):
    """Return typed phase-validity bounds with legacy metadata fallback."""

    typed_value = getattr(
        setup.condensate_setup,
        "temperature_validity_upper",
        None,
    )
    if typed_value is not None:
        return typed_value
    metadata = getattr(setup.condensate_setup, "metadata", None) or {}
    return metadata.get("temperature_validity_upper")
