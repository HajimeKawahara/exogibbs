"""Chemical setup preparation for the built-in MELTYQ model."""

from dataclasses import dataclass
from typing import Mapping, Optional, Tuple, Union

import jax.numpy as jnp
import numpy as np

from exogibbs.thermo.fugacity import LogFugacityCoefficientFunction
from exogibbs.thermo.models import ChemicalSetup


Array = jnp.ndarray
Scalar = Union[float, Array]

MELTYQ_ELEMENTS = ("H", "C", "O", "N", "He")
MELTYQ_SPECIES = (
    "H2",
    "He",
    "O2",
    "H2O",
    "CO",
    "CO2",
    "CH4",
    "N2",
    "NH3",
)

_MELTYQ_FORMULA_MATRIX = np.asarray(
    [
        [2, 0, 0, 2, 0, 0, 4, 0, 3],
        [0, 0, 0, 0, 1, 1, 1, 0, 0],
        [0, 0, 2, 1, 1, 2, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 2, 1],
        [0, 1, 0, 0, 0, 0, 0, 0, 0],
    ],
    dtype=float,
)


@dataclass(frozen=True)
class PreparedMagmaGasChemistry:
    """Canonical MELTYQ gas chemistry and its fugacity model."""

    setup: ChemicalSetup
    lnphi_func: Optional[LogFugacityCoefficientFunction]
    source_species_indices: Tuple[int, ...]
    source_element_indices: Tuple[int, ...]


PreparedMeltyqChemistry = PreparedMagmaGasChemistry


def _require_unique(names: Tuple[str, ...], label: str) -> None:
    if len(set(names)) != len(names):
        duplicates = sorted({name for name in names if names.count(name) > 1})
        raise ValueError(f"{label} names must be unique; duplicates: {duplicates}.")


def _subset_last_axis(
    values: Array,
    indices: Array,
    expected_size: int,
    source_name: str,
) -> Array:
    array = jnp.asarray(values)
    if array.ndim == 0 or array.shape[-1] != expected_size:
        raise ValueError(
            f"{source_name} must return one value per source species on its "
            f"last axis; expected size {expected_size}, got shape {array.shape}."
        )
    return jnp.take(array, indices, axis=-1)


def prepare_meltyq_chemistry(
    chemical_setup: ChemicalSetup,
    *,
    lnphi_func: Optional[LogFugacityCoefficientFunction] = None,
    species_map: Optional[Mapping[str, str]] = None,
) -> PreparedMagmaGasChemistry:
    """Reduce a full gas setup to the canonical MELTYQ chemistry.

    ``species_map`` maps MELTYQ species names to names in
    ``chemical_setup.species``. Missing entries retain the MELTYQ name.
    Preparation and validation run on the host; returned thermochemical
    callables remain JAX-compatible.
    """

    formula_matrix = np.asarray(chemical_setup.formula_matrix)
    if formula_matrix.ndim != 2:
        raise ValueError("chemical_setup.formula_matrix must be two-dimensional.")
    element_count, species_count = formula_matrix.shape

    if chemical_setup.elements is None:
        raise ValueError("chemical_setup.elements is required.")
    if chemical_setup.species is None:
        raise ValueError("chemical_setup.species is required.")
    source_elements = tuple(chemical_setup.elements)
    source_species = tuple(chemical_setup.species)
    if len(source_elements) != element_count:
        raise ValueError(
            "chemical_setup.elements must match the formula-matrix rows: "
            f"expected {element_count}, got {len(source_elements)}."
        )
    if len(source_species) != species_count:
        raise ValueError(
            "chemical_setup.species must match the formula-matrix columns: "
            f"expected {species_count}, got {len(source_species)}."
        )
    _require_unique(source_elements, "Element")
    _require_unique(source_species, "Species")

    supplied_map = dict(species_map or {})
    unknown_names = sorted(set(supplied_map) - set(MELTYQ_SPECIES))
    if unknown_names:
        raise ValueError(
            "species_map contains unknown MELTYQ species: "
            f"{unknown_names}."
        )
    mapped_species = tuple(
        supplied_map.get(name, name) for name in MELTYQ_SPECIES
    )
    if not all(isinstance(name, str) and name for name in mapped_species):
        raise ValueError("species_map values must be non-empty source names.")
    _require_unique(mapped_species, "Mapped species")

    missing_elements = [
        name for name in MELTYQ_ELEMENTS if name not in source_elements
    ]
    if missing_elements:
        raise ValueError(
            "chemical_setup is missing MELTYQ elements: "
            f"{missing_elements}."
        )
    missing_species = [name for name in mapped_species if name not in source_species]
    if missing_species:
        raise ValueError(
            "chemical_setup is missing mapped source species: "
            f"{missing_species}."
        )

    element_indices = tuple(source_elements.index(name) for name in MELTYQ_ELEMENTS)
    species_indices = tuple(source_species.index(name) for name in mapped_species)
    reduced_formula = formula_matrix[np.ix_(element_indices, species_indices)]

    extra_element_indices = tuple(
        index
        for index, name in enumerate(source_elements)
        if name not in MELTYQ_ELEMENTS
    )
    if extra_element_indices:
        extra_stoichiometry = formula_matrix[
            np.ix_(extra_element_indices, species_indices)
        ]
        if np.any(extra_stoichiometry != 0):
            raise ValueError(
                "Selected magma-gas species must not contain elements outside "
                f"{MELTYQ_ELEMENTS}."
            )

    if not np.array_equal(reduced_formula, _MELTYQ_FORMULA_MATRIX):
        mismatched = [
            MELTYQ_SPECIES[index]
            for index in range(len(MELTYQ_SPECIES))
            if not np.array_equal(
                reduced_formula[:, index],
                _MELTYQ_FORMULA_MATRIX[:, index],
            )
        ]
        raise ValueError(
            "Mapped species stoichiometry does not match MELTYQ formulae: "
            f"{mismatched}."
        )

    rank = int(np.linalg.matrix_rank(reduced_formula))
    reaction_count = len(MELTYQ_SPECIES) - rank
    if rank != len(MELTYQ_ELEMENTS) or reaction_count != 4:
        raise ValueError(
            "MELTYQ magma-gas chemistry must have formula-matrix rank 5 "
            f"and four independent reactions; got rank {rank} and "
            f"{reaction_count} reactions."
        )

    reduced_formula_array = jnp.asarray(
        reduced_formula,
        dtype=jnp.result_type(reduced_formula, jnp.float32),
    )
    source_index_array = jnp.asarray(species_indices, dtype=jnp.int32)

    def reduced_hvector(temperature: Scalar) -> Array:
        values = _subset_last_axis(
            chemical_setup.hvector_func(temperature),
            source_index_array,
            species_count,
            "chemical_setup.hvector_func",
        )
        dtype = jnp.result_type(
            reduced_formula_array,
            jnp.asarray(temperature),
            jnp.float32,
        )
        return jnp.asarray(values, dtype=dtype)

    reduced_lnphi_func: Optional[LogFugacityCoefficientFunction]
    if lnphi_func is None:
        reduced_lnphi_func = None
    else:

        def reduced_lnphi_func(
            temperature: Scalar,
            pressure_bar: Scalar,
            mole_fractions: Optional[Array],
        ) -> Array:
            if mole_fractions is not None:
                raise ValueError(
                    "Prepared magma-gas fugacity coefficients currently support "
                    "pure-component mode only (mole_fractions=None)."
                )
            return _subset_last_axis(
                lnphi_func(temperature, pressure_bar, None),
                source_index_array,
                species_count,
                "lnphi_func",
            )

    element_vector_reference = chemical_setup.element_vector_reference
    if element_vector_reference is not None:
        reference = jnp.asarray(element_vector_reference)
        if reference.shape != (element_count,):
            raise ValueError(
                "chemical_setup.element_vector_reference must have one value "
                f"per source element; expected {(element_count,)}, got "
                f"{reference.shape}."
            )
        element_index_array = jnp.asarray(element_indices, dtype=jnp.int32)
        element_vector_reference = jnp.take(reference, element_index_array)

    temperature_validity_upper = chemical_setup.temperature_validity_upper
    if temperature_validity_upper is not None:
        if len(temperature_validity_upper) != species_count:
            raise ValueError(
                "chemical_setup.temperature_validity_upper must have one value "
                f"per source species; expected {species_count}, got "
                f"{len(temperature_validity_upper)}."
            )
        temperature_validity_upper = tuple(
            temperature_validity_upper[index] for index in species_indices
        )

    reduced_setup = ChemicalSetup(
        formula_matrix=reduced_formula_array,
        hvector_func=reduced_hvector,
        elements=MELTYQ_ELEMENTS,
        species=MELTYQ_SPECIES,
        element_vector_reference=element_vector_reference,
        metadata=chemical_setup.metadata,
        temperature_validity_upper=temperature_validity_upper,
    )
    return PreparedMagmaGasChemistry(
        setup=reduced_setup,
        lnphi_func=reduced_lnphi_func,
        source_species_indices=species_indices,
        source_element_indices=element_indices,
    )


CANONICAL_ELEMENTS = MELTYQ_ELEMENTS
CANONICAL_SPECIES = MELTYQ_SPECIES


__all__ = (
    "CANONICAL_ELEMENTS",
    "CANONICAL_SPECIES",
    "MELTYQ_ELEMENTS",
    "MELTYQ_SPECIES",
    "PreparedMagmaGasChemistry",
    "PreparedMeltyqChemistry",
    "prepare_meltyq_chemistry",
)
