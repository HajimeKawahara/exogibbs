"""Tests for canonical magma-gas chemistry preparation."""

from dataclasses import replace

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from exogibbs.experimental.magma_gas.setup import (
    CANONICAL_ELEMENTS,
    CANONICAL_SPECIES,
    prepare_meltyq_chemistry,
)
from exogibbs.thermo.models import ChemicalSetup


_SOURCE_ELEMENTS = ("O", "H", "S", "He", "N", "C")
_SOURCE_SPECIES = (
    "CO2",
    "H2O",
    "SO2",
    "NH3",
    "H2",
    "CH4",
    "He",
    "CO",
    "N2",
    "O2",
)
_FORMULAE = {
    "H2": {"H": 2},
    "He": {"He": 1},
    "O2": {"O": 2},
    "H2O": {"H": 2, "O": 1},
    "CO": {"C": 1, "O": 1},
    "CO2": {"C": 1, "O": 2},
    "CH4": {"C": 1, "H": 4},
    "N2": {"N": 2},
    "NH3": {"N": 1, "H": 3},
    "SO2": {"S": 1, "O": 2},
}


def _source_setup(*, renamed: bool = False) -> ChemicalSetup:
    species = tuple(
        f"raw_{name}" if renamed and name in CANONICAL_SPECIES else name
        for name in _SOURCE_SPECIES
    )
    formula_matrix = np.zeros((len(_SOURCE_ELEMENTS), len(species)))
    for species_index, source_name in enumerate(species):
        formula_name = source_name.removeprefix("raw_")
        for element, count in _FORMULAE[formula_name].items():
            formula_matrix[_SOURCE_ELEMENTS.index(element), species_index] = count

    source_count = len(species)

    def hvector_func(temperature):
        temperature = jnp.asarray(temperature)
        return temperature[..., None] + jnp.arange(source_count)

    return ChemicalSetup(
        formula_matrix=jnp.asarray(formula_matrix),
        hvector_func=hvector_func,
        elements=_SOURCE_ELEMENTS,
        species=species,
        element_vector_reference=jnp.arange(1, len(_SOURCE_ELEMENTS) + 1),
        metadata={"source": "test"},
        temperature_validity_upper=tuple(
            1000.0 + index for index in range(source_count)
        ),
    )


def _lnphi_func(temperature, pressure_bar, mole_fractions):
    assert mole_fractions is None
    return (
        jnp.asarray(temperature)
        + jnp.asarray(pressure_bar)
        + 2.0 * jnp.arange(len(_SOURCE_SPECIES))
    )


def test_prepare_reorders_and_subsets_all_species_outputs() -> None:
    source = _source_setup()
    prepared = prepare_meltyq_chemistry(source, lnphi_func=_lnphi_func)
    species_indices = tuple(_SOURCE_SPECIES.index(name) for name in CANONICAL_SPECIES)
    element_indices = tuple(_SOURCE_ELEMENTS.index(name) for name in CANONICAL_ELEMENTS)

    assert prepared.setup.elements == CANONICAL_ELEMENTS
    assert prepared.setup.species == CANONICAL_SPECIES
    assert prepared.source_species_indices == species_indices
    assert prepared.source_element_indices == element_indices
    assert np.linalg.matrix_rank(np.asarray(prepared.setup.formula_matrix)) == 5
    assert len(CANONICAL_SPECIES) - 5 == 4
    np.testing.assert_allclose(
        prepared.setup.hvector_func(3.0),
        3.0 + np.asarray(species_indices),
    )
    assert prepared.lnphi_func is not None
    np.testing.assert_allclose(
        prepared.lnphi_func(3.0, 7.0, None),
        10.0 + 2.0 * np.asarray(species_indices),
    )
    np.testing.assert_allclose(
        prepared.setup.element_vector_reference,
        np.arange(1, len(_SOURCE_ELEMENTS) + 1)[np.asarray(element_indices)],
    )
    assert prepared.setup.temperature_validity_upper == tuple(
        source.temperature_validity_upper[index] for index in species_indices
    )
    assert prepared.setup.metadata is source.metadata


def test_prepared_callables_are_jittable_and_differentiable() -> None:
    prepared = prepare_meltyq_chemistry(
        _source_setup(),
        lnphi_func=_lnphi_func,
    )
    assert prepared.lnphi_func is not None

    hvector = jax.jit(prepared.setup.hvector_func)(2.0)
    lnphi = jax.jit(
        lambda temperature: prepared.lnphi_func(temperature, 4.0, None)
    )(2.0)
    gradient = jax.grad(
        lambda temperature: jnp.sum(prepared.setup.hvector_func(temperature))
    )(2.0)

    assert hvector.shape == (len(CANONICAL_SPECIES),)
    assert lnphi.shape == (len(CANONICAL_SPECIES),)
    np.testing.assert_allclose(gradient, len(CANONICAL_SPECIES))


def test_prepared_hvector_matches_formula_and_temperature_dtype() -> None:
    source = replace(
        _source_setup(),
        formula_matrix=jnp.asarray(
            _source_setup().formula_matrix,
            dtype=jnp.float32,
        ),
        hvector_func=lambda temperature: jnp.arange(
            len(_SOURCE_SPECIES),
            dtype=jnp.float64,
        ),
    )
    prepared = prepare_meltyq_chemistry(source)

    calculated = prepared.setup.hvector_func(jnp.asarray(1700.0, jnp.float32))

    assert prepared.setup.formula_matrix.dtype == jnp.float32
    assert calculated.dtype == jnp.float32


def test_species_map_selects_explicit_raw_names() -> None:
    source = _source_setup(renamed=True)
    species_map = {name: f"raw_{name}" for name in CANONICAL_SPECIES}

    prepared = prepare_meltyq_chemistry(source, species_map=species_map)

    assert prepared.setup.species == CANONICAL_SPECIES
    assert prepared.source_species_indices == tuple(
        source.species.index(species_map[name]) for name in CANONICAL_SPECIES
    )
    assert prepared.lnphi_func is None


@pytest.mark.parametrize(
    ("row_name", "species_name", "message"),
    [
        ("C", "CO", "stoichiometry"),
        ("S", "H2", "outside"),
    ],
)
def test_prepare_rejects_invalid_selected_formulae(
    row_name: str,
    species_name: str,
    message: str,
) -> None:
    source = _source_setup()
    row = source.elements.index(row_name)
    column = source.species.index(species_name)
    invalid_formula = source.formula_matrix.at[row, column].add(1)

    with pytest.raises(ValueError, match=message):
        prepare_meltyq_chemistry(
            replace(source, formula_matrix=invalid_formula)
        )


def test_prepare_rejects_non_unique_species_mapping() -> None:
    with pytest.raises(ValueError, match="Mapped species names must be unique"):
        prepare_meltyq_chemistry(
            _source_setup(),
            species_map={"H2": "H2O"},
        )


def test_prepared_callables_validate_source_output_shapes() -> None:
    source = replace(
        _source_setup(),
        hvector_func=lambda temperature: jnp.zeros(2),
    )
    prepared = prepare_meltyq_chemistry(
        source,
        lnphi_func=lambda temperature, pressure_bar, mole_fractions: jnp.zeros(2),
    )

    with pytest.raises(ValueError, match="expected size 10"):
        prepared.setup.hvector_func(1000.0)
    assert prepared.lnphi_func is not None
    with pytest.raises(ValueError, match="expected size 10"):
        prepared.lnphi_func(1000.0, 1.0, None)
    with pytest.raises(ValueError, match="pure-component mode"):
        prepared.lnphi_func(1000.0, 1.0, jnp.ones(9) / 9.0)
