"""Condensate equilibrium initialization policies."""

from dataclasses import replace

import jax
import jax.numpy as jnp
import pytest

from exogibbs.api.condensate import (
    FixedSupportCondensateEquilibriumGrid,
    GridCondensateEquilibriumInitializer,
)
from exogibbs.api.condensate_equilibrium import (
    FixedSupportCondensateEquilibriumGrid as CompatibilityFixedSupportGrid,
    GridCondensateEquilibriumInitializer as CompatibilityGridInitializer,
)
from exogibbs.equilibrium.condensate.setup import CondensateChemicalSetup
from exogibbs.equilibrium.condensate.types import (
    CondensateEquilibriumInit,
    CondensateEquilibriumInitRequest,
)
from exogibbs.equilibrium.gas.grid.service import (
    EquilibriumGrid,
    EquilibriumGridInterpolationResult,
    EquilibriumGridMetadata,
    EquilibriumGridOutputs,
    compute_physical_log10_z_over_z_sun,
)
from exogibbs.thermo.models import ChemicalSetup


def _condensate_setup() -> CondensateChemicalSetup:
    elements = ("H", "He", "O", "e-")
    gas_species = ("H2", "H2O")
    condensate_species = ("O[s]", "H2O[s]", "O2[s]")
    gas_formula_matrix = jnp.zeros((len(elements), len(gas_species)))
    condensate_formula_matrix = jnp.zeros(
        (len(elements), len(condensate_species))
    )
    gas_setup = ChemicalSetup(
        formula_matrix=gas_formula_matrix,
        hvector_func=lambda temperature: jnp.zeros((len(gas_species),)),
        elements=elements,
        species=gas_species,
        element_vector_reference=jnp.asarray([1.0, 0.1, 0.01, 0.0]),
        metadata={"source": "test", "dataset": "gas"},
    )
    condensate_setup = ChemicalSetup(
        formula_matrix=condensate_formula_matrix,
        hvector_func=lambda temperature: jnp.zeros(
            (len(condensate_species),)
        ),
        elements=elements,
        species=condensate_species,
        metadata={"source": "test", "dataset": "condensates"},
    )
    return CondensateChemicalSetup(
        gas_setup=gas_setup,
        condensate_setup=condensate_setup,
        formula_matrix=gas_formula_matrix,
        formula_matrix_cond=condensate_formula_matrix,
        gas_species=gas_species,
        condensate_species=condensate_species,
        elements=elements,
    )


def _gas_grid(
    setup: CondensateChemicalSetup,
    *,
    species=None,
) -> EquilibriumGrid:
    gas_species = tuple(species or setup.gas_species)
    species_count = len(gas_species)
    return EquilibriumGrid(
        temperature_axis=jnp.asarray([500.0, 1500.0]),
        pressure_axis=jnp.asarray([0.1, 1.0]),
        log10_z_over_z_sun_axis=jnp.asarray([-1.0, 1.0]),
        outputs=EquilibriumGridOutputs(
            ln_n=jnp.zeros((2, 2, 2, species_count)),
            n=jnp.ones((2, 2, 2, species_count)),
            x=jnp.full(
                (2, 2, 2, species_count),
                1.0 / species_count,
            ),
            ntot=jnp.ones((2, 2, 2)),
        ),
        metadata=EquilibriumGridMetadata(
            preset_name="test",
            preset_setup_metadata=setup.gas_setup.metadata,
            preset_elements=setup.gas_setup.elements,
            preset_species=gas_species,
            source="exogibbs",
            verify_exogibbs_against_fastchem=False,
        ),
    )


def _fixed_support_grid(
    setup: CondensateChemicalSetup,
    *,
    gas_grid=None,
    condensate_amounts=None,
    support_indices=(0, 2),
    condensate_setup_metadata=None,
) -> FixedSupportCondensateEquilibriumGrid:
    active_gas_grid = gas_grid or _gas_grid(setup)
    if condensate_amounts is None:
        values = jnp.asarray([0.2, 0.4])[: len(support_indices)]
        condensate_amounts = jnp.broadcast_to(
            values,
            (2, 2, 2, len(support_indices)),
        )
    if condensate_setup_metadata is None:
        condensate_setup_metadata = EquilibriumGridMetadata.from_setup(
            setup.condensate_setup,
            preset_name="test",
            source="exogibbs",
            verify_exogibbs_against_fastchem=False,
        )
    return FixedSupportCondensateEquilibriumGrid(
        gas_grid=active_gas_grid,
        condensate_amounts=condensate_amounts,
        support_indices=tuple(support_indices),
        condensate_setup_metadata=condensate_setup_metadata,
    )


def test_grid_condensate_initializer_maps_linear_total_and_preserves_user_init(
    monkeypatch,
):
    setup = _condensate_setup()
    grid = _fixed_support_grid(setup)
    captured = {}

    def fake_interpolate(
        grid_in,
        temperature,
        pressure,
        log10_z_over_z_sun,
        *,
        options=None,
    ):
        captured.update(
            grid=grid_in,
            temperature=temperature,
            pressure=pressure,
            composition=log10_z_over_z_sun,
            options=options,
        )
        return EquilibriumGridInterpolationResult(
            ln_n=jnp.asarray([-3.0, -4.0]),
            x=jnp.asarray([0.75, 0.25]),
            ntot=jnp.asarray(2.5),
        )

    monkeypatch.setattr(
        "exogibbs.equilibrium.gas.grid.service.interpolate_equilibrium_grid",
        fake_interpolate,
    )
    user_init = CondensateEquilibriumInit(
        gas_ln_n=jnp.asarray([8.0, 9.0]),
        gas_ntot=jnp.asarray(10.0),
        condensate_amounts=jnp.asarray([0.9, 0.8, 0.7]),
        support_indices=(1,),
        support_amounts=(0.8,),
        element_potential=jnp.arange(4.0),
        rho=jnp.asarray([0.3, 0.4]),
        barrier_epsilon=jnp.asarray(1.0e-4),
    )
    initializer = GridCondensateEquilibriumInitializer(
        grid=grid,
        preset_name="test",
    )

    result = initializer(
        CondensateEquilibriumInitRequest(
            setup=setup,
            T=900.0,
            P=0.3,
            b=jnp.asarray([1.0, 0.1, 0.02, 0.0]),
            user_init=user_init,
            explicit_log10_z_over_z_sun=0.25,
        )
    )

    assert captured["grid"] is grid.gas_grid
    assert captured["temperature"] == 900.0
    assert captured["pressure"] == 0.3
    assert captured["composition"] == 0.25
    assert captured["options"] is None
    assert jnp.allclose(result.gas_ln_n, jnp.asarray([-3.0, -4.0]))
    assert jnp.isclose(result.gas_ntot, 2.5)
    assert jnp.allclose(
        result.condensate_amounts,
        jnp.asarray([0.2, 0.0, 0.4]),
    )
    assert result.support_indices == (0, 2)
    assert jnp.allclose(result.support_amounts, jnp.asarray([0.2, 0.4]))
    assert jnp.allclose(result.element_potential, user_init.element_potential)
    assert jnp.allclose(result.rho, user_init.rho)
    assert jnp.isclose(result.barrier_epsilon, user_init.barrier_epsilon)


def test_grid_condensate_initializer_infers_coordinate_and_preserves_previous(
    monkeypatch,
):
    setup = _condensate_setup()
    grid = _fixed_support_grid(setup)
    element_vector = jnp.asarray([1.0, 0.1, 0.02, 0.0])
    captured = {}

    def fake_interpolate(
        grid_in,
        temperature,
        pressure,
        log10_z_over_z_sun,
        *,
        options=None,
    ):
        del grid_in, temperature, pressure, options
        captured["composition"] = log10_z_over_z_sun
        return EquilibriumGridInterpolationResult(
            ln_n=jnp.asarray([-1.0, -2.0]),
            x=jnp.asarray([0.6, 0.4]),
            ntot=jnp.asarray(1.5),
        )

    monkeypatch.setattr(
        "exogibbs.equilibrium.gas.grid.service.interpolate_equilibrium_grid",
        fake_interpolate,
    )
    previous = CondensateEquilibriumInit(
        condensate_amounts=jnp.asarray([0.9, 0.8, 0.7]),
        support_indices=(1,),
        support_amounts=(0.8,),
        element_potential=jnp.arange(4.0),
        rho=jnp.asarray([0.3, 0.4]),
        barrier_epsilon=jnp.asarray(1.0e-4),
    )

    result = GridCondensateEquilibriumInitializer(grid, "test")(
        CondensateEquilibriumInitRequest(
            setup=setup,
            T=1000.0,
            P=0.5,
            b=element_vector,
            previous_solution=previous,
        )
    )

    expected = compute_physical_log10_z_over_z_sun(
        setup.gas_setup,
        element_vector,
    )
    assert jnp.isclose(captured["composition"], expected)
    assert jnp.allclose(result.gas_ln_n, jnp.asarray([-1.0, -2.0]))
    assert jnp.isclose(result.gas_ntot, 1.5)
    assert jnp.allclose(
        result.condensate_amounts,
        jnp.asarray([0.2, 0.0, 0.4]),
    )
    assert result.support_indices == (0, 2)
    assert jnp.allclose(result.support_amounts, jnp.asarray([0.2, 0.4]))
    assert jnp.allclose(result.element_potential, previous.element_potential)
    assert jnp.allclose(result.rho, previous.rho)
    assert jnp.isclose(result.barrier_epsilon, previous.barrier_epsilon)


def test_grid_condensate_initializer_rejects_incompatible_gas_grid():
    setup = _condensate_setup()
    grid = _fixed_support_grid(
        setup,
        gas_grid=_gas_grid(setup, species=("H2",)),
    )
    request = CondensateEquilibriumInitRequest(
        setup=setup,
        T=1000.0,
        P=0.5,
        b=jnp.asarray([1.0, 0.1, 0.02, 0.0]),
        explicit_log10_z_over_z_sun=0.0,
    )

    with pytest.raises(ValueError, match="species mismatch"):
        GridCondensateEquilibriumInitializer(grid, "test")(request)


def test_grid_condensate_initializer_requires_fixed_support_grid():
    setup = _condensate_setup()
    request = CondensateEquilibriumInitRequest(
        setup=setup,
        T=1000.0,
        P=0.5,
        b=jnp.asarray([1.0, 0.1, 0.02, 0.0]),
        explicit_log10_z_over_z_sun=0.0,
    )

    with pytest.raises(TypeError, match="FixedSupportCondensateEquilibriumGrid"):
        GridCondensateEquilibriumInitializer(_gas_grid(setup), "test")(
            request
        )


def test_grid_condensate_initializer_rejects_condensate_setup_mismatch():
    setup = _condensate_setup()
    grid = _fixed_support_grid(setup)
    incompatible_condensate_setup = replace(
        setup.condensate_setup,
        metadata={"source": "different", "dataset": "condensates"},
    )
    incompatible_setup = replace(
        setup,
        condensate_setup=incompatible_condensate_setup,
    )
    request = CondensateEquilibriumInitRequest(
        setup=incompatible_setup,
        T=1000.0,
        P=0.5,
        b=jnp.asarray([1.0, 0.1, 0.02, 0.0]),
        explicit_log10_z_over_z_sun=0.0,
    )

    with pytest.raises(ValueError, match="setup metadata mismatch"):
        GridCondensateEquilibriumInitializer(grid, "test")(request)


def test_grid_condensate_initializer_rejects_condensate_species_mismatch():
    setup = _condensate_setup()
    grid = _fixed_support_grid(setup)
    reordered_species = ("H2O[s]", "O[s]", "O2[s]")
    incompatible_setup = replace(
        setup,
        condensate_setup=replace(
            setup.condensate_setup,
            species=reordered_species,
        ),
        condensate_species=reordered_species,
    )
    request = CondensateEquilibriumInitRequest(
        setup=incompatible_setup,
        T=1000.0,
        P=0.5,
        b=jnp.asarray([1.0, 0.1, 0.02, 0.0]),
        explicit_log10_z_over_z_sun=0.0,
    )

    with pytest.raises(ValueError, match="species mismatch"):
        GridCondensateEquilibriumInitializer(grid, "test")(request)


def test_grid_condensate_initializer_rejects_amount_shape_mismatch():
    setup = _condensate_setup()
    grid = _fixed_support_grid(
        setup,
        condensate_amounts=jnp.ones((2, 2, 2, 3)),
    )
    request = CondensateEquilibriumInitRequest(
        setup=setup,
        T=1000.0,
        P=0.5,
        b=jnp.asarray([1.0, 0.1, 0.02, 0.0]),
        explicit_log10_z_over_z_sun=0.0,
    )

    with pytest.raises(ValueError, match="amounts shape mismatch"):
        GridCondensateEquilibriumInitializer(grid, "test")(request)


def test_grid_condensate_initializer_is_trace_safe():
    setup = _condensate_setup()
    temperatures = jnp.asarray([500.0, 1500.0])
    pressures = jnp.asarray([0.1, 1.0])
    compositions = jnp.asarray([-1.0, 1.0])
    temperature_values = temperatures[:, None, None, None]
    condensate_amounts = jnp.broadcast_to(
        temperature_values / 1000.0,
        (2, 2, 2, 2),
    )
    grid = _fixed_support_grid(
        setup,
        condensate_amounts=condensate_amounts,
    )
    initializer = GridCondensateEquilibriumInitializer(grid, "test")

    @jax.jit
    def lookup(temperature, pressure, composition):
        result = initializer(
            CondensateEquilibriumInitRequest(
                setup=setup,
                T=temperature,
                P=pressure,
                b=jnp.asarray([1.0, 0.1, 0.02, 0.0]),
                explicit_log10_z_over_z_sun=composition,
            )
        )
        return (
            result.gas_ln_n,
            result.gas_ntot,
            result.condensate_amounts,
            result.support_amounts,
        )

    gas_ln_n, gas_ntot, condensates, support_amounts = lookup(
        jnp.asarray(1000.0),
        jnp.asarray(0.5),
        jnp.asarray(0.0),
    )

    assert jnp.allclose(gas_ln_n, jnp.zeros((2,)))
    assert jnp.isclose(gas_ntot, 1.0)
    assert jnp.allclose(condensates, jnp.asarray([1.0, 0.0, 1.0]))
    assert jnp.allclose(support_amounts, jnp.asarray([1.0, 1.0]))
    slope = jax.grad(
        lambda temperature: jnp.sum(
            lookup(temperature, 0.5, 0.0)[3]
        )
    )(1000.0)
    assert jnp.isclose(slope, 2.0e-3)


def test_grid_condensate_initializer_is_exported_by_compatibility_api():
    assert CompatibilityGridInitializer is GridCondensateEquilibriumInitializer
    assert (
        CompatibilityFixedSupportGrid
        is FixedSupportCondensateEquilibriumGrid
    )
