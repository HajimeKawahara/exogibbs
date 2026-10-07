"""Condensate equilibrium initialization policies."""

from dataclasses import replace
import math

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from exogibbs.api.condensate import (
    FixedSupportCondensateEquilibriumGrid,
    GridCondensateEquilibriumInitializer,
    regauge_gas_only_warm_start,
)
from exogibbs.api.condensate_equilibrium import (
    FixedSupportCondensateEquilibriumGrid as CompatibilityFixedSupportGrid,
    GridCondensateEquilibriumInitializer as CompatibilityGridInitializer,
    regauge_gas_only_warm_start as compatibility_regauge_gas_only_warm_start,
)
from exogibbs.equilibrium.condensate import lifecycle as _lifecycle
from exogibbs.equilibrium.condensate.setup import CondensateChemicalSetup
from exogibbs.equilibrium.condensate.types import (
    CondensateEquilibriumInit,
    CondensateEquilibriumInitRequest,
    CondensateEquilibriumPoint,
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
    gas_formula_matrix = jnp.asarray(
        [
            [2.0, 2.0],
            [0.0, 0.0],
            [0.0, 1.0],
            [0.0, 0.0],
        ]
    )
    condensate_formula_matrix = jnp.asarray(
        [
            [0.0, 2.0, 0.0],
            [0.0, 0.0, 0.0],
            [1.0, 1.0, 2.0],
            [0.0, 0.0, 0.0],
        ]
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


def _warm_start_setup() -> CondensateChemicalSetup:
    elements = ("H", "C", "O", "e-")
    gas_species = ("H", "C", "HO", "e-")
    gas_formula_matrix = jnp.asarray(
        [
            [1.0, 0.0, 1.0, 0.0],
            [0.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0, -1.0],
        ],
        dtype=jnp.float64,
    )
    condensate_formula_matrix = jnp.asarray(
        [[1.0], [0.0], [1.0], [0.0]],
        dtype=jnp.float64,
    )
    gas_setup = ChemicalSetup(
        formula_matrix=gas_formula_matrix,
        hvector_func=lambda temperature: jnp.zeros((len(gas_species),)),
        elements=elements,
        species=gas_species,
        metadata={"source": "test", "dataset": "gas"},
    )
    condensate_setup = ChemicalSetup(
        formula_matrix=condensate_formula_matrix,
        hvector_func=lambda temperature: jnp.zeros((1,)),
        elements=elements,
        species=("HO[s]",),
        metadata={"source": "test", "dataset": "condensates"},
    )
    return CondensateChemicalSetup(
        gas_setup=gas_setup,
        condensate_setup=condensate_setup,
        formula_matrix=gas_formula_matrix,
        formula_matrix_cond=condensate_formula_matrix,
        gas_species=gas_species,
        condensate_species=("HO[s]",),
        elements=elements,
    )


def test_gas_only_warm_start_preserves_finite_log_ratios_and_floors_absent():
    setup = _warm_start_setup()
    gas_log_amounts = np.asarray([-0.7, -721.0, -2.0, -np.inf])
    inventory = np.asarray([0.8, 0.2, 0.0, 0.0])

    initial = regauge_gas_only_warm_start(
        setup,
        jnp.asarray(gas_log_amounts),
        jnp.asarray(inventory),
    )

    usable = np.asarray([True, True, False, False])
    reference = float(np.max(gas_log_amounts[usable]))
    relative = np.zeros_like(gas_log_amounts)
    relative[usable] = np.exp(gas_log_amounts[usable] - reference)
    represented = float(
        np.sum((np.asarray(setup.formula_matrix) @ relative)[:2])
    )
    shift = math.log(np.sum(inventory[:2]) / represented) - reference
    expected_usable = gas_log_amounts[usable] + shift
    expected_total = float(np.sum(np.exp(expected_usable)))
    relative_floor = math.log(expected_total * 1.0e-300)
    represented_floor = float(
        np.min(expected_usable) + math.log(np.finfo(np.float64).eps)
    )
    expected_floor = min(relative_floor, represented_floor)
    result_logs = np.asarray(initial.gas_ln_n)

    np.testing.assert_array_equal(result_logs[usable], expected_usable)
    np.testing.assert_array_equal(result_logs[~usable], expected_floor)
    assert float(initial.gas_ntot) == np.sum(np.exp(result_logs))
    assert result_logs[1] < relative_floor
    assert compatibility_regauge_gas_only_warm_start is (
        regauge_gas_only_warm_start
    )


def test_gas_only_warm_start_is_source_and_target_gauge_covariant():
    setup = _warm_start_setup()
    gas_logs = jnp.asarray([-0.7, -721.0, -2.0, -np.inf])
    inventory = jnp.asarray([0.8, 0.2, 0.0, 0.0])
    target_scale = 1.0e-12

    base = regauge_gas_only_warm_start(setup, gas_logs, inventory)
    shifted_source = regauge_gas_only_warm_start(
        setup,
        gas_logs + 17.0,
        inventory,
    )
    scaled_target = regauge_gas_only_warm_start(
        setup,
        gas_logs,
        inventory * target_scale,
    )

    np.testing.assert_allclose(
        shifted_source.gas_ln_n,
        base.gas_ln_n,
        rtol=0.0,
        atol=1.0e-12,
    )
    np.testing.assert_allclose(
        np.asarray(scaled_target.gas_ln_n) - math.log(target_scale),
        base.gas_ln_n,
        rtol=0.0,
        atol=1.0e-12,
    )
    assert float(scaled_target.gas_ntot) == pytest.approx(
        float(base.gas_ntot) * target_scale,
        rel=1.0e-12,
    )


def test_gas_only_warm_start_avoids_large_source_gauge_cancellation():
    setup = _warm_start_setup()
    inventory = jnp.asarray([0.8, 0.2, 0.0, 0.0])
    base_logs = jnp.asarray([0.0, -1.0, -np.inf, -np.inf])

    base = regauge_gas_only_warm_start(setup, base_logs, inventory)
    shifted = regauge_gas_only_warm_start(
        setup,
        base_logs + 1.0e15,
        inventory,
    )

    np.testing.assert_array_equal(shifted.gas_ln_n, base.gas_ln_n)
    assert float(shifted.gas_ntot) == float(base.gas_ntot)


def test_gas_only_warm_start_requires_neither_atoms_nor_electron() -> None:
    elements = ("H", "O")
    gas_species = ("H2", "O2", "H2O")
    gas_formula_matrix = jnp.asarray(
        [[2.0, 0.0, 2.0], [0.0, 2.0, 1.0]],
        dtype=jnp.float64,
    )
    condensate_formula_matrix = jnp.asarray(
        [[2.0], [1.0]],
        dtype=jnp.float64,
    )
    gas_setup = ChemicalSetup(
        formula_matrix=gas_formula_matrix,
        hvector_func=lambda temperature: jnp.zeros((len(gas_species),)),
        elements=elements,
        species=gas_species,
        metadata={"source": "test", "dataset": "molecules_only"},
    )
    condensate_setup = ChemicalSetup(
        formula_matrix=condensate_formula_matrix,
        hvector_func=lambda temperature: jnp.zeros((1,)),
        elements=elements,
        species=("H2O[s]",),
        metadata={"source": "test", "dataset": "condensates"},
    )
    setup = CondensateChemicalSetup(
        gas_setup=gas_setup,
        condensate_setup=condensate_setup,
        formula_matrix=gas_formula_matrix,
        formula_matrix_cond=condensate_formula_matrix,
        gas_species=gas_species,
        condensate_species=("H2O[s]",),
        elements=elements,
    )
    source_logs = np.asarray([-2.0, -3.0, -5.0])

    initial = regauge_gas_only_warm_start(
        setup,
        jnp.asarray(source_logs),
        jnp.asarray([0.8, 0.2]),
    )

    result_logs = np.asarray(initial.gas_ln_n)
    shifts = result_logs - source_logs
    assert "e-" not in setup.elements
    assert set(setup.gas_species).isdisjoint(setup.elements)
    np.testing.assert_allclose(shifts, shifts[0], rtol=0.0, atol=1.0e-12)
    assert float(initial.gas_ntot) == pytest.approx(
        np.sum(np.exp(result_logs)),
        rel=1.0e-15,
    )


@pytest.mark.parametrize(
    ("gas_logs", "inventory", "message"),
    (
        ([0.0], [0.8, 0.2, 0.0, 0.0], "one value per gas species"),
        ([np.nan, 0.0, 0.0, 0.0], [0.8, 0.2, 0.0, 0.0], "must not"),
        ([np.inf, 0.0, 0.0, 0.0], [0.8, 0.2, 0.0, 0.0], "must not"),
        ([0.0, 0.0, 0.0, 0.0], [0.8, 0.2, 0.0], "one value per element"),
        (
            [0.0, 0.0, 0.0, 0.0],
            [0.8, -0.2, 0.0, 0.0],
            "finite non-negative",
        ),
        (
            [0.0, 0.0, 0.0, 0.0],
            [0.8, np.nan, 0.0, 0.0],
            "finite non-negative",
        ),
        (
            [0.0, 0.0, 0.0, 0.0],
            [0.8, np.inf, 0.0, 0.0],
            "finite non-negative",
        ),
        ([0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0], "positive"),
        (
            [-np.inf, -np.inf, -np.inf, -np.inf],
            [0.8, 0.2, 0.0, 0.0],
            "no finite species",
        ),
        (
            [-np.inf, -np.inf, -np.inf, 0.0],
            [0.8, 0.0, 0.0, 0.0],
            "cannot be regauged",
        ),
    ),
)
def test_gas_only_warm_start_rejects_invalid_inputs(
    gas_logs,
    inventory,
    message,
):
    with pytest.raises(ValueError, match=message):
        regauge_gas_only_warm_start(
            _warm_start_setup(),
            jnp.asarray(gas_logs),
            jnp.asarray(inventory),
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


def _grid_to_caller_amount_ratio(
    setup,
    element_vector,
    gas_ln_n,
    condensate_amounts,
):
    grid_inventory = (
        setup.formula_matrix @ jnp.exp(jnp.asarray(gas_ln_n))
        + setup.formula_matrix_cond @ jnp.asarray(condensate_amounts)
    )
    return jnp.sum(jnp.asarray(element_vector)[:3]) / jnp.sum(
        grid_inventory[:3]
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
        inventory_bridge_origin=CondensateEquilibriumPoint(
            temperature=800.0,
            pressure=0.2,
            element_inventory=jnp.asarray([1.0, 0.1, 0.02, 0.0]),
        ),
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
    raw_gas_ln_n = jnp.asarray([-3.0, -4.0])
    raw_condensates = jnp.asarray([0.2, 0.0, 0.4])
    amount_ratio = _grid_to_caller_amount_ratio(
        setup,
        jnp.asarray([1.0, 0.1, 0.02, 0.0]),
        raw_gas_ln_n,
        raw_condensates,
    )
    assert jnp.allclose(result.gas_ln_n, raw_gas_ln_n + jnp.log(amount_ratio))
    assert jnp.isclose(result.gas_ntot, 2.5 * amount_ratio)
    assert jnp.allclose(
        result.condensate_amounts,
        raw_condensates * amount_ratio,
    )
    assert result.support_indices == (0, 2)
    assert result.inventory_bridge_origin is None
    assert jnp.allclose(
        result.support_amounts,
        jnp.asarray([0.2, 0.4]) * amount_ratio,
    )
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
    raw_gas_ln_n = jnp.asarray([-1.0, -2.0])
    raw_condensates = jnp.asarray([0.2, 0.0, 0.4])
    amount_ratio = _grid_to_caller_amount_ratio(
        setup,
        element_vector,
        raw_gas_ln_n,
        raw_condensates,
    )
    assert jnp.allclose(result.gas_ln_n, raw_gas_ln_n + jnp.log(amount_ratio))
    assert jnp.isclose(result.gas_ntot, 1.5 * amount_ratio)
    assert jnp.allclose(
        result.condensate_amounts,
        raw_condensates * amount_ratio,
    )
    assert result.support_indices == (0, 2)
    assert jnp.allclose(
        result.support_amounts,
        jnp.asarray([0.2, 0.4]) * amount_ratio,
    )
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

    base_element_vector = jnp.asarray([1.0, 0.1, 0.02, 0.0])

    @jax.jit
    def lookup(temperature, pressure, composition, amount_scale):
        result = initializer(
            CondensateEquilibriumInitRequest(
                setup=setup,
                T=temperature,
                P=pressure,
                b=amount_scale * base_element_vector,
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
        jnp.asarray(1.0),
    )

    raw_gas_ln_n = jnp.zeros((2,))
    raw_condensates = jnp.asarray([1.0, 0.0, 1.0])
    amount_ratio = _grid_to_caller_amount_ratio(
        setup,
        base_element_vector,
        raw_gas_ln_n,
        raw_condensates,
    )
    assert jnp.allclose(gas_ln_n, raw_gas_ln_n + jnp.log(amount_ratio))
    assert jnp.isclose(gas_ntot, amount_ratio)
    assert jnp.allclose(condensates, raw_condensates * amount_ratio)
    assert jnp.allclose(
        support_amounts,
        jnp.asarray([1.0, 1.0]) * amount_ratio,
    )
    scaled = lookup(1000.0, 0.5, 0.0, 1.0e-12)
    assert jnp.allclose(scaled[0], gas_ln_n + jnp.log(1.0e-12))
    assert jnp.isclose(scaled[1] / 1.0e-12, gas_ntot)
    assert jnp.allclose(scaled[2] / 1.0e-12, condensates)
    assert jnp.allclose(scaled[3] / 1.0e-12, support_amounts)
    slope = jax.grad(
        lambda temperature: jnp.sum(
            lookup(temperature, 0.5, 0.0, 1.0)[3]
        )
    )(1000.0)
    gas_inventory_scale = jnp.sum(
        (setup.formula_matrix @ jnp.ones((2,)))[:3]
    )
    condensate_inventory_scale = jnp.sum(
        (
            setup.formula_matrix_cond
            @ jnp.asarray([1.0, 0.0, 1.0])
        )[:3]
    )
    expected_slope = (
        jnp.sum(base_element_vector[:3])
        * 2.0
        * gas_inventory_scale
        / (gas_inventory_scale + condensate_inventory_scale) ** 2
        / 1000.0
    )
    assert jnp.isclose(slope, expected_slope)


def test_grid_condensate_initializer_matches_lifecycle_amount_gauge():
    setup = _condensate_setup()
    initializer = GridCondensateEquilibriumInitializer(
        _fixed_support_grid(setup),
        "test",
    )
    base_element_vector = jnp.asarray([1.0, 0.1, 0.02, 0.0])
    canonical_states = []

    for scale in (1.0e-12, 1.0, 1.0e8):
        caller_inventory = scale * base_element_vector
        amount_scale = _lifecycle._inventory_amount_gauge_scale(
            setup,
            caller_inventory,
        )
        initial = initializer(
            CondensateEquilibriumInitRequest(
                setup=setup,
                T=1000.0,
                P=0.5,
                b=caller_inventory,
                explicit_log10_z_over_z_sun=0.0,
                user_init=CondensateEquilibriumInit(
                    element_potential=jnp.arange(4.0),
                    rho=jnp.asarray([0.3, 0.4]),
                    barrier_epsilon=jnp.asarray(
                        -13.0 + jnp.log(amount_scale)
                    ),
                ),
            )
        )
        canonical_states.append(
            _lifecycle._normalize_condensate_init_amount_gauge(
                initial,
                amount_scale,
            )
        )

    reference = canonical_states[1]
    for state in canonical_states:
        assert jnp.allclose(state.gas_ln_n, reference.gas_ln_n)
        assert jnp.allclose(state.gas_ntot, reference.gas_ntot)
        assert jnp.allclose(
            state.condensate_amounts,
            reference.condensate_amounts,
        )
        assert jnp.allclose(
            jnp.asarray(state.support_amounts),
            jnp.asarray(reference.support_amounts),
        )
        assert jnp.allclose(
            state.element_potential,
            reference.element_potential,
        )
        assert jnp.allclose(state.rho, reference.rho)
        assert jnp.isclose(state.barrier_epsilon, -13.0)


def test_grid_condensate_initializer_is_exported_by_compatibility_api():
    assert CompatibilityGridInitializer is GridCondensateEquilibriumInitializer
    assert (
        CompatibilityFixedSupportGrid
        is FixedSupportCondensateEquilibriumGrid
    )
