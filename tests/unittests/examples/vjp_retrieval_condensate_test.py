"""Focused, dependency-light tests for the condensate NUTS example."""

from __future__ import annotations

import ast
import importlib.util
from pathlib import Path
from types import SimpleNamespace
import sys

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from exogibbs.equilibrium.condensate.fixed_support.types import (
    DifferentiableFixedSupportResult,
)
from exogibbs.equilibrium.condensate.types import CondensateEquilibriumInit
from exogibbs.equilibrium.gas.types import EquilibriumInit


REPOSITORY_ROOT = Path(__file__).resolve().parents[3]
EXAMPLE_PATH = (
    REPOSITORY_ROOT
    / "examples"
    / "retrievals"
    / "exojax_nuts_condensate_fixed_support.py"
)
GRID_EXAMPLE_PATH = (
    REPOSITORY_ROOT
    / "examples"
    / "retrievals"
    / "exojax_nuts_condensate_grid.py"
)


@pytest.fixture(scope="module")
def condensate_example():
    spec = importlib.util.spec_from_file_location(
        "exojax_nuts_condensate_fixed_support_test_module",
        EXAMPLE_PATH,
    )
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_example_is_main_guarded_and_states_local_contract():
    source = EXAMPLE_PATH.read_text(encoding="utf-8")
    tree = ast.parse(source, filename=str(EXAMPLE_PATH))

    compile(source, str(EXAMPLE_PATH), "exec")
    assert "minimize_gibbs_fixed_support" in source
    assert "minimize_gibbs(" in source
    assert "does not differentiate" in source
    assert "rainout" in source
    assert "C/O = 2" in source
    assert "full-catalog equilibrium is not supported" in source
    assert "fixed_gas_log_amounts_init=fixed_gas_log_amounts_init" in source

    guarded_main = [
        node
        for node in tree.body
        if isinstance(node, ast.If)
        and isinstance(node.test, ast.Compare)
        and any(
            isinstance(statement, ast.Expr)
            and isinstance(statement.value, ast.Call)
            and isinstance(statement.value.func, ast.Name)
            and statement.value.func.id == "main"
            for statement in node.body
        )
    ]
    assert len(guarded_main) == 1


def test_condensate_grid_wrapper_selects_shared_grid_runner():
    source = GRID_EXAMPLE_PATH.read_text(encoding="utf-8")
    common_source = EXAMPLE_PATH.read_text(encoding="utf-8")

    compile(source, str(GRID_EXAMPLE_PATH), "exec")
    assert "use_grid_initializer=True" in source
    assert "run_condensate_demo" in source
    assert "use_grid_initializer=False" in common_source
    assert "interpolate_graphite_grid_initial_values" in common_source
    assert "GridCondensateEquilibriumInitializer" in common_source


def test_pressure_and_temperature_profiles_match_exojax_convention(
    condensate_example,
):
    module = condensate_example
    pressure = module.pressure_profile(8)
    temperature = module.powerlaw_temperature(pressure, 1160.0, 0.03)

    np.testing.assert_allclose(pressure, np.logspace(-3.0, 1.0, 8))
    np.testing.assert_allclose(temperature, 1160.0 * pressure**0.03)
    assert float(pressure[0]) == pytest.approx(1.0e-3)
    assert float(pressure[-1]) == pytest.approx(10.0)
    assert bool(jnp.all(jnp.diff(pressure) > 0.0))


def test_carbon_oxygen_scale_is_ratio_preserving_and_differentiable(
    condensate_example,
):
    module = condensate_example
    reference = jnp.asarray([2.0, 1.0, 7.0])

    def total(log_scale):
        scaled = module.scale_carbon_and_oxygen(
            reference,
            carbon_index=0,
            oxygen_index=1,
            log_co_scale=log_scale,
        )
        return jnp.sum(scaled[:2])

    scaled = module.scale_carbon_and_oxygen(reference, 0, 1, 0.25)
    gradient = jax.grad(total)(0.0)

    assert float(scaled[0] / scaled[1]) == pytest.approx(2.0)
    assert float(gradient) == pytest.approx(3.0 * np.log(10.0))


def test_condensate_model_declares_a_graphite_only_reduced_catalog(
    condensate_example,
):
    module = condensate_example
    setup = module.graphite_only_chemical_setup()
    metadata = setup.condensate_setup.metadata

    assert setup.condensate_species == (module.GRAPHITE_SPECIES,)
    assert setup.formula_matrix_cond.shape[1] == 1
    assert metadata["model_scope"] == module.CONDENSATE_MODEL_SCOPE
    assert metadata["condensate_catalog_mode"] == "reduced_explicit"
    assert metadata["reduced_condensate_catalog"] is True
    assert metadata["source_condensate_catalog_count"] > 1
    assert metadata["excluded_condensate_species_count"] == (
        metadata["source_condensate_catalog_count"] - 1
    )
    assert metadata["full_catalog_equilibrium_claimed"] is False
    assert metadata["full_catalog_support_closure_checked"] is False
    assert setup.condensate_setup.temperature_validity_upper == (6000.0,)
    assert metadata["temperature_validity_upper"] == (6000.0,)
    assert np.asarray(setup.condensate_setup.hvector_func(1160.0)).shape == (1,)
    carbon = setup.elements.index("C")
    oxygen = setup.elements.index("O")
    reference = setup.gas_setup.element_vector_reference
    assert float(reference[carbon] / reference[oxygen]) == pytest.approx(2.0)


def test_grid_initial_values_are_interpolated_for_each_layer(
    condensate_example,
):
    module = condensate_example

    class FakeGasGridInitializer:
        def __call__(self, request):
            return EquilibriumInit(
                ln_nk=jnp.asarray([request.T, request.P]),
                ln_ntot=jnp.log(jnp.asarray(2.0)),
            )

    class FakeFixedGridInitializer:
        def __call__(self, request):
            return CondensateEquilibriumInit(
                gas_ln_n=jnp.asarray([-request.T, -request.P]),
                gas_ntot=jnp.asarray(3.0),
                support_indices=(0,),
                support_amounts=jnp.asarray([request.P]),
            )

    plan = SimpleNamespace(
        grid_initializer=SimpleNamespace(
            gas_initializer=FakeGasGridInitializer(),
            fixed_initializers=(
                FakeFixedGridInitializer(),
                FakeFixedGridInitializer(),
            ),
        ),
        setup=SimpleNamespace(
            gas_setup=SimpleNamespace(),
            gas_species=("A", "B"),
        ),
        pressures_bar=jnp.asarray([0.1, 1.0, 10.0]),
        active_indices=(0, 2),
        graphite_amounts_init=jnp.ones((3,)),
    )
    temperatures = jnp.asarray([900.0, 1000.0, 1100.0])

    initial = module.interpolate_graphite_grid_initial_values(
        plan,
        temperatures,
        jnp.asarray([1.0, 0.1]),
    )

    np.testing.assert_allclose(initial.gas_log_amounts[:, 0], temperatures)
    np.testing.assert_allclose(
        initial.gas_log_amounts[:, 1], plan.pressures_bar
    )
    np.testing.assert_allclose(initial.gas_total_log_amounts, np.log(2.0))
    np.testing.assert_allclose(
        initial.fixed_gas_log_amounts[:, 0],
        [-900.0, 1000.0, -1100.0],
    )
    np.testing.assert_allclose(
        initial.fixed_total_log_amounts,
        [np.log(3.0), np.log(2.0), np.log(3.0)],
    )
    np.testing.assert_allclose(initial.graphite_amounts, [0.1, 1.0, 10.0])


def test_hybrid_solver_uses_static_condensate_and_gas_partitions(
    condensate_example,
    monkeypatch: pytest.MonkeyPatch,
):
    module = condensate_example
    setup = SimpleNamespace(
        formula_matrix=jnp.eye(2),
        formula_matrix_cond=jnp.asarray([[1.0], [0.0]]),
        gas_setup=SimpleNamespace(
            hvector_func=lambda temperature: jnp.zeros((2,)),
        ),
        condensate_setup=SimpleNamespace(
            hvector_func=lambda temperature: jnp.zeros((1,)),
        ),
    )
    q_init = jnp.zeros((3, 2))
    plan = module.GraphiteProfilePlan(
        setup=setup,
        pressures_bar=jnp.asarray([0.1, 1.0, 10.0]),
        nominal_temperatures=jnp.asarray([1000.0, 1100.0, 1200.0]),
        reference_element_vector=jnp.asarray([2.0, 1.0]),
        carbon_index=0,
        oxygen_index=1,
        co_species_index=0,
        graphite_species_index=0,
        active_indices=(0,),
        inactive_indices=(1, 2),
        gas_only_log_amounts=q_init,
        gas_only_total_log_amounts=jnp.zeros((3,)),
        hybrid_log_amounts_init=q_init,
        hybrid_total_log_amounts_init=jnp.zeros((3,)),
        graphite_amounts_init=jnp.ones((3,)),
        graphite_seed_amount=1.0e-3,
        nominal_graphite_driving_margin=jnp.zeros((3,)),
        nominal_fixed_support_residual=jnp.zeros((3,)),
    )

    def fake_fixed_support(
        state,
        gas_log_amounts_init,
        condensate_amounts_init,
        total_gas_log_amount_init,
        gas_formula_matrix,
        condensate_formula_matrix,
        gas_hvector_func,
        condensate_hvector_func,
        **kwargs,
    ):
        del (
            state,
            total_gas_log_amount_init,
            gas_formula_matrix,
            condensate_formula_matrix,
            gas_hvector_func,
            condensate_hvector_func,
            kwargs,
        )
        return DifferentiableFixedSupportResult(
            gas_log_amounts=gas_log_amounts_init + 10.0,
            condensate_amounts=condensate_amounts_init,
        )

    def fake_gas(
        state,
        gas_log_amounts_init,
        total_gas_log_amount_init,
        gas_formula_matrix,
        gas_hvector_func,
        **kwargs,
    ):
        del (
            state,
            total_gas_log_amount_init,
            gas_formula_matrix,
            gas_hvector_func,
            kwargs,
        )
        return gas_log_amounts_init + 20.0

    monkeypatch.setattr(
        module, "minimize_gibbs_fixed_support", fake_fixed_support
    )
    monkeypatch.setattr(module, "minimize_gibbs", fake_gas)

    result = module.solve_hybrid_log_amounts(
        plan,
        jnp.asarray([1000.0, 1100.0, 1200.0]),
        0.0,
    )

    np.testing.assert_allclose(result[0], np.full((2,), 10.0))
    np.testing.assert_allclose(result[1:], np.full((2, 2), 20.0))


def test_condensate_cli_defaults_to_preflighted_eight_layer_grid(
    condensate_example,
):
    args = condensate_example.build_parser().parse_args(["--preflight-only"])

    assert args.preflight_only
    assert args.nlayer == 8
    assert args.co_database is None
    assert "vjp_retrieval" in str(args.output_dir)
