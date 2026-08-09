"""Regression tests for the reduced Fe--FeS rainout demo."""

import ast
from pathlib import Path
import runpy

import numpy as np
import pytest


REPOSITORY_ROOT = Path(__file__).resolve().parents[3]
EXAMPLE_PATH = (
    REPOSITORY_ROOT
    / "examples"
    / "comparisons"
    / "demo_fe_fes_rainout.py"
)


@pytest.fixture(scope="module")
def demo():
    matplotlib = pytest.importorskip("matplotlib")
    matplotlib.use("Agg", force=True)
    return runpy.run_path(
        EXAMPLE_PATH,
        run_name="fe_fes_rainout_demo_test_module",
    )


@pytest.fixture(scope="module")
def setup(demo):
    return demo["build_reduced_setup"]()


@pytest.fixture(scope="module")
def solutions(demo, setup):
    temperatures = demo["DEFAULT_TEMPERATURES_K"]
    return {
        "local": demo["solve_exogibbs"](
            setup,
            temperatures,
            rainout=False,
        ),
        "rainout": demo["solve_exogibbs"](
            setup,
            temperatures,
            rainout=True,
        ),
    }


def test_example_is_main_guarded_and_compiles() -> None:
    source = EXAMPLE_PATH.read_text(encoding="utf-8")
    tree = ast.parse(source, filename=str(EXAMPLE_PATH))

    compile(source, str(EXAMPLE_PATH), "exec")
    guarded_calls = [
        node
        for node in tree.body
        if isinstance(node, ast.If)
        and any(
            isinstance(statement, ast.Expr)
            and isinstance(statement.value, ast.Call)
            and isinstance(statement.value.func, ast.Name)
            and statement.value.func.id == "main"
            for statement in node.body
        )
    ]
    assert len(guarded_calls) == 1


def test_literature_fit_reproduces_the_one_bar_iron_temperature(demo) -> None:
    assert float(demo["iron_condensation_temperature"](1.0)) == pytest.approx(
        1838.2352941
    )
    temperatures = demo["iron_condensation_temperature"](
        np.asarray([0.1, 1.0, 10.0])
    )
    assert np.all(np.diff(temperatures) > 0.0)
    with pytest.raises(ValueError, match="finite and positive"):
        demo["iron_condensation_temperature"](0.0)


def test_reduced_setup_uses_the_exact_h_fe_s_catalog(demo, setup) -> None:
    assert setup.elements == ("H", "Fe", "S")
    assert setup.gas_species == ("H1", "Fe1", "S1", "H2", "H2S1")
    assert setup.condensate_species == ("Fe(s,l)", "FeS(s,l)")
    assert setup.formula_matrix.shape == (3, 5)
    assert setup.formula_matrix_cond.shape == (3, 2)

    budget = np.asarray(demo["solar_element_budget"](setup))
    assert np.sum(budget) == pytest.approx(1.0)
    assert budget[setup.elements.index("Fe")] > budget[
        setup.elements.index("S")
    ]


def test_local_and_rainout_profiles_resolve_the_fe_fes_contrast(
    demo,
    setup,
    solutions,
) -> None:
    local = solutions["local"]
    rainout = solutions["rainout"]
    assert np.all(local.converged), local.status
    assert np.all(rainout.converged), rainout.status
    assert local.method == "vmap_cold"
    assert local.rainout is False
    assert rainout.method == "scan_hot_from_bottom"
    assert rainout.rainout is True

    iron_index = setup.condensate_species.index(demo["IRON"])
    fes_index = setup.condensate_species.index(demo["IRON_SULFIDE"])
    assert demo["condensation_bracket"](
        local.temperatures_k,
        local.condensate_amounts[:, iron_index],
    ) == (1850.0, 1875.0)
    assert demo["condensation_bracket"](
        local.temperatures_k,
        local.condensate_amounts[:, fes_index],
    ) == (650.0, 700.0)

    cold_index = 0
    h2s_index = setup.gas_species.index("H2S1")
    element_iron_index = setup.elements.index("Fe")
    initial_iron = float(
        demo["solar_element_budget"](setup)[element_iron_index]
    )
    assert local.condensate_amounts[cold_index, fes_index] > 1.0e-5
    assert rainout.condensate_amounts[cold_index, fes_index] < 1.0e-15
    assert local.gas_x[cold_index, h2s_index] < 1.0e-6
    assert rainout.gas_x[cold_index, h2s_index] > 1.0e-5
    assert (
        rainout.element_inventory_target[cold_index, element_iron_index]
        / initial_iron
        < 1.0e-15
    )


def test_profiles_conserve_each_target_and_rainout_handoffs_are_adjacent(
    demo,
    setup,
    solutions,
) -> None:
    gas_matrix = np.asarray(setup.formula_matrix)
    condensate_matrix = np.asarray(setup.formula_matrix_cond)
    for solution in solutions.values():
        reconstructed = (
            solution.gas_n @ gas_matrix.T
            + solution.condensate_amounts @ condensate_matrix.T
        )
        target = solution.element_inventory_target
        assert np.all(target > 0.0)
        relative_residual = np.abs(reconstructed - target) / target
        assert np.max(relative_residual) < 1.0e-3

    rainout = solutions["rainout"]
    np.testing.assert_allclose(
        rainout.element_inventory_target[:-1],
        rainout.element_inventory_out[1:],
        rtol=1.0e-12,
        atol=1.0e-25,
    )
