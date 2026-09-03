"""Regression tests for the forsterite-enstatite competition demo."""

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
    / "comparison_with_visscher_2010_forsterite_enstatite.py"
)
REGRESSION_TEMPERATURES_K = np.asarray(
    [
        1300.0,
        1550.0,
        1574.0,
        1576.0,
        1580.0,
        1582.0,
        1702.0,
        1704.0,
    ]
)


@pytest.fixture(scope="module")
def demo():
    matplotlib = pytest.importorskip("matplotlib")
    matplotlib.use("Agg", force=True)
    return runpy.run_path(
        EXAMPLE_PATH,
        run_name="silicate_competition_demo_test_module",
    )


@pytest.fixture(scope="module")
def setups(demo):
    return {
        run.key: demo["build_reduced_setup"](run)
        for run in demo["COMPETITION_RUNS"]
    }


@pytest.fixture(scope="module")
def solutions(demo, setups):
    return {
        run.key: demo["solve_exogibbs"](
            setups[run.key],
            REGRESSION_TEMPERATURES_K,
        )
        for run in demo["COMPETITION_RUNS"]
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
    main_source = ast.get_source_segment(
        source,
        next(
            node
            for node in tree.body
            if isinstance(node, ast.FunctionDef) and node.name == "main"
        ),
    )
    assert main_source is not None
    assert main_source.index("unlink(missing_ok=True)") < main_source.index(
        "build_reduced_setup("
    )


def test_literature_fits_reproduce_one_bar_reference_temperatures(demo) -> None:
    assert float(
        demo["forsterite_condensation_temperature"](1.0)
    ) == pytest.approx(1697.7928693)
    assert float(
        demo["enstatite_condensation_temperature"](1.0)
    ) == pytest.approx(1597.4440895)

    pressures = np.asarray([0.1, 1.0, 10.0])
    forsterite = demo["forsterite_condensation_temperature"](pressures)
    enstatite = demo["enstatite_condensation_temperature"](pressures)
    assert np.all(np.diff(forsterite) > 0.0)
    assert np.all(np.diff(enstatite) > 0.0)


def test_reduced_setups_differ_only_by_enstatite(demo, setups) -> None:
    run_a = setups[demo["RUN_A"].key]
    run_b = setups[demo["RUN_B"].key]

    assert run_a.elements == ("H", "He", "C", "O", "Mg", "Si")
    assert run_a.gas_species == (
        "H2",
        "He1",
        "C1O1",
        "H2O1",
        "Mg1",
        "O1Si1",
    )
    assert run_b.elements == run_a.elements
    assert run_b.gas_species == run_a.gas_species
    assert run_a.condensate_species == (
        "Mg2SiO4(s,l)",
        "MgSiO3(s,l)",
        "SiO2(s,l)",
    )
    assert run_b.condensate_species == (
        "Mg2SiO4(s,l)",
        "SiO2(s,l)",
    )
    np.testing.assert_allclose(
        demo["solar_element_budget"](run_a),
        demo["solar_element_budget"](run_b),
    )
    assert run_a.formula_matrix.shape == (6, 6)
    assert run_a.formula_matrix_cond.shape == (6, 3)
    assert run_b.formula_matrix_cond.shape == (6, 2)


def test_production_solver_resolves_the_three_phase_boundaries(
    demo,
    setups,
    solutions,
) -> None:
    for solution in solutions.values():
        assert np.all(solution.converged), solution.status

    run_a_setup = setups[demo["RUN_A"].key]
    run_b_setup = setups[demo["RUN_B"].key]
    run_a = solutions[demo["RUN_A"].key]
    run_b = solutions[demo["RUN_B"].key]

    def bracket(solution, setup, phase):
        index = setup.condensate_species.index(phase)
        return demo["condensation_bracket"](
            solution.temperatures_k,
            solution.condensate_amounts[:, index],
        )

    assert bracket(run_a, run_a_setup, demo["FORSTERITE"]) == (
        1702.0,
        1704.0,
    )
    assert bracket(run_b, run_b_setup, demo["FORSTERITE"]) == (
        1702.0,
        1704.0,
    )
    assert bracket(run_a, run_a_setup, demo["ENSTATITE"]) == (
        1580.0,
        1582.0,
    )
    assert bracket(run_b, run_b_setup, demo["QUARTZ"]) == (
        1574.0,
        1576.0,
    )


def test_enstatite_suppresses_quartz_and_changes_the_cold_phase_split(
    demo,
    setups,
    solutions,
) -> None:
    run_a_setup = setups[demo["RUN_A"].key]
    run_b_setup = setups[demo["RUN_B"].key]
    run_a = solutions[demo["RUN_A"].key]
    run_b = solutions[demo["RUN_B"].key]

    quartz_index = run_a_setup.condensate_species.index(demo["QUARTZ"])
    assert np.all(run_a.condensate_amounts[:, quartz_index] == 0.0)

    cold_a = demo["phase_element_fractions"](
        run_a,
        run_a_setup,
        "Si",
    )
    cold_b = demo["phase_element_fractions"](
        run_b,
        run_b_setup,
        "Si",
    )
    assert cold_a[demo["ENSTATITE"]][0] > 0.97
    assert cold_a[demo["QUARTZ"]][0] == 0.0

    budget = np.asarray(demo["solar_element_budget"](run_b_setup))
    mg_to_si = (
        budget[run_b_setup.elements.index("Mg")]
        / budget[run_b_setup.elements.index("Si")]
    )
    expected_forsterite = mg_to_si / 2.0
    assert cold_b[demo["FORSTERITE"]][0] == pytest.approx(
        expected_forsterite,
        rel=5.0e-4,
    )
    assert cold_b[demo["QUARTZ"]][0] == pytest.approx(
        1.0 - expected_forsterite,
        rel=5.0e-4,
    )


def test_saturation_diagnostic_exposes_the_excluded_stable_phase(
    demo,
    setups,
    solutions,
) -> None:
    diagnostic_setup = setups[demo["RUN_A"].key]
    run_a = solutions[demo["RUN_A"].key]
    run_b = solutions[demo["RUN_B"].key]
    log_saturation_a = demo["log_saturation_ratios"](
        run_a,
        diagnostic_setup,
    )
    log_saturation_b = demo["log_saturation_ratios"](
        run_b,
        diagnostic_setup,
    )
    temperature_index = list(run_a.temperatures_k).index(1550.0)
    forsterite_index = diagnostic_setup.condensate_species.index(
        demo["FORSTERITE"]
    )
    enstatite_index = diagnostic_setup.condensate_species.index(
        demo["ENSTATITE"]
    )
    quartz_index = diagnostic_setup.condensate_species.index(demo["QUARTZ"])

    assert abs(log_saturation_a[temperature_index, forsterite_index]) < 1.0e-8
    assert abs(log_saturation_a[temperature_index, enstatite_index]) < 1.0e-8
    assert log_saturation_a[temperature_index, quartz_index] < -0.1
    assert abs(log_saturation_b[temperature_index, forsterite_index]) < 1.0e-8
    assert abs(log_saturation_b[temperature_index, quartz_index]) < 1.0e-8
    assert log_saturation_b[temperature_index, enstatite_index] > 0.05


def test_release_criteria_fail_closed(
    demo,
    setups,
    solutions,
    monkeypatch,
) -> None:
    validate = demo["_validate_release_criteria"]
    validate(setups=setups, solutions=solutions)

    original = validate.__globals__["log_saturation_ratios"]

    def invalid_saturation(solution, setup):
        values = original(solution, setup).copy()
        if solution is solutions[demo["RUN_A"].key]:
            temperature_index = list(solution.temperatures_k).index(1550.0)
            quartz_index = setup.condensate_species.index(demo["QUARTZ"])
            values[temperature_index, quartz_index] = 0.0
        return values

    monkeypatch.setitem(
        validate.__globals__,
        "log_saturation_ratios",
        invalid_saturation,
    )
    with pytest.raises(RuntimeError, match="saturation criteria"):
        validate(setups=setups, solutions=solutions)


@pytest.mark.parametrize(
    ("run_key", "phase"),
    (
        ("with_enstatite", "SiO2(s,l)"),
        ("without_enstatite", "MgSiO3(s,l)"),
    ),
)
def test_release_criteria_reject_nonfinite_inactive_saturation(
    demo,
    setups,
    solutions,
    monkeypatch,
    run_key,
    phase,
) -> None:
    validate = demo["_validate_release_criteria"]
    original = validate.__globals__["log_saturation_ratios"]

    def invalid_saturation(solution, setup):
        values = original(solution, setup).copy()
        if solution is solutions[run_key]:
            temperature_index = list(solution.temperatures_k).index(1550.0)
            phase_index = setup.condensate_species.index(phase)
            values[temperature_index, phase_index] = np.nan
        return values

    monkeypatch.setitem(
        validate.__globals__,
        "log_saturation_ratios",
        invalid_saturation,
    )
    with pytest.raises(RuntimeError, match="saturation criteria"):
        validate(setups=setups, solutions=solutions)


def test_both_runs_conserve_the_shared_element_inventory(
    demo,
    setups,
    solutions,
) -> None:
    for run in demo["COMPETITION_RUNS"]:
        setup = setups[run.key]
        solution = solutions[run.key]
        gas_inventory = np.exp(solution.gas_ln_n) @ np.asarray(
            setup.formula_matrix
        ).T
        condensed_inventory = solution.condensate_amounts @ np.asarray(
            setup.formula_matrix_cond
        ).T
        residual = (
            gas_inventory
            + condensed_inventory
            - np.asarray(demo["solar_element_budget"](setup))[None, :]
        )
        assert np.max(np.abs(residual)) < 1.0e-10
