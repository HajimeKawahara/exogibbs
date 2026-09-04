"""Regression tests for the reduced KCl and Na2S comparison demo."""

import ast
from dataclasses import replace
from pathlib import Path
import runpy
from types import SimpleNamespace

import numpy as np
import pytest


REPOSITORY_ROOT = Path(__file__).resolve().parents[3]
EXAMPLE_PATH = (
    REPOSITORY_ROOT
    / "examples"
    / "comparisons"
    / "comparison_with_visscher_2006_na2s_morley_2012_kcl.py"
)
REGRESSION_TEMPERATURES_K = np.asarray(
    [700.0, 750.0, 800.0, 805.0, 950.0, 995.0, 1005.0, 1050.0]
)


@pytest.fixture(scope="module")
def demo():
    matplotlib = pytest.importorskip("matplotlib")
    matplotlib.use("Agg", force=True)
    return runpy.run_path(EXAMPLE_PATH, run_name="kcl_na2s_demo_test_module")


@pytest.fixture(scope="module")
def setups(demo):
    return {
        case.label: demo["build_reduced_setup"](case)
        for case in demo["BENCHMARK_CASES"]
    }


@pytest.fixture(scope="module")
def solutions(demo, setups):
    return {
        case.label: demo["solve_exogibbs"](
            setups[case.label],
            REGRESSION_TEMPERATURES_K,
        )
        for case in demo["BENCHMARK_CASES"]
    }


def test_example_is_main_guarded_and_offers_independent_fastchem4() -> None:
    source = EXAMPLE_PATH.read_text(encoding="utf-8")
    tree = ast.parse(source, filename=str(EXAMPLE_PATH))

    compile(source, str(EXAMPLE_PATH), "exec")
    assert "run_fastchem_executable" in source
    assert 'chemistry_mode="equilibrium_condensation"' in source
    assert "--fastchem-executable" in source
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
    resolve_index = main_source.index("resolve(strict=True)")
    invalidate_index = main_source.index("unlink(missing_ok=True)")
    solve_index = main_source.index("build_reduced_setup(")
    assert resolve_index < invalidate_index < solve_index
    assert main_source.index("os.access(") < invalidate_index


def test_literature_fits_reproduce_one_bar_reference_temperatures(demo) -> None:
    assert float(demo["kcl_condensation_temperature"](1.0)) == pytest.approx(
        801.3462617
    )
    assert float(demo["na2s_condensation_temperature"](1.0)) == pytest.approx(
        995.0248756
    )
    kcl = demo["kcl_condensation_temperature"](
        np.asarray([0.1, 1.0, 10.0])
    )
    na2s = demo["na2s_condensation_temperature"](
        np.asarray([0.1, 1.0, 10.0])
    )
    assert np.all(np.diff(kcl) > 0.0)
    assert np.all(np.diff(na2s) > 0.0)


def test_reduced_setups_use_exact_independent_reaction_catalogs(demo, setups) -> None:
    kcl = setups[demo["KCL_CASE"].label]
    na2s = setups[demo["NA2S_CASE"].label]

    assert kcl.elements == ("H", "He", "K", "Cl")
    assert kcl.gas_species == (
        "H1",
        "He1",
        "K1",
        "Cl1",
        "H2",
        "Cl1K1",
    )
    assert kcl.condensate_species == ("KCl(s,l)",)
    assert na2s.elements == ("H", "He", "Na", "S")
    assert na2s.gas_species == (
        "H1",
        "He1",
        "Na1",
        "S1",
        "H2",
        "H2S1",
    )
    assert na2s.condensate_species == ("Na2S(s,l)",)
    assert kcl.formula_matrix.shape == (4, 6)
    assert na2s.formula_matrix.shape == (4, 6)
    assert kcl.formula_matrix_cond.shape == (4, 1)
    assert na2s.formula_matrix_cond.shape == (4, 1)


@pytest.mark.parametrize("case_name", ["KCL_CASE", "NA2S_CASE"])
def test_fastchem_inputs_are_filtered_to_the_same_reduced_catalog(
    demo,
    tmp_path,
    case_name,
) -> None:
    case = demo[case_name]
    abundance_path, gas_path, condensate_path = demo[
        "_write_reduced_fastchem_inputs"
    ](tmp_path, case)

    abundance_elements = tuple(
        line.split()[0]
        for line in abundance_path.read_text(encoding="utf-8").splitlines()
        if line and not line.startswith("#")
    )
    gas_records = tuple(
        match.group(1)
        for line in gas_path.read_text(encoding="utf-8").splitlines()
        if (match := demo["_FASTCHEM_RECORD"].match(line)) is not None
    )
    condensate_records = tuple(
        match.group(1)
        for line in condensate_path.read_text(encoding="utf-8").splitlines()
        if (match := demo["_FASTCHEM_RECORD"].match(line)) is not None
    )

    assert abundance_elements == case.elements
    assert set(gas_records) == set(case.molecule_species)
    assert condensate_records == (case.condensate_species,)


def test_fastchem_result_is_aligned_and_normalized_on_the_shared_catalog(
    demo,
    setups,
    monkeypatch,
) -> None:
    gas_names = ("Cl", "H", "He", "K", "Cl1K1", "H2")
    gas_density = np.asarray(
        [
            [2.0, 50.0, 5.0, 3.0, 7.0, 20.0],
            [4.0, 40.0, 6.0, 2.0, 8.0, 10.0],
        ]
    )

    def fake_fastchem_runner(**kwargs):
        assert kwargs["chemistry_mode"] == "equilibrium_condensation"
        return SimpleNamespace(
            gas_names=gas_names,
            condensate_names=("KCl(s,l)",),
            gas_number_densities=gas_density,
            condensate_number_densities=np.asarray([[4.0], [6.0]]),
            total_element_density=np.asarray([100.0, 200.0]),
            converged=np.asarray([True, True]),
            elements_conserved=np.asarray([True, True]),
            convergence_status=np.asarray(["ok", "ok"]),
            element_conservation_status=np.asarray(["ok", "ok"]),
        )

    monkeypatch.setitem(
        demo["solve_fastchem4"].__globals__,
        "run_fastchem_executable",
        fake_fastchem_runner,
    )
    case = demo["KCL_CASE"]
    solution = demo["solve_fastchem4"](
        Path("unused-fastchem"),
        case,
        setups[case.label],
        np.asarray([700.0, 800.0]),
    )

    expected_first_row = np.asarray([50.0, 5.0, 3.0, 2.0, 20.0, 7.0])
    expected_first_row /= np.sum(expected_first_row)
    np.testing.assert_allclose(solution.gas_x[0], expected_first_row)
    np.testing.assert_allclose(np.sum(solution.gas_x, axis=1), 1.0)
    np.testing.assert_allclose(solution.condensate_amounts[:, 0], [0.04, 0.03])
    assert np.all(solution.converged)


def test_production_solver_brackets_both_literature_transitions(
    demo,
    setups,
    solutions,
) -> None:
    for solution in solutions.values():
        assert np.all(solution.converged), solution.status

    kcl = solutions[demo["KCL_CASE"].label]
    na2s = solutions[demo["NA2S_CASE"].label]
    kcl_bracket = demo["condensation_bracket"](
        kcl.temperatures_k,
        kcl.condensate_amounts[:, 0],
    )
    na2s_bracket = demo["condensation_bracket"](
        na2s.temperatures_k,
        na2s.condensate_amounts[:, 0],
    )

    assert kcl_bracket == (800.0, 805.0)
    assert na2s_bracket == (995.0, 1005.0)
    assert kcl_bracket[0] <= float(
        demo["kcl_condensation_temperature"](1.0)
    ) <= kcl_bracket[1]
    assert na2s_bracket[0] <= float(
        demo["na2s_condensation_temperature"](1.0)
    ) <= na2s_bracket[1]


def test_release_criteria_fail_closed(demo, setups, solutions) -> None:
    demo["_validate_release_criteria"](
        setups=setups,
        exogibbs=solutions,
        fastchem4=None,
    )

    kcl = solutions[demo["KCL_CASE"].label]
    changed_gas = kcl.gas_x.copy()
    temperature_index = list(kcl.temperatures_k).index(750.0)
    species_index = setups[demo["KCL_CASE"].label].gas_species.index("Cl1K1")
    changed_gas[temperature_index, species_index] *= 2.0
    invalid_literature = {
        **solutions,
        demo["KCL_CASE"].label: replace(kcl, gas_x=changed_gas),
    }
    with pytest.raises(RuntimeError, match="vapor-pressure fit"):
        demo["_validate_release_criteria"](
            setups=setups,
            exogibbs=invalid_literature,
            fastchem4=None,
        )

    changed_gas = kcl.gas_x.copy()
    changed_gas[:, species_index] *= 1.01
    invalid_fastchem = {
        **solutions,
        demo["KCL_CASE"].label: replace(kcl, gas_x=changed_gas),
    }
    with pytest.raises(RuntimeError, match="gas difference"):
        demo["_validate_release_criteria"](
            setups=setups,
            exogibbs=solutions,
            fastchem4=invalid_fastchem,
        )


def test_cold_kcl_gas_follows_saturation_vapor_pressure(
    demo,
    setups,
    solutions,
) -> None:
    setup = setups[demo["KCL_CASE"].label]
    solution = solutions[demo["KCL_CASE"].label]
    temperature_index = list(solution.temperatures_k).index(750.0)
    kcl_index = setup.gas_species.index("Cl1K1")
    calculated_pressure = (
        solution.gas_x[temperature_index, kcl_index] * demo["PRESSURE_BAR"]
    )
    reference_pressure = float(demo["kcl_saturation_pressure"]([750.0])[0])

    assert calculated_pressure == pytest.approx(reference_pressure, rel=0.03)


def test_na2s_is_sodium_limited_and_leaves_most_sulfur(
    demo,
    setups,
    solutions,
) -> None:
    setup = setups[demo["NA2S_CASE"].label]
    solution = solutions[demo["NA2S_CASE"].label]
    fractions = demo["condensed_element_fractions"](solution, setup)
    cold_index = list(solution.temperatures_k).index(700.0)

    assert fractions["Na in Na2S"][cold_index] > 0.99
    assert fractions["S in Na2S"][cold_index] == pytest.approx(
        10.0 ** (6.37 - 7.26) / 2.0,
        rel=0.02,
    )
    assert fractions["S in Na2S"][cold_index] < 0.07
