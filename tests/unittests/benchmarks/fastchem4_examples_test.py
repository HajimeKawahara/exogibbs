import ast
import hashlib
import importlib.util
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest


REPOSITORY_ROOT = Path(__file__).resolve().parents[3]
EXAMPLE_ROOT = REPOSITORY_ROOT / "examples" / "comparisons"


@pytest.fixture(scope="module")
def fastchem4_condensate_example():
    matplotlib = pytest.importorskip("matplotlib")
    matplotlib.use("Agg", force=True)
    path = EXAMPLE_ROOT / "comparison_with_fastchem4_condensates.py"
    spec = importlib.util.spec_from_file_location(
        "comparison_with_fastchem4_condensates_test_module",
        path,
    )
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def fastchem4_gas_example():
    matplotlib = pytest.importorskip("matplotlib")
    matplotlib.use("Agg", force=True)
    path = EXAMPLE_ROOT / "comparison_with_fastchem4_gas.py"
    spec = importlib.util.spec_from_file_location(
        "comparison_with_fastchem4_gas_test_module",
        path,
    )
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _guarded_main_call_count(tree: ast.Module) -> int:
    return sum(
        1
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
    )


@pytest.mark.parametrize(
    ("filename", "chemistry_mode", "public_api"),
    [
        (
            "comparison_with_fastchem4_gas.py",
            '"gas"',
            "solve_gas_profile",
        ),
        (
            "comparison_with_fastchem4_condensates.py",
            '"equilibrium_condensation"',
            "solve_condensate_profile",
        ),
    ],
)
def test_fastchem4_comparison_example_is_current_and_main_guarded(
    filename,
    chemistry_mode,
    public_api,
):
    path = EXAMPLE_ROOT / filename
    source = path.read_text(encoding="utf-8")
    tree = ast.parse(source, filename=str(path))

    compile(source, str(path), "exec")
    assert "pyfastchem" not in source
    assert "exojax" not in source
    assert "run_fastchem_executable" in source
    assert f"chemistry_mode={chemistry_mode}" in source
    assert public_api in source
    assert "build_aligned_abundance_vector" in source
    assert "occurrence_keys" in source
    assert "elements_conserved" in source

    assert _guarded_main_call_count(tree) == 1
    main_node = next(
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name == "main"
    )
    main_source = ast.get_source_segment(source, main_node)
    assert main_source is not None
    resolve_index = main_source.index("resolve(strict=True)")
    invalidate_index = main_source.index("unlink(missing_ok=True)")
    solve_index = main_source.index("run_fastchem_executable(")
    assert resolve_index < invalidate_index < solve_index
    assert main_source.index("os.access(") < invalidate_index


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("finite", False, "non-finite amounts"),
        ("finite", np.nan, "non-finite amounts"),
        ("jaccard", 0.5, "major-gas set"),
        ("gas_difference", 2.0e-3, "gas abundances"),
        ("budget", 3.0e-4, "elemental-budget"),
        ("total_gas", 2.0e-8, "total gas amounts"),
    ],
)
def test_fastchem4_gas_release_metrics_fail_closed(
    fastchem4_gas_example,
    field,
    value,
    message,
) -> None:
    module = fastchem4_gas_example
    row = {
        "gas": {
            "finite": True,
            "major_set_jaccard": 1.0,
            "max_absolute_log10_ratio": 1.0e-6,
        },
        "exogibbs_budget": 1.0e-8,
        "fastchem_budget": 1.0e-8,
        "total_gas_relative_difference": 1.0e-10,
    }
    module._validate_release_metrics([row])
    if field == "finite":
        row["gas"]["finite"] = value
    elif field == "jaccard":
        row["gas"]["major_set_jaccard"] = value
    elif field == "gas_difference":
        row["gas"]["max_absolute_log10_ratio"] = value
    elif field == "budget":
        row["exogibbs_budget"] = value
    else:
        row["total_gas_relative_difference"] = value
    with pytest.raises(RuntimeError, match=message):
        module._validate_release_metrics([row])


@pytest.mark.parametrize(
    ("target", "field", "value", "message"),
    [
        ("gas", "finite", False, "non-finite amounts"),
        ("gas", "finite", np.nan, "non-finite amounts"),
        ("condensate", "finite", False, "non-finite amounts"),
        ("condensate", "finite", np.nan, "non-finite amounts"),
        ("gas", "major_set_jaccard", 0.5, "major-gas set"),
        ("gas", "max_absolute_log10_ratio", 2.0e-3, "gas abundances"),
        ("condensate", "active_set_jaccard", 0.5, "active condensate set"),
        (
            "condensate",
            "max_absolute_log10_ratio",
            2.0e-3,
            "condensate amounts",
        ),
    ],
)
def test_fastchem4_condensate_release_metrics_fail_closed(
    fastchem4_condensate_example,
    target,
    field,
    value,
    message,
) -> None:
    module = fastchem4_condensate_example
    gas = {
        "finite": True,
        "major_set_jaccard": 1.0,
        "mean_absolute_log10_ratio": 1.0e-6,
        "max_absolute_log10_ratio": 1.0e-6,
    }
    condensate = {
        "finite": True,
        "active_set_jaccard": 1.0,
        "max_absolute_log10_ratio": 1.0e-6,
    }
    module._validate_release_metrics(
        gas_metrics=gas,
        condensate_metrics=condensate,
    )
    metrics = gas if target == "gas" else condensate
    metrics[field] = value
    with pytest.raises(RuntimeError, match=message):
        module._validate_release_metrics(
            gas_metrics=gas,
            condensate_metrics=condensate,
        )


def test_l_dwarf_gas_only_runs_remain_independent_of_condensation_runs():
    path = EXAMPLE_ROOT / "comparison_with_fastchem4_condensates.py"
    source = path.read_text(encoding="utf-8")
    tree = ast.parse(source, filename=str(path))
    calls = [node for node in ast.walk(tree) if isinstance(node, ast.Call)]

    fastchem_modes = []
    for call in calls:
        if not isinstance(call.func, ast.Name):
            continue
        if call.func.id != "run_fastchem_executable":
            continue
        for keyword in call.keywords:
            if keyword.arg == "chemistry_mode":
                assert isinstance(keyword.value, ast.Constant)
                fastchem_modes.append(keyword.value.value)
    assert sorted(fastchem_modes) == ["equilibrium_condensation", "gas"]

    exogibbs_solve_calls = [
        call
        for call in calls
        if isinstance(call.func, ast.Name)
        and call.func.id in {"solve_condensate_profile", "solve_gas_profile"}
    ]
    assert len(exogibbs_solve_calls) == 2
    for call in exogibbs_solve_calls:
        assert "fastchem" not in ast.unparse(call).lower()
    gas_solve_call = next(
        call
        for call in exogibbs_solve_calls
        if isinstance(call.func, ast.Name)
        and call.func.id == "solve_gas_profile"
    )
    assert isinstance(gas_solve_call.args[0], ast.Attribute)
    assert isinstance(gas_solve_call.args[0].value, ast.Name)
    assert gas_solve_call.args[0].value.id == "setup"
    assert gas_solve_call.args[0].attr == "gas_setup"
    main_node = next(
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name == "main"
    )
    main_source = ast.get_source_segment(source, main_node)
    assert main_source is not None
    assert main_source.index(
        "_validate_gas_profile_release_metrics("
    ) < main_source.index("_plot_l_dwarf_profile_comparison(")


def test_l_dwarf_gas_only_profile_uses_shared_release_gate(
    fastchem4_condensate_example,
) -> None:
    module = fastchem4_condensate_example
    left = np.asarray([[0.9, 0.1], [0.8, 0.2]])
    right = left.copy()
    module._validate_gas_profile_release_metrics(
        names=("H2", "H1"),
        left_values=left,
        right_values=right,
        comparison_label="gas-only",
    )

    right[1, 0] *= 0.5
    with pytest.raises(RuntimeError, match="gas-only.*gas abundances"):
        module._validate_gas_profile_release_metrics(
            names=("H2", "H1"),
            left_values=left,
            right_values=right,
            comparison_label="gas-only",
        )


def test_l_dwarf_profile_is_positive_monotonic_and_reproducible(
    fastchem4_condensate_example,
):
    module = fastchem4_condensate_example
    validation_temperatures, validation_pressures = module._profile_conditions(
        "validation"
    )
    temperatures, pressures = module._profile_conditions("l-dwarf")

    np.testing.assert_array_equal(
        validation_temperatures,
        np.asarray([1800.0, 1600.0, 1400.0, 1200.0]),
    )
    np.testing.assert_array_equal(
        validation_pressures,
        np.full(4, 0.1),
    )
    np.testing.assert_allclose(pressures, np.logspace(-4.0, 2.0, 13))
    assert temperatures.shape == pressures.shape
    assert np.all(np.isfinite(temperatures))
    assert np.all(np.diff(temperatures) > 0.0)
    assert np.all(np.diff(pressures) > 0.0)
    assert temperatures[0] == pytest.approx(1100.0)
    assert temperatures[-1] == pytest.approx(2600.0)
    assert module._default_output_path("l-dwarf").name == (
        "comparison_with_fastchem4_ldwarf_profile.png"
    )


def test_l_dwarf_figure_has_shared_solver_columns(
    fastchem4_condensate_example,
):
    module = fastchem4_condensate_example
    temperatures, pressures = module._profile_conditions("l-dwarf")
    setup = SimpleNamespace(
        gas_species=list(module.GAS_SPECIES),
        condensate_species=list(module.CONDENSATE_SPECIES),
    )
    gas_shape = (pressures.size, len(setup.gas_species))
    condensate_shape = (pressures.size, len(setup.condensate_species))
    gas_scale = np.geomspace(1.0, 1.0e-8, gas_shape[1])
    condensate_scale = np.geomspace(1.0e-3, 1.0e-12, condensate_shape[1])
    fastchem_x = np.broadcast_to(gas_scale, gas_shape).copy()
    exogibbs_x = fastchem_x * 0.99
    fastchem_gas_only_x = fastchem_x * 0.8
    exogibbs_gas_only_x = exogibbs_x * 0.81
    fastchem_condensates = np.broadcast_to(
        condensate_scale,
        condensate_shape,
    ).copy()
    exogibbs_condensates = fastchem_condensates * 1.01

    fig = module._make_l_dwarf_profile_figure(
        pressures=pressures,
        temperatures=temperatures,
        setup=setup,
        exogibbs_gas_only_x=exogibbs_gas_only_x,
        exogibbs_x=exogibbs_x,
        exogibbs_condensates=exogibbs_condensates,
        fastchem_gas_only_x=fastchem_gas_only_x,
        fastchem_x=fastchem_x,
        fastchem_condensates=fastchem_condensates,
    )
    try:
        assert len(fig.axes) == 4
        fastchem_gas_axis, exogibbs_gas_axis = fig.axes[:2]
        fastchem_cond_axis, exogibbs_cond_axis = fig.axes[2:]
        assert fastchem_gas_axis.get_shared_x_axes().joined(
            fastchem_gas_axis,
            exogibbs_gas_axis,
        )
        assert fastchem_cond_axis.get_shared_x_axes().joined(
            fastchem_cond_axis,
            exogibbs_cond_axis,
        )
        assert fastchem_gas_axis.get_shared_y_axes().joined(
            fastchem_gas_axis,
            exogibbs_cond_axis,
        )
        assert fastchem_gas_axis.get_xscale() == "log"
        assert fastchem_gas_axis.get_yscale() == "log"
        assert fastchem_gas_axis.get_ylim()[0] > fastchem_gas_axis.get_ylim()[1]
        assert "FastChem 4" in fastchem_gas_axis.get_title()
        assert "ExoGibbs" in exogibbs_gas_axis.get_title()
        assert len(fastchem_gas_axis.lines) == 2 * len(module.GAS_SPECIES)
        assert len(exogibbs_gas_axis.lines) == 2 * len(module.GAS_SPECIES)
        assert len(fastchem_cond_axis.lines) == len(module.CONDENSATE_SPECIES)
        assert len(exogibbs_cond_axis.lines) == len(module.CONDENSATE_SPECIES)

        fastchem_condensed_line = fastchem_gas_axis.lines[0]
        fastchem_gas_only_line = fastchem_gas_axis.lines[1]
        np.testing.assert_allclose(
            fastchem_gas_only_line.get_xdata(),
            fastchem_gas_only_x[:, 0],
        )
        np.testing.assert_allclose(
            fastchem_condensed_line.get_xdata(),
            fastchem_x[:, 0],
        )
        np.testing.assert_allclose(
            fastchem_gas_only_line.get_ydata(),
            pressures,
        )
        assert fastchem_gas_only_line.get_color() == (
            fastchem_condensed_line.get_color()
        )
        assert fastchem_gas_only_line.get_linestyle() == "--"
        assert fastchem_condensed_line.get_linestyle() == "-"
        assert fastchem_condensed_line.get_marker() == "o"
        gas_state_labels = {
            text.get_text()
            for text in exogibbs_gas_axis.get_legend().get_texts()
        }
        assert gas_state_labels == {"Gas-only", "With condensates"}
    finally:
        module.plt.close(fig)


@pytest.mark.parametrize(
    ("legacy_filename", "current_module"),
    [
        ("comparison_with_fastchem.py", "comparison_with_fastchem4_gas"),
        (
            "comparison_with_fastchem_extended.py",
            "comparison_with_fastchem4_gas",
        ),
        (
            "comparison_with_fastchem_cond.py",
            "comparison_with_fastchem4_condensates",
        ),
    ],
)
def test_historical_fastchem_example_path_delegates_to_current_example(
    legacy_filename,
    current_module,
):
    path = EXAMPLE_ROOT / legacy_filename
    source = path.read_text(encoding="utf-8")

    compile(source, str(path), "exec")
    assert f"from {current_module} import main" in source
    assert 'if __name__ == "__main__":' in source


def test_restored_fastchem_initializer_example_keeps_solver_inputs_independent():
    path = EXAMPLE_ROOT / "comparison_with_fastchem_initializer.py"
    source = path.read_text(encoding="utf-8")
    tree = ast.parse(source, filename=str(path))

    compile(source, str(path), "exec")
    assert "pyfastchem" not in source
    assert "exojax" not in source
    assert "GridEquilibriumInitializer" in source
    assert "solve_gas_profile" in source
    assert "build_aligned_abundance_vector" in source
    assert "run_fastchem_executable" in source
    assert 'chemistry_mode="gas"' in source
    assert 'path="fastchem/logK/logK.dat"' in source
    assert source.index("grid_result, grid_diagnostics") < source.index(
        "fastchem = run_fastchem_executable"
    )
    assert source.index("uniform_result, uniform_diagnostics") < source.index(
        "fastchem = run_fastchem_executable"
    )
    assert _guarded_main_call_count(tree) == 1


@pytest.mark.parametrize(
    ("filename", "required_fragments"),
    [
        (
            "comparison_with_hsystem.py",
            (
                "from exogibbs.api.gas import EquilibriumOptions, solve",
                "HSystem",
                "jax.jacrev",
                "temperature_gradient_reference",
                "pressure_gradient_reference",
            ),
        ),
        (
            "comparison_with_hcosystem.py",
            (
                "from exogibbs.api.gas import EquilibriumOptions, solve",
                "HCOSystem",
                "jax.jacrev",
                "_bisect_analytic_co",
                "derivative_dlnnCO_db",
            ),
        ),
        (
            "comparison_with_ykcode.py",
            (
                "from exogibbs.api.gas import EquilibriumOptions, solve",
                "LEGACY_ELEMENT_BUDGET",
                "EXPECTED_SPECIES_ORDER_SHA256",
                "result.n",
                "MAX_RELATIVE_ERROR_LIMIT = 0.051",
            ),
        ),
    ],
)
def test_restored_trace_example_uses_current_api_and_main_guard(
    filename,
    required_fragments,
):
    path = EXAMPLE_ROOT / filename
    source = path.read_text(encoding="utf-8")
    tree = ast.parse(source, filename=str(path))

    compile(source, str(path), "exec")
    assert "exogibbs.optimize" not in source
    for fragment in required_fragments:
        assert fragment in source
    assert _guarded_main_call_count(tree) == 1


def test_historical_ykb4_reference_payload_is_exact_and_well_formed():
    path = REPOSITORY_ROOT / "examples" / "data" / "p10.txt"
    payload = path.read_bytes()
    values = np.loadtxt(path, delimiter=",")

    assert hashlib.sha256(payload).hexdigest() == (
        "062a0d21768f85871b7980ae3883d34f2466b9e11618255060e19d32c4a8612b"
    )
    assert values.shape == (160,)
    assert np.all(np.isfinite(values))
    assert np.all(values >= 0.0)
    assert np.count_nonzero(values > 1.0e-14) == 16
