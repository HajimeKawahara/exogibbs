"""Regression tests for the Ito et al. (2025) comparison example."""

import ast
from pathlib import Path
import runpy
from types import SimpleNamespace
from zipfile import ZipFile

import numpy as np
import pytest


REPOSITORY_ROOT = Path(__file__).resolve().parents[3]
EXAMPLE_PATH = (
    REPOSITORY_ROOT
    / "examples"
    / "comparisons"
    / "comparison_with_ito_2025.py"
)
RAINOUT_EXAMPLE_PATH = (
    REPOSITORY_ROOT
    / "examples"
    / "comparisons"
    / "comparison_with_ito_2025_rainout.py"
)


@pytest.fixture(scope="module")
def comparison_module():
    matplotlib = pytest.importorskip("matplotlib")
    matplotlib.use("Agg", force=True)
    return runpy.run_path(EXAMPLE_PATH, run_name="ito_2025_comparison_test_module")


@pytest.fixture(scope="module")
def rainout_comparison_module():
    matplotlib = pytest.importorskip("matplotlib")
    matplotlib.use("Agg", force=True)
    return runpy.run_path(
        RAINOUT_EXAMPLE_PATH,
        run_name="ito_2025_rainout_comparison_test_module",
    )


def _write_minimal_workbook(path: Path) -> None:
    headers = (
        "Layer number",
        "P [bar]",
        "T[K]",
        "Fraction of H2",
        "Fraction of He",
        "Fraction of H2O",
        "Fracton of O2",
        "Fraction of SiO",
        "Fraction of SiH4",
    )
    data = (
        (
            1,
            38935.466,
            3000.0,
            0.7705057324637681,
            0.14613039753623192,
            0.0052410987,
            2.4430258e-12,
            0.0068915286,
            0.071231245,
        ),
        (
            2,
            38561.028,
            2995.0518,
            0.7846548576811594,
            0.14881385231884062,
            0.00019717632,
            3.2954038e-15,
            0.00022926757,
            0.066104847,
        ),
    )
    shared_items = "".join(f"<si><t>{header}</t></si>" for header in headers)
    shared = (
        '<?xml version="1.0" encoding="UTF-8"?>'
        '<sst xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main">'
        f"{shared_items}</sst>"
    )
    workbook = (
        '<?xml version="1.0" encoding="UTF-8"?>'
        '<workbook xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main" '
        'xmlns:r="http://schemas.openxmlformats.org/officeDocument/2006/relationships">'
        '<sheets><sheet name="Sheet1" sheetId="1" r:id="rId1"/></sheets>'
        "</workbook>"
    )
    relationships = (
        '<?xml version="1.0" encoding="UTF-8"?>'
        '<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">'
        '<Relationship Id="rId1" '
        'Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/worksheet" '
        'Target="worksheets/sheet1.xml"/>'
        "</Relationships>"
    )
    header_cells = "".join(
        f'<c r="{chr(ord("A") + index)}1" t="s"><v>{index}</v></c>'
        for index in range(len(headers))
    )
    data_rows = []
    for row_number, values in enumerate(data, 2):
        cells = "".join(
            f'<c r="{chr(ord("A") + index)}{row_number}"><v>{value:.17g}</v></c>'
            for index, value in enumerate(values)
        )
        data_rows.append(f'<row r="{row_number}">{cells}</row>')
    worksheet = (
        '<?xml version="1.0" encoding="UTF-8"?>'
        '<worksheet xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main">'
        f'<sheetData><row r="1">{header_cells}</row>{"".join(data_rows)}</sheetData>'
        "</worksheet>"
    )
    with ZipFile(path, "w") as archive:
        archive.writestr("xl/workbook.xml", workbook)
        archive.writestr("xl/_rels/workbook.xml.rels", relationships)
        archive.writestr("xl/sharedStrings.xml", shared)
        archive.writestr("xl/worksheets/sheet1.xml", worksheet)


def test_example_is_main_guarded_and_documents_ito_anchored_rainout() -> None:
    source = EXAMPLE_PATH.read_text(encoding="utf-8")
    tree = ast.parse(source, filename=str(EXAMPLE_PATH))

    compile(source, str(EXAMPLE_PATH), "exec")
    assert "one-grid-higher-pressure Layer" in source
    assert 'chemistry_mode="equilibrium_condensation"' in source
    assert "HE_TO_H2_RATIO" in source
    assert "Layer 1 excluded" in source
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


def test_propagated_rainout_example_uses_both_native_profile_modes() -> None:
    source = RAINOUT_EXAMPLE_PATH.read_text(encoding="utf-8")
    tree = ast.parse(source, filename=str(RAINOUT_EXAMPLE_PATH))

    compile(source, str(RAINOUT_EXAMPLE_PATH), "exec")
    assert "Ito Layer 1 is used only as the lower-boundary gas inventory" in source
    assert 'rainout=True' in source
    assert 'profile_method="scan_hot_from_bottom"' in source
    assert 'chemistry_mode="rainout_condensation"' in source
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


def test_propagated_rainout_target_excludes_layer1(
    rainout_comparison_module,
    tmp_path: Path,
) -> None:
    workbook = tmp_path / "ito.xlsx"
    _write_minimal_workbook(workbook)
    profile = rainout_comparison_module["load_ito_profile"](workbook)

    target = rainout_comparison_module["_target_profile"](profile, None)
    boundary = rainout_comparison_module["reactive_element_abundances"](
        profile.gas_fractions[0]
    )

    np.testing.assert_array_equal(target.layer, [2])
    np.testing.assert_allclose(target.gas_fractions, profile.gas_fractions[1:])
    assert target.layer[0] != profile.layer[0]
    assert boundary.sum() == pytest.approx(1.0)


def test_propagated_rainout_masks_exactly_depleted_gases(
    rainout_comparison_module,
) -> None:
    fractions = np.asarray(
        [
            [0.70, 0.10, 0.05, 0.05, 0.10],
            [0.60, 0.10, 0.10, 0.10, 0.10],
        ],
        dtype=np.float64,
    )
    inventory = np.asarray(
        [[0.9, 0.0, 0.1], [0.8, 0.1, 0.1]],
        dtype=np.float64,
    )
    formula = np.asarray(
        [
            [2.0, 2.0, 0.0, 0.0, 4.0],
            [0.0, 1.0, 2.0, 1.0, 0.0],
            [0.0, 0.0, 0.0, 1.0, 1.0],
        ],
        dtype=np.float64,
    )

    masked = rainout_comparison_module[
        "_mask_exactly_depleted_gases"
    ](fractions, inventory, formula)

    np.testing.assert_allclose(masked[0], [0.875, 0.0, 0.0, 0.0, 0.125])
    np.testing.assert_allclose(masked[1], fractions[1])
    np.testing.assert_allclose(np.sum(masked, axis=1), 1.0)
    np.testing.assert_allclose(fractions[0], [0.70, 0.10, 0.05, 0.05, 0.10])


def test_sio_saturation_diagnostic_uses_profile_thermochemistry(
    rainout_comparison_module,
) -> None:
    setup = rainout_comparison_module["build_ito_exogibbs_setup"]()
    temperature = 1200.0
    pressure = 10.0
    sio_index = setup.condensate_species.index(
        rainout_comparison_module["SIO_CONDENSATE"]
    )
    sio_formula = np.asarray(
        setup.formula_matrix_cond[:, sio_index], dtype=np.float64
    )
    hcond = np.asarray(setup.condensate_setup.hvector_func(temperature))
    target_log_saturation = np.log(2.0)
    element_potential = (
        (hcond[sio_index] + target_log_saturation)
        * sio_formula
        / np.dot(sio_formula, sio_formula)
    )
    hgas = np.asarray(setup.gas_setup.hvector_func(temperature))
    gas_ln_n = (
        np.asarray(setup.formula_matrix).T @ element_potential
        - hgas
        - np.log(pressure)
    )
    jnp = rainout_comparison_module["jnp"]
    layer = SimpleNamespace(
        gas_ln_n=jnp.asarray(gas_ln_n),
        gas_ntot=jnp.asarray(1.0),
        condensate_support_indices=jnp.asarray([sio_index]),
    )

    log_saturation, saturation, support = rainout_comparison_module[
        "_sio_saturation_diagnostics"
    ](
        setup,
        np.asarray([temperature]),
        np.asarray([pressure]),
        (layer,),
    )

    assert log_saturation[0] == pytest.approx(target_log_saturation, abs=1.0e-11)
    assert saturation[0] == pytest.approx(2.0, rel=1.0e-11)
    np.testing.assert_array_equal(support, [True])


def test_rainout_inventory_diagnostics_expose_depletion_and_reintroduction(
    rainout_comparison_module,
) -> None:
    solution_type = rainout_comparison_module["RainoutSolution"]
    target_inventory = np.asarray(
        [[0.9, 1.0e-8, 0.1], [0.9, 0.0, 0.1]], dtype=np.float64
    )
    gas_inventory = np.asarray(
        [[0.9, 0.0, 0.1 - 1.0e-8], [0.9, 1.0e-8, 0.1]],
        dtype=np.float64,
    )
    inventory_out = np.asarray(
        [[0.9, 0.0, 0.1], [0.9, 0.0, 0.1]], dtype=np.float64
    )
    condensates = np.asarray(
        [[1.0e-8, 0.0], [0.0, 0.0]], dtype=np.float64
    )
    solution = solution_type(
        gas_fractions=np.full((2, 6), 1.0 / 6.0),
        atomic_gas_fractions=np.zeros((2, 3)),
        condensate_amounts=condensates,
        reactive_pressure_bar=np.asarray([100.0, 10.0]),
        converged=np.asarray([True, True]),
        status=("converged", "converged"),
        acceptance_tier=("fixed_support_v2_accepted",) * 2,
        fixed_point_iterations=1,
        solver_iterations=np.asarray([-1, -1]),
        element_inventory_target=target_inventory,
        gas_element_inventory=gas_inventory,
        element_inventory_out=inventory_out,
        abundance_scale=np.ones(2),
        sio_log_saturation_ratio=np.asarray([0.0, -1.0]),
        sio_saturation_ratio=np.asarray([1.0, np.exp(-1.0)]),
        sio_support_active=np.asarray([True, False]),
        sio_condensate_positive=np.asarray([True, False]),
    )

    diagnostics = rainout_comparison_module[
        "_exogibbs_profile_diagnostics"
    ](solution)
    profile_type = rainout_comparison_module["ItoProfile"]
    profile = profile_type(
        layer=np.asarray([10, 11]),
        pressure_bar=np.asarray([100.0, 10.0]),
        temperature_k=np.asarray([1200.0, 1000.0]),
        gas_fractions=np.full((2, 6), 1.0 / 6.0),
        atomic_fractions=np.zeros((2, 4)),
    )
    summary = rainout_comparison_module[
        "_stability_diagnostic_summary"
    ](profile, solution, diagnostics)

    np.testing.assert_allclose(
        diagnostics["conservative_gas_element_inventory"][0],
        gas_inventory[0],
    )
    assert diagnostics["positive_target_below_budget_floor_element_mask"][
        0, 1
    ]
    assert diagnostics["exact_depletion_event_element_mask"][0, 1]
    assert diagnostics[
        "gas_reintroduced_into_exact_zero_target_element_mask"
    ][1, 1]
    assert np.isnan(
        diagnostics[
            "gas_vs_conservative_inventory_positive_relative_mismatch"
        ][1, 1]
    )
    assert summary["sio_saturation"]["support_transition_count"] == 1
    assert (
        summary["numerical_depletion"]["positive_to_exact_zero_events"][
            "first_layer"
        ]
        == 10
    )
    assert not summary["rerun_pass_criteria"]["overall_passed"]


def test_rainout_budget_floor_uses_caller_inventory_not_scheduler_scale(
    rainout_comparison_module,
) -> None:
    solution_type = rainout_comparison_module["RainoutSolution"]
    target = np.asarray([[9.0, 5.0e-6, 1.0]], dtype=np.float64)
    gas = target.copy()
    solution = solution_type(
        gas_fractions=np.full((1, 6), 1.0 / 6.0),
        atomic_gas_fractions=np.zeros((1, 3)),
        condensate_amounts=np.zeros((1, 2)),
        reactive_pressure_bar=np.asarray([1.0]),
        converged=np.asarray([True]),
        status=("converged",),
        acceptance_tier=("fixed_support_v2_accepted",),
        fixed_point_iterations=1,
        solver_iterations=np.asarray([-1]),
        element_inventory_target=target,
        gas_element_inventory=gas,
        element_inventory_out=target,
        abundance_scale=np.asarray([3.0e8]),
        sio_log_saturation_ratio=np.asarray([-1.0]),
        sio_saturation_ratio=np.asarray([np.exp(-1.0)]),
        sio_support_active=np.asarray([False]),
        sio_condensate_positive=np.asarray([False]),
    )

    diagnostics = rainout_comparison_module[
        "_exogibbs_profile_diagnostics"
    ](solution)

    assert diagnostics["positive_target_below_budget_floor_element_mask"][
        0, 1
    ]
    expected_floor = 1.0e-6 * np.sum(target[0])
    mismatch = diagnostics["gas_vs_conservative_inventory_absolute_mismatch"]
    floor_scaled = diagnostics[
        "gas_vs_conservative_inventory_floor_scaled_mismatch"
    ]
    np.testing.assert_allclose(
        floor_scaled[0],
        mismatch[0] / np.maximum(np.abs(target[0]), expected_floor),
    )


def test_rainout_outputs_report_depletion_and_actual_conservation_mask(
    rainout_comparison_module,
    tmp_path: Path,
) -> None:
    profile_type = rainout_comparison_module["ItoProfile"]
    solution_type = rainout_comparison_module["RainoutSolution"]
    gas = np.asarray(
        [
            [0.75, 0.14, 0.02, 1.0e-6, 0.02, 0.069999],
            [0.76, 0.14, 0.01, 1.0e-7, 0.01, 0.0799999],
        ],
        dtype=np.float64,
    )
    target = profile_type(
        layer=np.asarray([2, 3], dtype=np.int64),
        pressure_bar=np.asarray([100.0, 10.0]),
        temperature_k=np.asarray([2000.0, 1500.0]),
        gas_fractions=gas,
        atomic_fractions=np.zeros((2, 4), dtype=np.float64),
    )
    common = {
        "gas_fractions": gas.copy(),
        "atomic_gas_fractions": np.zeros((2, 3), dtype=np.float64),
        "condensate_amounts": np.zeros((2, 2), dtype=np.float64),
        "reactive_pressure_bar": np.asarray([90.0, 9.0]),
        "converged": np.asarray([True, True]),
        "status": ("converged", "converged"),
        "acceptance_tier": ("fixed_support_v2_accepted",) * 2,
        "fixed_point_iterations": 2,
        "solver_iterations": np.asarray([-1, -1], dtype=np.int64),
    }
    inventory_target = np.asarray(
        [[0.90, 0.05, 0.05], [0.95, 0.0, 0.05]],
        dtype=np.float64,
    )
    inventory_out = np.asarray(
        [[0.95, 0.0, 0.05], [0.95, 0.0, 0.05]],
        dtype=np.float64,
    )
    exogibbs = solution_type(
        **common,
        element_inventory_target=inventory_target,
        gas_element_inventory=inventory_out.copy(),
        element_inventory_out=inventory_out,
        abundance_scale=np.ones(2, dtype=np.float64),
        sio_log_saturation_ratio=np.zeros(2, dtype=np.float64),
        sio_saturation_ratio=np.ones(2, dtype=np.float64),
        sio_support_active=np.zeros(2, dtype=bool),
        sio_condensate_positive=np.zeros(2, dtype=bool),
    )
    fastchem = solution_type(
        **{
            **common,
            "acceptance_tier": ("native_cr",) * 2,
            "solver_iterations": np.asarray([3, 4], dtype=np.int64),
        },
        elements_conserved=np.asarray([True, False]),
    )
    input_path = tmp_path / "ito.xlsx"
    executable = tmp_path / "fastchem"
    input_path.write_bytes(b"input")
    executable.write_bytes(b"executable")

    arrays = rainout_comparison_module["_solution_archive"](
        target,
        np.asarray([0.90, 0.05, 0.05]),
        exogibbs,
        fastchem,
    )
    table_path = tmp_path / "comparison.csv"
    rainout_comparison_module["_write_table"](
        table_path,
        arrays,
        exogibbs,
        fastchem,
    )
    payload = rainout_comparison_module["_write_summary"](
        tmp_path / "summary.json",
        input_path=input_path,
        executable=executable,
        target=target,
        layer1_abundance=np.asarray([0.90, 0.05, 0.05]),
        exogibbs=exogibbs,
        fastchem=fastchem,
    )
    figure = rainout_comparison_module["make_comparison_figure"](
        target,
        exogibbs,
        fastchem,
    )
    try:
        np.testing.assert_array_equal(
            arrays["fastchem_elements_conserved"],
            [True, False],
        )
        assert payload["fastchem4"]["elements_conserved_layers"] == 1
        assert payload["schema"].endswith("_v2")
        assert payload["exogibbs"]["first_exact_depletion_layer"] == 3
        assert payload["exogibbs"]["first_exact_depletion_elements"] == ["O"]
        assert "exogibbs_sio_log_saturation_ratio" in arrays
        assert "exogibbs_conservative_gas_element_inventory" in arrays
        table_header = table_path.read_text(encoding="utf-8").splitlines()[0]
        assert "exogibbs_sio_support_state" in table_header
        assert "exogibbs_conservative_gas_element_inventory_O" in table_header
        assert (
            "exogibbs_gas_vs_conservative_inventory_floor_scaled_mismatch_O"
            in table_header
        )
        stability = payload["exogibbs"]["stability_diagnostics"]
        assert stability["schema"].endswith("_v1")
        assert not stability["rerun_pass_criteria"]["overall_passed"]
        figure_text = "\n".join(text.get_text() for text in figure.texts)
        assert "First exact depletion: Layer 3 (O)" in figure_text
        assert "SiO support transitions" in figure_text
    finally:
        rainout_comparison_module["plt"].close(figure)


def test_standard_library_xlsx_loader_and_previous_layer_abundance(
    comparison_module,
    tmp_path: Path,
) -> None:
    workbook = tmp_path / "ito.xlsx"
    _write_minimal_workbook(workbook)

    profile = comparison_module["load_ito_profile"](workbook)
    target_indices = comparison_module["_target_indices"](
        profile,
        step=1,
        max_layers=None,
    )
    checkpoint = comparison_module["_new_checkpoint"](
        profile,
        target_indices,
        input_sha256="input",
        executable_sha256="executable",
    )

    np.testing.assert_array_equal(profile.layer, [1, 2])
    np.testing.assert_array_equal(target_indices, [1])
    np.testing.assert_array_equal(checkpoint["source_layers"], [1])
    np.testing.assert_array_equal(checkpoint["layers"], [2])
    expected = comparison_module["reactive_element_abundances"](
        profile.gas_fractions[0]
    )
    current = comparison_module["reactive_element_abundances"](
        profile.gas_fractions[1]
    )
    np.testing.assert_allclose(checkpoint["input_abundances"][0], expected)
    assert not np.allclose(checkpoint["input_abundances"][0], current)


def test_atomic_inventory_matches_workbook_formula(comparison_module) -> None:
    gas = np.asarray(
        [
            0.7705057324637681,
            0.14613039753623192,
            0.0052410987,
            2.4430258e-12,
            0.0068915286,
            0.071231245,
        ]
    )
    inventory = comparison_module["reconstruct_atomic_inventory"](gas)
    fractions = inventory / np.sum(inventory)

    np.testing.assert_allclose(
        fractions,
        [
            0.8859584658389388,
            0.005853242624464339,
            0.03768940864051308,
            0.07049888289608385,
        ],
        rtol=2.0e-15,
    )


def test_helium_conversion_preserves_total_and_fixed_ratio(
    comparison_module,
) -> None:
    common_names = comparison_module["COMMON_GAS_SPECIES"]
    reactive = np.asarray(
        [0.001, 0.002, 0.003, 0.85, 0.02, 0.004, 0.02, 0.1]
    )
    gas, atoms, pressure_fraction = comparison_module[
        "_convert_reactive_gas"
    ](reactive)
    species = comparison_module["ITO_SPECIES"]

    assert tuple(common_names[:3]) == ("H1", "O1", "Si1")
    assert np.sum(gas) + np.sum(atoms) == pytest.approx(1.0)
    assert pressure_fraction == pytest.approx(1.0 - gas[species.index("He")])
    assert gas[species.index("He")] / gas[species.index("H2")] == pytest.approx(
        comparison_module["HE_TO_H2_RATIO"]
    )


def test_fastchem_record_filter_is_exact_and_source_ordered(
    comparison_module,
) -> None:
    text = """# header
A1 A : A 1
  1 2 3 4 5

B1 B : B 1
  6 7 8 9 10

C1 C : C 1
  11 12 13 14 15
"""

    filtered = comparison_module["filter_fastchem_records"](
        text,
        ("C1", "A1"),
    )

    assert filtered.startswith("# header\nA1")
    assert "B1" not in filtered
    assert filtered.index("A1") < filtered.index("C1")
    with pytest.raises(ValueError, match="missing"):
        comparison_module["filter_fastchem_records"](text, ("D1",))


def test_exogibbs_subset_uses_exact_ito_species_catalog(comparison_module) -> None:
    setup = comparison_module["build_ito_exogibbs_setup"]()

    assert setup.elements == comparison_module["ELEMENTS"]
    assert setup.gas_species == comparison_module["EXOGIBBS_GAS_SPECIES"]
    assert setup.condensate_species == comparison_module["CONDENSATE_SPECIES"]
    assert setup.formula_matrix.shape == (3, 5)
    assert setup.formula_matrix_cond.shape == (3, 2)


def test_exogibbs_abundance_gauge_preserves_ratios(comparison_module) -> None:
    abundance = np.asarray([1.2, 2.0e-8, 4.0e-4])

    scales = comparison_module["_exogibbs_abundance_scales"](abundance)

    assert scales == pytest.approx((5.0e4, 5.0e3, 5.0e2, 5.0e1, 5.0))
    scale = scales[0]
    np.testing.assert_allclose(
        (abundance * scale) / (abundance[0] * scale),
        abundance / abundance[0],
    )


def test_difference_summary_counts_converged_zero_values(
    comparison_module,
) -> None:
    model = np.ones((2, 6), dtype=np.float64)
    reference = np.ones((2, 6), dtype=np.float64)
    model[0, 3] = 0.0

    summary = comparison_module["_difference_summary"](model, reference)

    assert summary["O2"]["compared_count"] == 2
    assert summary["O2"]["model_zero_count"] == 1
    assert summary["O2"]["maximum_absolute_dex"] == pytest.approx(45.0)


def test_comparison_figure_reports_nonconvergence(comparison_module) -> None:
    state = {
        "layers": np.asarray([2, 3]),
        "pressure_bar": np.asarray([3.8e4, 3.7e4]),
        "ito_gas_fractions": np.asarray(
            [
                [0.78, 0.148, 2.0e-4, 3.0e-15, 2.0e-4, 0.0716],
                [0.79, 0.149, 1.8e-4, 2.8e-15, 1.9e-4, 0.0606],
            ]
        ),
        "exogibbs_gas_fractions": np.asarray(
            [
                [0.78, 0.148, 1.9e-4, 3.1e-15, 2.1e-4, 0.0716],
                [np.nan] * 6,
            ]
        ),
        "fastchem_gas_fractions": np.asarray(
            [
                [0.78, 0.148, 1.9e-4, 3.1e-15, 2.1e-4, 0.0716],
                [0.79, 0.149, 1.7e-4, 2.9e-15, 1.8e-4, 0.0606],
            ]
        ),
        "exogibbs_converged": np.asarray([True, False]),
    }

    figure = comparison_module["make_comparison_figure"](state)
    try:
        assert len(figure.axes) == 6
        assert figure.axes[0].get_xscale() == "log"
        assert figure.axes[0].get_yscale() == "log"
        assert figure.axes[0].get_ylim()[0] > figure.axes[0].get_ylim()[1]
        assert "ExoGibbs converged 1/2" in figure._suptitle.get_text()
    finally:
        comparison_module["plt"].close(figure)
