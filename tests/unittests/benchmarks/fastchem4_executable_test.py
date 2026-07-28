from pathlib import Path
import runpy
import stat
import subprocess

import numpy as np
import pytest

REPOSITORY_ROOT = Path(__file__).resolve().parents[3]
_module = runpy.run_path(
    REPOSITORY_ROOT / "benchmarks" / "fastchem4" / "fastchem_executable.py"
)
_parse_chemistry_output = _module["_parse_chemistry_output"]
_parse_condensate_output = _module["_parse_condensate_output"]
_parse_monitor_output = _module["_parse_monitor_output"]
_validate_profile = _module["_validate_profile"]
run_fastchem_executable = _module["run_fastchem_executable"]


CHEMISTRY_OUTPUT = """\
#p(bar) T(K) n_<tot>(cm-3) n_g(cm-3) m(u) H H2
1.0000000000e-01 1.0000000000e+03 9.0000000000e+17 7.0000000000e+17 2.3000000000e+00 1.0000000000e+17 3.0000000000e+17
2.0000000000e-01 1.2000000000e+03 1.1000000000e+18 8.0000000000e+17 2.4000000000e+00 2.0000000000e+17 3.5000000000e+17
"""

MONITOR_OUTPUT = """\
#grid_point iterations chem_iter cond_iter converged elem_conserved p(bar) T(K) n_<tot>(cm-3) n_g(cm-3) m(u) H Zn
0 2 13 67 ok ok 1.0000000000e-01 1.0000000000e+03 9.0000000000e+17 7.0000000000e+17 2.3000000000e+00 ok ok
1 3 17 71 fail fail 2.0000000000e-01 1.2000000000e+03 1.1000000000e+18 8.0000000000e+17 2.4000000000e+00 ok fail
"""

CONDENSATE_OUTPUT = """\
#p(bar) T(K) H Zn Zn(s,l) Zn(s,l) MgO(s)
1.0000000000e-01 1.0000000000e+03 0.0 0.5 1.0000000000e+12 2.0000000000e+12 3.0000000000e+12
2.0000000000e-01 1.2000000000e+03 0.0 0.6 4.0000000000e+12 5.0000000000e+12 6.0000000000e+12
"""


def _make_input_files(tmp_path: Path) -> dict[str, Path]:
    paths = {
        "executable": tmp_path / "fastchem",
        "element_abundance_file": tmp_path / "elements.dat",
        "gas_logk_file": tmp_path / "gas.dat",
        "condensate_logk_file": tmp_path / "condensates.dat",
    }
    for name, path in paths.items():
        path.write_text(name + "\n", encoding="utf-8")
    paths["executable"].chmod(
        paths["executable"].stat().st_mode | stat.S_IXUSR
    )
    return paths


def test_chemistry_parser_returns_nd_columns_without_name_mapping():
    output = _parse_chemistry_output(CHEMISTRY_OUTPUT)

    assert output.gas_names == ("H", "H2")
    assert output.gas_number_densities.shape == (2, 2)
    np.testing.assert_allclose(
        output.gas_number_densities,
        [[1.0e17, 3.0e17], [2.0e17, 3.5e17]],
    )
    np.testing.assert_allclose(output.total_element_density, [9.0e17, 1.1e18])
    np.testing.assert_allclose(output.ideal_gas_density, [7.0e17, 8.0e17])


def test_monitor_parser_preserves_per_layer_status_and_iteration_counts():
    output = _parse_monitor_output(MONITOR_OUTPUT)

    assert output.element_names == ("H", "Zn")
    assert output.convergence_status == ("ok", "fail")
    assert output.element_conservation_status == ("ok", "fail")
    np.testing.assert_array_equal(output.total_iterations, [2, 3])
    np.testing.assert_array_equal(output.chemistry_iterations, [13, 17])
    np.testing.assert_array_equal(output.condensation_iterations, [67, 71])


def test_parsers_retain_failed_layer_with_zero_solver_outputs():
    chemistry = CHEMISTRY_OUTPUT.replace(
        "1.1000000000e+18 8.0000000000e+17 2.4000000000e+00 "
        "2.0000000000e+17 3.5000000000e+17",
        "0.0000000000e+00 8.0000000000e+17 0.0000000000e+00 "
        "0.0000000000e+00 0.0000000000e+00",
    )
    monitor = MONITOR_OUTPUT.replace(
        "1.1000000000e+18 8.0000000000e+17 2.4000000000e+00",
        "0.0000000000e+00 8.0000000000e+17 0.0000000000e+00",
    )

    chemistry_output = _parse_chemistry_output(chemistry)
    monitor_output = _parse_monitor_output(monitor)

    assert chemistry_output.total_element_density[1] == 0.0
    assert chemistry_output.mean_molecular_weight[1] == 0.0
    assert monitor_output.convergence_status[1] == "fail"


def test_condensate_parser_preserves_duplicate_zinc_names_and_columns():
    output = _parse_condensate_output(
        CONDENSATE_OUTPUT,
        element_names=("H", "Zn"),
    )

    assert output.condensate_names == ("Zn(s,l)", "Zn(s,l)", "MgO(s)")
    assert output.condensate_number_densities.shape == (2, 3)
    np.testing.assert_allclose(
        output.condensate_number_densities,
        [[1.0e12, 2.0e12, 3.0e12], [4.0e12, 5.0e12, 6.0e12]],
    )


@pytest.mark.parametrize(
    ("temperatures", "pressures", "message"),
    [
        ([[1000.0]], [0.1], "one-dimensional"),
        ([1000.0], [[0.1]], "one-dimensional"),
        ([1000.0, 1200.0], [0.1], "same one-dimensional shape"),
        ([], [], "at least one layer"),
        ([0.0], [0.1], "strictly positive"),
        ([1000.0], [-0.1], "strictly positive"),
        ([np.nan], [0.1], "finite"),
    ],
)
def test_profile_validation_rejects_invalid_shapes_and_values(
    temperatures,
    pressures,
    message,
):
    with pytest.raises(ValueError, match=message):
        _validate_profile(temperatures, pressures)


def test_table_parser_rejects_row_width_mismatch():
    malformed = """\
#p(bar) T(K) n_<tot>(cm-3) n_g(cm-3) m(u) H
1.0 1000.0 9.0e17 7.0e17 2.3
"""

    with pytest.raises(ValueError, match="has 5 columns; expected 6"):
        _parse_chemistry_output(malformed)


@pytest.mark.parametrize(
    ("option", "value", "message"),
    [
        ("verbosity", 0, "verbosity"),
        ("verbosity", 5, "verbosity"),
        ("verbosity", 1.5, "verbosity"),
        ("chemistry_accuracy", 0.0, "chemistry_accuracy"),
        ("chemistry_accuracy", np.nan, "chemistry_accuracy"),
        ("chemistry_accuracy", True, "chemistry_accuracy"),
        (
            "element_conservation_accuracy",
            -1.0e-4,
            "element_conservation_accuracy",
        ),
        ("max_chemistry_iterations", 0, "max_chemistry_iterations"),
        ("max_internal_iterations", 1.5, "max_internal_iterations"),
        ("chemistry_mode", "rainout", "chemistry_mode"),
    ],
)
def test_executable_adapter_rejects_invalid_options(
    tmp_path,
    option,
    value,
    message,
):
    paths = _make_input_files(tmp_path)
    kwargs = {
        "executable": paths["executable"],
        "temperatures": [1000.0],
        "pressures": [0.1],
        "element_abundance_file": paths["element_abundance_file"],
        "gas_logk_file": paths["gas_logk_file"],
        "condensate_logk_file": paths["condensate_logk_file"],
        option: value,
    }

    with pytest.raises(ValueError, match=message):
        run_fastchem_executable(**kwargs)


def test_executable_adapter_writes_ce_nd_inputs_and_parses_outputs(
    tmp_path,
    monkeypatch,
):
    paths = _make_input_files(tmp_path)

    def fake_run(args, *, cwd, check, capture_output, text):
        work_directory = Path(cwd)
        assert args == [str(paths["executable"].resolve()), "config.input"]
        assert check is False
        assert capture_output is True
        assert text is True

        profile = (work_directory / "profile.dat").read_text(encoding="utf-8")
        profile_rows = [line.split() for line in profile.splitlines()]
        np.testing.assert_allclose(
            np.asarray(profile_rows, dtype=float),
            [[0.1, 1000.0], [0.2, 1200.0]],
        )

        config = (work_directory / "config.input").read_text(encoding="utf-8")
        config_values = [
            line.strip()
            for line in config.splitlines()
            if line.strip() and not line.startswith("#")
        ]
        assert config_values[:6] == [
            "profile.dat",
            "ce",
            "chemistry.dat condensates.dat",
            "monitor.dat",
            "2",
            "ND",
        ]
        assert "element_abundances.dat" in config_values
        assert "gas_logk.dat condensate_logk.dat" in config_values
        assert (work_directory / "element_abundances.dat").read_text(
            encoding="utf-8"
        ) == "element_abundance_file\n"
        assert (work_directory / "gas_logk.dat").read_text(
            encoding="utf-8"
        ) == "gas_logk_file\n"
        assert (work_directory / "condensate_logk.dat").read_text(
            encoding="utf-8"
        ) == "condensate_logk_file\n"

        (work_directory / "chemistry.dat").write_text(
            CHEMISTRY_OUTPUT, encoding="utf-8"
        )
        (work_directory / "monitor.dat").write_text(
            MONITOR_OUTPUT, encoding="utf-8"
        )
        (work_directory / "condensates.dat").write_text(
            CONDENSATE_OUTPUT, encoding="utf-8"
        )
        return subprocess.CompletedProcess(
            args=args,
            returncode=0,
            stdout="FastChem finished!\n",
            stderr="",
        )

    monkeypatch.setattr(subprocess, "run", fake_run)

    result = run_fastchem_executable(
        executable=paths["executable"],
        temperatures=np.asarray([1000.0, 1200.0]),
        pressures=np.asarray([0.1, 0.2]),
        element_abundance_file=paths["element_abundance_file"],
        gas_logk_file=paths["gas_logk_file"],
        condensate_logk_file=paths["condensate_logk_file"],
        verbosity=2,
    )

    assert result.gas_names == ("H", "H2")
    assert result.condensate_names == ("Zn(s,l)", "Zn(s,l)", "MgO(s)")
    assert result.gas_number_densities.shape == (2, 2)
    assert result.condensate_number_densities.shape == (2, 3)
    np.testing.assert_array_equal(result.convergence_status, ["ok", "fail"])
    np.testing.assert_array_equal(
        result.element_conservation_status, ["ok", "fail"]
    )
    np.testing.assert_array_equal(result.converged, [True, False])
    np.testing.assert_array_equal(result.elements_conserved, [True, False])
    assert result.status is result.convergence_status
    assert result.iterations is result.total_iterations
    assert result.stdout == "FastChem finished!\n"
    assert result.chemistry_mode == "equilibrium_condensation"


def test_executable_adapter_runs_gas_mode_without_condensate_files(
    tmp_path,
    monkeypatch,
):
    paths = _make_input_files(tmp_path)
    gas_monitor_output = MONITOR_OUTPUT.replace(
        "0 2 13 67 ok ok",
        "0 2 13 0 ok ok",
    ).replace(
        "1 3 17 71 fail fail",
        "1 3 17 0 fail fail",
    )

    def fake_run(args, *, cwd, check, capture_output, text):
        work_directory = Path(cwd)
        assert args == [str(paths["executable"].resolve()), "config.input"]
        assert check is False
        assert capture_output is True
        assert text is True

        config = (work_directory / "config.input").read_text(encoding="utf-8")
        config_values = [
            line.strip()
            for line in config.splitlines()
            if line.strip() and not line.startswith("#")
        ]
        assert config_values[:6] == [
            "profile.dat",
            "g",
            "chemistry.dat",
            "monitor.dat",
            "2",
            "ND",
        ]
        assert "gas_logk.dat" in config_values
        assert "condensate_logk.dat" not in config_values
        assert not (work_directory / "condensate_logk.dat").exists()

        (work_directory / "chemistry.dat").write_text(
            CHEMISTRY_OUTPUT, encoding="utf-8"
        )
        (work_directory / "monitor.dat").write_text(
            gas_monitor_output, encoding="utf-8"
        )
        return subprocess.CompletedProcess(
            args=args,
            returncode=0,
            stdout="FastChem gas finished!\n",
            stderr="",
        )

    monkeypatch.setattr(subprocess, "run", fake_run)

    result = run_fastchem_executable(
        executable=paths["executable"],
        temperatures=np.asarray([1000.0, 1200.0]),
        pressures=np.asarray([0.1, 0.2]),
        element_abundance_file=paths["element_abundance_file"],
        gas_logk_file=paths["gas_logk_file"],
        chemistry_mode="gas",
        verbosity=2,
    )

    assert result.chemistry_mode == "gas"
    assert result.condensate_names == ()
    assert result.condensate_number_densities.shape == (2, 0)
    np.testing.assert_array_equal(result.condensation_iterations, [0, 0])
    assert result.stdout == "FastChem gas finished!\n"


def test_executable_adapter_raises_on_nonzero_return_code(
    tmp_path,
    monkeypatch,
):
    paths = _make_input_files(tmp_path)

    def fake_run(*args, **kwargs):
        return subprocess.CompletedProcess(
            args=args[0],
            returncode=7,
            stdout="model output",
            stderr="model error",
        )

    monkeypatch.setattr(subprocess, "run", fake_run)

    with pytest.raises(RuntimeError, match="return code 7") as exc_info:
        run_fastchem_executable(
            executable=paths["executable"],
            temperatures=[1000.0],
            pressures=[0.1],
            element_abundance_file=paths["element_abundance_file"],
            gas_logk_file=paths["gas_logk_file"],
            condensate_logk_file=paths["condensate_logk_file"],
        )

    assert "model output" in str(exc_info.value)
    assert "model error" in str(exc_info.value)


def test_executable_adapter_rejects_output_layer_mismatch(
    tmp_path,
    monkeypatch,
):
    paths = _make_input_files(tmp_path)

    def fake_run(args, *, cwd, **kwargs):
        work_directory = Path(cwd)
        (work_directory / "chemistry.dat").write_text(
            "\n".join(CHEMISTRY_OUTPUT.splitlines()[:2]) + "\n",
            encoding="utf-8",
        )
        (work_directory / "monitor.dat").write_text(
            MONITOR_OUTPUT, encoding="utf-8"
        )
        (work_directory / "condensates.dat").write_text(
            CONDENSATE_OUTPUT, encoding="utf-8"
        )
        return subprocess.CompletedProcess(
            args=args,
            returncode=0,
            stdout="",
            stderr="",
        )

    monkeypatch.setattr(subprocess, "run", fake_run)

    with pytest.raises(
        ValueError,
        match="chemistry output has 1 layers; expected exactly 2",
    ):
        run_fastchem_executable(
            executable=paths["executable"],
            temperatures=[1000.0, 1200.0],
            pressures=[0.1, 0.2],
            element_abundance_file=paths["element_abundance_file"],
            gas_logk_file=paths["gas_logk_file"],
            condensate_logk_file=paths["condensate_logk_file"],
        )
