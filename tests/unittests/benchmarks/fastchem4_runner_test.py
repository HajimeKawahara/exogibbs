"""Unit tests for the FastChem4 production comparison orchestrator."""

import argparse
import json
from pathlib import Path
import runpy
import sys
from types import SimpleNamespace

import numpy as np
import pytest


REPOSITORY_ROOT = Path(__file__).resolve().parents[3]
_module = runpy.run_path(
    REPOSITORY_ROOT
    / "benchmarks"
    / "fastchem4"
    / "run_production_comparison.py"
)
_common_gibbs_over_rt = _module["_common_gibbs_over_rt"]
_fastchem_common_basis_state = _module["_fastchem_common_basis_state"]
_main = _module["main"]
_preflight = _module["_preflight"]
_preflight_path = _module["_preflight_path"]
_parse_point = _module["_parse_point"]
_source_data_parity = _module["_source_data_parity"]
_summary = _module["_summary"]
_write_failure_report = _module["_write_failure_report"]


def _complete_layer() -> dict:
    return {
        "status": {
            "exogibbs": {"converged": True},
            "fastchem": {
                "converged": True,
                "elements_conserved": True,
            },
        },
        "element_budget": {
            "exogibbs": {
                "finite": True,
                "max_absolute_relative_residual": 1.0e-12,
            },
            "fastchem": {
                "finite": True,
                "max_absolute_relative_residual": 2.0e-10,
            },
        },
        "total_gas": {
            "exogibbs_normalized_amount": 0.5,
            "fastchem_normalized_amount": 0.5,
        },
        "gas_major_species": {
            "finite": True,
            "max_absolute_log10_ratio": 0.25,
        },
        "condensates": {
            "1e-20": {"finite": True},
            "1e-12": {"finite": True},
            "1e-08": {"finite": True},
        },
        "gibbs_over_rt": {
            "exogibbs": -10.0,
            "fastchem_state": -10.00000003,
            "exogibbs_minus_fastchem": 3.0e-8,
        },
    }


def test_parse_point_preserves_temperature_pressure_order() -> None:
    assert _parse_point("1400,0.1") == (1400.0, 0.1)


def test_preflight_path_is_specific_to_output_stem(tmp_path: Path) -> None:
    output = tmp_path / "temperature_scan.json"

    assert _preflight_path(output) == tmp_path / "temperature_scan.preflight.json"


@pytest.mark.parametrize(
    "value",
    ("1400", "1400,0.1,1.0", "nan,0.1", "1400,0"),
)
def test_parse_point_rejects_invalid_values(value: str) -> None:
    with pytest.raises(argparse.ArgumentTypeError):
        _parse_point(value)


@pytest.mark.parametrize(
    "version_label",
    (
        "",
        "unknown",
        "3.1.3",
        "latest",
        "4.0.3 (wrong commit)",
        "4.1.0 (ae67cbd)",
    ),
)
def test_preflight_rejects_nonreference_version_labels(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    version_label: str,
) -> None:
    monkeypatch.setitem(
        _preflight.__globals__,
        "_source_data_parity",
        lambda source_root: {
            "files": {},
            "all_source_files_present": True,
            "all_byte_identical": True,
        },
    )
    monkeypatch.setitem(
        _preflight.__globals__,
        "_git_metadata",
        lambda source_root: {
            "available": True,
            "commit": "ae67cbd559bc64a3233a1cee6030b8e6b50520de",
            "describe": "v4.0.3",
            "worktree_clean": True,
        },
    )
    args = SimpleNamespace(
        fastchem_executable=Path(sys.executable),
        fastchem_source_root=tmp_path,
        fastchem_version_label=version_label,
    )

    report = _preflight(args)

    assert (
        report["checks"]["fastchem_version_label_identifies_reference"]
        is False
    )
    assert report["passed"] is False


def test_preflight_accepts_recorded_clean_v4_source(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.setitem(
        _preflight.__globals__,
        "_source_data_parity",
        lambda source_root: {
            "files": {},
            "all_source_files_present": True,
            "all_byte_identical": True,
        },
    )
    monkeypatch.setitem(
        _preflight.__globals__,
        "_git_metadata",
        lambda source_root: {
            "available": True,
            "commit": "ae67cbd559bc64a3233a1cee6030b8e6b50520de",
            "describe": "v4.0.3",
            "worktree_clean": True,
        },
    )
    args = SimpleNamespace(
        fastchem_executable=Path(sys.executable),
        fastchem_source_root=tmp_path,
        fastchem_version_label="4.0.3 (ae67cbd)",
    )

    report = _preflight(args)

    assert all(report["checks"].values())
    assert report["passed"] is True


def test_preflight_rejects_nonreference_source_commit(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.setitem(
        _preflight.__globals__,
        "_source_data_parity",
        lambda source_root: {
            "files": {},
            "all_source_files_present": True,
            "all_byte_identical": True,
        },
    )
    monkeypatch.setitem(
        _preflight.__globals__,
        "_git_metadata",
        lambda source_root: {
            "available": True,
            "commit": "different",
            "describe": "v4.0.3",
            "worktree_clean": True,
        },
    )
    args = SimpleNamespace(
        fastchem_executable=Path(sys.executable),
        fastchem_source_root=tmp_path,
        fastchem_version_label="4.0.3 (ae67cbd)",
    )

    report = _preflight(args)

    assert report["checks"]["fastchem_source_commit_is_reference"] is False
    assert report["passed"] is False


def test_source_data_parity_records_missing_packaged_file(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    packaged = tmp_path / "packaged"
    source = tmp_path / "source"
    source_abundance = (
        source / "input" / "element_abundances" / "asplund_2021.dat"
    )
    source_gas = source / "input" / "logK" / "logK_wo_ions.dat"
    source_cond = source / "input" / "logK" / "logK_condensates.dat"
    for path in (source_abundance, source_gas, source_cond):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("shared\n", encoding="utf-8")

    packaged_gas = packaged / "logK_wo_ions.dat"
    packaged_cond = packaged / "logK_condensates.dat"
    packaged.mkdir()
    packaged_gas.write_text("shared\n", encoding="utf-8")
    packaged_cond.write_text("shared\n", encoding="utf-8")
    monkeypatch.setitem(
        _source_data_parity.__globals__,
        "ELEMENT_ABUNDANCE_FILE",
        packaged / "missing_abundances.dat",
    )
    monkeypatch.setitem(
        _source_data_parity.__globals__,
        "GAS_LOGK_FILE",
        packaged_gas,
    )
    monkeypatch.setitem(
        _source_data_parity.__globals__,
        "CONDENSATE_LOGK_FILE",
        packaged_cond,
    )

    report = _source_data_parity(source)

    assert report["all_packaged_files_present"] is False
    assert report["all_byte_identical"] is False
    assert report["files"]["element_abundances"]["packaged_sha256"] is None


def test_source_data_parity_detects_byte_mismatch(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    packaged = tmp_path / "packaged"
    source = tmp_path / "source"
    source_abundance = (
        source / "input" / "element_abundances" / "asplund_2021.dat"
    )
    source_gas = source / "input" / "logK" / "logK_wo_ions.dat"
    source_cond = source / "input" / "logK" / "logK_condensates.dat"
    packaged_abundance = packaged / "asplund_2021.dat"
    packaged_gas = packaged / "logK_wo_ions.dat"
    packaged_cond = packaged / "logK_condensates.dat"
    for path in (
        source_abundance,
        source_gas,
        source_cond,
        packaged_abundance,
        packaged_gas,
        packaged_cond,
    ):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("shared\n", encoding="utf-8")
    source_cond.write_text("different\n", encoding="utf-8")
    monkeypatch.setitem(
        _source_data_parity.__globals__,
        "ELEMENT_ABUNDANCE_FILE",
        packaged_abundance,
    )
    monkeypatch.setitem(
        _source_data_parity.__globals__,
        "GAS_LOGK_FILE",
        packaged_gas,
    )
    monkeypatch.setitem(
        _source_data_parity.__globals__,
        "CONDENSATE_LOGK_FILE",
        packaged_cond,
    )

    report = _source_data_parity(source)

    assert report["all_packaged_files_present"] is True
    assert report["all_source_files_present"] is True
    assert report["files"]["condensate_logk"]["byte_identical"] is False
    assert report["all_byte_identical"] is False


def test_common_gibbs_objective_uses_one_recorded_basis() -> None:
    setup = SimpleNamespace(
        gas_setup=SimpleNamespace(
            hvector_func=lambda temperature: np.asarray([1.0, 2.0])
        ),
        condensate_setup=SimpleNamespace(
            hvector_func=lambda temperature: np.asarray([3.0])
        ),
    )
    gas = np.asarray([1.0, 3.0])
    condensates = np.asarray([2.0])

    result = _common_gibbs_over_rt(
        setup=setup,
        temperature=1000.0,
        pressure=1.0,
        gas_amounts=gas,
        condensate_amounts=condensates,
    )

    expected = (
        gas[0] * (1.0 + np.log(gas[0]) - np.log(np.sum(gas)))
        + gas[1] * (2.0 + np.log(gas[1]) - np.log(np.sum(gas)))
        + condensates[0] * 3.0
    )
    assert result == pytest.approx(expected)


def test_fastchem_common_basis_state_normalizes_all_amounts() -> None:
    gas, condensates, mixing_ratios = _fastchem_common_basis_state(
        gas_number_densities=np.asarray([[2.0, 6.0]]),
        condensate_number_densities=np.asarray([[4.0]]),
        total_element_density=np.asarray([4.0]),
    )

    np.testing.assert_allclose(gas, [[0.5, 1.5]])
    np.testing.assert_allclose(condensates, [[1.0]])
    np.testing.assert_allclose(mixing_ratios, [[0.25, 0.75]])


def test_fastchem_common_basis_state_rejects_zero_element_density() -> None:
    with pytest.raises(RuntimeError, match="invalid layers: \\[0\\]"):
        _fastchem_common_basis_state(
            gas_number_densities=np.ones((1, 1)),
            condensate_number_densities=np.zeros((1, 1)),
            total_element_density=np.zeros(1),
        )


def test_summary_marks_complete_run_without_claiming_scientific_agreement() -> None:
    layer = _complete_layer()

    report = _summary(
        preflight={"passed": True},
        layers=[layer],
        gas_catalog_match=True,
        condensate_catalog_match=True,
    )

    assert report["status"] == "complete"
    assert report["comparison_completed"] is True
    assert report["scientific_acceptance_thresholds_applied"] is False
    assert report["max_major_gas_abs_log10_ratio"] == pytest.approx(0.25)


def test_summary_rejects_nonfinite_condensate_metrics() -> None:
    layer = _complete_layer()
    layer["condensates"]["1e-08"]["finite"] = False

    report = _summary(
        preflight={"passed": True},
        layers=[layer],
        gas_catalog_match=True,
        condensate_catalog_match=True,
    )

    assert report["status"] == "incomplete"
    assert report["comparison_completed"] is False
    assert report["all_condensate_metrics_finite"] is False
    assert report["all_comparison_metrics_finite"] is False


def test_summary_rejects_nonfinite_total_gas_metrics() -> None:
    layer = _complete_layer()
    layer["total_gas"]["fastchem_normalized_amount"] = np.nan

    report = _summary(
        preflight={"passed": True},
        layers=[layer],
        gas_catalog_match=True,
        condensate_catalog_match=True,
    )

    assert report["status"] == "incomplete"
    assert report["comparison_completed"] is False
    assert report["all_total_gas_metrics_finite"] is False
    assert report["all_comparison_metrics_finite"] is False


def test_failure_report_replaces_both_output_formats(tmp_path: Path) -> None:
    output = tmp_path / "summary.json"
    output.write_text("stale JSON", encoding="utf-8")
    output.with_suffix(".md").write_text("stale Markdown", encoding="utf-8")

    _write_failure_report(
        output_path=output,
        preflight={"passed": True},
        stage="comparison",
        error=RuntimeError("catalog mismatch"),
    )

    payload = json.loads(output.read_text(encoding="utf-8"))
    assert payload["summary"]["status"] == "failed"
    assert payload["summary"]["comparison_completed"] is False
    assert payload["failure"] == {
        "stage": "comparison",
        "error_type": "RuntimeError",
        "message": "catalog mismatch",
    }
    markdown = output.with_suffix(".md").read_text(encoding="utf-8")
    assert "Status: `failed` during `comparison`." in markdown
    assert "stale" not in markdown


def test_main_writes_failure_artifact_when_preflight_fails(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    output = tmp_path / "summary.json"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "run_production_comparison.py",
            "--fastchem-executable",
            sys.executable,
            "--fastchem-version-label",
            "4.0.3 (ae67cbd)",
            "--output",
            str(output),
        ],
    )
    monkeypatch.setitem(
        _main.__globals__,
        "_preflight",
        lambda args: {"passed": False},
    )

    with pytest.raises(RuntimeError, match="preflight failed"):
        _main()

    payload = json.loads(output.read_text(encoding="utf-8"))
    assert payload["failure"]["stage"] == "preflight"
    assert payload["summary"]["comparison_completed"] is False
    assert (tmp_path / "summary.preflight.json").is_file()


def test_main_writes_failure_artifact_when_comparison_raises(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    output = tmp_path / "summary.json"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "run_production_comparison.py",
            "--fastchem-executable",
            sys.executable,
            "--fastchem-version-label",
            "4.0.3 (ae67cbd)",
            "--output",
            str(output),
        ],
    )
    monkeypatch.setitem(
        _main.__globals__,
        "_preflight",
        lambda args: {"passed": True},
    )
    monkeypatch.setitem(
        _main.__globals__,
        "_configure_jax",
        lambda platform: None,
    )

    def fail_comparison(args: SimpleNamespace, preflight: dict) -> None:
        raise ValueError("synthetic comparison failure")

    monkeypatch.setitem(_main.__globals__, "_run", fail_comparison)

    with pytest.raises(ValueError, match="synthetic comparison failure"):
        _main()

    payload = json.loads(output.read_text(encoding="utf-8"))
    assert payload["failure"]["stage"] == "comparison"
    assert payload["failure"]["error_type"] == "ValueError"
    assert payload["summary"]["comparison_completed"] is False
