"""Regression checks for the gallery worker's independent acceptance gates."""

from pathlib import Path
import sys
from types import SimpleNamespace

import numpy as np
import pytest

from benchmarks.documented_examples import check_exoeos


def _setup_and_result():
    formula = np.array([[1.0, 2.0]])
    fractions = np.array([0.5, 0.5])
    setup = SimpleNamespace(
        formula_matrix=formula,
        hvector_func=lambda temperature: -0.5 * formula[0] - np.log(fractions),
    )
    return setup, SimpleNamespace(n=np.ones(2), x=fractions)


def test_stationarity_rejects_finite_conserved_nonequilibrium() -> None:
    setup, result = _setup_and_result()
    valid = check_exoeos._audit_result(setup, result, 1500.0, 1.0, np.array([3.0]))
    assert valid["accepted"]
    invalid = SimpleNamespace(n=np.array([2.0, 0.5]), x=np.array([0.8, 0.2]))
    rejected = check_exoeos._audit_result(setup, invalid, 1500.0, 1.0, np.array([3.0]))
    assert rejected["maximum_relative_element_error"] == 0.0
    assert rejected["normalization_error"] == 0.0
    assert not rejected["accepted"]


@pytest.mark.parametrize("create_figure", [False, True])
def test_worker_requires_fresh_artifact_and_audits_original_solve(
    monkeypatch, tmp_path, create_figure,
) -> None:
    setup, result = _setup_and_result()
    calls = []

    def solve(*args, **kwargs):
        calls.append(kwargs)
        return result

    example = SimpleNamespace(solve=solve, PRESSURES_BAR=np.array([1.0]))

    def main():
        example.solve(setup, 1500.0, 1.0, np.array([3.0]), Pref=1.0)
        example.solve(
            setup, 1500.0, 1.0, np.array([3.0]),
            lnphi_func=lambda temperature, pressure, fractions: np.zeros(2),
        )
        if create_figure:
            Path(sys.argv[-1]).write_bytes(b"fresh figure")

    example.main = main
    monkeypatch.setattr(check_exoeos, "_load_example", lambda: example)
    (tmp_path / "exoeos_pure_fugacity.png").write_bytes(b"stale figure")
    original_argv = sys.argv
    audit = check_exoeos.run(tmp_path)
    assert audit["accepted"] == create_figure
    assert len(calls) == 2
    assert all("return_diagnostics" not in call for call in calls)
    assert [layer["mode"] for layer in audit["layers"]] == ["ideal", "pure_fugacity"]
    assert sys.argv is original_argv
    assert (tmp_path / "exoeos_audit.json").is_file()


def test_missing_optional_provider_is_a_failed_run(monkeypatch, tmp_path) -> None:
    example = SimpleNamespace(solve=lambda: None, main=lambda: None, PRESSURES_BAR=[1.0])
    monkeypatch.setattr(check_exoeos, "_load_example", lambda: example)
    audit = check_exoeos.run(tmp_path)
    assert not audit["accepted"]
    assert not audit["layers"]


def test_nonfinite_result_produces_strict_failed_audit() -> None:
    setup, result = _setup_and_result()
    result.n[0] = np.nan
    audit = check_exoeos._audit_result(setup, result, 1500.0, 1.0, np.array([3.0]))
    assert not audit["accepted"]
    assert audit["stationarity_error"] is None
