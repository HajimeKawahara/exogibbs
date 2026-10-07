"""Lightweight coverage and failure contracts for executable RST examples."""

import json
from types import SimpleNamespace

import numpy as np
import pytest

from benchmarks.documented_examples import check_inline


@pytest.mark.parametrize("case", check_inline.CASES.values(), ids=check_inline.CASES)
def test_registered_document_blocks_exist_and_compile(case) -> None:
    blocks = check_inline.extract_blocks(check_inline.REPOSITORY_ROOT / case["document"])
    for index in case["blocks"]:
        assert check_inline.ANCHORS[case["kind"]] in blocks[index]
        compile(blocks[index], case["document"], "exec")


def test_rainout_runs_after_its_documented_imports_and_setup() -> None:
    case = check_inline.CASES["condensate_profile"]
    blocks = check_inline.extract_blocks(check_inline.REPOSITORY_ROOT / case["document"])
    first, second = (blocks[index] for index in case["blocks"])
    assert "setup = condensate_chemical_setup(" in first
    assert "rainout=True" in second


def test_trace_element_budget_has_no_acceptance_floor() -> None:
    with pytest.raises(RuntimeError, match="Positive element budget error"):
        check_inline._budget_audit(np.array([1.0, 2e-40]), np.array([1.0, 1e-40]), 1e-3)


def test_gas_stationarity_rejects_finite_conserved_nonequilibrium() -> None:
    formula = np.array([[1.0, 2.0]])
    setup = SimpleNamespace(
        formula_matrix=formula,
        hvector_func=lambda temperature: -0.5 * formula[0] - np.log(0.5),
    )
    amounts = np.array([2.0, 0.5])
    namespace = {
        "setup": setup, "T": 1500.0, "P": 1.0, "b": np.array([3.0]),
        "result": SimpleNamespace(n=amounts, ln_n=np.log(amounts), x=amounts / 2.5, ntot=2.5),
    }
    audit = check_inline.audit_result("gas", namespace)
    assert not audit["accepted"]
    assert audit["amount_weighted_stationarity_norm"] > audit["effective_epsilon_crit"]


@pytest.mark.parametrize("trace_amount, expected", [(1e-15, True), (1e-4, False)])
def test_gas_budget_gate_matches_kernel_absolute_amount_norm(trace_amount, expected) -> None:
    from exogibbs.equilibrium.gas.kernel.solver import compute_residuals_with_at_pi

    target = np.array([1.0, trace_amount])
    amounts = target * np.array([1.0, 1.5])
    fractions = amounts / amounts.sum()
    setup = SimpleNamespace(formula_matrix=np.eye(2), hvector_func=lambda temperature: -np.log(fractions))
    result = SimpleNamespace(n=amounts, x=fractions, ln_n=np.log(amounts), ntot=amounts.sum())
    audit = check_inline._gas_audit(setup, result, 1500.0, 1.0, target)
    kernel_residual = float(compute_residuals_with_at_pi(
        amounts, amounts.sum(), target, np.zeros(2), amounts, np.zeros(2),
    ))
    np.testing.assert_allclose(audit["projected_amount_residual"], kernel_residual, rtol=1e-10)
    assert audit["maximum_relative_element_error"] == pytest.approx(0.5)
    assert audit["accepted"] is expected


def test_condensate_uses_public_budget_floor_and_reports_raw_trace_error() -> None:
    budget = np.array([1.0, 1e-12])
    amounts = np.array([1.0, 2e-12])
    setup = SimpleNamespace(
        formula_matrix=np.eye(2), formula_matrix_cond=np.zeros((2, 1)), elements=("H", "C"),
    )
    result = SimpleNamespace(
        converged=True, status="converged", gas_n=amounts, gas_x=amounts / amounts.sum(),
        gas_ntot=amounts.sum(), condensate_amounts=np.zeros(1), acceptance_tier="converged", diagnostics=None,
    )
    audit = check_inline._condensate_audit(setup, result, budget)
    assert audit["accepted"]
    assert audit["maximum_relative_element_error"] == pytest.approx(1.0)
    assert audit["max_relative_element_error"] < 1e-3


def test_nonconverged_condensate_is_rejected_before_plot_values() -> None:
    result = SimpleNamespace(converged=False, status="not_converged")
    with pytest.raises(RuntimeError, match="Condensate solve failed"):
        check_inline.audit_result("condensate", {"setup": None, "result": result, "b": None})


@pytest.mark.parametrize("value", [0.25, 0.0])
def test_run_executes_extracted_source_and_writes_strict_audit(monkeypatch, tmp_path, value) -> None:
    source = f"def h2_hirschmann2012():\n    return {value}\nx_h2 = h2_hirschmann2012()\n"
    monkeypatch.setattr(check_inline, "extract_blocks", lambda path: (source,))
    report = check_inline.run("solubility", tmp_path)
    assert report["accepted"] == (value > 0)
    saved = json.loads((tmp_path / "inline_audit.json").read_text())
    assert saved["accepted"] == report["accepted"]
    if report["accepted"]:
        assert saved["checks"][0]["x_h2"] == value
        assert len(saved["checks"][0]["source_sha256"]) == 64


def test_changed_block_manifest_fails_without_executing(monkeypatch, tmp_path) -> None:
    monkeypatch.setattr(check_inline, "extract_blocks", lambda path: ("raise AssertionError('should not execute')",))
    report = check_inline.run("solubility", tmp_path)
    assert not report["accepted"]
    assert "explicit manifest" in report["error"]
