"""Verify the archived C2H substitution without an external solver or MCEG."""

import json
from pathlib import Path
import runpy
import sys

import numpy as np
import pytest
from scipy.special import logsumexp

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "mceg/reviews"))
from gas_source_comparison import file_hash

DATA = ROOT / "tests/data/validation/c2h_equilibrium_impact/20261102"


def test_correction_rebuild_and_archived_execution_hashes():
    builder = runpy.run_path(str(ROOT / "mceg/reviews/build_c2h_correction.py"))
    expected = (ROOT / "src/exogibbs/data/curated/c2h_v1.json").read_text()
    assert json.dumps(builder["build_data"](), indent=2, allow_nan=False)+"\n" == expected
    receipt = json.loads((DATA / "receipt.json").read_text())
    assert receipt["result"] == "passed"
    for name, digest in receipt["selected_file_sha256"].items():
        assert file_hash(ROOT / name) == digest, name
    for name, digest in receipt["output_sha256"].items():
        assert file_hash(DATA / name) == digest, name


def test_full_archive_retains_fixed_network_budgets_and_all_amounts():
    with np.load(DATA / "equilibria.npz") as saved:
        assert saved["ln_n"].shape == saved["h"].shape == (2, 75, 537)
        assert np.isfinite(saved["ln_n"]).all()
        np.testing.assert_allclose(saved["n"], np.exp(saved["ln_n"]), atol=0, rtol=1e-14)
        logx = saved["ln_n"]-logsumexp(saved["ln_n"], axis=-1, keepdims=True)
        np.testing.assert_allclose(saved["ln_x"], logx, atol=1e-13, rtol=0)
        np.testing.assert_allclose(saved["x"], np.exp(logx), atol=0, rtol=1e-14)
        np.testing.assert_allclose(saved["delta_log10_x"], (logx[1]-logx[0])/np.log(10), atol=1e-13)
        index = list(saved["species"]).index("C2H1")
        np.testing.assert_allclose(np.delete(saved["h"][1]-saved["h"][0], index, axis=-1), 0, atol=1e-12)
        c, o, hydrogen = [list(saved["elements"]).index(e) for e in ("C", "O", "H")]
        budget = saved["budget"]
        np.testing.assert_allclose(budget[:, c]/budget[:, o], saved["C_over_O"])
        relative = budget/budget[:, hydrogen, None]
        np.testing.assert_allclose(np.delete(relative, c, axis=1),
                                   np.broadcast_to(np.delete(relative[0], c), (75, 27)), atol=1e-15)
        for values in (saved["ln_n"], saved["strict_ln_n"]):
            amounts, matrix = np.exp(values), saved["formula_matrix"]
            positive = budget[0] > 0
            assert np.max(abs(amounts@matrix.T-budget)[..., positive]/budget[:, positive]) < 1e-5
            charge = matrix[list(saved["elements"]).index("e-")]
            assert np.max(abs(amounts@charge)/(amounts@abs(charge))) < 1e-12
            refs = [list(saved["species"]).index("e1-" if e == "e-" else e+"1") for e in saved["elements"]]
            logp = values-logsumexp(values, axis=-1, keepdims=True)+np.log(saved["pressure_bar"])[None, :, None]
            assert np.max(abs(logp-logp[..., refs]@matrix+saved["h"])) < 1e-8
        np.testing.assert_allclose(saved["carbon_fraction"].sum(axis=-1), 1, rtol=1e-5)


def test_carbon_redistribution_and_independent_solver_checks():
    summary = json.loads((DATA / "summary.json").read_text())
    assert summary["paired_cases"] == 75
    assert summary["endpoint_solves"] == summary["tighter_cold_solves"] == 150
    assert summary["tolerance_stability"]["max_abs_delta_x"] < 1e-12
    assert len(summary["independent_solver"]) == 2
    assert max(r["max_abs_delta_log10_x"] for r in summary["independent_solver"]) < 1e-6
    with np.load(DATA / "equilibria.npz") as saved:
        index = np.flatnonzero((saved["C_over_O"] == 1.5) & (saved["temperature_K"] == 2000)
                               & (saved["pressure_bar"] == 1e-6))[0]
        np.testing.assert_allclose(saved["lambda_ln_n"][[0, -1]], saved["strict_ln_n"][:, index], atol=2e-9)
        i = list(saved["species"]).index("C2H1")
        assert saved["carbon_fraction"][0, index, i] == pytest.approx(.20212973, rel=1e-6)
        assert saved["carbon_fraction"][1, index, i] == pytest.approx(.00660778, rel=1e-6)
    diagnostics = json.loads((DATA / "diagnostics.json").read_text())
    assert len(diagnostics) == 300 and all(r["converged"] for r in diagnostics)
    assert all(r["converged"] for r in json.loads((DATA / "lambda_diagnostics.json").read_text()))
    assert json.loads((DATA / "precision_probe.json").read_text())["converged"] is False
