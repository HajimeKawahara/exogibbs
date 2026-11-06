"""Independently replay the archived water substitution without new solves."""

from hashlib import sha256
import json
from pathlib import Path
import sys

import numpy as np
import pytest
from scipy.special import logsumexp


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "mceg/reviews"))
from curated_tio_history import historical_source
import h2o_candidate as candidate

DATA = ROOT / "tests/data/validation/h2o_equilibrium_impact/20261104"
BASELINE = ROOT / "tests/data/validation/curated_tio_adoption/20261103/ions.npz"


@pytest.fixture(scope="module")
def saved():
    with np.load(DATA / "equilibria.npz") as archive:
        return {name: archive[name].copy() for name in archive.files}


def log_fractions(values):
    return values-logsumexp(values, axis=-1, keepdims=True)


def check_constraints(saved, ln_n, h, budget, pressure):
    assert np.isfinite(ln_n).all() and np.isfinite(h).all()
    n = np.exp(ln_n)
    assert np.all(n > 0) and np.isfinite(n).all()
    matrix = saved["formula_matrix"]
    elements, species = saved["elements"].tolist(), saved["species"].tolist()
    positive = np.asarray(budget).reshape(-1, len(elements))[0] > 0
    residual = np.abs(n @ matrix.T-budget)
    assert np.max(residual[..., positive]/np.asarray(budget)[..., positive]) < 1e-5
    # The thermal-scope response is small; the major H/C/O reservoirs must
    # close much more tightly than the permitted rare-element residual.
    for element in ("H", "C", "O"):
        j = elements.index(element)
        assert np.max(residual[..., j]/np.asarray(budget)[..., j]) < 1e-9
    charge = matrix[elements.index("e-")]
    assert np.max(abs(n @ charge)/(n @ abs(charge))) < 1e-12
    references = [species.index("e1-" if e == "e-" else e+"1") for e in elements]
    logp = log_fractions(ln_n)+np.log(np.asarray(pressure))[..., None]
    assert np.max(abs(logp-logp[..., references] @ matrix+h)) < 1e-8


def test_fixed_current_network_and_only_water_potential_changes(saved):
    assert saved["variants"].tolist() == ["current", "evaluated", "discrete"]
    assert saved["ln_n"].shape == saved["h"].shape == (3, 75, 537)
    with np.load(BASELINE) as baseline:
        for name in ("formula_matrix", "species", "elements", "budget", "temperature_K", "pressure_bar", "C_over_O"):
            np.testing.assert_array_equal(saved[name], baseline[name])
        np.testing.assert_array_equal(saved["ln_n"][0], baseline["ln_n"])
        np.testing.assert_allclose(saved["h"][0], baseline["h"], atol=1e-12, rtol=0)
    water = saved["species"].tolist().index("H2O1")
    delta = saved["h"][1:]-saved["h"][0]
    np.testing.assert_allclose(np.delete(delta, water, axis=-1), 0, atol=1e-12, rtol=0)
    assert np.max(abs(delta[..., water])) > .01
    for index, thermal in enumerate(("evaluated", "discrete"), 1):
        expected = [candidate.formation_h(float(t), thermal=thermal) for t in saved["temperature_K"]]
        np.testing.assert_allclose(saved["h"][index, :, water], expected, atol=2e-12, rtol=0)
    for amounts, h in zip(saved["ln_n"], saved["h"]):
        check_constraints(saved, amounts, h, saved["budget"], saved["pressure_bar"])
    assert log_fractions(saved["ln_n"]).min() < np.log(1e-100)


def test_tighter_controls_resolve_the_water_scope_response(saved):
    selected = saved["tight_case_indices"]
    assert selected.tolist() == sorted(set(selected.tolist())) and 1 <= len(selected) <= 4
    assert saved["tight_ln_n"].shape == (3, len(selected), 537)
    for index, amounts in enumerate(saved["tight_ln_n"]):
        check_constraints(saved, amounts, saved["h"][index, selected],
                          saved["budget"][selected], saved["pressure_bar"][selected])
    tight = log_fractions(saved["tight_ln_n"])
    original = log_fractions(saved["ln_n"][:, selected])
    water = saved["species"].tolist().index("H2O1")
    assert np.max(abs(tight-original)) < 1e-4
    assert np.max(abs(tight[..., water]-original[..., water])) < 1e-7
    visible = np.maximum(np.exp(tight), np.exp(original)) >= 1e-12
    assert np.max(abs((tight-original)[visible])) < 2e-6
    assert np.max(abs(np.exp(tight)-np.exp(original))) < 1e-9
    response_error = (tight[2, :, water]-tight[1, :, water])-(original[2, :, water]-original[1, :, water])
    assert np.max(abs(response_error)) < 2e-7


def test_finite_path_obeys_constraints_and_water_amount_monotonicity(saved):
    point = int(saved["highlighted_case_index"])
    assert point in saved["tight_case_indices"]
    fractions = saved["lambda_values"]
    np.testing.assert_array_equal(fractions, np.linspace(0, 1, 5))
    initial, final = saved["h"][[0, 1], point]
    path_h = initial+fractions[:, None]*(final-initial)
    check_constraints(saved, saved["lambda_ln_n"], path_h, saved["budget"][point], saved["pressure_bar"][point])
    np.testing.assert_allclose(log_fractions(saved["lambda_ln_n"][[0, -1]]),
                               log_fractions(saved["ln_n"][[0, 1], point]), atol=1e-4, rtol=0)
    water = saved["species"].tolist().index("H2O1")
    # Concavity of the minimized Gibbs energy fixes the response sign of
    # the extensive water amount when its own chemical potential varies.
    assert abs(final[water]-initial[water]) > 1e-6
    assert np.max(np.diff(saved["lambda_ln_n"][:, water])*(final[water]-initial[water])) < 1e-10


def test_each_endpoint_minimizes_its_own_gibbs_function(saved):
    logs = saved["ln_n"].astype(np.longdouble)
    n = np.exp(logs)
    logp = logs-np.log(n.sum(axis=-1))[..., None]+np.log(saved["pressure_bar"].astype(np.longdouble))[None, :, None]
    h = saved["h"].astype(np.longdouble)
    current, variant = [], []
    for index in (1, 2):
        current.append(np.sum(n[index]*(h[0]+logp[index])-n[0]*(h[0]+logp[0]), axis=-1))
        variant.append(np.sum(n[index]*(h[index]+logp[index])-n[0]*(h[index]+logp[0]), axis=-1))
    np.testing.assert_allclose(current, saved["delta_g_in_current"], atol=5e-14, rtol=1e-7)
    np.testing.assert_allclose(variant, saved["delta_g_in_variant"], atol=5e-14, rtol=1e-7)
    assert np.min(current) > -2e-12 and np.max(variant) < 2e-12
    refs = [saved["species"].tolist().index("e1-" if e == "e-" else e+"1") for e in saved["elements"]]
    budget_delta = (n[1:]-n[0]) @ saved["formula_matrix"].T
    potentials = h+logp
    # Remove the first-order effect of tiny feasibility residuals before
    # checking the remaining ideal-gas convexity gap.
    corrected_current = np.asarray(current)-(budget_delta*potentials[0][..., refs]).sum(axis=-1)
    corrected_variant = np.asarray(variant)-(budget_delta*potentials[1:][..., refs]).sum(axis=-1)
    assert corrected_current.min() > -5e-14 and corrected_variant.max() < 5e-14


def test_reported_responses_and_independent_solver_results(saved):
    summary = json.loads((DATA/"summary.json").read_text())
    assert summary["reused_baseline_cases"] == 75 and summary["new_endpoint_solves"] == 150
    assert summary["species"] == 537 and summary["curated_update"] == "none"
    ln_x = log_fractions(saved["ln_n"])
    water = saved["species"].tolist().index("H2O1")
    oxygen = saved["elements"].tolist().index("O")
    share = np.exp(saved["ln_n"][:, :, water])/saved["budget"][:, oxygen]
    point = int(np.argmax(abs(share[1]-share[0])))
    assert point == int(saved["highlighted_case_index"])
    np.testing.assert_allclose([summary["highlighted_case"]["oxygen_share_before"],
                               summary["highlighted_case"]["oxygen_share_after"]], share[:2, point], rtol=1e-12)
    for index, result in enumerate(summary["variants_summary"], 1):
        ratios = np.exp(ln_x[index, :, water]-ln_x[0, :, water])
        assert result["water_ratio_min"] == pytest.approx(ratios.min())
        assert result["water_ratio_max"] == pytest.approx(ratios.max())
    scope = np.exp(ln_x[2, :, water]-ln_x[1, :, water])
    assert summary["discrete_over_evaluated_water_ratio_min"] == pytest.approx(scope.min())
    assert summary["discrete_over_evaluated_water_ratio_max"] == pytest.approx(scope.max())
    checks = summary["independent_solver"]
    assert isinstance(checks, list) and 1 <= len(checks) <= 2
    for point in checks:
        assert len(point["comparisons"]) == 3
        for check in point["comparisons"]:
            assert check["max_abs_delta_log10_x"] < 1e-4
            assert check["element_relative_residual"] < 1e-5
            assert check["charge_relative_residual"] < 1e-7
            assert check["mass_action_max_abs_ln"] < 1e-7
    rows = json.loads((DATA/"diagnostics.json").read_text())
    assert len(rows) == 150 and all(row["converged"] for row in rows)
    extra = json.loads((DATA/"selected_diagnostics.json").read_text())
    assert len(extra["tight"]) == 3*len(saved["tight_case_indices"]) and len(extra["lambda"]) == 5
    assert all(row["converged"] for rows in extra.values() for row in rows)


def test_execution_receipt_pins_every_selected_input_and_output():
    receipt = json.loads((DATA/"receipt.json").read_text())
    assert receipt["result"] == "passed" and receipt["dtype"] == "float64"
    assert receipt["fastchem_binary_sha256"]
    for name, digest in receipt["selected_file_sha256"].items():
        assert sha256(historical_source(ROOT, name, digest).read_bytes()).hexdigest() == digest, name
    for name, digest in receipt["output_sha256"].items():
        assert sha256((DATA/name).read_bytes()).hexdigest() == digest, name
