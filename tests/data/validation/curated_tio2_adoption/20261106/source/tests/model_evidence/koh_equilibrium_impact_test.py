"""Replay KOH substitution invariants from frozen amounts without new solves."""

from hashlib import sha256
import json
from pathlib import Path
import sys

import numpy as np
import pytest
from scipy.special import logsumexp


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "mceg/reviews"))
import koh_candidate as candidate

DATA = ROOT / "tests/data/validation/koh_equilibrium_impact/20261105"
BASELINE = ROOT / "tests/data/validation/curated_cos_adoption/20261105"


@pytest.fixture(scope="module", params=["ions"])
def saved(request):
    with np.load(DATA/f"{request.param}.npz") as archive:
        return request.param, {key: archive[key].copy() for key in archive.files}


def fractions(values):
    return values-logsumexp(values, axis=-1, keepdims=True)


def check_constraints(data, amounts, h, budget, pressure):
    n, matrix = np.exp(amounts), data["formula_matrix"]
    elements, species = data["elements"].tolist(), data["species"].tolist()
    positive = np.asarray(budget).reshape(-1, len(elements))[0] > 0
    residual = abs(n@matrix.T-budget)
    assert np.isfinite(amounts).all() and np.all(n > 0)
    assert np.max(residual[..., positive]/np.asarray(budget)[..., positive]) < 1e-5
    for element in ("H", "C", "O"):
        j = elements.index(element)
        assert np.max(residual[..., j]/np.asarray(budget)[..., j]) < 1e-9
    j = elements.index('K')
    assert np.max(residual[..., j]/np.asarray(budget)[..., j]) < 1e-8
    if "e-" in elements:
        charge = matrix[elements.index("e-")]
        assert np.max(abs(n@charge)/(n@abs(charge))) < 1e-12
    references = [species.index("e1-" if e == "e-" else e+"1") for e in elements]
    logp = fractions(amounts)+np.log(np.asarray(pressure))[..., None]
    assert np.max(abs(logp-logp[..., references]@matrix+h)) < 1e-8


def test_only_koh_changes_on_the_pinned_current_network(saved):
    name, data = saved
    assert data["ln_n"].shape == data["h"].shape == (2, 75, 537 if name == "ions" else 421)
    with np.load(BASELINE/f"{name}.npz") as baseline:
        for key in ("species", "elements", "formula_matrix", "budget", "temperature_K", "pressure_bar", "C_over_O"):
            np.testing.assert_array_equal(data[key], baseline[key])
        np.testing.assert_array_equal(data["ln_n"][0], baseline["ln_n"])
        np.testing.assert_allclose(data["h"][0], baseline["h"], rtol=0, atol=1e-12)
    kh = data["species"].tolist().index('H1K1O1')
    np.testing.assert_array_equal(np.delete(data["h"][1]-data["h"][0], kh, axis=-1), 0)
    np.testing.assert_allclose(data["h"][1, :, kh], [candidate.formation_h(t) for t in data["temperature_K"]], atol=2e-12, rtol=0)
    check_constraints(data, data["ln_n"], data["h"], data["budget"], data["pressure_bar"])


def test_cold_tolerance_and_nasa_controls_obey_the_same_constraints(saved):
    _, data = saved
    selected = data["tight_case_indices"]
    kh = data["species"].tolist().index('H1K1O1')
    tight, original = fractions(data["tight_ln_n"]), fractions(data["ln_n"][:, selected])
    delta = tight-original
    assert abs(delta).max() < 1e-4 and abs(delta[..., kh]).max() < 1e-8
    assert abs(np.exp(tight)-np.exp(original)).max() < 1e-9
    visible = np.maximum(np.exp(tight), np.exp(original)) >= 1e-12
    assert abs(delta[visible]).max() < 2e-6
    check_constraints(data, data["tight_ln_n"], data["h"][:, selected], data["budget"][selected], data["pressure_bar"][selected])
    check_constraints(data, data["nasa_control_ln_n"], data["nasa_control_h"], data["budget"][selected], data["pressure_bar"][selected])
    # JAX compiles the three-node control separately from the 75-node grid.
    np.testing.assert_allclose(np.delete(data["nasa_control_h"]-data["h"][1, selected], kh, axis=-1), 0, atol=1e-12, rtol=0)
    np.testing.assert_allclose(data["nasa_control_h"][:, kh], [candidate.formation_h(t, variant=candidate.CONTROL) for t in data["temperature_K"][selected]], atol=2e-12, rtol=0)


def test_finite_path_and_gibbs_ordering(saved):
    _, data = saved
    case, lambdas = int(data["highlighted_case_index"]), data["lambda_values"]
    np.testing.assert_array_equal(lambdas, np.linspace(0, 1, 5))
    initial, final = data["h"][:, case]
    check_constraints(data, data["lambda_ln_n"], initial+lambdas[:, None]*(final-initial), data["budget"][case], data["pressure_bar"][case])
    np.testing.assert_allclose(fractions(data["lambda_ln_n"][[0, -1]]), fractions(data["ln_n"][:, case]), atol=1e-4, rtol=0)
    kh = data["species"].tolist().index('H1K1O1')
    assert abs(final[kh]-initial[kh]) > .01
    assert np.max(np.diff(data["lambda_ln_n"][:, kh])*(final[kh]-initial[kh])) < 1e-10
    logs = data["ln_n"].astype(np.longdouble)
    n = np.exp(logs)
    logp = logs-np.log(n.sum(axis=-1))[..., None]+np.log(data["pressure_bar"].astype(np.longdouble))[None, :, None]
    h = data["h"].astype(np.longdouble)
    raw = np.asarray([np.sum(n[1]*(h[i]+logp[1])-n[0]*(h[i]+logp[0]), axis=-1) for i in (0, 1)])
    refs = [data["species"].tolist().index("e1-" if e == "e-" else e+"1") for e in data["elements"]]
    corrected = raw-((n[1]-n[0])@data["formula_matrix"].T*(h+logp)[..., refs]).sum(axis=-1)
    np.testing.assert_allclose(raw, data["delta_g_raw"], atol=5e-14, rtol=1e-7)
    np.testing.assert_allclose(corrected, data["delta_g_corrected"], atol=5e-14, rtol=1e-7)
    assert corrected[0].min() >= -5e-14 and corrected[1].max() <= 5e-14


def test_reported_response_matches_archived_amounts(saved):
    name, data = saved
    all_summary = json.loads((DATA/"summary.json").read_text())
    summary = next(row for row in all_summary["networks"] if row["network"] == name)
    assert all_summary["curated_update"] == "none" and all_summary["new_endpoint_solves"] == 75
    kh, potassium = data["species"].tolist().index('H1K1O1'), data["elements"].tolist().index('K')
    logx = fractions(data["ln_n"])
    ratios = np.exp(logx[1, :, kh]-logx[0, :, kh])
    assert summary["KOH_ratio_min"] == pytest.approx(ratios.min())
    assert summary["KOH_ratio_max"] == pytest.approx(ratios.max())
    share = np.exp(data["ln_n"][:, :, kh])/data["budget"][:, potassium]
    case = int(np.argmax(abs(share[1]-share[0])))
    assert summary["highlighted_case"]["case_index"] == case
    np.testing.assert_allclose([summary["highlighted_case"]["potassium_share_before"], summary["highlighted_case"]["potassium_share_after"]], share[:, case], rtol=1e-12)
    for point in summary["species_extrema"]:
        i, j = point["case_index"], data["species"].tolist().index(point["species"])
        assert point["x_before"] == pytest.approx(np.exp(logx[0, i, j]), rel=1e-12, abs=0)
        assert point["x_after"] == pytest.approx(np.exp(logx[1, i, j]), rel=1e-12, abs=0)
    thermal_ratio = np.exp(fractions(data["nasa_control_ln_n"])[:, kh]-fractions(data["tight_ln_n"][1])[:, kh])
    assert summary["nasa_control_KOH_ratio_min"] == pytest.approx(thermal_ratio.min())
    assert summary["nasa_control_KOH_ratio_max"] == pytest.approx(thermal_ratio.max())


def test_execution_receipt_convergence_and_independent_solver():
    receipt = json.loads((DATA/"receipt.json").read_text())
    assert receipt["result"] == "passed" and receipt["dtype"] == "float64"
    assert receipt["fastchem_binary_sha256"]
    for path, digest in receipt["selected_file_sha256"].items():
        assert sha256((ROOT/path).read_bytes()).hexdigest() == digest, path
    for path, digest in receipt["output_sha256"].items():
        assert sha256((DATA/path).read_bytes()).hexdigest() == digest, path
    rows = json.loads((DATA/"diagnostics.json").read_text())
    assert len(rows) == 75 and all(row["converged"] for row in rows)
    controls = json.loads((DATA/"selected_diagnostics.json").read_text())
    assert all(row["converged"] for row in controls)
    summary = json.loads((DATA/"summary.json").read_text())
    assert len(summary["independent_solver"]) == 2
    for point in summary["independent_solver"]:
        assert len(point["comparisons"]) == 2
        for row in point["comparisons"]:
            assert row["max_abs_delta_log10_x"] < 1e-4
            assert row["element_relative_residual"] < 1e-5
            assert row["charge_relative_residual"] < 1e-7
            assert row["mass_action_max_abs_ln"] < 1e-7
