"""Check the archived TiO substitution without rerunning equilibrium solvers."""

from hashlib import sha256
import json
from pathlib import Path

import numpy as np
import pytest
from scipy.special import logsumexp


ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "tests/data/validation/tio_equilibrium_impact/20261103"
BASELINE = ROOT / "tests/data/validation/curated_default_adoption/20261102/ions.npz"


@pytest.fixture(scope="module")
def saved():
    with np.load(DATA / "equilibria.npz") as archive:
        return {name: archive[name] for name in archive.files}


def log_fractions(ln_n):
    return ln_n-logsumexp(ln_n, axis=-1, keepdims=True)


def check_equilibrium(saved, ln_n, h, budget, pressure):
    """Independently check every species, positive budget and charge carrier."""
    assert np.isfinite(ln_n).all() and np.isfinite(h).all()
    n = np.exp(ln_n)
    assert np.isfinite(n).all() and np.all(n > 0)
    matrix = saved["formula_matrix"]
    elements, species = saved["elements"].tolist(), saved["species"].tolist()
    positive = np.asarray(budget).reshape(-1, len(elements))[0] > 0
    residual = np.abs(n @ matrix.T-budget)[..., positive]/np.asarray(budget)[..., positive]
    assert np.max(residual) <= 1e-5
    charge = matrix[elements.index("e-")]
    assert np.max(np.abs(n @ charge)/(n @ np.abs(charge))) <= 1e-12
    references = [species.index("e1-" if element == "e-" else element+"1")
                  for element in elements]
    logp = log_fractions(ln_n)+np.log(np.asarray(pressure))[..., None]
    assert np.max(np.abs(logp-logp[..., references] @ matrix+h)) <= 1e-8


def test_fixed_network_reuses_all_baseline_cases_and_changes_only_tio(saved):
    assert saved["variants"].tolist() == ["current", "toto", "nasa_proxy"]
    assert saved["ln_n"].shape == saved["h"].shape == (3, 75, 537)
    with np.load(BASELINE) as baseline:
        for name in ("formula_matrix", "budget", "species", "elements", "temperature_K",
                     "pressure_bar", "C_over_O"):
            np.testing.assert_array_equal(saved[name], baseline[name])
        np.testing.assert_array_equal(saved["ln_n"][0], baseline["ln_n"])
        np.testing.assert_allclose(saved["h"][0], baseline["h"], atol=1e-12, rtol=0)
    column = saved["species"].tolist().index("O1Ti1")
    delta = saved["h"][1:]-saved["h"][0]
    np.testing.assert_allclose(np.delete(delta, column, axis=-1), 0, atol=1e-12, rtol=0)
    assert np.any(np.abs(delta[..., column]) > 1e-3)
    for ln_n, h in zip(saved["ln_n"], saved["h"]):
        check_equilibrium(saved, ln_n, h, saved["budget"], saved["pressure_bar"])
    # Trace species remain in logarithmic calculations; no reporting floor is applied.
    assert np.min(log_fractions(saved["ln_n"])) < np.log(1e-100)
    titanium = saved["formula_matrix"][saved["elements"].tolist().index("Ti")]
    ti_budget = saved["budget"][:, saved["elements"].tolist().index("Ti")]
    shares = np.exp(saved["ln_n"])*titanium/ti_budget[None, :, None]
    np.testing.assert_allclose(shares.sum(axis=-1), 1, rtol=1e-8, atol=0)
    oxygen = saved["elements"].tolist().index("O")
    np.testing.assert_allclose(np.exp(saved["ln_n"]) @ saved["formula_matrix"][oxygen],
                               np.broadcast_to(saved["budget"][:, oxygen], (3, 75)),
                               rtol=1e-8, atol=0)


def test_selected_tight_solutions_and_finite_path_obey_the_same_constraints(saved):
    selected = saved["tight_case_indices"]
    assert selected.tolist() == sorted(set(selected.tolist()))
    assert 1 <= len(selected) <= 3
    assert saved["tight_ln_n"].shape == (2, len(selected), 537)
    budget, pressure = saved["budget"], saved["pressure_bar"]
    for variant, ln_n in enumerate(saved["tight_ln_n"], 1):
        check_equilibrium(saved, ln_n, saved["h"][variant, selected],
                          budget[selected], pressure[selected])
        assert np.max(np.abs(log_fractions(ln_n)
                             -log_fractions(saved["ln_n"][variant, selected]))) < 1e-4

    point = int(saved["highlighted_case_index"])
    assert point in selected
    fractions = saved["lambda_values"]
    np.testing.assert_array_equal(fractions, np.linspace(0, 1, 5))
    initial, final = saved["h"][[0, 1], point]
    path_h = initial+fractions[:, None]*(final-initial)
    assert saved["lambda_ln_n"].shape == (5, 537)
    check_equilibrium(saved, saved["lambda_ln_n"], path_h, budget[point], pressure[point])
    np.testing.assert_allclose(log_fractions(saved["lambda_ln_n"][[0, -1]]),
                               log_fractions(saved["ln_n"][[0, 1], point]), atol=1e-4, rtol=0)
    column = saved["species"].tolist().index("O1Ti1")
    # Raising just TiO's formation potential cannot increase its equilibrium amount.
    assert final[column] > initial[column]
    assert np.max(np.diff(saved["lambda_ln_n"][:, column])) < 1e-7


def test_energy_sensitivity_holds_thermal_inventory_fixed(saved):
    point = int(saved["highlighted_case_index"])
    energies = saved["uncertainty_D0_eV"]
    np.testing.assert_array_equal(energies, [6.814, 6.834])
    assert saved["uncertainty_ln_n"].shape == (2, 537)
    column = saved["species"].tolist().index("O1Ti1")
    h = np.broadcast_to(saved["h"][1, point], (2, 537)).copy()
    # Stronger D0 increases ln K and therefore decreases h=-ln K.
    ev_over_k = 1.602176634e-19/1.380649e-23
    h[:, column] -= (energies-6.824)*ev_over_k/saved["temperature_K"][point]
    check_equilibrium(saved, saved["uncertainty_ln_n"], h,
                      saved["budget"][point], saved["pressure_bar"][point])
    low, high = saved["uncertainty_ln_n"][:, column]
    center = saved["ln_n"][1, point, column]
    assert low <= center+1e-7 and center <= high+1e-7


def test_same_model_gibbs_ordering_is_reproduced_from_all_endpoints(saved):
    def objective(log_amounts, potential):
        logs = np.asarray(log_amounts, dtype=np.longdouble)
        n = np.exp(logs)
        total = n.sum(axis=-1)
        # Extensive ideal-gas form, independent of the review's helper.
        return (np.sum(n*(logs+np.asarray(potential, dtype=np.longdouble)), axis=-1)
                -total*np.log(total)
                +total*np.log(saved["pressure_bar"].astype(np.longdouble)))

    current, variant = [], []
    for index in (1, 2):
        old, new = saved["ln_n"][[0, index]]
        old_h, new_h = saved["h"][[0, index]]
        current.append(objective(new, old_h)-objective(old, old_h))
        variant.append(objective(new, new_h)-objective(old, new_h))
    np.testing.assert_allclose(current, saved["delta_g_in_current"], atol=5e-14, rtol=1e-7)
    np.testing.assert_allclose(variant, saved["delta_g_in_variant"], atol=5e-14, rtol=1e-7)
    assert np.min(current) >= -2e-12
    assert np.max(variant) <= 2e-12
    # Small primal feasibility differences can change raw G more than summation
    # roundoff. Remove their first-order elemental-potential term before checking
    # the residual ideal-gas ordering; this is not a new physical error estimate.
    logs = saved["ln_n"].astype(np.longdouble)
    amounts = np.exp(logs)
    logp = (logs-np.log(amounts.sum(axis=-1))[..., None]
            +np.log(saved["pressure_bar"].astype(np.longdouble))[None, :, None])
    refs = [saved["species"].tolist().index("e1-" if e == "e-" else e+"1")
            for e in saved["elements"]]
    potentials = logp+saved["h"].astype(np.longdouble)
    budget_delta = (amounts[1:]-amounts[0]) @ saved["formula_matrix"].T
    current_correction = (budget_delta*potentials[0][..., refs]).sum(axis=-1)
    variant_correction = (budget_delta*potentials[1:][..., refs]).sum(axis=-1)
    assert np.min(np.asarray(current)-current_correction) >= -5e-14
    assert np.max(np.asarray(variant)-variant_correction) <= 5e-14


def test_summary_and_independent_comparator_preserve_the_review_scope(saved):
    summary = json.loads((DATA / "summary.json").read_text())
    assert summary["reused_baseline_cases"] == summary["cases"] == 75
    assert summary["new_endpoint_solves"] == 150
    assert summary["species"] == 537 and summary["curated_update"] == "none"
    assert summary["minimum_archived_ln_n"] == pytest.approx(float(saved["ln_n"].min()))
    point = int(saved["highlighted_case_index"])
    column = saved["species"].tolist().index("O1Ti1")
    titanium = saved["elements"].tolist().index("Ti")
    shares = np.exp(saved["ln_n"][:, :, column])/saved["budget"][:, titanium]
    assert point == int(np.argmax(abs(shares[1]-shares[0]))) == 4
    assert saved["temperature_K"][point] == 1500 and saved["pressure_bar"][point] == 100
    np.testing.assert_allclose(shares[:2, point], [.6057227507449432, .4895723641233141],
                               rtol=1e-8, atol=0)
    ratios = np.exp(log_fractions(saved["ln_n"])[1:, :, column]
                    -log_fractions(saved["ln_n"])[0, :, column])
    for index, variant in enumerate(summary["variants_summary"]):
        assert variant["tio_ratio_min"] == pytest.approx(ratios[index].min())
        assert variant["tio_ratio_max"] == pytest.approx(ratios[index].max())
    assert np.max(ratios) < 1 and .62 < np.min(ratios) < .63
    # The finite abundance response differs from a constant K multiplier.
    assert .80 < ratios[0, point] < .82
    assert summary["independent_solver_case_index"] == point
    checks = summary["independent_solver"]
    assert len(checks) == 3 and isinstance(checks, list)
    for check in checks:
        assert check["max_abs_delta_log10_x"] < 1e-4
        assert check["element_relative_residual"] < 1e-5
        assert check["charge_relative_residual"] < 1e-7
        assert check["mass_action_max_abs_ln"] < 1e-7
    rows = json.loads((DATA / "diagnostics.json").read_text())
    assert len(rows) == 150 and all(row["converged"] for row in rows)
    extra = json.loads((DATA / "selected_diagnostics.json").read_text())
    assert len(extra["tight"]) == 2*len(saved["tight_case_indices"])
    assert len(extra["lambda"]) == 5 and len(extra["energy_sensitivity"]) == 2
    assert all(row["converged"] for group in extra.values() for row in group)


def test_selected_inputs_and_outputs_are_pinned():
    receipt = json.loads((DATA / "receipt.json").read_text())
    assert receipt["result"] == "passed"
    inputs = receipt["selected_file_sha256"]
    assert str(BASELINE.relative_to(ROOT)) in inputs
    for name, digest in inputs.items():
        assert sha256((ROOT/name).read_bytes()).hexdigest() == digest, name
    for name, digest in receipt["output_sha256"].items():
        assert sha256((DATA/name).read_bytes()).hexdigest() == digest, name
