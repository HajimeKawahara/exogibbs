"""Replay joint production adoption with independent frozen candidate functions."""

from hashlib import sha256
import json
from pathlib import Path
import sys

import numpy as np
import pytest
from scipy.special import logsumexp


ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "tests/data/validation/curated_h2o_adoption/20261104"
BASELINE = ROOT / "tests/data/validation/curated_tio_adoption/20261103"
sys.path.insert(0, str(ROOT / "mceg/reviews"))
import co2_candidate
import h2o_candidate
from curated_adoption_history import historical_source


def test_production_receipt_identifies_current_inputs_and_completed_outputs():
    receipt = json.loads((DATA / "receipt.json").read_text())
    assert receipt["result"] == "passed"
    assert receipt["backend"] == "cpu" and receipt["dtype"] == "float64"
    for name, expected in receipt["selected_file_sha256"].items():
        assert sha256(historical_source(ROOT, name, expected).read_bytes()).hexdigest() == expected, name
    for name, expected in receipt["output_sha256"].items():
        assert sha256((DATA / name).read_bytes()).hexdigest() == expected, name
    summary = json.loads((DATA / "summary.json").read_text())
    assert summary["dataset"] == "curated-h2o-v1"
    assert summary["base_dataset"] == "curated-co2-v1"
    assert summary["new_production_solves"] == 150
    assert summary["tighter_joint_reference_solves"] == 6
    assert summary["strict_epsilon_crit"] < summary["epsilon_crit"]
    assert len(summary["independent_solver"]) == 2
    for comparison in summary["independent_solver"]:
        assert comparison["comparisons"][0]["max_abs_delta_log10_x"] < 1e-4
    rows = json.loads((DATA / "diagnostics.json").read_text())
    assert len(rows) == 156 and all(row["converged"] for row in rows)


def check_constraints(data, ln_n, h, budget, pressure):
    assert np.isfinite(ln_n).all() and np.isfinite(h).all()
    n = np.exp(ln_n)
    matrix, elements = data["formula_matrix"], data["elements"].tolist()
    species = data["species"].tolist()
    positive = budget[0] > 0
    residual = abs(n @ matrix.T - budget)
    assert np.max(residual[:, positive] / budget[:, positive]) < 1e-5
    for name in ("H", "C", "O"):
        j = elements.index(name)
        assert np.max(residual[:, j] / budget[:, j]) < 1e-9
    logp = ln_n - logsumexp(ln_n, axis=-1, keepdims=True) + np.log(pressure)[:, None]
    references = [species.index("e1-" if name == "e-" else name+"1") for name in elements]
    assert np.max(abs(logp - logp[:, references] @ matrix + h)) < 1e-8
    if "e-" in elements:
        charge = matrix[elements.index("e-")]
        assert np.max(abs(n @ charge) / (n @ abs(charge))) < 1e-12


@pytest.mark.parametrize("network,count", [("ions", 537), ("neutral", 421)])
def test_joint_factory_changes_only_reviewed_species_and_closes_both_networks(network, count):
    with np.load(DATA / f"{network}.npz") as data, np.load(BASELINE / f"{network}.npz") as old:
        assert data["ln_n"].shape == (75, count) and data["converged"].all()
        for name in ("formula_matrix", "budget", "species", "elements", "temperature_K", "pressure_bar", "C_over_O"):
            np.testing.assert_array_equal(data[name], old[name])
        water, carbon = (data["species"].tolist().index(s) for s in ("H2O1", "C1O2"))
        expected = old["h"].copy()
        expected[:, water] = [h2o_candidate.formation_h(t, thermal="evaluated") for t in data["temperature_K"]]
        expected[:, carbon] = [co2_candidate.formation_h(t, model="nist626") for t in data["temperature_K"]]
        np.testing.assert_allclose(data["h"], expected, atol=1e-12, rtol=0)
        np.testing.assert_allclose(data["reference_h"], expected, atol=1e-12, rtol=0)
        check_constraints(data, data["ln_n"], expected, data["budget"], data["pressure_bar"])
        ln_x = data["ln_n"] - logsumexp(data["ln_n"], axis=-1, keepdims=True)
        np.testing.assert_allclose(data["ln_x"], ln_x, atol=1e-13, rtol=0)
        selected = data["tighter_case_indices"]
        tight = data["tighter_reference_ln_n"]
        check_constraints(data, tight, expected[selected], data["budget"][selected], data["pressure_bar"][selected])
        tight_x = tight - logsumexp(tight, axis=-1, keepdims=True)
        assert np.max(abs(tight_x - ln_x[selected])) < 1e-4
        assert np.max(abs(tight_x[:, [water, carbon]] - ln_x[selected][:, [water, carbon]])) < 1e-7
        assert np.max(abs(np.exp(tight_x) - np.exp(ln_x[selected]))) < 1e-9
        changed = expected.copy()
        changed[:, water] += 1e-3
        with pytest.raises(AssertionError):
            check_constraints(data, data["ln_n"], changed, data["budget"], data["pressure_bar"])
