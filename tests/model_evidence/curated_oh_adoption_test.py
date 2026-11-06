"""Replay the adopted OH model, conserved responses and immutable receipts."""

from hashlib import sha256
import json
from pathlib import Path
import runpy
import sys

import numpy as np
import pytest
from scipy.special import logsumexp


ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "tests/data/validation/curated_oh_adoption/20261105"
BASELINE = ROOT / "tests/data/validation/curated_h2o_adoption/20261104"
CANDIDATE = ROOT / "tests/data/validation/oh_equilibrium_impact/20261105"
sys.path.insert(0, str(ROOT / "mceg/reviews"))
from curated_adoption_history import historical_source
import oh_candidate
from oh_equilibrium_impact import constraints


def test_production_receipt_pins_current_inputs_and_completed_outputs():
    receipt = json.loads((DATA / "receipt.json").read_text())
    assert receipt["result"] == "passed"
    assert receipt["backend"] == "cpu" and receipt["dtype"] == "float64"
    for name, expected in receipt["selected_file_sha256"].items():
        assert sha256(historical_source(ROOT, name, expected).read_bytes()).hexdigest() == expected, name
    for name, expected in receipt["output_sha256"].items():
        assert sha256((DATA / name).read_bytes()).hexdigest() == expected, name
    summary = json.loads((DATA / "summary.json").read_text())
    assert (summary["dataset"], summary["base_dataset"]) == ("curated-oh-v1", "curated-h2o-v1")
    assert summary["new_production_solves"] == 150 and summary["tighter_reference_solves"] == 6
    assert summary["strict_epsilon_crit"] < summary["epsilon_crit"]
    assert len(summary["independent_solver"]) == 2
    for comparison in summary["independent_solver"]:
        assert comparison["comparisons"][0]["max_abs_delta_log10_x"] < 1e-4
    rows = json.loads((DATA / "diagnostics.json").read_text())
    assert len(rows) == 156 and all(row["converged"] for row in rows)


def test_packaged_model_rebuilds_from_pinned_candidate_and_refuses_changed_inputs(monkeypatch):
    build = runpy.run_path(str(ROOT / "mceg/reviews/build_oh_adoption.py"))["build_data"]
    assert build() == json.loads((ROOT / "src/exogibbs/data/curated/oh_v1.json").read_text())
    original = Path.read_bytes
    candidate = ROOT / "tests/data/validation/oh_candidate/20261105/candidate.json"
    monkeypatch.setattr(Path, "read_bytes", lambda p: original(p) + (b"\n" if p == candidate else b""))
    with pytest.raises(ValueError, match="reviewed candidate"):
        build()


@pytest.mark.parametrize("network,count", [("ions", 537), ("neutral", 421)])
def test_oh_only_factory_reproduces_candidate_and_conserves_both_networks(network, count):
    with np.load(DATA / f"{network}.npz") as data, np.load(BASELINE / f"{network}.npz") as old, np.load(CANDIDATE / f"{network}.npz") as candidate:
        assert data["ln_n"].shape == (75, count) and data["converged"].all()
        for name in ("formula_matrix", "budget", "species", "elements", "temperature_K", "pressure_bar", "C_over_O"):
            np.testing.assert_array_equal(data[name], old[name])
            np.testing.assert_array_equal(data[name], candidate[name])
        oh = data["species"].tolist().index("H1O1")
        expected = old["h"].copy()
        expected[:, oh] = [oh_candidate.formation_h(t) for t in data["temperature_K"]]
        np.testing.assert_allclose(data["h"], expected, atol=1e-12, rtol=0)
        np.testing.assert_array_equal(np.delete(data["h"] - old["h"], oh, axis=-1), 0)
        matrix, elements, species = data["formula_matrix"], data["elements"].tolist(), data["species"].tolist()
        constraints(matrix, elements, species, data["budget"], data["ln_n"], expected, data["pressure_bar"])
        ln_x = data["ln_n"] - logsumexp(data["ln_n"], axis=-1, keepdims=True)
        np.testing.assert_allclose(data["ln_x"], ln_x, atol=1e-13, rtol=0)
        selected, tight = data["tighter_case_indices"], data["tighter_reference_ln_n"]
        constraints(matrix, elements, species, data["budget"][selected], tight,
                    expected[selected], data["pressure_bar"][selected])
        tight_x = tight - logsumexp(tight, axis=-1, keepdims=True)
        archived_x = candidate["ln_n"][1] - logsumexp(candidate["ln_n"][1], axis=-1, keepdims=True)
        for actual, reference in ((ln_x, archived_x), (ln_x[selected], tight_x)):
            assert np.max(abs(actual - reference)) < 1e-4
            assert np.max(abs(actual[:, oh] - reference[:, oh])) < 1e-8
            assert np.max(abs(np.exp(actual) - np.exp(reference))) < 1e-9
        ratios = np.exp(ln_x[:, oh] - old["ln_x"][:, oh])
        assert 1.038 < ratios.min() < 1.04 and 1.129 < ratios.max() < 1.13
        changed = expected.copy()
        changed[:, oh] += 1e-3
        with pytest.raises(AssertionError):
            constraints(matrix, elements, species, data["budget"], data["ln_n"], changed, data["pressure_bar"])
