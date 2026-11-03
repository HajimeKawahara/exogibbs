"""Check the CO2 production execution against the candidate and both networks."""

from hashlib import sha256
import json
from pathlib import Path
import sys

import numpy as np
import pytest
from scipy.special import logsumexp


ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "tests/data/validation/curated_co2_adoption/20261104"
sys.path.insert(0, str(ROOT / "mceg/reviews"))
from build_co2_adoption import build_data


def test_co2_receipt_pins_original_sources_and_completed_outputs():
    receipt = json.loads((DATA / "receipt.json").read_text())
    assert receipt["result"] == "passed"
    for name, expected in receipt["selected_file_sha256"].items():
        assert sha256((ROOT / name).read_bytes()).hexdigest() == expected, name
    for name, expected in receipt["output_sha256"].items():
        assert sha256((DATA / name).read_bytes()).hexdigest() == expected, name
    summary = json.loads((DATA / "summary.json").read_text())
    assert summary["dataset"] == "curated-co2-v1"
    assert summary["legacy_dataset"] == "curated-tio-v1"
    assert [r["cases_passed"] for r in summary["networks"]] == [75, 75]
    assert summary["epsilon_crit"] == 5e-14


@pytest.mark.parametrize("network,count", [("ions", 537), ("neutral", 421)])
def test_every_adopted_default_case_satisfies_saved_constraints(network, count):
    previous_path = ROOT / "tests/data/validation/co2_equilibrium_impact/20261104/original.npz"
    with np.load(DATA / f"{network}.npz") as saved, np.load(previous_path) as previous:
        assert saved["ln_n"].shape == (75, count)
        assert saved["converged"].all() and np.isfinite(saved["ln_n"]).all()
        elements, species = list(saved["elements"]), list(saved["species"])
        rows = [list(previous["elements"]).index(name) for name in elements]
        columns = [list(previous["species"]).index(name) for name in species]
        np.testing.assert_array_equal(saved["budget"], previous["budget"][:, rows])
        np.testing.assert_array_equal(saved["formula_matrix"], previous["formula_matrix"][rows][:, columns])
        np.testing.assert_allclose(saved["h"], previous["h"][1][:, columns], atol=1e-12, rtol=0)
        for key in ("temperature_K", "pressure_bar"):
            np.testing.assert_array_equal(saved[key], previous[key])
        n = np.exp(saved["ln_n"])
        a, b = saved["formula_matrix"], saved["budget"]
        positive = b[0] > 0
        assert np.max(abs(n @ a.T - b)[:, positive] / b[:, positive]) < 1e-5
        for element in ("C", "O"):
            index = elements.index(element)
            assert np.max(abs(n @ a[index]-b[:, index])/b[:, index]) < 1e-12
        logx = saved["ln_n"] - logsumexp(saved["ln_n"], axis=-1, keepdims=True)
        np.testing.assert_allclose(saved["ln_x"], logx, atol=1e-13, rtol=0)
        refs = [species.index("e1-" if e == "e-" else e+"1") for e in elements]
        logp = logx + np.log(saved["pressure_bar"])[:, None]
        assert np.max(abs(logp - logp[:, refs] @ a + saved["h"])) < 1e-8
        if network == "ions":
            charge = a[elements.index("e-")]
            assert np.max(abs(n @ charge) / (n @ abs(charge))) < 1e-12
            reference = previous["ln_n"][1].copy()
            reference -= logsumexp(reference, axis=-1, keepdims=True)
            np.testing.assert_allclose(logx, reference, atol=1e-4, rtol=0)
            co2 = species.index("C1O2")
            np.testing.assert_allclose(logx[:, co2], reference[:, co2], atol=1e-9, rtol=0)
        else:
            assert "e-" not in elements and "e1-" not in species


def test_packaged_co2_exactly_adopts_reviewed_candidate():
    path = ROOT / "src/exogibbs/data/curated/co2_v1.json"
    assert json.loads(path.read_text()) == build_data()
    data = json.loads(path.read_text())
    candidate = json.loads((ROOT / "tests/data/validation/co2_candidate/20261104/candidate.json").read_text())
    for name in ("knots_K", "coefficients"):
        assert data[name] == candidate["models"]["nist626"][name]
    assert data["D0_J_mol"] == 1598180.
    assert data["base_dataset"] == "curated-tio-v1"
