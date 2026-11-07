"""Keep the default migration tied to its own execution and both networks."""

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
DATA = ROOT / "tests/data/validation/curated_default_adoption/20261102"


def test_default_receipt_pins_current_sources_and_completed_outputs():
    receipt = json.loads((DATA / "receipt.json").read_text())
    assert receipt["result"] == "passed"
    for name, expected in receipt["selected_file_sha256"].items():
        assert sha256(historical_source(ROOT, name, expected).read_bytes()).hexdigest() == expected, name
    for name, expected in receipt["output_sha256"].items():
        assert sha256((DATA / name).read_bytes()).hexdigest() == expected, name
    summary = json.loads((DATA / "summary.json").read_text())
    assert summary["dataset"] == "curated-c2h-v1"
    assert summary["legacy_dataset"] == "curated-gas-v1"
    assert [r["cases_passed"] for r in summary["networks"]] == [75, 75]
    assert summary["epsilon_crit"] == 5e-14


@pytest.mark.parametrize("network,count", [("ions", 537), ("neutral", 421)])
def test_every_adopted_default_case_satisfies_saved_constraints(network, count):
    previous_path = ROOT / "tests/data/validation/c2h_equilibrium_impact/20261102/equilibria.npz"
    with np.load(DATA / f"{network}.npz") as saved, np.load(previous_path) as previous:
        assert saved["ln_n"].shape == (75, count)
        assert saved["converged"].all() and np.isfinite(saved["ln_n"]).all()
        elements, species = list(saved["elements"]), list(saved["species"])
        rows = [list(previous["elements"]).index(name) for name in elements]
        columns = [list(previous["species"]).index(name) for name in species]
        np.testing.assert_array_equal(saved["budget"], previous["budget"][:, rows])
        np.testing.assert_array_equal(saved["formula_matrix"], previous["formula_matrix"][rows][:, columns])
        np.testing.assert_allclose(saved["h"], previous["h"][1][:, columns], atol=1e-12, rtol=0)
        for key in ("temperature_K", "pressure_bar", "C_over_O"):
            np.testing.assert_array_equal(saved[key], previous[key])
        n = np.exp(saved["ln_n"])
        a, b = saved["formula_matrix"], saved["budget"]
        positive = b[0] > 0
        assert np.max(abs(n @ a.T - b)[:, positive] / b[:, positive]) < 1e-5
        logx = saved["ln_n"] - logsumexp(saved["ln_n"], axis=-1, keepdims=True)
        np.testing.assert_allclose(saved["ln_x"], logx, atol=1e-13, rtol=0)
        refs = [species.index("e1-" if e == "e-" else e+"1") for e in elements]
        logp = logx + np.log(saved["pressure_bar"])[:, None]
        assert np.max(abs(logp - logp[:, refs] @ a + saved["h"])) < 1e-8
        if network == "ions":
            charge = a[elements.index("e-")]
            assert np.max(abs(n @ charge) / (n @ abs(charge))) < 1e-12
            reference = previous["strict_ln_n"][1]
            reference -= logsumexp(reference, axis=-1, keepdims=True)
            np.testing.assert_allclose(logx, reference, atol=1e-4, rtol=0)
        else:
            assert "e-" not in elements and "e1-" not in species
