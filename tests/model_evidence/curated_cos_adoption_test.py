"""Replay COS production on its HO2 base and separate the old candidate background."""

from hashlib import sha256
import json
from pathlib import Path
import runpy
import sys

import numpy as np
import pytest
from scipy.special import logsumexp


ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT/"tests/data/validation/curated_cos_adoption/20261105"
BASELINE = ROOT/"tests/data/validation/curated_ho2_adoption/20261105"
OLD_BASELINE = ROOT/"tests/data/validation/curated_oh_adoption/20261105"
CANDIDATE = ROOT/"tests/data/validation/cos_equilibrium_impact/20261105"
sys.path.insert(0, str(ROOT/"mceg/reviews"))
import cos_candidate
from cos_equilibrium_impact import constraints
from curated_adoption_history import historical_source


def fractions(logs):
    return logs-logsumexp(logs, axis=-1, keepdims=True)


def test_production_receipt_pins_inputs_outputs_and_actual_completed_execution():
    receipt = json.loads((DATA/"receipt.json").read_text())
    assert receipt["result"] == "passed"
    assert receipt["backend"] == "cpu" and receipt["dtype"] == "float64"
    assert receipt["command"][1] == "mceg/reviews/curated_cos_adoption.py"
    for name, expected in receipt["selected_file_sha256"].items():
        assert sha256(historical_source(ROOT, name, expected).read_bytes()).hexdigest() == expected, name
    for name, expected in receipt["output_sha256"].items():
        assert sha256((DATA/name).read_bytes()).hexdigest() == expected, name
    summary = json.loads((DATA/"summary.json").read_text())
    assert (summary["dataset"], summary["base_dataset"]) == ("curated-cos-v1", "curated-ho2-v1")
    assert summary["new_production_solves"] == 150 and summary["tighter_reference_solves"] == 6
    assert summary["strict_epsilon_crit"] < summary["epsilon_crit"]
    assert len(summary["independent_solver"]) == 2
    for point in summary["independent_solver"]:
        assert point["comparisons"][0]["max_abs_delta_log10_x"] < 1e-4
    rows = json.loads((DATA/"diagnostics.json").read_text())
    assert len(rows) == 156 and all(row["converged"] for row in rows)


def test_packaged_model_rebuilds_and_rejects_changed_reviewed_input(monkeypatch):
    build = runpy.run_path(str(ROOT/"mceg/reviews/build_cos_adoption.py"))["build_data"]
    assert build() == json.loads((ROOT/"src/exogibbs/data/curated/cos_v1.json").read_text())
    original = Path.read_bytes
    candidate = ROOT/"tests/data/validation/cos_candidate/20261105/candidate.json"
    monkeypatch.setattr(Path, "read_bytes", lambda p: original(p)+(b"\n" if p == candidate else b""))
    with pytest.raises(ValueError, match="reviewed candidate"):
        build()


@pytest.mark.parametrize("network,count", [("ions", 537), ("neutral", 421)])
def test_cos_only_adoption_matches_target_h_and_conserves_the_cumulative_network(network, count):
    with np.load(DATA/f"{network}.npz") as data, np.load(BASELINE/f"{network}.npz") as base, np.load(CANDIDATE/f"{network}.npz") as candidate:
        assert data["ln_n"].shape == (75, count) and data["converged"].all()
        for name in ("formula_matrix", "budget", "species", "elements", "temperature_K", "pressure_bar", "C_over_O"):
            np.testing.assert_array_equal(data[name], base[name])
            np.testing.assert_array_equal(data[name], candidate[name])
        species, elements = data["species"].tolist(), data["elements"].tolist()
        cos = species.index("C1O1S1")
        expected = base["h"].copy()
        expected[:, cos] = [cos_candidate.formation_h(t) for t in data["temperature_K"]]
        np.testing.assert_allclose(data["h"], expected, atol=1e-12, rtol=0)
        np.testing.assert_array_equal(np.delete(data["h"]-base["h"], cos, axis=-1), 0)
        background = [species.index(s) for s in ("H1K1", "H1O2")]
        assert np.max(abs(data["h"][:, background]-candidate["h"][1, :, background].T)) > .1
        np.testing.assert_array_equal(np.delete(data["h"]-candidate["h"][1], background+[cos], axis=-1), 0)
        constraints(data["formula_matrix"], elements, species, data["budget"], data["ln_n"], expected, data["pressure_bar"])
        ln_x = fractions(data["ln_n"])
        np.testing.assert_allclose(data["ln_x"], ln_x, atol=1e-13, rtol=0)
        selected, tight = data["tighter_case_indices"], data["tighter_reference_ln_n"]
        constraints(data["formula_matrix"], elements, species, data["budget"][selected], tight, expected[selected], data["pressure_bar"][selected])
        tight_x = fractions(tight)
        assert np.max(abs(ln_x[selected]-tight_x)) < 1e-4
        assert np.max(abs(ln_x[selected, cos]-tight_x[:, cos])) < 1e-8
        assert np.max(abs(np.exp(ln_x[selected])-np.exp(tight_x))) < 1e-9
        ratios = np.exp(ln_x[:, cos]-fractions(base["ln_n"])[:, cos])
        assert 1.183 < ratios.min() < 1.185 and 1.568 < ratios.max() < 1.569
        changed = expected.copy(); changed[:, cos] += 1e-3
        with pytest.raises(AssertionError):
            constraints(data["formula_matrix"], elements, species, data["budget"], data["ln_n"], changed, data["pressure_bar"])


@pytest.mark.parametrize("network", ["ions", "neutral"])
def test_summary_separates_incremental_and_cumulative_background_effects(network):
    with np.load(DATA/f"{network}.npz") as data, np.load(BASELINE/f"{network}.npz") as base, np.load(OLD_BASELINE/f"{network}.npz") as original, np.load(CANDIDATE/f"{network}.npz") as candidate:
        summary = next(row for row in json.loads((DATA/"summary.json").read_text())["networks"] if row["ions"] == (network == "ions"))
        logx, basex, oldx, candx = map(fractions, (data["ln_n"], base["ln_n"], original["ln_n"], candidate["ln_n"][1]))
        species, elements = data["species"].tolist(), data["elements"].tolist()
        cos, sulfur = species.index("C1O1S1"), elements.index("S")
        ratios = np.exp(logx[:, cos]-basex[:, cos]);r = summary["cos_incremental_response"]
        assert [r["min_x_ratio"], r["max_x_ratio"]] == pytest.approx([ratios.min(), ratios.max()], rel=1e-12)
        shares = np.array([np.exp(base["ln_n"][:, cos]), np.exp(data["ln_n"][:, cos])])/data["budget"][:, sulfur]
        i = int(np.argmax(abs(shares[1]-shares[0])))
        assert r["highlighted_case_index"] == i
        assert [r["sulfur_share_before"], r["sulfur_share_after"]] == pytest.approx(shares[:, i], rel=1e-12)
        for name, item in summary["cumulative_response_from_curated_oh"].items():
            j = species.index(name);ratio = np.exp(logx[:, j]-oldx[:, j])
            assert [item["production_over_old_oh_min"], item["production_over_old_oh_max"]] == pytest.approx([ratio.min(), ratio.max()], rel=1e-12)
        effect = summary["background_effect_relative_to_old_cos_only_candidate"]
        ratio = np.exp(logx[:, cos]-candx[:, cos])
        assert [effect["COS_ratio_min"], effect["COS_ratio_max"]] == pytest.approx([ratio.min(), ratio.max()], rel=1e-12)
        assert effect["max_absolute_delta_x_all_species"] == pytest.approx(np.max(abs(np.exp(logx)-np.exp(candx))), rel=1e-12)
