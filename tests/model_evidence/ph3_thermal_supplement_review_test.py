"""Keep retrieved PH3 data defects reproducible without production adoption."""

from hashlib import sha256
import json
from pathlib import Path
import sys

import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "mceg/reviews"))
import ph3_thermal_supplement_review as study


def test_raw_moments_violate_positive_state_sum_constraints():
    moments, _ = study.read_sources()
    q, q1, q2 = moments[100, 300]
    assert (q, q1, q2) == (3319.5, 5285, 5437.1)
    # Cauchy--Schwarz requires Q*Q2 >= Q1**2 for any nonnegative weights.
    assert q * q2 < q1**2
    assert moments[41, 300][2] == 15010
    assert moments[42, 300][2] == 5437.1
    for row in study.review()["samples"]:
        assert row["Q2_drop_lower_after_source_rounding"] > 0
        if row["temperature_K"] <= 2000:
            assert row["variance_upper_after_source_rounding"] < 0


def test_caloric_closure_defect_is_large_and_stable_at_two_steps():
    row = next(r for r in study.review()["samples"] if r["temperature_K"] == 3000)
    values = row["closure"]
    assert abs(values[0]["dH_dT_J_mol_K"] - values[1]["dH_dT_J_mol_K"]) < .1
    for v in values:
        assert v["dH_dT_minus_published_Cp_J_mol_K"] > 4
        assert v["dH_dT_rounding_bound_J_mol_K"] < .3
        assert abs(v["dH_dT_J_mol_K"] - v["T_dS_dT_J_mol_K"]) < .2
    # The derivative stencil has an independent exact polynomial control.
    ts = {t: float(t**3) for t in (180, 240, 360, 420)}
    assert study.derivative(ts, 300, 60)[0] == pytest.approx(3 * 300**2)


def test_direct_q_and_scope_do_not_imply_candidate_validation():
    report = study.review()
    assert abs(report["final_row_Q296"] - report["paper_direct_Q296"]) < report["Q296_rounding_half_width"]
    assert report["production_change"] == "none"
    assert report["candidate_created"] is False
    assert report["equilibrium_substitution"] == "not_run"
    assert report["nuclear_spin_divisor_for_thermochemical_Q"] == 16


def test_saved_replay_and_receipt():
    assert study.review() == json.loads((study.DATA / "comparison.json").read_text())
    receipt = json.loads((study.DATA / "receipt.json").read_text())
    for group in ("selected_input_sha256", "output_sha256"):
        for name, digest in receipt[group].items():
            assert sha256((ROOT / name).read_bytes()).hexdigest() == digest, name
