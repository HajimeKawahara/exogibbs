"""Verify NS conventions, distinct energy evaluations and archived replay."""
import json
import math
from hashlib import sha256
from pathlib import Path
import sys

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "mceg/reviews"))
import ns_source_review as study


@pytest.fixture(scope="module")
def report():
    return study.review()


def test_corrected_source_is_not_old_nasa_or_peebles(report):
    assert report["balanced_D0_J_mol"]["nasa9"] == pytest.approx(478999.947)
    assert report["Peebles_published_D0_J_mol"] == 468200
    inputs = json.loads((study.DATA / "inputs.json").read_text())
    assert inputs["lodders_hf298_J_mol"] == 283400
    assert inputs["peebles_hf298_J_mol"] == 278000
    for r in report["source_samples"]:
        t = r["temperature_K"]
        assert r["lodders"]["Peebles_H298_shift_only_K_ratio"] == pytest.approx(math.exp(5400/(study.major.R*t)))
        if t >= 1500:
            assert .98 < r["lodders"]["curated_over_reconstructed"] < 1
    assert report["source_samples"][3]["curated_over_nasa"] == pytest.approx(.27299437179538)
    with pytest.raises(ValueError, match="outside"):
        study.curated_ln_k(1000)


def test_spin_normalization_preserves_state_structure(report):
    with np.load(study.DATA / "states.npz") as f:
        states = f["states"]
    assert states.shape == (31502, 4)
    assert np.min(states[:, 0]) == 0
    assert np.all(states[:, 1] == 3*(2*states[:, 2]+1))
    for r in report["source_samples"]:
        p = r["modern_partition"]
        assert abs(p["Q_over_published"]-1) < 7e-7
        assert p["isotope_translation_K_proxy"] == pytest.approx(p["Q_over_nasa"]*(45.975145/46.0717)**1.5)
    top = report["source_samples"][-1]["modern_partition"]
    assert top["Q_over_nasa"] == pytest.approx(1.0050089356)
    assert top["Q_fraction_above_24524_cm-1"] < .00185
    assert top["Q_fraction_J_above_220"] < 3e-6
    dt = .5
    hp = study.moments(states, 3000+dt, 3)["H_increment_J_mol"]
    hm = study.moments(states, 3000-dt, 3)["H_increment_J_mol"]
    assert (hp-hm)/(2*dt) == pytest.approx(study.moments(states, 3000, 3)["Cp_J_mol_K"], abs=1e-8)


def test_current_footprint_and_no_substitution(report):
    f = report["footprint"]
    assert (f["model"],f["cases"],f["species"],f["all_converged"]) == ("curated-oh-v1",75,537,True)
    assert f["maximum_abs_h_error"] < 2e-12
    assert f["maximum_relative_sulfur_conservation_residual"] < 3e-12
    assert f["maximum_sulfur_fraction"] == pytest.approx(4.3453908654014924e-5)
    assert f["case_index"] == 24
    assert report["production_change"] == "none"
    assert report["equilibrium_substitution"] == "not_run"


def test_frozen_report_and_receipt(report):
    expected = json.loads((study.DATA / "comparison.json").read_text())
    def equal(a,b):
        if isinstance(a,dict):
            assert a.keys()==b.keys()
            for k in a: equal(a[k],b[k])
        elif isinstance(a,list):
            assert len(a)==len(b)
            for x,y in zip(a,b):equal(x,y)
        elif isinstance(a,float):assert a == pytest.approx(b,rel=2e-11,abs=2e-10)
        else:assert a==b
    equal(report,expected)
    receipt = json.loads((study.DATA/"receipt.json").read_text())
    for group in ("selected_input_sha256","output_sha256"):
        for name,digest in receipt[group].items():
            assert sha256((ROOT/name).read_bytes()).hexdigest()==digest,name
