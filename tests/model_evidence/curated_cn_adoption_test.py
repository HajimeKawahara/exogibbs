"""Replay CN production on its COS base and isolate the unchanged background."""

from hashlib import sha256
import json
from pathlib import Path
import runpy
import sys

import numpy as np
import pytest
from scipy.special import logsumexp


ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT/"tests/data/validation/curated_cn_adoption/20261106"
BASELINE = ROOT/"tests/data/validation/curated_cos_adoption/20261105"
OLD_BASELINE = ROOT/"tests/data/validation/curated_cos_adoption/20261105"
CANDIDATE = ROOT/"tests/data/validation/cn_equilibrium_impact/20261106"
sys.path.insert(0,str(ROOT/"mceg/reviews"))
import cn_candidate
from cn_equilibrium_impact import constraints
from curated_adoption_history import historical_source


def fractions(logs):
    return logs-logsumexp(logs,axis=-1,keepdims=True)


def test_production_receipt_pins_actual_execution():
    receipt=json.loads((DATA/"receipt.json").read_text())
    assert receipt["result"] == "passed"
    assert receipt["backend"] == "cpu" and receipt["dtype"] == "float64"
    assert receipt["command"][1] == "mceg/reviews/curated_cn_adoption.py"
    for name,expected in receipt["selected_file_sha256"].items():
        assert sha256(historical_source(ROOT,name,expected).read_bytes()).hexdigest() == expected,name
    for name,expected in receipt["output_sha256"].items():
        assert sha256((DATA/name).read_bytes()).hexdigest() == expected,name
    summary=json.loads((DATA/"summary.json").read_text())
    assert (summary["dataset"],summary["base_dataset"]) == ("curated-cn-v1","curated-cos-v1")
    assert summary["new_production_solves"] == 150 and summary["tighter_reference_solves"] == 6
    assert summary["strict_epsilon_crit"] < summary["epsilon_crit"]
    assert len(summary["independent_solver"]) == 2
    for point in summary["independent_solver"]:
        assert len(point["comparisons"]) == 1
        assert point["comparisons"][0]["max_abs_delta_log10_x"] < 1e-4
    rows=json.loads((DATA/"diagnostics.json").read_text())
    assert len(rows) == 156 and all(row["converged"] for row in rows)


def test_packaged_model_rebuilds_and_rejects_changed_source(monkeypatch):
    build=runpy.run_path(str(ROOT/"mceg/reviews/build_cn_adoption.py"))["build_data"]
    packaged=json.loads((ROOT/"src/exogibbs/data/curated/cn_v1.json").read_text())
    assert build() == packaged
    model=cn_candidate.load_candidate()
    for name in ("base_D0_J_mol", "D0_J_mol", "delta_D0_J_mol", "formal_fit_intercept_Ra_J_mol"):
        assert packaged[name] == model[name]
    assert packaged["delta_D0_J_mol"] == -4801
    original=Path.read_bytes
    candidate=ROOT/"tests/data/validation/cn_candidate/20261106/candidate.json"
    monkeypatch.setattr(Path,"read_bytes",lambda p:original(p)+(b"\n" if p==candidate else b""))
    with pytest.raises(ValueError,match="reviewed candidate"):
        build()


@pytest.mark.parametrize("network,count",[("ions",537),("neutral",421)])
def test_cn_only_update_conserves_the_cumulative_network(network,count):
    with np.load(DATA/f"{network}.npz") as data,np.load(BASELINE/f"{network}.npz") as base:
        assert data["ln_n"].shape == (75,count) and data["converged"].all()
        for name in ("formula_matrix","budget","species","elements","temperature_K","pressure_bar","C_over_O"):
            np.testing.assert_array_equal(data[name],base[name])
        species,elements=data["species"].tolist(),data["elements"].tolist()
        target=species.index("C1N1")
        expected=base["h"].copy()
        expected[:,target]=[cn_candidate.formation_h(t) for t in data["temperature_K"]]
        np.testing.assert_allclose(data["h"],expected,atol=1e-12,rtol=0)
        np.testing.assert_array_equal(np.delete(data["h"]-base["h"],target,axis=-1),0)
        constraints(data["formula_matrix"],elements,species,data["budget"],data["ln_n"],expected,data["pressure_bar"])
        logx=fractions(data["ln_n"])
        np.testing.assert_allclose(data["ln_x"],logx,atol=1e-13,rtol=0)
        selected,tight=data["tighter_case_indices"],data["tighter_reference_ln_n"]
        constraints(data["formula_matrix"],elements,species,data["budget"][selected],tight,expected[selected],data["pressure_bar"][selected])
        tight_x=fractions(tight)
        assert abs(logx[selected]-tight_x).max() < 1e-4
        assert abs(logx[selected,target]-tight_x[:,target]).max() < 1e-8
        assert abs(np.exp(logx[selected])-np.exp(tight_x)).max() < 1e-9
        ratio=np.exp(logx[:,target]-fractions(base["ln_n"])[:,target])
        assert .68 < ratio.min() < .69 and .87 < ratio.max() < .88
        changed=expected.copy();changed[:,target]+=1e-3
        with pytest.raises(AssertionError):
            constraints(data["formula_matrix"],elements,species,data["budget"],data["ln_n"],changed,data["pressure_bar"])


@pytest.mark.parametrize("network",["ions","neutral"])
def test_summary_distinguishes_incremental_and_cumulative_responses(network):
    with np.load(DATA/f"{network}.npz") as data,np.load(BASELINE/f"{network}.npz") as base,np.load(OLD_BASELINE/f"{network}.npz") as original:
        summary=next(row for row in json.loads((DATA/"summary.json").read_text())["networks"] if row["ions"]==(network=="ions"))
        logx,basex,oldx=map(fractions,(data["ln_n"],base["ln_n"],original["ln_n"]))
        species,elements=data["species"].tolist(),data["elements"].tolist()
        target,nitrogen=species.index("C1N1"),elements.index("N")
        ratios=np.exp(logx[:,target]-basex[:,target]);r=summary["cn_incremental_response"]
        assert [r["min_x_ratio"],r["max_x_ratio"]] == pytest.approx([ratios.min(),ratios.max()],rel=1e-12)
        shares=np.array([np.exp(base["ln_n"][:,target]),np.exp(data["ln_n"][:,target])])/data["budget"][:,nitrogen]
        i=int(np.argmax(abs(np.exp(logx[:,target])-np.exp(basex[:,target]))))
        assert r["highlighted_case_index"] == i
        assert [r["nitrogen_share_before"],r["nitrogen_share_after"]] == pytest.approx(shares[:,i],rel=1e-12)
        for name,item in summary["cumulative_response_from_curated_cos"].items():
            j=species.index(name);ratio=np.exp(logx[:,j]-oldx[:,j])
            assert [item["production_over_old_cos_min"],item["production_over_old_cos_max"]] == pytest.approx([ratio.min(),ratio.max()],rel=1e-12)
        effect=summary["background_effect_relative_to_old_cn_only_candidate"]
        if network=="ions":
            with np.load(CANDIDATE/"ions.npz") as candidate:
                candx=fractions(candidate["ln_n"][1]);ratio=np.exp(logx[:,target]-candx[:,target])
                assert [effect["CN_ratio_min"],effect["CN_ratio_max"]] == pytest.approx([ratio.min(),ratio.max()],rel=1e-12)
                assert effect["max_absolute_delta_x_all_species"] == pytest.approx(abs(np.exp(logx)-np.exp(candx)).max(),rel=1e-12)
        else:
            assert effect.startswith("not_available:")
