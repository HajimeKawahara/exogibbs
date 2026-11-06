"""Replay TiO2 production on its PS base and isolate the CN/PS background."""

from hashlib import sha256
import json
from pathlib import Path
import runpy
import sys

import numpy as np
import pytest
from scipy.special import logsumexp


ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT/"tests/data/validation/curated_tio2_adoption/20261106"
BASELINE = ROOT/"tests/data/validation/curated_ps_adoption/20261106"
OLD_BASELINE = ROOT/"tests/data/validation/curated_cos_adoption/20261105"
CANDIDATE = ROOT/"tests/data/validation/tio2_equilibrium_impact/20261106"
sys.path.insert(0,str(ROOT/"mceg/reviews"))
import tio2_candidate
from tio2_equilibrium_impact import constraints
from curated_adoption_history import historical_source


def fractions(logs):
    return logs-logsumexp(logs,axis=-1,keepdims=True)


def test_production_receipt_pins_actual_execution():
    receipt=json.loads((DATA/"receipt.json").read_text())
    assert receipt["result"] == "passed"
    assert receipt["backend"] == "cpu" and receipt["dtype"] == "float64"
    assert receipt["command"][1] == "mceg/reviews/curated_tio2_adoption.py"
    for name,expected in receipt["selected_file_sha256"].items():
        assert sha256(historical_source(ROOT,name,expected).read_bytes()).hexdigest() == expected,name
    for name,expected in receipt["output_sha256"].items():
        assert sha256((DATA/name).read_bytes()).hexdigest() == expected,name
    summary=json.loads((DATA/"summary.json").read_text())
    assert (summary["dataset"],summary["base_dataset"]) == ("curated-tio2-v1","curated-ps-v1")
    assert summary["new_production_solves"] == 150 and summary["tighter_reference_solves"] == 6
    assert summary["strict_epsilon_crit"] < summary["epsilon_crit"]
    assert len(summary["independent_solver"]) == 2
    for point in summary["independent_solver"]:
        assert len(point["comparisons"]) == 1
        assert point["comparisons"][0]["max_abs_delta_log10_x"] < 1e-4
    rows=json.loads((DATA/"diagnostics.json").read_text())
    assert len(rows) == 156 and all(row["converged"] for row in rows)


def test_packaged_model_rebuilds_and_rejects_changed_source(monkeypatch):
    build=runpy.run_path(str(ROOT/"mceg/reviews/build_tio2_adoption.py"))["build_data"]
    packaged=json.loads((ROOT/"src/exogibbs/data/curated/tio2_v1.json").read_text())
    assert build() == packaged
    model=tio2_candidate.load_candidate()
    for name in ("nasa_molecular_thermal_coefficients","rotational_constants_MHz","frequencies_cm-1",
                 "mass_u","original_source_D0_J_mol","delta_D0_J_mol","symmetry_number"):
        assert packaged[name] == model[name]
    assert packaged["delta_D0_J_mol"] == 0
    original=Path.read_bytes
    candidate=ROOT/"tests/data/validation/tio2_candidate/20261106/candidate.json"
    monkeypatch.setattr(Path,"read_bytes",lambda p:original(p)+(b"\n" if p==candidate else b""))
    with pytest.raises(ValueError,match="reviewed candidate"):
        build()


@pytest.mark.parametrize("network,count",[("ions",537),("neutral",421)])
def test_tio2_only_update_conserves_the_cumulative_network(network,count):
    with np.load(DATA/f"{network}.npz") as data,np.load(BASELINE/f"{network}.npz") as base:
        assert data["ln_n"].shape == (75,count) and data["converged"].all()
        for name in ("formula_matrix","budget","species","elements","temperature_K","pressure_bar","C_over_O"):
            np.testing.assert_array_equal(data[name],base[name])
        species,elements=data["species"].tolist(),data["elements"].tolist()
        target=species.index("O2Ti1")
        expected=base["h"].copy()
        expected[:,target]=[tio2_candidate.formation_h(t) for t in data["temperature_K"]]
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
        assert .89 < ratio.min() < .90 and .95 < ratio.max() < .96
        changed=expected.copy();changed[:,target]+=1e-3
        with pytest.raises(AssertionError):
            constraints(data["formula_matrix"],elements,species,data["budget"],data["ln_n"],changed,data["pressure_bar"])


@pytest.mark.parametrize("network",["ions","neutral"])
def test_summary_distinguishes_incremental_and_cumulative_responses(network):
    with np.load(DATA/f"{network}.npz") as data,np.load(BASELINE/f"{network}.npz") as base,np.load(OLD_BASELINE/f"{network}.npz") as original:
        summary=next(row for row in json.loads((DATA/"summary.json").read_text())["networks"] if row["ions"]==(network=="ions"))
        logx,basex,oldx=map(fractions,(data["ln_n"],base["ln_n"],original["ln_n"]))
        species,elements=data["species"].tolist(),data["elements"].tolist()
        target,titanium=species.index("O2Ti1"),elements.index("Ti")
        ratios=np.exp(logx[:,target]-basex[:,target]);r=summary["tio2_incremental_response"]
        assert [r["min_x_ratio"],r["max_x_ratio"]] == pytest.approx([ratios.min(),ratios.max()],rel=1e-12)
        shares=np.array([np.exp(base["ln_n"][:,target]),np.exp(data["ln_n"][:,target])])/data["budget"][:,titanium]
        i=int(np.argmax(abs(shares[1]-shares[0])))
        assert r["highlighted_case_index"] == i
        assert [r["titanium_share_before"],r["titanium_share_after"]] == pytest.approx(shares[:,i],rel=1e-12)
        for name,item in summary["cumulative_response_from_curated_cos"].items():
            j=species.index(name);ratio=np.exp(logx[:,j]-oldx[:,j])
            assert [item["production_over_old_cos_min"],item["production_over_old_cos_max"]] == pytest.approx([ratio.min(),ratio.max()],rel=1e-12)
        effect=summary["background_effect_relative_to_old_tio2_only_candidate"]
        if network=="ions":
            with np.load(CANDIDATE/"ions.npz") as candidate:
                candx=fractions(candidate["ln_n"][1]);ratio=np.exp(logx[:,target]-candx[:,target])
                assert [effect["TiO2_ratio_min"],effect["TiO2_ratio_max"]] == pytest.approx([ratio.min(),ratio.max()],rel=1e-12)
                assert effect["max_absolute_delta_x_all_species"] == pytest.approx(abs(np.exp(logx)-np.exp(candx)).max(),rel=1e-12)
        else:
            assert effect.startswith("not_available:")
