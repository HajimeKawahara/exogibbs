"""Replay the actual C2H2 energy-only production adoption and its retained base."""

from hashlib import sha256
import json
from pathlib import Path
import runpy
import sys

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT/"tests/data/validation/curated_c2h2_adoption/20261106"
BASELINE = ROOT/"tests/data/validation/curated_hcp_adoption/20261106"
sys.path.insert(0,str(ROOT/"mceg/reviews"))
import c2h2_candidate as candidate
from curated_energy_adoption import constraints, fractions
from curated_adoption_history import historical_source


def test_production_receipt_and_finite_controls():
    receipt=json.loads((DATA/"receipt.json").read_text())
    assert receipt["result"] == "passed" and receipt["backend"] == "cpu" and receipt["dtype"] == "float64"
    assert receipt["command"][1] == "mceg/reviews/curated_c2h2_adoption.py"
    for name,expected in receipt["selected_file_sha256"].items():
        assert sha256(historical_source(ROOT,name,expected).read_bytes()).hexdigest() == expected,name
    for name,expected in receipt["output_sha256"].items():
        assert sha256((DATA/name).read_bytes()).hexdigest() == expected,name
    summary=json.loads((DATA/"summary.json").read_text())
    assert (summary["dataset"],summary["base_dataset"]) == ("curated-c2h2-v1","curated-hcp-v1")
    assert summary["new_production_solves"] == 150 and summary["tighter_reference_solves"] == 6
    assert summary["energy_increase_J_mol"] == 1135
    assert len(summary["independent_solver"]) == 2
    for point in summary["independent_solver"]:
        assert len(point["comparisons"]) == 1
        assert point["comparisons"][0]["max_abs_delta_log10_x"] < 1e-4
    rows=json.loads((DATA/"diagnostics.json").read_text())
    assert len(rows) == 156 and all(row["converged"] for row in rows)


def test_packaged_model_rebuilds_and_rejects_changed_source(monkeypatch):
    build=runpy.run_path(str(ROOT/"mceg/reviews/build_c2h2_adoption.py"))["build_data"]
    packaged=json.loads((ROOT/"src/exogibbs/data/curated/c2h2_v1.json").read_text())
    assert build() == packaged
    assert packaged["D0_J_mol"]-packaged["base_D0_J_mol"] == -1135
    assert packaged["delta_D0_J_mol"] == -1135
    original=Path.read_bytes
    source=ROOT/"tests/data/validation/c2h2_candidate/20261106/candidate.json"
    monkeypatch.setattr(Path,"read_bytes",lambda p:original(p)+(b"\n" if p==source else b""))
    with pytest.raises(ValueError,match="reviewed candidate"):
        build()


@pytest.mark.parametrize("network,count",[("ions",537),("neutral",421)])
def test_target_only_update_closure_and_recorded_response(network,count):
    summary=json.loads((DATA/"summary.json").read_text())["networks"][network=="neutral"]
    with np.load(DATA/f"{network}.npz") as data,np.load(BASELINE/f"{network}.npz") as base:
        assert data["ln_n"].shape == (75,count) and data["converged"].all()
        for name in ("formula_matrix","budget","species","elements","temperature_K","pressure_bar","C_over_O"):
            np.testing.assert_array_equal(data[name],base[name])
        species,elements=data["species"].tolist(),data["elements"].tolist()
        target=species.index("C2H2")
        expected=base["h"].copy()
        expected[:,target]=[candidate.formation_h(t,"retained") for t in data["temperature_K"]]
        np.testing.assert_allclose(data["h"],expected,atol=1e-12,rtol=0)
        np.testing.assert_array_equal(np.delete(data["h"]-base["h"],target,axis=-1),0)
        constraints(data["formula_matrix"],elements,species,data["budget"],data["ln_n"],expected,data["pressure_bar"])
        logx=fractions(data["ln_n"]);np.testing.assert_allclose(data["ln_x"],logx,atol=1e-13,rtol=0)
        selected,tight=data["tighter_case_indices"],data["tighter_reference_ln_n"]
        constraints(data["formula_matrix"],elements,species,data["budget"][selected],tight,expected[selected],data["pressure_bar"][selected])
        difference=logx[selected]-fractions(tight)
        assert abs(difference).max()<1e-4 and abs(difference[:,target]).max()<1e-8
        assert abs(np.exp(logx[selected])-np.exp(fractions(tight))).max()<1e-9
        ratios=np.exp(logx[:,target]-fractions(base["ln_n"])[:,target])
        recorded=summary["incremental_response"]
        assert ratios.min()==pytest.approx(recorded["min_x_ratio"],abs=1e-12,rel=1e-12)
        assert ratios.max()==pytest.approx(recorded["max_x_ratio"],abs=1e-12,rel=1e-12)
        atom=elements.index("C");stoich=data["formula_matrix"][atom,target]
        shares=stoich*np.exp(data["ln_n"][:,target])/data["budget"][:,atom]
        assert shares.max()==pytest.approx(recorded["maximum_element_share_after"],abs=1e-12,rel=1e-12)
        changed=expected.copy();changed[:,target]+=1e-3
        with pytest.raises(AssertionError):
            constraints(data["formula_matrix"],elements,species,data["budget"],data["ln_n"],changed,data["pressure_bar"])
