"""Replay compact PS production on the CN base and isolate background changes."""

from hashlib import sha256
import json
from pathlib import Path
import sys

import numpy as np
import pytest
from scipy.special import logsumexp

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT/"tests/data/validation/curated_ps_adoption/20261106"
BASELINE = ROOT/"tests/data/validation/curated_cn_adoption/20261106"
OLD_BASELINE = ROOT/"tests/data/validation/curated_cos_adoption/20261105"
CANDIDATE = ROOT/"tests/data/validation/ps_equilibrium_impact/20261106"
sys.path.insert(0,str(ROOT/"mceg/reviews"))
import ps_candidate
from ps_equilibrium_impact import constraints
from curated_adoption_history import historical_source


def fractions(logs):
    return logs-logsumexp(logs,axis=-1,keepdims=True)


def test_production_receipt_records_actual_fresh_endpoints_and_controls():
    receipt=json.loads((DATA/"receipt.json").read_text())
    assert receipt["result"]=="passed" and receipt["backend"]=="cpu" and receipt["dtype"]=="float64"
    assert receipt["command"][1]=="mceg/reviews/curated_ps_adoption.py"
    for name,digest in receipt["selected_file_sha256"].items():
        assert sha256(historical_source(ROOT,name,digest).read_bytes()).hexdigest()==digest,name
    for name,digest in receipt["output_sha256"].items():
        assert sha256((DATA/name).read_bytes()).hexdigest()==digest,name
    summary=json.loads((DATA/"summary.json").read_text())
    assert (summary["dataset"],summary["base_dataset"])==("curated-ps-v1","curated-cn-v1")
    assert summary["new_production_solves"]==150 and summary["tighter_reference_solves"]==6
    assert summary["strict_epsilon_crit"] < summary["epsilon_crit"]
    assert len(summary["independent_solver"])==2
    for point in summary["independent_solver"]:
        assert len(point["comparisons"])==1
        row=point["comparisons"][0]
        assert row["max_abs_delta_log10_x"]<1e-4
        assert row["element_relative_residual"]<1e-5
        assert row["charge_relative_residual"]<1e-7
        assert row["mass_action_max_abs_ln"]<1e-7
    rows=json.loads((DATA/"diagnostics.json").read_text())
    assert len(rows)==156 and all(row["converged"] for row in rows)


@pytest.mark.parametrize("network,count",[("ions",537),("neutral",421)])
def test_ps_only_update_matches_direct_candidate_and_conserves_network(network,count):
    with np.load(DATA/f"{network}.npz") as data,np.load(BASELINE/f"{network}.npz") as base:
        assert data["ln_n"].shape==(75,count) and data["converged"].all()
        for name in ("formula_matrix","budget","species","elements","temperature_K","pressure_bar","C_over_O"):
            np.testing.assert_array_equal(data[name],base[name])
        species,elements=data["species"].tolist(),data["elements"].tolist()
        target=species.index("P1S1")
        expected=base["h"].copy()
        expected[:,target]=[ps_candidate.formation_h(t) for t in data["temperature_K"]]
        np.testing.assert_allclose(data["h"],expected,atol=1e-12,rtol=0)
        np.testing.assert_array_equal(np.delete(data["h"]-base["h"],target,axis=-1),0)
        constraints(data["formula_matrix"],elements,species,data["budget"],data["ln_n"],expected,data["pressure_bar"])
        logx=fractions(data["ln_n"])
        np.testing.assert_allclose(data["ln_x"],logx,atol=1e-13,rtol=0)
        selected,tight=data["tighter_case_indices"],data["tighter_reference_ln_n"]
        assert len(selected)==3
        constraints(data["formula_matrix"],elements,species,data["budget"][selected],tight,expected[selected],data["pressure_bar"][selected])
        tight_x=fractions(tight)
        assert abs(logx[selected]-tight_x).max()<1e-4
        assert abs(logx[selected,target]-tight_x[:,target]).max()<1e-8
        assert abs(np.exp(logx[selected])-np.exp(tight_x)).max()<1e-9
        delta=logx-fractions(base["ln_n"])
        predicted=delta[:,species.index("P1")]+delta[:,species.index("S1")]-(data["h"][:,target]-base["h"][:,target])
        np.testing.assert_allclose(delta[:,target],predicted,atol=1e-10,rtol=0)
        changed=expected.copy();changed[:,target]+=1e-3
        with pytest.raises(AssertionError):
            constraints(data["formula_matrix"],elements,species,data["budget"],data["ln_n"],changed,data["pressure_bar"])
        if network=="ions":
            with np.load(CANDIDATE/"ions.npz") as old:
                np.testing.assert_allclose(data["h"][:,target],old["h"][1,:,target],atol=1e-12,rtol=0)
                cn=species.index("C1N1")
                assert abs(data["h"][:,cn]-old["h"][1,:,cn]).max()>.3
                np.testing.assert_allclose(np.delete(data["h"]-old["h"][1],[target,cn],axis=-1),0,atol=1e-12,rtol=0)


@pytest.mark.parametrize("network",["ions","neutral"])
def test_response_summary_distinguishes_incremental_and_cumulative_effects(network):
    with np.load(DATA/f"{network}.npz") as data,np.load(BASELINE/f"{network}.npz") as base,np.load(OLD_BASELINE/f"{network}.npz") as original:
        row=next(r for r in json.loads((DATA/"summary.json").read_text())["networks"] if r["ions"]==(network=="ions"))
        actual,prior,old=map(fractions,(data["ln_n"],base["ln_n"],original["ln_n"]))
        species,elements=data["species"].tolist(),data["elements"].tolist()
        target,phosphorus=species.index("P1S1"),elements.index("P")
        ratio=np.exp(actual[:,target]-prior[:,target]);r=row["ps_incremental_response"]
        assert [r["min_x_ratio"],r["max_x_ratio"]]==pytest.approx([ratio.min(),ratio.max()],rel=1e-12)
        shares=np.array([np.exp(base["ln_n"][:,target]),np.exp(data["ln_n"][:,target])])/data["budget"][:,phosphorus]
        case=int(np.argmax(abs(shares[1]-shares[0])))
        assert r["highlighted_case_index"]==case
        assert [r["phosphorus_share_before"],r["phosphorus_share_after"]]==pytest.approx(shares[:,case],rel=1e-12)
        assert row["max_dense_h_difference_from_candidate_on_cn"]<1e-12
        for name,reported in row["cumulative_response_from_curated_cos"].items():
            index=species.index(name);q=np.exp(actual[:,index]-old[:,index])
            assert [reported["production_over_old_cos_min"],reported["production_over_old_cos_max"]]==pytest.approx([q.min(),q.max()],rel=1e-12)
        effect=row["background_effect_relative_to_old_ps_only_candidate"]
        if network=="ions":
            with np.load(CANDIDATE/"ions.npz") as packet:
                earlier=fractions(packet["ln_n"][1]);q=np.exp(actual[:,target]-earlier[:,target])
                assert [effect["PS_ratio_min"],effect["PS_ratio_max"]]==pytest.approx([q.min(),q.max()],rel=1e-12)
                assert effect["max_absolute_delta_x_all_species"]==pytest.approx(abs(np.exp(actual)-np.exp(earlier)).max(),rel=1e-12)
        else:
            assert effect.startswith("not_available:")
