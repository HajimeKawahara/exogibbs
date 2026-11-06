"""Verify P4S3 source anchors, moments, inherited fit and immutable evidence."""

from copy import deepcopy
from hashlib import sha256
import json
from pathlib import Path
import re
import sys

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "mceg/reviews"))
import p4s3_source_review as study
from curated_adoption_history import historical_source


def compare(actual, expected, path=""):
    if isinstance(expected, dict):
        assert actual.keys() == expected.keys(), path
        for key in expected:
            compare(actual[key], expected[key], path + "." + key)
    elif isinstance(expected, list):
        assert len(actual) == len(expected), path
        for i, value in enumerate(expected):
            compare(actual[i], value, path + f"[{i}]")
    elif isinstance(expected, float):
        assert actual == pytest.approx(expected, rel=2e-12,
            abs=1e-8 if "_J_mol" in path else 2e-11), path
    else:
        assert actual == expected, path


def test_source_replay_and_dimensional_regression():
    expected = json.loads((study.DATA / "comparison.json").read_text())
    compare(study.review(), expected)
    changed = deepcopy(expected)
    changed["source_samples"][0]["source"]["molecular"]["Cp_J_mol_K"] += 1e-4
    with pytest.raises(AssertionError):
        compare(changed, expected)
    for row in expected["source_samples"]:
        source = row["source"]
        assert abs(source["closure_ln_K"]) < 5e-12
        assert source["native_janaf_D0_J_mol"] is None
        assert source["energy_alignment"] == "H298"
        for label, gas_r in [("janaf",study.R)]:
            reaction = source["reaction"][label]
            assert reaction["G_J_mol"] == pytest.approx(reaction["H_J_mol"]-row["temperature_K"]*reaction["S_J_mol_K"],abs=1e-8)
            assert reaction["ln_K"] == pytest.approx(-reaction["G_J_mol"]/(gas_r*row["temperature_K"]),abs=2e-12)


def test_printed_janaf_and_explicit_nasa_absence():
    config = study.inputs()
    assert config["nasa9"] == {} and "no matching neutral P4S3 gas" in config["nasa_comparison_status"]
    baseline = json.loads((ROOT / "tests/data/validation/gas_source_comparison/baseline-001/inputs.json").read_text())
    assert "P4S3" not in baseline["nasa9_records"]
    packaged = np.load(ROOT / "src/exogibbs/data/ykb4/gibbs_matrices.npz",allow_pickle=True)["arr_0"].item()
    for name,f in config["tables"].items():
        sid = study.SPECIES if name == study.NAME else baseline["atomic_references"][name]["janaf_id"]
        frame,table = packaged[sid],study.read_table(study.DATA/f)
        for t in [298.15,1500.,2000.,3000.,4000.]:
            np.testing.assert_array_equal(frame[frame["T(K)"]==t].iloc[0].to_numpy(dtype=float)[1:],table[t])


def test_actual_named_factory_and_analytic_calorics():
    import jax
    from exogibbs.presets import curated_tio2
    jax.config.update("jax_enable_x64", True)
    setup = curated_tio2.chemsetup(ions=True)
    j = setup.species.index(study.SPECIES)
    for t in [1500., 2000., 3000., 4000.]:
        p = study.curated_properties(t)
        assert p["ln_K"] == pytest.approx(-float(setup.hvector_func(t)[j]), abs=3e-12)
        assert p["G_J_mol"] == pytest.approx(p["H_J_mol"]-t*p["S_J_mol_K"], abs=1e-8)
    t, dt = 2500., .05
    left, mid, right = [study.curated_properties(x) for x in [t-dt,t,t+dt]]
    assert (right["H_J_mol"]-left["H_J_mol"])/(2*dt) == pytest.approx(mid["Cp_J_mol_K"], abs=2e-7)
    assert (right["S_J_mol_K"]-left["S_J_mol_K"])/(2*dt) == pytest.approx(mid["Cp_J_mol_K"]/t, abs=2e-10)
    with pytest.raises(ValueError, match="outside"):
        study.curated_properties(1000)


def test_molecule_specific_source_diagnostic():
    facts = json.loads((study.DATA / "literature.json").read_text())
    table = study.read_table(study.DATA / study.inputs()["tables"][study.NAME])
    assert 0. not in table and facts["janaf_printed"]["zero_K_molecular_anchor"] is None
    assert sum(study.inputs()["composition"].values()) == 7
    for row in study.review()["source_samples"]:
        modern = row["modern"]
        assert row["curated_over_nasa"] is None
        assert abs(modern["constant_Cp_H_increment_minus_table_J_mol"]) < 1
        assert abs(modern["constant_Cp_S_minus_table_J_mol_K"]) < .001
        assert modern["constant_Cp_minus_nonlinear_RRHO_high_T_limit_J_mol_K"] < -3
        for atom,value in [("P",5360.),("S",4412.)]:
            assert modern["elemental_reference_heat_contents_J_mol"][atom] == pytest.approx(value,abs=1e-8)
        for key,delta in [("preferred",56542.),("atomization_route",47142.)]:
            energy = modern["energy_alternatives"][key]
            assert energy["H298_shift_J_mol"] == delta
            assert energy["delta_G_J_mol"] == energy["delta_H_J_mol"] == delta
            assert energy["delta_S_J_mol_K"] == energy["delta_Cp_J_mol_K"] == 0
            assert np.log(energy["energy_only_K_factor"]) == pytest.approx(-delta/(study.R*row["temperature_K"]))
        assert modern["cycle_3_to_9_spread_J_mol"] == 4800.


def test_current_footprint_scope_and_receipt():
    result = study.review()
    f = result["footprint"]
    assert (f["dataset"],f["cases"],f["species"],f["all_converged"]) == ("curated-tio2-v1",75,537,True)
    assert f["maximum_abs_h_error"] < 3e-12
    with np.load(study.GRID) as g:
        j = g["species"].tolist().index(study.SPECIES)
        for element, detail in f["element_shares"].items():
            e = g["elements"].tolist().index(element)
            i = detail["case_index"]
            assert detail["maximum_share"] == pytest.approx(study.inputs()["composition"][element]*np.exp(g["ln_n"][i,j])/g["budget"][i,e],rel=1e-12,abs=0)
    assert result["new_equilibrium_substitution"] == "not_run" and result["curated_change"] == "none"
    receipt = json.loads((study.DATA / "receipt.json").read_text())
    for name, expected in receipt["selected_input_sha256"].items():
        assert sha256(historical_source(ROOT, name, expected).read_bytes()).hexdigest() == expected, name
    for name, expected in receipt["output_sha256"].items():
        assert sha256((study.DATA / name).read_bytes()).hexdigest() == expected, name
