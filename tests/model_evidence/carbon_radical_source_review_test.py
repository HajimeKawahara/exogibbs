"""Independent normalization and archive checks for available carbon packets."""

import json
import math
from pathlib import Path
import sys

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "mceg/reviews"))
import carbon_radical_source_review as study

AVAILABLE = [s for s in ("cn", "ch", "ch3")
             if (ROOT/f"tests/data/validation/{s}_source_review/20261106/inputs.json").is_file()]


@pytest.fixture(scope="module", params=AVAILABLE)
def packet(request):
    slug = request.param
    data = ROOT/f"tests/data/validation/{slug}_source_review/20261106"
    return slug, data, json.loads((data/"inputs.json").read_text()), study.review(slug)


def test_balanced_energy_and_native_constant_decomposition(packet):
    slug, data, config, report = packet
    expected = {"cn": (745355, -4801), "ch": (334443, -2095), "ch3": (1209580, -680)}[slug]
    assert report["balanced_ATcT_D0_J_mol"] == pytest.approx(expected[0])
    assert report["ATcT_minus_janaf_D0_J_mol"] == pytest.approx(expected[1])
    for row in report["samples"]:
        assert abs(row["source"]["decomposition_closure_ln_K"]) < 5e-13
        assert row["same_thermal_ATcT_energy_only_K_ratio"] == pytest.approx(
            math.exp(expected[1]/(study.R*row["temperature_K"])))
    with pytest.raises(ValueError, match="outside"):
        study.curated_properties(config["species_id"], 1000)


def test_exact_official_nasa_extraction(packet):
    slug, data, config, report = packet
    nasa = json.loads((data/"nasa_records.json").read_text())
    baseline = json.loads((ROOT/"tests/data/validation/gas_source_comparison/baseline-001/inputs.json").read_text())
    blocks = (data/"nasa_selected_records.txt").read_text().strip().split("\n\n")
    for name, record in nasa.items():
        lines = next(b.splitlines() for b in blocks if b[:18].strip() == name)
        assert record["hf298_J_mol"] == float(lines[1][65:])
        assert record["h298_minus_h0_J_mol"] == float(lines[2][65:])
        for i, interval in enumerate(record["intervals"]):
            printed = lines[3+3*i]+lines[4+3*i]
            expected = [float(printed[k:k+16].replace("D", "E")) for k in [0,16,32,48,64,80,96,128,144]]
            assert interval["coefficients"] == expected
            assert interval["bounds_K"] == [float(v) for v in lines[2+3*i][:22].split()]
            assert interval["coefficients"] == baseline["nasa9_records"][name]["intervals"][i]["coefficients"]


def test_production_identity_and_archived_footprint(packet):
    import jax
    from exogibbs.presets import curated_cos
    jax.config.update("jax_enable_x64", True)
    slug, data, config, report = packet
    setup = curated_cos.chemsetup(ions=True)
    col = setup.species.index(config["species_id"])
    assert {e:float(setup.formula_matrix[i,col]) for i,e in enumerate(setup.elements)
            if setup.formula_matrix[i,col]} == config["composition"]
    for t in [1500.,2000.,3000.,4000.]:
        p = study.curated_properties(config["species_id"],t)
        assert p["ln_K"] == pytest.approx(-float(setup.hvector_func(t)[col]),abs=3e-12)
        assert p["ln_K"] == pytest.approx(-(p["H_J_mol"]/t-p["S_J_mol_K"])/study.R,abs=5e-13)
    f=report["footprint"]
    assert (f["cases"],f["species_count"],f["all_converged"]) == (75,537,True)
    assert f["maximum_abs_h_error"] < 3e-12
    with np.load(study.GRID) as g:
        c=g["species"].tolist().index(config["species_id"])
        e=g["elements"].tolist().index(f["element"])
        i=f["case_index"]
        assert f["maximum_element_share"] == pytest.approx(
            config["composition"][f["element"]]*np.exp(g["ln_n"][i,c])/g["budget"][i,e])
    assert report["production_change"] == "none"
    assert report["equilibrium_substitution"] == "not_run"


def test_partition_scope_and_caloric_discrimination(packet):
    slug, data, config, report = packet
    rows = {r["temperature_K"]:r for r in report["samples"]}
    if slug == "cn":
        assert report["state_inventory"]["rows"] == 28004
        assert report["state_inventory"]["minimum_energy_cm1"] == 0
        assert report["state_inventory"]["maximum_spin_weight_error"] == 0
        for row in rows.values():
            assert row["partition"]["states"]["Q_over_published"] == pytest.approx(1,abs=8e-7)
        with np.load(data/"states.npz") as states:
            e,g=states["energy_cm1"],states["gtot"]/3
            t,dt=3000.,0.05
            left=study.partition_properties(*study.state_moments(e,g,t-dt),t-dt,26.0174)
            mid=study.partition_properties(*study.state_moments(e,g,t),t,26.0174)
            right=study.partition_properties(*study.state_moments(e,g,t+dt),t+dt,26.0174)
            assert (right["HminusH0_J_mol"]-left["HminusH0_J_mol"])/(2*dt) == pytest.approx(mid["Cp_J_mol_K"],abs=2e-7)
            assert t*(right["S_J_mol_K"]-left["S_J_mol_K"])/(2*dt) == pytest.approx(mid["Cp_J_mol_K"],abs=2e-7)
        assert rows[4000]["partition"]["states"]["J_above_100_Q_fraction"] > 0.001
    elif slug == "ch":
        assert report["state_inventory"]["rows"] == 2680
        assert report["state_inventory"]["electronic_states"]["B(2SIGMA-)"]["minimum_energy_cm1"] < 15
        definition=json.loads((data/"exomol_definition.json").read_text())
        assert definition["dataset"]["states"]["number_of_states"] == 2681
        for t,row in rows.items():
            part=row["partition"]
            assert "states" not in part
            if t>=1000:
                assert part["published_spin_normalized_Q"] == pytest.approx(part["masseron2014_analytic_fit"]["properties_at_NASA_mass"]["Q"],abs=0.000026)
        assert rows[4000]["partition"]["masseron2014_analytic_fit"]["minus_NASA_thermal"]["Cp_J_mol_K"] < -2
        assert report["ATcT_quartet_minus_doublet_kJ_mol"] == pytest.approx(72.049)
    else:
        assert config["partition"]["nuclear_spin_divisor"] == 8
        assert rows[1500]["partition"]["published_Q_over_nasa"] == pytest.approx(0.9843541027651328)
        assert rows[2000]["partition"]["published_spin_normalized_Q"] is not None
        assert rows[3000]["partition"]["published_spin_normalized_Q"] is None
        assert rows[4000]["partition"]["published_spin_normalized_Q"] is None
        assert report["state_inventory"] is None


def test_archived_replay_and_receipt(packet):
    slug, data, config, report = packet
    def compare(a,b):
        if isinstance(a,dict):
            assert a.keys()==b.keys()
            for k in a: compare(a[k],b[k])
        elif isinstance(a,list):
            assert len(a)==len(b)
            for x,y in zip(a,b): compare(x,y)
        elif isinstance(a,float):
            assert a == pytest.approx(b,rel=2e-11,abs=2e-10)
        else:
            assert a == b
    compare(report,json.loads((data/"comparison.json").read_text()))
    receipt=json.loads((data/"receipt.json").read_text())
    for group in ["selected_input_sha256","output_sha256"]:
        for path,expected in receipt[group].items():
            assert study.file_hash(ROOT/path)==expected,path
