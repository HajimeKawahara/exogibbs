"""Check NH source normalization, archived scope and independent identities."""

import json
import math
from pathlib import Path
import re
import sys

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "mceg/reviews"))
import nh_source_review as study


@pytest.fixture(scope="module")
def report():
    return study.review()


def test_balanced_source_energy_and_thermal_decomposition(report):
    assert report["balanced_ATcT_D0_J_mol"] == pytest.approx(327809)
    for row in report["source_samples"]:
        t = row["temperature_K"]
        source = row["table_source"]
        assert abs(source["decomposition_closure_ln_K"]) < 4e-13
        if study.NAME == "N2":
            assert row["same_thermal_ATcT_energy_only_K_ratio"] == pytest.approx(math.exp(144/(study.R*t)),abs=2e-13)
        if row["curated"] is not None:
            assert row["curated_over_nasa"] == pytest.approx(
                math.exp(row["curated"]["ln_K"]-source["nasa_ln_K"]))
    with pytest.raises(ValueError, match="outside"):
        study.curated_properties(1000)


def test_nasa_records_match_original_printed_coefficients_and_frozen_screening():
    nasa = json.loads((study.DATA / "nasa_records.json").read_text())
    frozen = json.loads((ROOT / "tests/data/validation/gas_source_comparison/baseline-001/inputs.json").read_text())
    raw = (study.DATA / "nasa_selected_records.txt").read_text().split("\n\n")
    for name, record in nasa.items():
        lines = next(block.splitlines() for block in raw if block.startswith(name+" "))
        coefficients = [a for interval in record["intervals"] for a in interval["coefficients"]]
        printed = []
        for i in range(len(record["intervals"])):
            printed.extend(float(v.replace("D", "E")) for v in re.findall(
                r"[-+]?\d+\.\d+D[-+]\d+", "\n".join(lines[3+3*i:5+3*i])))
        assert coefficients == printed
        assert coefficients == [a for interval in frozen["nasa9_records"][name]["intervals"] for a in interval["coefficients"]]
        assert record["hf298_J_mol"] == float(lines[1][65:])
        assert record["h298_minus_h0_J_mol"] == float(lines[2][65:])


def test_literal_curated_function_matches_named_production_and_caloric_identities():
    import jax
    from exogibbs.presets import curated_cos
    jax.config.update("jax_enable_x64", True)
    setup = curated_cos.chemsetup(ions=True)
    index = setup.species.index(study.SPECIES)
    composition = {e: float(setup.formula_matrix[i,index]) for i,e in enumerate(setup.elements)
                   if setup.formula_matrix[i,index]}
    assert composition == study.COMPOSITION
    for t in [1500., 2000., 3000., 4000.]:
        actual = study.curated_properties(t)
        assert actual["ln_K"] == pytest.approx(-float(setup.hvector_func(t)[index]),abs=3e-12)
        assert actual["ln_K"] == pytest.approx(-(actual["H_J_mol"]/t-actual["S_J_mol_K"])/study.R,abs=3e-13)
    t, dt = 2500., 0.05
    left, middle, right = [study.curated_properties(x) for x in [t-dt,t,t+dt]]
    assert (right["H_J_mol"]-left["H_J_mol"])/(2*dt) == pytest.approx(middle["Cp_J_mol_K"],abs=2e-7)
    assert (right["ln_K"]-left["ln_K"])/(2*dt) == pytest.approx(middle["H_J_mol"]/(study.R*t*t),abs=2e-10)


def test_saved_grid_scope_and_stoichiometric_share(report):
    f = report["footprint"]
    assert (f["model"],f["cases"],f["species_count"],f["all_converged"]) == ("curated-cos-v1",75,537,True)
    assert f["maximum_abs_h_error"] < 3e-12
    assert report["production_change"] == "none"
    assert report["equilibrium_substitution"] == "not_run"
    with np.load(study.GRID) as g:
        c = g["species"].tolist().index(study.SPECIES)
        e = g["elements"].tolist().index(f["element"])
        i = f["case_index"]
        assert f["atoms_per_molecule"] == study.COMPOSITION[f["element"]]
        assert f["maximum_element_share"] == pytest.approx(
            study.COMPOSITION[f["element"]]*np.exp(g["ln_n"][i,c])/g["budget"][i,e],rel=1e-12,abs=0)


def test_partition_spin_ground_zero_and_caloric_moments(report):
    levels = study.states()
    assert levels[:,1].min() == 0
    gns = levels[:,2]/(2*levels[:,3]+1)
    if study.NAME == "N2":
        assert np.array_equal(gns,np.where(levels[:,3]%2==0,6,3))
        assert len(levels) == 810
    else:
        assert np.array_equal(gns,np.full(len(levels),6.))
        assert len(levels) == 4552
    record = json.loads((study.DATA/"nasa_records.json").read_text())[study.NAME]
    t,dt=2500.,.05
    a,b,c=[study.partition_diagnostic(x,record,report["balanced_ATcT_D0_J_mol"])
           for x in [t-dt,t,t+dt]]
    assert (c["H_minus_H0_J_mol"]-a["H_minus_H0_J_mol"])/(2*dt) == pytest.approx(b["Cp_J_mol_K"],abs=1e-7)
    assert (c["S_J_mol_K"]-a["S_J_mol_K"])/(2*dt) == pytest.approx(b["Cp_J_mol_K"]/t,abs=1e-10)
    for row in report["source_samples"]:
        q=row["partition"]
        assert q["spin_normalized_Q"] == pytest.approx(q["raw_state_Q"]/q["nuclear_spin_divisor"])
        assert q["Cp_J_mol_K"] > 2.5*study.R


def test_source_specific_decision(report):
    rows={r["temperature_K"]:r for r in report["source_samples"]}
    if study.NAME == "N2":
        for t,row in rows.items():
            if t>=1500:
                assert row["curated"]["ln_K"] == pytest.approx(row["adopted_NASA_ATcT_reconstruction_ln_K"],abs=3e-12)
        assert rows[4000]["partition"]["state_Q_over_tabulated_Q"] < .9996
        assert rows[4000]["partition"]["spin_normalized_Q_over_nasa"] < .99
        assert rows[4000]["partition"]["state_minus_nasa_Cp_J_mol_K"] < -2
    else:
        for t,row in rows.items():
            assert abs(row["partition"]["state_Q_over_tabulated_Q"]-1)<1e-6
            if t>=1500:
                assert .97 < row["NASA_thermal_ATcT_K_over_curated"] < 1
        assert rows[1500]["table_source"]["source_over_nasa"] < .22
        assert rows[1500]["table_source"]["energy_aligned_source_over_nasa"] == pytest.approx(1,abs=3e-5)
        assert report["inherited_ATcT_D0_J_mol"] is None


def test_archived_output_and_actual_input_hashes(report):
    expected = json.loads((study.DATA/"comparison.json").read_text())
    def compare(a,b):
        if isinstance(a,dict):
            assert a.keys() == b.keys()
            for k in a: compare(a[k],b[k])
        elif isinstance(a,list):
            assert len(a) == len(b)
            for x,y in zip(a,b):compare(x,y)
        elif isinstance(a,float):assert a == pytest.approx(b,rel=2e-11,abs=2e-10)
        else:assert a == b
    compare(report,expected)
    receipt = json.loads((study.DATA/"receipt.json").read_text())
    for group in ["selected_input_sha256","output_sha256"]:
        for path,expected in receipt[group].items():
            assert study.file_hash(ROOT/path) == expected,path
