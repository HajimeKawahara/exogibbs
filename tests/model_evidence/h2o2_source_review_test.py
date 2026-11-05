"""Check H2O2 source normalization, archived scope and independent identities."""

import json
import math
from pathlib import Path
import re
import sys

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "mceg/reviews"))
import h2o2_source_review as study


@pytest.fixture(scope="module")
def report():
    return study.review()


def test_balanced_source_energy_and_thermal_decomposition(report):
    assert report["balanced_ATcT_D0_J_mol"] == pytest.approx(1055154)
    assert report["energy_only_delta_D0_J_mol"] == pytest.approx(-388, abs=1e-8)
    for row in report["source_samples"]:
        t = row["temperature_K"]
        source = row["table_source"]
        assert abs(source["decomposition_closure_ln_K"]) < 4e-13
        assert row["same_thermal_ATcT_energy_only_K_ratio"] == pytest.approx(
            math.exp(-388/(study.R*t)), abs=2e-13)
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


def test_source_specific_decision(report):
    rows = {r["temperature_K"]:r for r in report["source_samples"]}
    assert rows[1500]["partition"]["spin_normalized_Q"] == pytest.approx(rows[1500]["partition"]["raw_Q"]/4)
    assert rows[2000]["partition"]["isotope_translation_K_proxy"] == pytest.approx(0.8898057235113578)
    assert rows[3000]["partition"] is None
    assert rows[2000]["legacy_janaf"] is None
    table = json.loads((study.DATA/"dorofeeva2003_nodes.json").read_text())["rows"]
    for ts,node in table.items():
        t = float(ts)
        if t:
            assert node[2] == pytest.approx(node[1]-1000*node[3]/t,abs=0.0015)


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
