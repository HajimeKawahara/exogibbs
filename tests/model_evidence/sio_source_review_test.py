"""Check SiO balanced energy, independently tabulated thermal data and Si budgets."""

from hashlib import sha256
import json
from pathlib import Path
import re
import sys

import numpy as np
import pytest


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "mceg/reviews"))
import sio_source_review as study
from gas_source_comparison import check_equal


@pytest.fixture(scope="module")
def report():
    return study.review()


def test_source_replay_and_balanced_energy_reproduce_adoption(report):
    check_equal(report, json.loads((study.DATA/"comparison.json").read_text()))
    baseline = json.loads((study.BASELINE/"comparison.json").read_text())
    assert report["balanced_D0_J_mol"] == pytest.approx(
        {"nasa9": 792456.865, "janaf": 794029., "atct": 795554.}, abs=1e-6)
    assert report["balanced_D0_J_mol"]["atct"] == pytest.approx((450.34+246.844+98.37)*1000)
    assert report["adopted_minus_native_nasa_D0_over_k_K"] == pytest.approx(373.0428665669757)
    for row in report["source_samples"]:
        t = row["temperature_K"]
        old = next((r for r in baseline["samples"]
                    if r["species_id"] == "O1Si1" and r["temperature_K"] == t), None)
        if old:
            assert abs(row["baseline_closure_ln_K"]) < 2e-11
            for source in ("nasa", "janaf", "curated"):
                if row[source+"_ln_K"] is not None:
                    assert row[source+"_ln_K"] == pytest.approx(old["ln_Kp"][source], abs=2e-11)
        if t >= 1500:
            assert row["curated_ln_K"] == pytest.approx(row["reconstructed_hybrid_ln_K"], abs=3e-12)
            assert t*(row["curated_ln_K"]-row["nasa_ln_K"]) == pytest.approx(
                report["adopted_minus_native_nasa_D0_over_k_K"], abs=1e-9)
            assert .9945 < row["same_D0_same_atoms_janaf_over_nasa"] < .9972
        else:
            assert row["curated_ln_K"] is None
    with pytest.raises(ValueError, match="outside"):
        study.curated_ln_k(1000)


def test_official_tables_and_coefficients_match_existing_adoption_inputs():
    reference = json.loads(study.major.REFERENCE_PATH.read_text())
    for name, filename in (("SiO", "O-012.txt"), ("Si", "Si-005.txt"), ("O", "O-001.txt")):
        fresh = study.read_table(study.DATA/filename)
        for t, values in reference["janaf"][name]["rows"].items():
            for actual, expected in zip(fresh[float(t)], values):
                if expected is not None:
                    assert actual == expected
    for block in (study.DATA/"nasa9_excerpt.txt").read_text().strip().split("\n\n"):
        lines = block.splitlines()
        name = lines[0].split()[0]
        record = reference["nasa9_records"][name]
        assert float(lines[1].split()[-1]) == record["hf298_J_mol"]
        for i, interval in enumerate(record["intervals"]):
            offset = 2+3*i
            assert float(lines[offset].split()[-1]) == record["h298_minus_h0_J_mol"]
            numbers = [float(v.replace("D", "E")) for v in re.findall(
                r"[+-]?\d+\.\d+D[+-]\d+", " ".join(lines[offset+1:offset+3]))]
            assert numbers == interval["coefficients"]


def test_state_moments_close_published_partition_and_heat_capacity(report):
    literature = json.loads((study.DATA/"literature.json").read_text())
    assert literature["exomol"]["nuclear_spin_factor"] == 1
    assert literature["exomol"]["states"] == 174250
    assert literature["exomol"]["fixed_De_eV"] == 8.26
    moments = json.loads((study.DATA/"partition_samples.json").read_text())
    for row, moment in zip(report["source_samples"], moments):
        # Independent published PF/Cp values check the full-state moment summary.
        assert row["Q_states_over_published"] == pytest.approx(1, abs=8e-7)
        assert abs(row["Cp_states_J_mol_K"]-row["Cp_exomol_published_J_mol_K"]) < .0025
        assert row["Cp_states_J_mol_K"] > 3.5*study.major.R
        assert moment["variance_energy_K2"] > 0
        assert .9979 < row["exomol_over_nasa_internal_Q"] < .9987
        if row["temperature_K"] >= 1500:
            assert abs(row["Cp_states_J_mol_K"]-row["Cp_nasa_J_mol_K"]) < .064
    # Changing only translational mass must not silently change energy or spin weights.
    q1 = study.internal_q(1500, -30., 44.0849)
    q2 = study.internal_q(1500, -30., 43.97184115)
    assert q1/q2 == pytest.approx((43.97184115/44.0849)**1.5)


def test_existing_grid_conserves_silicon_and_matches_actual_hybrid(report):
    result = report["footprint"]
    point = result["maximum_silicon_share"]
    assert result["cases"] == 75 and result["species_count"] == 537
    assert result["all_cases_converged"]
    assert point["case_index"] == 5
    assert point["silicon_fraction"] == pytest.approx(.9999233282073725, rel=1e-12)
    with np.load(study.GRID) as grid:
        element = grid["elements"].tolist().index("Si")
        column = grid["species"].tolist().index("O1Si1")
        n = np.exp(grid["ln_n"])
        assert n @ grid["formula_matrix"][element] == pytest.approx(
            grid["budget"][:, element], rel=1e-7, abs=0)
        i = point["case_index"]
        assert point["mole_fraction"] == pytest.approx(n[i, column]/n[i].sum(), rel=2e-12, abs=0)
        for t, h in zip(grid["temperature_K"], grid["h"][:, column]):
            assert -h == pytest.approx(study.curated_ln_k(t), abs=3e-12)
    assert report["equilibrium_substitution"] == "not_run"
    assert report["production_change"] == "none"


def test_evidence_packet_and_selected_inputs_are_pinned():
    receipt = json.loads((study.DATA/"receipt.json").read_text())
    for group in ("selected_input_sha256", "output_sha256"):
        for path, digest in receipt[group].items():
            assert sha256((ROOT/path).read_bytes()).hexdigest() == digest, path
    sources = json.loads((study.DATA/"sources.json").read_text())
    for source in sources["downloads"]:
        if source["archived"]:
            assert sha256((study.DATA/source["file"]).read_bytes()).hexdigest() == source["sha256"]
