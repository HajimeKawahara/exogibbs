"""Guard PO2 reference balancing, source transcriptions and saved P accounting."""

from hashlib import sha256
import json
from pathlib import Path
import re
import sys

import numpy as np
import pytest


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "mceg/reviews"))
import po2_source_review as study
from gas_source_comparison import check_equal


@pytest.fixture(scope="module")
def report():
    return study.review()


def test_fresh_tables_reproduce_baseline_and_decompose_reaction(report):
    check_equal(report, json.loads((study.DATA / "comparison.json").read_text()))
    baseline = json.loads((study.BASELINE / "comparison.json").read_text())
    for row in report["source_samples"]:
        t = row["temperature_K"]
        components = row["reaction_delta_G_components_J_mol"]
        assert sum(components.values()) == pytest.approx(row["reaction_delta_G_J_mol"], abs=1e-8)
        assert components["anchor"] == pytest.approx(-32889.00548370392, abs=1e-7)
        assert 90 < row["atomic_delta_G_J_mol"] < 120
        assert row["molecular_delta_G_J_mol"] + row["atomic_delta_G_J_mol"] == pytest.approx(
            row["reaction_delta_G_J_mol"], abs=1e-8)
        old = next((v for v in baseline["samples"]
                    if v["species_id"] == "O2P1" and v["temperature_K"] == t), None)
        if old:
            assert abs(row["baseline_closure_ln_K"]) < 2e-11
            for source in ("janaf", "nasa", "curated"):
                if old["ln_Kp"][source] is not None:
                    assert row[source + "_ln_K"] == pytest.approx(old["ln_Kp"][source], abs=2e-11)
        else:
            assert row["baseline_closure_ln_K"] is None
        if t < 1500:
            assert row["curated_ln_K"] is None
    assert report["max_abs_curated_minus_janaf_ln_K"] < .011
    assert len(report["fit_samples"]) == 26
    hot = next(v for v in report["source_samples"] if v["temperature_K"] == 3000)
    assert hot["curated_over_nasa"] == pytest.approx(5.118671478929384)
    assert hot["H298_aligned_janaf_over_nasa"] == pytest.approx(1.3785174732826742)
    with pytest.raises(ValueError, match="outside"):
        study.curated_ln_k(1000)


def test_printed_nodes_and_nasa_coefficients_match_frozen_inputs():
    literature = json.loads((study.DATA / "literature.json").read_text())
    table = study.read_table(study.DATA / "O-032.txt")
    for t, node in literature["janaf_print"]["printed_nodes"].items():
        assert table[float(t)][1] == node["S_J_mol_K"]
        assert table[float(t)][3] == node["H_minus_H298_kJ_mol"]
    # Independent NASA report transcription: neutral species, not adjacent anions.
    baseline = study.load_inputs()
    for block in (study.DATA / "nasa9_excerpt.txt").read_text().strip().split("\n\n"):
        lines = block.splitlines()
        name = lines[0].split()[0]
        record = baseline["nasa9_records"][name]
        header = literature["nasa_headers"][name]
        assert float(lines[1].split()[-1]) == header["hf298_J_mol"]
        for i, interval in enumerate(record["intervals"]):
            offset = 2 + 3*i
            assert list(map(float, lines[offset].split()[:1])) == interval["bounds_K"][:1]
            assert float(lines[offset].split()[-1]) == header["H298_minus_H0_J_mol"]
            values = [float(v.replace("D", "E")) for v in re.findall(
                r"[+-]?\d+\.\d+D[+-]\d+", " ".join(lines[offset+1:offset+3]))]
            assert values == interval["coefficients"]
        h = study.nasa_properties(record, 298.15, 8.31451)[0]
        assert h == pytest.approx(header["hf298_J_mol"], abs=.02)


def test_ground_energy_balances_atoms_in_each_reference_gauge(report):
    assert report["balanced_H0_reaction_J_mol"] == pytest.approx(
        {"janaf": -1120275, "nasa_headers": -1087241.875}, abs=1e-6)
    delta = report["balanced_H0_reaction_J_mol"]["janaf"] - report["balanced_H0_reaction_J_mol"]["nasa_headers"]
    assert delta == pytest.approx(-33033.125)
    # Neither a molecular-only Hf(0) shift nor a quoted 1962 estimate is this reaction.
    assert delta != pytest.approx((-311.142 + 278.00)*1000)
    assert -report["balanced_H0_reaction_J_mol"]["janaf"] != pytest.approx(270*4184)


def test_archived_footprint_conserves_phosphorus_and_uses_this_fit(report):
    result = report["footprint"]
    point = result["maximum_phosphorus_share"]
    assert result["cases"] == 75 and result["species_count"] == 537
    assert result["all_cases_converged"]
    assert point["case_index"] == 0
    assert point["phosphorus_fraction"] == pytest.approx(.0007433127757996308)
    with np.load(study.GRID) as grid:
        p = grid["elements"].tolist().index("P")
        molecule = grid["species"].tolist().index("O2P1")
        n = np.exp(grid["ln_n"][0])
        assert grid["formula_matrix"][p] @ n == pytest.approx(grid["budget"][0, p], rel=1e-7)
        assert point["mole_fraction"] == pytest.approx(n[molecule]/n.sum(), rel=2e-12)
        for t, h in zip(grid["temperature_K"], grid["h"][:, molecule]):
            assert -h == pytest.approx(study.curated_ln_k(t), abs=3e-12)
    assert report["equilibrium_substitution"] == "not_run"
    assert report["production_change"] == "none"


def test_selected_inputs_and_outputs_are_pinned():
    receipt = json.loads((study.DATA / "receipt.json").read_text())
    for group in ("selected_input_sha256", "output_sha256"):
        for path, digest in receipt[group].items():
            assert sha256((ROOT / path).read_bytes()).hexdigest() == digest, path
    sources = json.loads((study.DATA / "sources.json").read_text())
    for source in sources["downloads"]:
        if source["archived"]:
            assert sha256((study.DATA/source["file"]).read_bytes()).hexdigest() == source["sha256"]
