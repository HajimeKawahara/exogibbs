"""Check N3 atom balancing, source identity and archived nitrogen accounting."""

from hashlib import sha256
import json
from pathlib import Path
import re
import sys

import numpy as np
import pytest


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "mceg/reviews"))
import n3_source_review as study
from gas_source_comparison import check_equal


@pytest.fixture(scope="module")
def report():
    return study.review()


def test_source_replay_closes_independent_baseline_and_decomposition(report):
    check_equal(report, json.loads((study.DATA / "comparison.json").read_text()))
    baseline = json.loads((study.BASELINE / "comparison.json").read_text())
    for row in report["source_samples"]:
        components = row["reaction_delta_G_components_J_mol"]
        assert sum(components.values()) == pytest.approx(row["reaction_delta_G_J_mol"], abs=1e-8)
        assert components["anchor"] == pytest.approx(-21798.596854205534, abs=1e-7)
        assert -31 < row["atomic_delta_G_J_mol"] < -18
        t = row["temperature_K"]
        old = next((r for r in baseline["samples"]
                    if r["species_id"] == "N3" and r["temperature_K"] == t), None)
        if old:
            assert abs(row["baseline_closure_ln_K"]) < 2e-11
            for source in ("janaf", "nasa", "curated"):
                if old["ln_Kp"][source] is not None:
                    assert row[source + "_ln_K"] == pytest.approx(old["ln_Kp"][source], abs=2e-11)
        else:
            assert row["baseline_closure_ln_K"] is None
        if t < 1500:
            assert row["curated_ln_K"] is None
        else:
            assert 2 < row["H298_aligned_janaf_over_nasa"] < 2.7
    assert report["max_abs_curated_minus_janaf_ln_K"] < .00861
    assert len(report["fit_samples"]) == 26
    with pytest.raises(ValueError, match="outside"):
        study.curated_ln_k(1000)


def test_official_coefficient_excerpt_and_neutral_identity():
    literature = json.loads((study.DATA / "literature.json").read_text())
    inputs = study.load_inputs()
    for block in (study.DATA / "nasa9_excerpt.txt").read_text().strip().split("\n\n"):
        lines = block.splitlines()
        name = lines[0].split()[0]
        header = literature["nasa_headers"][name]
        assert float(lines[1].split()[-1]) == header["hf298_J_mol"]
        for i, interval in enumerate(inputs["nasa9_records"][name]["intervals"]):
            offset = 2 + 3*i
            assert float(lines[offset].split()[-1]) == header["H298_minus_H0_J_mol"]
            values = [float(v.replace("D", "E")) for v in re.findall(
                r"[+-]?\d+\.\d+D[+-]\d+", " ".join(lines[offset+1:offset+3]))]
            assert values == interval["coefficients"]
    assert literature["identity"]["state"] == "X2Pi_g"
    assert literature["identity"]["modern_neutral_CAS"] != literature["identity"]["janaf_index_CAS"]
    assert literature["janaf_print"]["estimated_vibrational_wavenumbers_cm-1"] == [1800, 500, 1200]
    assert literature["janaf_print"]["vibrational_degeneracies"] == [1, 2, 1]


def test_atomization_energies_use_same_source_atoms_and_zero_gauge(report):
    values = report["balanced_D0_J_mol"]
    assert values == pytest.approx(
        {"janaf": 995495., "nasa_headers": 973018.622, "atct_1_172": 959369.}, abs=1e-6)
    assert values["atct_1_172"] == pytest.approx((3*470.583-452.38)*1000)
    assert values["janaf"]-values["atct_1_172"] == pytest.approx(36126.)
    # A molecular-only Hf(0) difference is not the atom-balanced correction.
    assert values["janaf"]-values["atct_1_172"] != pytest.approx((452.38-416.969)*1000)
    assert values["nasa_headers"]-values["atct_1_172"] == pytest.approx(13649.622)


def test_existing_grid_conserves_nitrogen_and_matches_current_fit(report):
    result = report["footprint"]
    point = result["maximum_nitrogen_share"]
    assert result["cases"] == 75 and result["species_count"] == 537
    assert result["all_cases_converged"]
    assert point["case_index"] == 24
    assert point["nitrogen_fraction"] == pytest.approx(6.698426863209913e-10, rel=1e-12, abs=0)
    with np.load(study.GRID) as grid:
        element = grid["elements"].tolist().index("N")
        molecule = grid["species"].tolist().index("N3")
        n = np.exp(grid["ln_n"])
        assert n @ grid["formula_matrix"][element] == pytest.approx(grid["budget"][:, element], rel=1e-7)
        i = point["case_index"]
        assert point["mole_fraction"] == pytest.approx(n[i, molecule]/n[i].sum(), rel=2e-12, abs=0)
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
