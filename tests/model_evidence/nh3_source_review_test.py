"""Verify balanced NH3 anchors and reject misleading partition substitutions."""

from hashlib import sha256
import json
import math
from pathlib import Path
import shutil
import sys

import numpy as np
import pytest


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "mceg/reviews"))
import nh3_source_review as study
from gas_source_comparison import check_equal


@pytest.fixture(scope="module")
def report():
    return study.review()


def test_balanced_energy_and_archived_reaction_match(report):
    check_equal(report, json.loads((study.DATA / "comparison.json").read_text()))
    assert report["balanced_D0_J_mol"] == pytest.approx({
        "atct": 1157249, "nasa9": 1157869.893, "janaf": 1157835}, abs=1e-6)
    assert report["adopted_minus_nasa_D0_over_k_K"] == pytest.approx(-73.8826647262)
    baseline = json.loads((ROOT / "tests/data/validation/gas_source_comparison/baseline-001/comparison.json").read_text())
    for row in report["source_samples"]:
        t = row["temperature_K"]
        if t >= 1500:
            assert row["curated_ln_K"] == pytest.approx(row["reconstructed_hybrid_ln_K"], abs=3e-12)
            assert t*(row["curated_ln_K"]-row["nasa_ln_K"]) == pytest.approx(
                report["adopted_minus_nasa_D0_over_k_K"], abs=2e-9)
        old = next((r for r in baseline["samples"] if r["species_id"] == "H3N1"
                    and r["temperature_K"] == t), None)
        if old:
            assert row["nasa_ln_K"] == pytest.approx(old["ln_Kp"]["nasa"], abs=3e-11)
            assert row["janaf"]["native_ln_K"] == pytest.approx(old["ln_Kp"]["janaf"], abs=3e-11)
    with pytest.raises(ValueError, match="outside"):
        study.curated_ln_k(1000)


def test_inversion_spin_normalization_and_thermal_identity(report):
    ground = report["lowest_allowed_state_check"]
    assert ground["single_state_Q_at_1K"] == pytest.approx(ground["distributed_Q_at_1K"], abs=5e-5)
    assert ground["reference_shift_energy_J_mol"] == pytest.approx(9.4908606887)
    literal = (study.DATA / "CoYuTe.def").read_text().splitlines()
    assert [int(literal[i].split()[0]) for i in (16, 19, 22, 25, 28, 31)] == [0, 12, 6, 0, 12, 6]
    for row in report["source_samples"]:
        t = row["temperature_K"]
        for source in row["modern"].values():
            ratio = source["Q_including_nuclear_spin"]/(24*row["Q_nasa_natural_mass"])
            assert source["same_D0_atoms_natural_mass_K_proxy"] == pytest.approx(ratio)
            assert math.exp(-source["delta_thermal_G_J_mol"]/(study.major.R*t)) == pytest.approx(ratio)
            assert source["isotope_mass_K_proxy"]/ratio == pytest.approx(report["isotope_to_natural_translation_factor"])
        if row["janaf"]:
            j = row["janaf"]
            delta_g = j["delta_H_increment_J_mol"]-t*j["delta_S_J_mol_K"]
            assert math.exp(-delta_g/(study.major.R*t)) == pytest.approx(j["same_D0_atoms_K_ratio"], abs=2e-14)


def test_modern_tables_do_not_establish_full_domain_replacement(report):
    rows = {r["temperature_K"]: r for r in report["source_samples"]}
    assert rows[1500]["modern"]["TIPS2025"]["Q_over_24_over_nasa"] == pytest.approx(.994369418324241)
    assert rows[4000]["modern"]["TIPS2025"]["Q_over_24_over_nasa"] == pytest.approx(.915291045509889)
    assert rows[500]["modern"]["TIPS2025"]["method"] == "direct_sum_distribution"
    assert all(r["modern"]["TIPS2025"]["method"] == "product_approximation" for t, r in rows.items() if t >= 1000)
    assert rows[1500]["modern"]["CoYuTe"]["within_published_CoYuTe_opacity_range"] is True
    assert rows[2000]["modern"]["CoYuTe"]["within_published_CoYuTe_opacity_range"] is False
    assert "CoYuTe" not in rows[2500]["modern"]
    mismatch = report["printed_fit_audit"]
    assert mismatch["printed_fit_Q"] == pytest.approx(1779.076858129, abs=1e-6)
    assert mismatch["relative_mismatch"] > .03
    assert mismatch["coefficient_last_digit_rounding_relative_bound"] < .0021
    assert mismatch["accepted_as_direct_sum_data"] is False
    assert report["production_change"] == "none"
    assert report["equilibrium_substitution"] == "not_run"


def test_saved_current_grid_conserves_nitrogen_and_hydrogen(report):
    grid = report["footprint"]
    assert (grid["model"], grid["cases"], grid["species"]) == ("curated-h2o-v1", 75, 537)
    assert grid["all_converged"] and grid["maximum_abs_h_error"] < 1e-12
    with np.load(study.GRID) as saved:
        n = np.exp(saved["ln_n"])
        column = saved["species"].tolist().index("H3N1")
        for element, count in (("N", 1), ("H", 3)):
            e = saved["elements"].tolist().index(element)
            p = grid["maximum_element_shares"][element]
            assert n @ saved["formula_matrix"][e] == pytest.approx(saved["budget"][:, e], rel=1e-10, abs=0)
            assert p["case_index"] == 4
            assert p["element_fraction"] == pytest.approx(count*n[4, column]/saved["budget"][4, e])
        assert grid["maximum_element_shares"]["N"]["element_fraction"] == pytest.approx(.26886641291557345)


def test_corrupted_partition_source_is_rejected(tmp_path, monkeypatch):
    shutil.copytree(study.DATA, tmp_path / "packet")
    monkeypatch.setattr(study, "DATA", tmp_path / "packet")
    path = study.DATA / "hapi_tips2025_nh3.txt"
    values = np.loadtxt(path)
    values[values[:, 0] == 1500, 1] *= 24
    np.savetxt(path, values)
    with pytest.raises(ValueError, match="Source hash mismatch"):
        study.review()


def test_literal_hapi_extraction_preserves_actual_nodes(tmp_path):
    source = tmp_path / "hapi.py"
    source.write_text("\n".join(
        f"TIPS_{year}_ISOT[7] = float64([1., 1500., 4000.])\n"
        f"M = 11\nI = 1\nTIPS_{year}_ISOT_HASH[(M,I)] = TIPS_{year}_ISOT[7]\n"
        f"TIPS_{year}_ISOQ_HASH[(M,I)] = float64([3.8, 61224., 4333347.])\n"
        for year in (2021, 2025)))
    study.extract_hapi(source, tmp_path / "output")
    values = np.loadtxt(tmp_path / "output/hapi_tips2025_nh3.txt")
    assert values.tolist() == [[1, 3.8], [1500, 61224], [4000, 4333347]]


def test_receipt_and_original_source_hashes_are_pinned():
    receipt = json.loads((study.DATA / "receipt.json").read_text())
    for group in ("selected_input_sha256", "output_sha256"):
        for path, digest in receipt[group].items():
            assert sha256((ROOT / path).read_bytes()).hexdigest() == digest, path
    sources = json.loads((study.DATA / "sources.json").read_text())
    for source in sources["downloads"]:
        if source["file"]:
            assert sha256((study.DATA / source["file"]).read_bytes()).hexdigest() == source["archive_sha256"]
