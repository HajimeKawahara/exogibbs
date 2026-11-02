"""Check methane source identities, spin normalization and unresolved coverage."""

import json
from pathlib import Path
import sys

import numpy as np
import pytest


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "mceg/reviews"))
import ch4_source_review as study
from gas_source_comparison import check_equal


@pytest.fixture(scope="module")
def report():
    return study.review()


def test_source_decomposition_replays_frozen_comparison(report):
    expected = json.loads((study.DATA / "comparison.json").read_text())
    check_equal(report, expected)
    for row in report["source_samples"]:
        if row["baseline_closure_ln_K"] is not None:
            assert abs(row["baseline_closure_ln_K"]) < 3e-11
        assert abs(row["atomic_delta_G_J_mol"]) < 50
    high = next(row for row in report["source_samples"] if row["temperature_K"] == 4000)
    assert high["H298_aligned_janaf_over_nasa"] == pytest.approx(0.58658503, abs=1e-8)
    assert high["properties"]["CH4"]["nasa_Cp_J_mol_K"] > 124
    assert high["properties"]["CH4"]["janaf_Cp_J_mol_K"] < 105


def test_balanced_ground_energy_uses_source_reference_gauges(report):
    energy = report["balanced_H0_reaction_J_mol"]
    assert energy == pytest.approx({"janaf": -1642239, "nasa": -1641965.907,
                                    "atct": -1642064}, abs=1e-6)
    # Molecular formation enthalpies alone would give a different correction.
    assert energy["atct"]-energy["janaf"] == pytest.approx(175)
    assert energy["atct"]-energy["janaf"] != pytest.approx((-66.547+66.911)*1000)


def test_curated_uses_existing_energy_and_atomic_thermal_background(report):
    expected = {1500: 0.9343938354017822, 3000: 0.7330470798085987}
    for row in report["source_samples"]:
        t = row["temperature_K"]
        if t in expected:
            assert row["curated_over_raw_nasa"] == pytest.approx(expected[t], rel=2e-12)
        if not 1500 <= t <= 4000:
            assert row["curated_over_raw_nasa"] is None
    with pytest.raises(ValueError, match="outside"):
        study.curated_ln_k(1000, {"knots_K": [1500, 2000, 3000, 4000]})


def test_spin_conversion_is_sixteen_not_symmetry_number(report):
    data = json.loads((study.DATA / "literature.json").read_text())
    assert data["identity"]["nuclear_spin_product"] == 2**4
    assert data["identity"]["janaf_symmetry_number"] == 12
    row = next(row for row in report["partition_samples"] if row["temperature_K"] == 300)
    # The independent spectroscopic and NASA values meet at room temperature
    # only after the full proton spin product has been removed.
    assert row["Q_over_16_over_nasa"]["MM"] == pytest.approx(1, abs=4e-5)
    assert row["spectroscopic_Q_including_nuclear_spin"]["MM"] == 602.8526


def test_partition_file_ranges_are_not_accuracy_ranges(report):
    definition = (study.DATA / "exomol_mm.def").read_text()
    assert "2000.00" in next(line for line in definition.splitlines()
                             if "Maximum temperature" in line)
    for row in report["partition_samples"]:
        if row["temperature_K"] > 2500:
            assert row["spectroscopic_Q_including_nuclear_spin"]["TIPS2021"] is None
            assert row["spectroscopic_Q_including_nuclear_spin"]["TIPS2025"] is None
            assert row["TIPS2025_over_2021"] is None
        assert row["MM_within_published_completeness_estimate"] == (row["temperature_K"] <= 2000)


def test_version_change_and_source_disagreement_are_preserved(report):
    rows = {row["temperature_K"]: row for row in report["partition_samples"]}
    assert rows[2000]["TIPS2025_over_2021"] == pytest.approx(0.9358962015710216)
    assert rows[2500]["TIPS2025_over_2021"] == pytest.approx(0.7871430256118235)
    assert rows[1500]["Q_over_16_over_janaf"]["MM"] > 1.05
    assert rows[2000]["Q_over_16_over_nasa"]["MM"] < 0.92


def test_archived_grid_footprint_is_a_conserved_carbon_share(report):
    row = report["footprint"]
    assert row["cases"] == 75 and row["all_cases_converged"]
    assert row["carbon_fraction"] == pytest.approx(0.9719754423216526)
    assert (row["temperature_K"], row["pressure_bar"], row["C_over_O"]) == (1500, 100, 1.5)
    with np.load(study.GRID) as grid:
        i = row["case_index"]
        carbon = grid["elements"].tolist().index("C")
        assert np.sum(np.exp(grid["ln_n"][i])*grid["formula_matrix"][carbon]) == pytest.approx(
            grid["budget"][i, carbon], rel=1e-10)
    assert row["new_equilibrium_substitution"] == "not_run"


def test_modified_partition_input_is_rejected(tmp_path, monkeypatch):
    sources = json.loads((study.DATA / "sources.json").read_text())
    first = sources["downloads"][0]
    (tmp_path / "sources.json").write_text(json.dumps(sources))
    (tmp_path / first["file"]).write_bytes(b"different source snapshot")
    monkeypatch.setattr(study, "DATA", tmp_path)
    with pytest.raises(ValueError, match="Source hash mismatch"):
        study.review()
