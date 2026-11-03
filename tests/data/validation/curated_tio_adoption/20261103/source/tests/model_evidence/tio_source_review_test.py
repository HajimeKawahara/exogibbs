"""Check TiO source zeros, spin normalization and archived titanium accounting."""

from hashlib import sha256
import json
from pathlib import Path
import sys

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "mceg/reviews"))
import tio_source_review as study
from gas_source_comparison import check_equal


@pytest.fixture(scope="module")
def report():
    return study.review()


def test_replay_and_source_balanced_decomposition(report):
    check_equal(report, json.loads((study.DATA / "comparison.json").read_text()))
    baseline = json.loads((study.BASELINE / "comparison.json").read_text())
    for row in report["samples"]:
        t = row["temperature_K"]
        assert row["selected_exact_node_ln_K"]-row["nasa_ln_K"] == pytest.approx(
            row["selected_minus_nasa_energy_ln_K"]
            + row["selected_minus_nasa_molecular_thermal_ln_K"], abs=2e-13)
        for old in baseline["samples"]:
            if old["species_id"] == "O1Ti1" and old["temperature_K"] == t:
                assert row["janaf_native_ln_K"] == pytest.approx(old["ln_Kp"]["janaf"], abs=3e-13)
                assert row["nasa_ln_K"] == pytest.approx(old["ln_Kp"]["nasa"], abs=3e-13)
                if row["curated_ln_K"] is not None:
                    assert abs(row["baseline_curated_residual"]) < 3e-13
        if 1500 <= t <= 4000:
            assert row["curated_ln_K"] == pytest.approx(row["selected_exact_node_ln_K"], abs=3e-13)
        else:
            assert row["curated_ln_K"] is None
    assert report["balanced_native_D0_eV"] == pytest.approx(
        {"janaf": 6.879553455355081, "nasa9": 6.923292251500468}, abs=1e-12)
    for t in (1000, 4001, float("nan")):
        with pytest.raises(ValueError, match="outside"):
            study.curated_ln_k(t)


def test_nuclear_spin_mass_and_ground_zero_are_not_energy_shifts(report):
    source = json.loads((study.DATA / "exomol_nodes.json").read_text())
    assert {k: v["nuclear_spin_weight"] for k, v in source["isotopologues"].items()} == {
        "46Ti-16O": 1, "47Ti-16O": 6, "48Ti-16O": 1, "49Ti-16O": 8, "50Ti-16O": 1}
    states = source["current_states_audit"]
    assert states["states_count"] == 301245 and states["negative_energy_count"] == 0
    for line in states["ground_rows"]:
        fields = line.split()
        assert fields[1:4] == ["0.000000", "3", "1"]
        assert fields[9:11] == ["X(3DELTA)", "0"]
    for row in report["samples"]:
        values = row["normalized_exomol_Q"]
        assert max(values.values())/min(values.values()) < 1.04
        assert abs(row["current_states_over_distributed_pf"]-1) < .00023
        if 1500 <= row["temperature_K"] <= 4000:
            assert .86 < row["exomol_over_janaf_Q"] < .89
            assert 1 < row["exomol_over_nasa_Q"] < 1.01
            assert abs(row["current_states_over_distributed_pf"]-1) < .00004
    assert report["molecular_only_translation_factor_48_over_janaf"] == pytest.approx(1.0014905433994374)
    # The small mass factor cannot explain the 12-13 percent hot partition discrepancy.
    assert report["mass_u"]["janaf"] != report["mass_u"]["nasa"]


def test_new_energy_is_D0_and_combined_diagnostic_is_not_nasa_fitting(report):
    facts = json.loads((study.DATA / "literature.json").read_text())
    new = facts["jain2023"]
    assert new["DR_endothermicity_eV"]+new["IE_TiO_eV"] == pytest.approx(new["D0_TiO_eV"], abs=.00021)
    assert new["D0_TiO_eV"] == 6.824 and new["uncertainty_eV"] == .010
    assert facts["identity"]["atomic_asymptote"] == ["Ti(a3F2)", "O(3P2)"]
    assert facts["janaf"]["a1Delta_assumed_Te_cm-1"] == 1000
    for row in report["samples"]:
        assert row["fixed_mass_combined_K_ratio"] == pytest.approx(
            row["new_energy_only_K_ratio"]*row["fixed_mass_Q_only_K_ratio"], abs=2e-15)
        if row["temperature_K"] in (1500, 3000):
            assert row["fixed_mass_diagnostic_over_nasa"] < row["curated_over_nasa"] < 1
    at1500 = next(row for row in report["samples"] if row["temperature_K"] == 1500)
    assert at1500["fixed_mass_combined_K_ratio"] == pytest.approx(.6244774502872336)
    assert report["curated_update"] == "none" and report["equilibrium_substitution"] == "not_run"


def test_archived_grid_matches_current_hybrid_and_conserves_titanium(report):
    result = report["footprint"]
    assert result["cases"] == 75 and result["species_count"] == 537 and result["all_converged"]
    assert result["case_index"] == 5
    assert result["maximum_titanium_fraction"] == pytest.approx(.9200750632507937)
    assert result["max_current_potential_residual"] < 1e-13
    with np.load(study.GRID) as grid:
        element = grid["elements"].tolist().index("Ti")
        molecule = grid["species"].tolist().index("O1Ti1")
        n = np.exp(grid["ln_n"])
        assert n @ grid["formula_matrix"][element] == pytest.approx(grid["budget"][:, element], rel=1e-7, abs=0)
        i = result["case_index"]
        assert result["mole_fraction"] == pytest.approx(n[i, molecule]/n[i].sum(), rel=2e-12, abs=0)


def test_scoped_input_and_output_receipts():
    receipt = json.loads((study.DATA / "receipt.json").read_text())
    for group in ("selected_input_sha256", "output_sha256"):
        for path, digest in receipt[group].items():
            assert sha256((ROOT / path).read_bytes()).hexdigest() == digest, path
