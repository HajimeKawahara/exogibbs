"""Protect source identity, balanced energy and archived-footprint scope for C5."""

from hashlib import sha256
import json
import math
from pathlib import Path
import shutil
import sys

import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "mceg/reviews"))
import c5_source_review as study
from gas_source_comparison import check_equal


@pytest.fixture(scope="module")
def report():
    return study.review()


def test_official_source_replay_and_receipt(report):
    expected = json.loads((study.DATA / "comparison.json").read_text())
    check_equal(report, expected)
    receipt = json.loads((study.DATA / "receipt.json").read_text())
    for group in ("selected_input_sha256", "output_sha256"):
        for path, digest in receipt[group].items():
            assert sha256((ROOT / path).read_bytes()).hexdigest() == digest, path
    closures = [row["frozen_janaf_closure_ln_K"] for row in report["source_samples"]
                if row["frozen_janaf_closure_ln_K"] is not None]
    assert len(closures) == 4
    assert max(abs(x) for x in closures) < 2e-11
    for row in report["source_samples"]:
        if row["frozen_curated_closure_ln_K"] is not None:
            assert abs(row["frozen_curated_closure_ln_K"]) < 2e-11


def test_energy_and_thermal_components_balance_in_native_R(report):
    for row in report["source_samples"]:
        for family in ("legacy_janaf", "production_source"):
            source = row[family]
            terms = source["source_minus_nasa_ln_K_components"]
            assert abs(source["decomposition_closure_ln_K"]) < 3e-13
            assert math.log(source["source_over_nasa"]) == pytest.approx(sum(terms.values()), abs=3e-13)
            assert source["energy_aligned_source_over_nasa"] == pytest.approx(
                math.exp(terms["molecular_thermal"] + terms["atomic_thermal"]), rel=1e-12)
            expected = (1000*source["native_source_D0_kJ_mol"]/study.R
                        -1000*source["native_nasa_D0_kJ_mol"]/study.NASA_R)/row["temperature_K"]
            assert terms["energy"] == pytest.approx(expected, abs=1e-13)


def test_analytic_fit_caloric_properties_are_derivatives_of_G():
    t, step = 2253., .05
    center = study.curated_properties(t)
    low, high = (study.curated_properties(t+x) for x in (-step, step))
    d_ln_k = (high["ln_K"]-low["ln_K"])/(2*step)
    assert center["H_J_mol"] == pytest.approx(study.R*t*t*d_ln_k, rel=2e-9)
    d_h = (high["H_J_mol"]-low["H_J_mol"])/(2*step)
    assert center["Cp_J_mol_K"] == pytest.approx(d_h, abs=1e-7)
    assert center["H_J_mol"]-t*center["S_J_mol_K"] == pytest.approx(-study.R*t*center["ln_K"], abs=2e-9)
    for outside in (0, 1000, 4000.01, float("nan")):
        with pytest.raises(ValueError, match="outside"):
            study.curated_properties(outside)


def test_tampered_source_is_rejected(tmp_path, monkeypatch):
    data = tmp_path / "packet"
    shutil.copytree(study.DATA, data)
    sources = json.loads((data / "sources.json").read_text())
    path = data / sources["molecular_janaf_table"]
    path.write_text(path.read_text() + "\nchanged source bytes\n")
    monkeypatch.setattr(study, "DATA", data)
    with pytest.raises(ValueError, match="Source hash mismatch"):
        study.review()


def test_current_archived_footprint_counts_atoms_and_is_not_a_substitution(report):
    f = report["footprint"]
    assert (f["cases"], f["species_count"], f["all_converged"]) == (75, 537, True)
    assert f["model"] == "curated-oh-v1"
    assert f["maximum_abs_h_error"] < 1e-12
    assert f["maximum_relative_element_residual"] < 4e-12
    assert f["maximum_element_share"] == pytest.approx(1.200227749279866e-06, rel=2e-12, abs=0)
    assert f["case_index"] == 55
    assert f["new_equilibrium_substitution"] == "not_run"
    assert report["adoption"] == "none"
    assert report["physical_accuracy"] == "unknown"


def test_reversal_and_state_identity_preclude_an_energy_only_replacement(report):
    rows = {row["temperature_K"]: row for row in report["source_samples"]}
    assert rows[1500]["curated_over_nasa"] == pytest.approx(4.3013986415405565)
    assert rows[3000]["curated_over_nasa"] == pytest.approx(.17799925555047977)
    assert rows[4000]["curated_over_nasa"] == pytest.approx(.08268587744000726)
    assert rows[1500]["legacy_janaf"]["energy_aligned_source_over_nasa"] < .019
    assert rows[3000]["legacy_janaf"]["energy_aligned_source_over_nasa"] < .013
    state = report["modern_diagnostic"]
    assert state["legacy_ground_multiplicity"] == 3
    assert state["W4_ground_multiplicity"] == 1
    assert state["legacy_vibrational_degrees"] == 3*5-5
    assert state["W4_linear_singlet_D0_kJ_mol"] == pytest.approx(596.64*4.184)
    assert rows[1500]["legacy_janaf"]["native_source_D0_kJ_mol"] == pytest.approx(2583.528)
    assert rows[1500]["legacy_janaf"]["native_nasa_D0_kJ_mol"] == pytest.approx(2515.988024)
    assert "state-consistent" in report["decision"]

