"""Check balanced CO energy, original state conventions and thermal comparisons."""

from copy import deepcopy
from hashlib import sha256
import json
import math
from pathlib import Path
import sys

import numpy as np
import pytest


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "mceg/reviews"))
import co_source_review as study
from gas_source_comparison import check_equal


def check_co_replay(actual: dict, expected: dict) -> None:
    """Allow cancellation roundoff only in differences of large enthalpies."""
    actual, expected = deepcopy(actual), deepcopy(expected)
    for measured, saved in zip(actual["source_samples"], expected["source_samples"]):
        for dataset in study.DATASETS:
            value = measured["modern"][dataset].pop("delta_H_J_mol")
            reference = saved["modern"][dataset].pop("delta_H_J_mol")
            # Subtracting O(1e5 J/mol) enthalpies varies by O(1e-11 J/mol)
            # across platforms. This absolute tolerance is still far below
            # the smallest physical delta; it does not alter other fields.
            assert value == pytest.approx(reference, abs=1e-8, rel=0), (
                f"delta_H_J_mol at {saved['temperature_K']} K, {dataset}")
    check_equal(actual, expected)


def test_co_replay_tolerance_is_limited_to_enthalpy_roundoff():
    expected = json.loads((study.DATA / "comparison.json").read_text())
    actual = deepcopy(expected)
    row = actual["source_samples"][5]["modern"]["4thplus"]
    # Exact value observed on CI at 3000 K, about 3e-11 J/mol from the archive.
    row["delta_H_J_mol"] = 3.395813737432707
    check_co_replay(actual, expected)
    row["delta_H_J_mol"] += 1e-6
    with pytest.raises(AssertionError, match="delta_H_J_mol"):
        check_co_replay(actual, expected)
    row["delta_H_J_mol"] = expected["source_samples"][5]["modern"]["4thplus"]["delta_H_J_mol"]
    row["Q_over_nasa_internal"] += 1e-8
    with pytest.raises(ValueError, match="Q_over_nasa_internal"):
        check_co_replay(actual, expected)


@pytest.fixture(scope="module")
def report():
    return study.review()


def test_balanced_atct_energy_and_hybrid_reproduce_the_original_screen(report):
    check_co_replay(report, json.loads((study.DATA / "comparison.json").read_text()))
    assert report["balanced_D0_J_mol"] == pytest.approx({
        "atct": (711.381+246.844+113.794)*1000,
        "nasa9": 1071800.005, "janaf": 1071780.0}, abs=1e-6)
    baseline = json.loads((ROOT / "tests/data/validation/gas_source_comparison/baseline-001/comparison.json").read_text())
    for row in report["source_samples"]:
        t = row["temperature_K"]
        if t >= 1500:
            assert row["curated_ln_K"] == pytest.approx(row["reconstructed_hybrid_ln_K"], abs=2e-12)
            assert t*(row["curated_ln_K"]-row["nasa_ln_K"]) == pytest.approx(
                report["adopted_minus_nasa_D0_over_k_K"], abs=1e-9)
        old = next((r for r in baseline["samples"]
                    if r["species_id"] == "C1O1" and r["temperature_K"] == t), None)
        if old:
            assert row["nasa_ln_K"] == pytest.approx(old["ln_Kp"]["nasa"], abs=2e-11)
            assert row["janaf"]["native_ln_K"] == pytest.approx(old["ln_Kp"]["janaf"], abs=2e-11)
    with pytest.raises(ValueError, match="outside"):
        study.curated_ln_k(1000)


def test_actual_state_weights_and_zero_override_inconsistent_metadata():
    inventory = json.loads((study.DATA / "state_inventory.json").read_text())
    with np.load(study.DATA / "states.npz") as states:
        for name, count in (("4thplus", 11264), ("Li2015", 6383)):
            rows = states[name]
            metadata = json.loads((study.DATA / (name + ".def.json")).read_text())
            assert len(rows) == count == inventory["datasets"][name]["states"]
            assert np.all(rows[:, 1] == 2*rows[:, 2]+1)
            assert rows[0].tolist() == [0, 1, 0, 0]
            assert metadata["irreducible_representations"] == {"Sigma+": 2, "Sigma-": 2}
            assert study.moments(rows, 1)["Q"] == pytest.approx(1.0119, abs=5e-5)
        assert states["4thplus"][:, 0].max() == 98839.941885
        assert json.loads((study.DATA / "4thplus.def.json").read_text())["dataset"]["states"]["max_energy"] == 98253.97
    printed = np.array(inventory["published_pf_rows"]["4thplus"])
    assert 1500 not in printed[:, 0] and 1500.2 in printed[:, 0]
    assert 4000.4 in printed[:, 0] and printed[-1, 0] == 9001


def test_state_thermodynamics_agree_with_independent_published_tables(report):
    for row in report["source_samples"]:
        t = row["temperature_K"]
        modern = row["modern"]["4thplus"]
        older = row["modern"]["Li2015"]
        # Separate rounded, printed-temperature PF distributions from direct moments.
        assert abs(modern["states_over_published_Q"]-1) < 1e-4
        assert abs(older["states_over_published_Q"]-1) < 1.2e-5
        assert abs(modern["equivalent_T_offset_K"]) < .05
        assert abs(older["states_over_published_Q"]-1) > older["printed_Q_rounding_relative_bound"]
        assert abs(modern["Q"]/older["Q"]-1) < 1.1e-6
        if row["janaf"]:
            assert abs(modern["Cp_J_mol_K"]-row["janaf"]["Cp_J_mol_K"]) < .014
            assert abs(row["states_over_BC2016_Q"]-1) < .00017
        if t >= 1500:
            assert 1.00014 < modern["Q_over_nasa_internal"] < 1.00027
            assert abs(modern["delta_Cp_J_mol_K"]) < .0045
            assert modern["A1Pi_fraction"] < 2.6e-10
            assert modern["above_adopted_D0_Q_fraction"] < 3.5e-13
        # Independent thermodynamic identity links moment calorics to inferred Q.
        inferred_ln_ratio = -(modern["delta_H_J_mol"]-t*modern["delta_S_J_mol_K"])/(study.major.R*t)
        assert inferred_ln_ratio == pytest.approx(math.log(modern["Q_over_nasa_internal"]), abs=2e-14)
        assert modern["same_D0_atoms_isotope_mass_K_proxy"] == pytest.approx(
            modern["Q_over_nasa_internal"]*(study.ISOTOPE_MASS/28.0101)**1.5)


def test_current_saved_grid_conserves_carbon_and_oxygen(report):
    result = report["footprint"]
    assert result["model"] == "curated-tio-v1"
    assert (result["cases"], result["species"], result["all_converged"]) == (75, 537, True)
    assert result["maximum_abs_h_error"] < 2e-12
    with np.load(study.GRID) as grid:
        n = np.exp(grid["ln_n"])
        column = grid["species"].tolist().index("C1O1")
        for name, case in (("C", 10), ("O", 55)):
            e = grid["elements"].tolist().index(name)
            assert n @ grid["formula_matrix"][e] == pytest.approx(grid["budget"][:, e], rel=1e-10, abs=0)
            point = result["maximum_element_shares"][name]
            assert point["case_index"] == case
            assert point["element_fraction"] > .99999
            assert point["mole_fraction"] == pytest.approx(n[case, column]/n[case].sum(), abs=1e-15)
    assert report["production_change"] == "none"
    assert report["equilibrium_substitution"] == "not_run"


def test_original_downloads_and_evidence_inputs_are_pinned():
    receipt = json.loads((study.DATA / "receipt.json").read_text())
    for group in ("selected_input_sha256", "output_sha256"):
        for path, digest in receipt[group].items():
            assert sha256((ROOT / path).read_bytes()).hexdigest() == digest, path
    sources = json.loads((study.DATA / "sources.json").read_text())
    for source in sources["downloads"]:
        if source["archived"]:
            assert sha256((study.DATA / source["file"]).read_bytes()).hexdigest() == source["archive_sha256"]
