"""Guard neutral hydroxide identity, thermal decomposition and sodium accounting."""

from hashlib import sha256
import json
from pathlib import Path
import runpy
import sys

import numpy as np
import pytest


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "mceg/reviews"))
import gas_source_comparison as baseline


@pytest.fixture(scope="module")
def study():
    return runpy.run_path(str(ROOT / "mceg/reviews/naoh_dimer_source_review.py"))


def test_exact_janaf_nodes_close_frozen_screen_and_decomposition(study):
    actual = study["review"]()
    saved = json.loads((study["DATA"] / "comparison.json").read_text())
    baseline.check_equal(actual, saved)
    for sample in actual["samples"]:
        assert abs(sample["baseline_closure_ln_K"]) < 3e-13
        components = sample["molecular_JANAF_minus_NASA_G_components_J_mol"]
        assert sum(components.values()) + sample["atomic_contribution_J_mol"] == pytest.approx(
            sample["reaction_JANAF_minus_NASA_G_J_mol"], abs=1e-9)
        assert sample["energy_aligned_janaf_over_nasa"] < .6
    assert actual["samples"][2]["curated_over_nasa"] == pytest.approx(.05162763562773825)
    assert actual["samples"][3]["curated_over_nasa"] == pytest.approx(.060854631819908446)
    assert actual["production_change"] == "none"
    assert actual["equilibrium_substitution"] == "not_run"


def test_source_mapping_excludes_peroxide_and_preserves_evaluation_constants(study):
    data = json.loads((study["DATA"] / "inputs.json").read_text())
    frozen = json.loads((study["BASELINE"] / "inputs.json").read_text())
    assert data["composition"] == {"H": 2, "Na": 2, "O": 2}
    assert data["charge"] == 0
    assert frozen["nasa9_records"][data["nasa_species"]]["composition"] == data["composition"]
    assert frozen["nasa9_records"][data["nasa_species"]]["reference"] == "Gurvich,1996b."
    table = study["read_table"](study["DATA"] / "H-062.txt")
    assert table[298.15][4] == -607.517
    assert data["janaf1974"]["hf298_kcal_mol"] * 4.184 == pytest.approx(-607.5168)
    assert len(data["gurvich1996"]["table_3_23"]["adopted_frequencies_cm-1"]) == 3 * 6 - 6
    assert data["gurvich1996"]["table_3_23"]["adopted_symmetry_number"] == 4
    with pytest.raises(ValueError, match="outside"):
        study["current_ln_k"](1000)


def test_nasa_matches_printed_absolute_functions_and_rrho_explains_thermal_change(study):
    result = study["review"]()
    assert result["rotation_new_minus_old_S_J_mol_K"] == pytest.approx(-6.399, abs=.001)
    for row in result["gurvich1996_replay"]:
        assert abs(row["NASA_minus_printed_H_J_mol"]) < 6
        assert abs(row["NASA_minus_printed_S_J_mol_K"]) < .003
        assert abs(row["NASA_minus_printed_Cp_J_mol_K"]) < .014
        assert row["harmonic_new_minus_old_Cp_J_mol_K"] == pytest.approx(
            row["printed_new_minus_JANAF_Cp_J_mol_K"], abs=.002)
        assert row["RRHO_new_minus_old_S_J_mol_K"] == pytest.approx(
            row["printed_new_minus_JANAF_S_J_mol_K"], abs=.003)
    # Do not silently substitute the inconsistent printed comparison-table cell.
    data = json.loads((study["DATA"] / "inputs.json").read_text())
    discrepancy = data["gurvich1996"]["printed_inconsistencies"]
    assert discrepancy["table_3_24_delta_S_at_1000_J_mol_K"] == 15.049
    assert discrepancy["table_3_27_minus_official_janaf_delta_S_at_1000_J_mol_K"] == 19.052


def test_harmonic_entropy_derivative_and_classical_mode_limit(study):
    frequencies = [200, 350, 3700]
    r = 8.31446261815324
    for t in (298.15, 1500, 3000):
        _, cp = study["vibrational_properties"](frequencies, t, r)
        plus = study["vibrational_properties"](frequencies, t + .01, r)[0]
        minus = study["vibrational_properties"](frequencies, t - .01, r)[0]
        assert t * (plus - minus) / .02 == pytest.approx(cp, rel=2e-9)
    assert study["vibrational_properties"](frequencies, 1e8, r)[1] == pytest.approx(3 * r, rel=1e-9)


def test_association_reaction_cancels_atoms_but_uses_both_monomer_sources(study):
    result = study["review"]()
    screen = json.loads((study["BASELINE"] / "comparison.json").read_text())
    for sample in result["samples"]:
        t = sample["temperature_K"]
        rows = {row["species_id"]: row["ln_Kp"] for row in screen["samples"]
                if row["temperature_K"] == t}
        for source in ("janaf", "nasa"):
            # Independent balanced-reaction combination from the frozen screen.
            assert sample["association_ln_K"][source] == pytest.approx(
                rows["H2Na2O2"][source] - 2 * rows["H1Na1O1"][source], abs=2e-13)
        assert sample["association_JANAF_over_NASA"] < sample["janaf_over_nasa"]


def test_archived_sodium_fraction_is_twice_dimer_number_and_model_matches(study):
    result = study["footprint"]()
    assert result["cases"] == 75 and result["species_count"] == 537
    assert result["all_cases_converged"]
    point = result["maximum_sodium_share"]
    assert point["case_index"] == 4
    assert point["sodium_fraction"] == pytest.approx(7.073911009008819e-9)
    with np.load(study["GRID"]) as grid:
        sodium = grid["elements"].tolist().index("Na")
        column = grid["species"].tolist().index("H2Na2O2")
        assert grid["formula_matrix"][sodium, column] == 2
        n = np.exp(grid["ln_n"][4])
        assert grid["formula_matrix"][sodium] @ n == pytest.approx(grid["budget"][4, sodium], rel=2e-8)
        assert point["sodium_fraction"] == pytest.approx(2 * n[column] / grid["budget"][4, sodium])
        assert point["mole_fraction"] == pytest.approx(n[column] / n.sum(), rel=2e-12)
        for t, h in zip(grid["temperature_K"], grid["h"][:, column]):
            assert -h == pytest.approx(study["current_ln_k"](t), abs=3e-12)


def test_selected_inputs_and_outputs_are_pinned(study):
    receipt = json.loads((study["DATA"] / "receipt.json").read_text())
    for group in ("selected_input_sha256", "output_sha256"):
        for path, digest in receipt[group].items():
            assert sha256((ROOT / path).read_bytes()).hexdigest() == digest, path
