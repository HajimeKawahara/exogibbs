"""Verify the C3 source audit and archived diagnostics without network access."""

from copy import deepcopy
from hashlib import sha256
import json
from pathlib import Path
import runpy
import sys

import pytest


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "mceg/reviews"))
import gas_source_comparison as baseline


@pytest.fixture(scope="module")
def study():
    return runpy.run_path(str(ROOT / "mceg/reviews/c3_source_review.py"))


@pytest.fixture(scope="module")
def report(study):
    return study["review"]()


def test_replay_and_reaction_decomposition(study, report):
    actual = deepcopy(report)
    expected = json.loads((study["DATA"] / "comparison.json").read_text())
    actual_cp, expected_cp = [value["partition_comparison"]["caloric_diagnostic"].pop("samples")
                              for value in (actual, expected)]
    baseline.check_equal(actual, expected)
    assert len(actual_cp) == len(expected_cp) == 4
    for rebuilt, saved in zip(actual_cp, expected_cp):
        # Second differences amplify libm rounding well below the tabulated
        # precision and the observed difference from full state-sum moments.
        assert rebuilt.pop("AtLast_Cp_J_mol_K_by_step_K") == pytest.approx(
            saved.pop("AtLast_Cp_J_mol_K_by_step_K"), abs=2e-7, rel=0)
        assert rebuilt.pop("step_spread_J_mol_K") == pytest.approx(
            saved.pop("step_spread_J_mol_K"), abs=4e-7, rel=0)
        baseline.check_equal(rebuilt, saved)
    for row in report["samples"]:
        assert abs(row["baseline_closure_ln_K"]) < 2e-11
        assert sum(row["reaction_delta_G_components_J_mol"].values()) == pytest.approx(
            row["reaction_delta_G_J_mol"], abs=2e-8)
        assert row["reaction_delta_G_components_J_mol"]["anchor"] == pytest.approx(-19862.11246034433)
        assert abs(row["atomic_delta_G_J_mol"]) < 20
        assert abs(row["janaf_internal_difference_dex"]) < .00006
    assert report["samples"][0]["ln_Kp"]["fastchem_fit"] is None
    assert report["samples"][1]["ln_Kp"]["fastchem_fit"] is None
    assert report["samples"][2]["curated_over_nasa"] == pytest.approx(2.6391066899)
    assert report["samples"][3]["curated_over_nasa"] == pytest.approx(.8089943349)
    assert report["samples"][2]["energy_aligned_janaf_over_nasa"] == pytest.approx(.5309047582)
    assert report["samples"][3]["energy_aligned_janaf_over_nasa"] == pytest.approx(.3667019159)


def test_official_nasa_records_and_valid_intervals(study, report):
    headers = study["nasa_headers"]()
    inputs = baseline.load_inputs()
    for name, source in headers.items():
        record = inputs["nasa9_records"][name]
        expected = [value for interval in record["intervals"] for value in interval["coefficients"]]
        assert source["coefficients"] == expected
        assert source["intervals_K"] == [row["bounds_K"] for row in record["intervals"]]
        with pytest.raises(ValueError, match="outside NASA intervals"):
            study["nasa_properties"](record, 0, inputs["gas_constant_J_mol_K"])
    assert headers["C3"]["Hf298_J_mol"] == 839948.646
    assert headers["C3"]["H298_minus_H0_J_mol"] == 12109.146
    assert len(report["NASA_interval_join_checks"]) == 4
    for row in report["NASA_interval_join_checks"]:
        assert abs(row["high_minus_low_H_J_mol"]) < .003
        assert abs(row["high_minus_low_S_J_mol_K"]) < 3e-6
        assert abs(row["high_minus_low_Cp_J_mol_K"]) < 3e-6
    for row in report["NASA_298_anchor_checks"]:
        assert abs(row["source_R_fit_residual_J_mol"]) < .001
        assert -5 < row["gas_constant_conversion_J_mol"] < -4


def test_printed_online_graphite_reference_and_frozen_nodes(study, report):
    audit = report["formation_reference_audit"]
    assert audit["printed_online_matching_nodes"] == 7
    assert len(audit["samples"]) == 61
    assert audit["max_abs_formation_H_error_kJ_mol"] < .0021
    assert audit["max_abs_formation_G_error_kJ_mol"] < .01
    printed = json.loads((study["DATA"] / "printed_c3_nodes.json").read_text())
    assert printed["notes"]["vibrational_frequencies_cm-1"] == [1225, 63.5, 2040]
    assert printed["notes"]["vibrational_degeneracies"] == [1, 2, 1]
    inputs = baseline.load_inputs()
    for name, identifier in (("C3", "C3"), ("C", "C1")):
        table = study["read_table"](study["DATA"] / study["TABLES"][name])
        frozen = inputs["janaf_records"][identifier]
        assert table[298.15][4] == frozen["hf298_kJ_mol"]
        for row in frozen["samples"]:
            node = table[row["temperature_K"]]
            assert node[1] == row["entropy_J_mol_K"]
            assert node[3] == row["enthalpy_increment_kJ_mol"]
            assert node[5] == row["formation_gibbs_kJ_mol"]


def test_current_fit_caloric_derivatives_and_independent_gibbs_identity(study, report):
    coefficients, _ = study["fastchem_record"]()
    inputs = baseline.load_inputs()
    r = inputs["gas_constant_J_mol_K"]
    for t in (1500., 3000., 4000.):
        ln_k, h, s, cp = study["fit_properties"](coefficients, t, r)
        plus = study["fit_properties"](coefficients, t+.1, r)
        minus = study["fit_properties"](coefficients, t-.1, r)
        assert (h-t*s)/(r*t) == pytest.approx(-ln_k, abs=2e-13)
        assert r*t*t*(plus[0]-minus[0])/.2 == pytest.approx(h, rel=1e-8)
        assert (plus[1]-minus[1])/.2 == pytest.approx(cp, rel=1e-8)
        for name in ("C3", "C"):
            record = inputs["nasa9_records"][name]
            nh, ns, _ = study["nasa_properties"](record, t, r)
            assert (nh-t*ns)/(r*t) == pytest.approx(baseline.nasa_g(record, t), abs=2e-12)
    fit = report["fastchem_fit"]
    assert len(fit["samples"]) == 26
    assert fit["max_abs_JANAF_residual_ln_K"] < .0111
    assert max(abs(row["fit_minus_JANAF_reaction_Cp_J_mol_K"]) for row in fit["samples"]) < 1.38


def test_atct_balances_same_version_atoms_and_preserves_isomers(study, report):
    source = json.loads((study["DATA"] / "atct_evidence.json").read_text())
    anchors = report["energy_anchors"]
    assert anchors["ATcT_version"] == "1.172"
    assert anchors["reported_C3_C_correlation"] == .228
    assert source["species"]["C3_linear"]["ATcT_ID"] != source["species"]["C3_cyclic"]["ATcT_ID"]
    assert source["species"]["C3_linear"]["Hf0_kJ_mol"] < source["species"]["C3_cyclic_triplet"]["Hf0_kJ_mol"]
    zero = anchors["anchors"]["0"]
    room = anchors["anchors"]["298"]
    assert zero["ATcT_reaction_H_J_mol"] == pytest.approx(-1319773)
    assert zero["ATcT_minus_JANAF_reaction_H_J_mol"] == pytest.approx(2326)
    assert zero["ATcT_minus_NASA_reaction_H_J_mol"] == pytest.approx(-17180.185)
    assert room["ATcT_minus_JANAF_reaction_H_J_mol"] == pytest.approx(2678)
    assert room["ATcT_minus_NASA_reaction_H_J_mol"] == pytest.approx(-17176.646)


def test_partition_conventions_and_energy_zero_independence(study, report):
    comparison = report["partition_comparison"]
    assert comparison["nuclear_spin_divisor"] == 1
    assert comparison["NASA_gas_constant_J_mol_K"] == 8.31451
    assert comparison["JANAF_gas_constant_J_mol_K"] == 8.31441
    hot = comparison["samples"][-1]
    assert hot["temperature_K"] == 4000
    assert hot["AtLast_over_JANAF"] == pytest.approx(1.7482314544)
    assert hot["AtLast_over_NASA"] == pytest.approx(.5583758182)
    tables = {name: study["read_table"](study["DATA"] / filename) for name, filename in study["TABLES"].items()}
    tables["C3"][298.15][4] += 20
    shifted = study["partition_comparison"](tables, baseline.load_inputs(), study["nasa_headers"]())
    for actual, expected in zip(shifted["samples"], comparison["samples"]):
        assert actual["JANAF_inferred_Q"] == pytest.approx(expected["JANAF_inferred_Q"], rel=1e-12)
    for row in comparison["caloric_diagnostic"]["samples"]:
        assert row["step_spread_J_mol_K"] < .0001


def test_historical_entropy_comparison_preserves_pressure_assumption(report):
    comparison = report["historical_entropy_comparison"]
    assert comparison["calorie_J"] == 4.184
    shift = comparison["conditional_1atm_to_1bar_shift_J_mol_K"]
    assert shift == pytest.approx(.109443159416017)
    rows = comparison["samples"]
    assert rows[0]["quoted_entropy_J_mol_K"] == pytest.approx(340.87048)
    assert rows[1]["quoted_entropy_J_mol_K"] == pytest.approx(346.89544)
    for row in rows:
        assert row["if_quoted_at_1atm_entropy_at_1bar_J_mol_K"] == pytest.approx(row["quoted_entropy_J_mol_K"]+shift)
        assert row["JANAF_entropy_J_mol_K"] < row["AtLast_published_Q_entropy_J_mol_K"] < row["NASA_entropy_J_mol_K"]
        assert abs(row["NASA_entropy_J_mol_K"]-row["quoted_entropy_J_mol_K"]) < 1.1
    primary = comparison["primary_abstract_samples"]
    assert [row["temperature_K"] for row in primary] == [2400, 2600]
    assert [row["quoted_entropy_J_mol_K"] for row in primary] == [row["quoted_entropy_J_mol_K"] for row in rows]
    assert primary[1]["NASA_entropy_J_mol_K"] == pytest.approx(344.3443653774663)


def test_archived_state_moments_preserve_published_partition_mismatch(study, report):
    states = json.loads((study["DATA"] / "state_sum_checks.json").read_text())
    ground = (study["DATA"] / "exomol_atlast_12C3_states_head.txt").read_text().splitlines()[0].split()
    assert [float(value) for value in ground[1:4]] == [0, 1, 0]
    definition = (study["DATA"] / "exomol_atlast_12C3.def").read_text().splitlines()
    assert [line.split()[0] for line in definition if "Nuclear spin degeneracy" in line] == ["6", "6"]
    assert states["state_count"] == 2166503
    assert states["all_weights_equal_2J_plus_1"]
    assert states["maximum_energy_cm-1"] < 20000
    assert states["maximum_J"] == 155 and states["maximum_l2"] == 12
    assert states["implementation_sha256"] == sha256((ROOT / "mceg/reviews/c3_source_review.py").read_bytes()).hexdigest()
    rows = {row["temperature_K"]: row for row in report["partition_comparison"]["independent_state_sum"]}
    assert rows[1]["state_sum_over_published_Q"] == pytest.approx(1, abs=.00005)
    assert rows[300]["state_sum_over_published_Q"] == pytest.approx(.8577946238)
    assert rows[1500]["state_sum_over_published_Q"] == pytest.approx(.9801569377)
    assert rows[4000]["state_sum_over_published_Q"] == pytest.approx(.9982338349)
    assert rows[300]["auxiliary_calculated_sum_over_published_Q"] < .86
    assert rows[4000]["state_sum_Cp_J_mol_K"] == pytest.approx(40.48158033)
    assert rows[4000]["state_sum_Cp_J_mol_K"] < rows[3000]["state_sum_Cp_J_mol_K"] < rows[1500]["state_sum_Cp_J_mol_K"]
    # Full state-file reduction is an optional acquisition-stage calculation,
    # not something the offline unit test pretends to repeat.
    for row in states["samples"]:
        assert row["mean_squared_E_over_kT"] >= row["mean_E_over_kT"]**2
        for key in ("fraction_E_above_15000_cm-1", "fraction_J_above_130", "fraction_l2_above_10"):
            assert 0 <= row[key] <= 1


def test_source_manifest_and_fresh_receipt_hashes(study):
    paths = study["archived_sources"]()
    receipt = json.loads((study["DATA"] / "receipt.json").read_text())
    assert receipt["execution_kind"] == "fresh_offline_execution"
    assert receipt["result"] == "passed"
    for hashes in (receipt["selected_input_sha256"], receipt["output_sha256"]):
        for path, expected in hashes.items():
            assert sha256((ROOT / path).read_bytes()).hexdigest() == expected
    assert set(str(path.relative_to(ROOT)) for path in paths) <= receipt["selected_input_sha256"].keys()
