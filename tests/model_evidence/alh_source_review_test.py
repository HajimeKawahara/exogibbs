"""Verify AlH source attribution and thermal diagnostics without network access."""

from hashlib import sha256
from copy import deepcopy
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
    return runpy.run_path(str(ROOT / "mceg/reviews/alh_source_review.py"))


@pytest.fixture(scope="module")
def report(study):
    return study["review"]()


def test_replay_and_reaction_decomposition(study, report):
    actual = deepcopy(report)
    expected = json.loads((study["DATA"] / "comparison.json").read_text())
    actual_cp, expected_cp = [value["partition_comparison"]["caloric_diagnostic"].pop("samples")
                              for value in (actual, expected)]
    baseline.check_equal(actual, expected)
    for rebuilt, saved in zip(actual_cp, expected_cp):
        # Second differences amplify last-bit libm rounding. Keep this much
        # tighter than the observed step-size spread and the table precision.
        assert rebuilt.pop("ExoMol_Cp_J_mol_K_by_step_K") == pytest.approx(
            saved.pop("ExoMol_Cp_J_mol_K_by_step_K"), abs=2e-7, rel=0)
        assert rebuilt.pop("step_spread_J_mol_K") == pytest.approx(
            saved.pop("step_spread_J_mol_K"), abs=4e-7, rel=0)
        baseline.check_equal(rebuilt, saved)
    assert report["reference_temperature_K"] == 298.15
    for row in report["samples"]:
        assert abs(row["baseline_closure_ln_K"]) < 2e-11
        assert sum(row["reaction_delta_G_components_J_mol"].values()) == pytest.approx(
            row["reaction_delta_G_J_mol"], abs=2e-8)
        assert row["reaction_delta_G_components_J_mol"]["anchor"] == pytest.approx(10456.321262149315)
        assert 280 < row["atomic_delta_G_J_mol"] < 310
        assert .98 < row["energy_aligned_janaf_over_nasa"] < 1.002
        assert abs(row["janaf_internal_difference_dex"]) < .00011
    assert report["samples"][0]["curated_over_nasa"] is None
    assert report["samples"][1]["curated_over_nasa"] is None
    assert report["samples"][2]["curated_over_nasa"] == pytest.approx(.4341211613145841)
    assert report["samples"][3]["curated_over_nasa"] == pytest.approx(.6469166404135734)


def test_official_nasa_intervals_coefficients_and_nonextrapolated_anchors(study, report):
    headers = study["nasa_headers"]()
    inputs = baseline.load_inputs()
    for name, source in headers.items():
        record = inputs["nasa9_records"][name]
        expected = [value for interval in record["intervals"] for value in interval["coefficients"]]
        assert source["coefficients"] == expected
        assert source["intervals_K"] == [row["bounds_K"] for row in record["intervals"]]
        with pytest.raises(ValueError, match="outside NASA intervals"):
            study["nasa_properties"](record, 0, inputs["gas_constant_J_mol_K"])
    assert headers["ALH"]["Hf298_J_mol"] == 249250.804
    assert headers["ALH"]["H298_minus_H0_J_mol"] == 8668.104
    intervals = report["source_intervals"]
    assert intervals["official_NASA_ALH"][0] == [200., 1000.]
    assert intervals["earlier_major_fixture_ALH"][0] == [300., 1000.]
    anchors = report["dissociation_anchors"]
    assert anchors["NASA_header_D0_eV"] == pytest.approx(3.0502048604)
    assert anchors["JANAF_D0_eV"] == pytest.approx(2.9418253920)
    assert anchors["Barklem_Collet_selected_D0_eV"] == 2.949
    assert anchors["Barklem_Collet_quoted_uncertainty_eV"] == .13
    assert len(report["NASA_interval_join_checks"]) == 3
    for row in report["NASA_interval_join_checks"]:
        assert abs(row["high_minus_low_H_J_mol"]) < .001
        assert abs(row["high_minus_low_S_J_mol_K"]) < 1e-6
        assert abs(row["high_minus_low_Cp_J_mol_K"]) < 1e-6


def test_printed_online_and_elemental_reference_agree(report):
    audit = report["formation_reference_audit"]
    assert audit["printed_online_matching_nodes"] == 6
    assert len(audit["samples"]) == 61
    assert audit["max_abs_formation_H_error_kJ_mol"] < .002
    assert audit["max_abs_formation_G_error_kJ_mol"] < .005
    # Include both sides of the aluminum boiling transition, not a fitted
    # continuation of the condensed reference through the gas region.
    temperatures = [row["temperature_K"] for row in audit["samples"]]
    assert 2000 in temperatures and 3000 in temperatures
    fit = report["fastchem_fit"]
    assert len(fit["samples"]) == 26
    assert fit["max_abs_JANAF_residual_ln_K"] < .0042
    assert max(abs(row["fit_minus_JANAF_reaction_Cp_J_mol_K"]) for row in fit["samples"]) < .71


def test_official_janaf_records_reproduce_frozen_nodes(study):
    inputs = baseline.load_inputs()
    for name, identifier in (("ALH", "AlH"), ("AL", "Al"), ("H", "H1")):
        table = study["read_table"](study["DATA"] / study["TABLES"][name])
        original = inputs["janaf_records"][identifier]
        assert table[298.15][4] == original["hf298_kJ_mol"]
        for sample in original["samples"]:
            node = table[sample["temperature_K"]]
            assert node[1] == sample["entropy_J_mol_K"]
            assert node[3] == sample["enthalpy_increment_kJ_mol"]
            assert node[5] == sample["formation_gibbs_kJ_mol"]


def test_caloric_derivatives_and_independent_gibbs_identity(study):
    coefficients, _ = study["fastchem_record"]()
    inputs = baseline.load_inputs()
    r = inputs["gas_constant_J_mol_K"]
    for t in (1500., 3000., 4000.):
        ln_k, h, s, cp = study["fit_properties"](coefficients, t, r)
        plus = study["fit_properties"](coefficients, t+.1, r)
        minus = study["fit_properties"](coefficients, t-.1, r)
        assert (h-t*s)/(r*t) == pytest.approx(-ln_k, abs=1e-13)
        assert r*t*t*(plus[0]-minus[0])/.2 == pytest.approx(h, rel=1e-8)
        assert (plus[1]-minus[1])/.2 == pytest.approx(cp, rel=1e-8)
        for name in ("ALH", "AL", "H"):
            record = inputs["nasa9_records"][name]
            nh, ns, _ = study["nasa_properties"](record, t, r)
            assert (nh-t*ns)/(r*t) == pytest.approx(baseline.nasa_g(record, t), abs=2e-12)


def test_partition_comparison_removes_nuclear_spin_and_is_energy_independent(study, report):
    comparison = report["partition_comparison"]
    assert comparison["nuclear_spin_divisor"] == 12
    assert comparison["NASA_gas_constant_J_mol_K"] == 8.31451
    assert comparison["JANAF_gas_constant_J_mol_K"] == 8.31441
    for row in comparison["samples"]:
        assert row["ExoMol_Q_without_nuclear_spin"]*12 == pytest.approx(row["ExoMol_Q_including_nuclear_spin"])
        assert .97 < row["ExoMol_over_JANAF"] < 1.03
        assert .97 < row["ExoMol_over_NASA"] < 1.03
    # Changing only the energy zero leaves Q unchanged: it uses H0-G(T).
    tables = {name: study["read_table"](study["DATA"] / filename) for name, filename in study["TABLES"].items()}
    inputs = baseline.load_inputs()
    headers = study["nasa_headers"]()
    tables["ALH"][298.15][4] += 20
    shifted = study["partition_comparison"](tables, inputs, headers, 8.31451)
    for actual, expected in zip(shifted["samples"], comparison["samples"]):
        assert actual["JANAF_inferred_Q"] == pytest.approx(expected["JANAF_inferred_Q"], rel=1e-12)
    caloric = comparison["caloric_diagnostic"]
    assert caloric["gas_constant_J_mol_K"] == inputs["gas_constant_J_mol_K"]
    assert len(caloric["samples"]) == 4
    for row in caloric["samples"]:
        assert row["step_spread_J_mol_K"] < .01
    hot = caloric["samples"][-1]
    assert hot["temperature_K"] == 4000
    assert hot["ExoMol_Cp_J_mol_K_by_step_K"]["20"] == pytest.approx(40.6025, abs=.001)
    assert hot["NASA_Cp_J_mol_K"] > hot["JANAF_Cp_J_mol_K"]+5


def test_source_manifest_and_fresh_receipt_hashes(study):
    paths = study["archived_sources"]()
    assert len(paths) >= 9
    receipt = json.loads((study["DATA"] / "receipt.json").read_text())
    assert receipt["execution_kind"] == "fresh_offline_execution"
    assert receipt["result"] == "passed"
    for hashes in (receipt["selected_input_sha256"], receipt["output_sha256"]):
        for path, expected in hashes.items():
            assert sha256((ROOT / path).read_bytes()).hexdigest() == expected
    assert set(str(path.relative_to(ROOT)) for path in paths) <= receipt["selected_input_sha256"].keys()
