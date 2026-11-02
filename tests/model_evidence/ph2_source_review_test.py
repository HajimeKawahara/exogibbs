"""Check PH2 source versions, elemental references and replayed evidence."""

import json
from hashlib import sha256
from pathlib import Path
import re
import runpy
import sys

import pytest


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "mceg/reviews"))
import gas_source_comparison as baseline


@pytest.fixture(scope="module")
def study():
    return runpy.run_path(str(ROOT / "mceg/reviews/ph2_source_review.py"))


@pytest.fixture(scope="module")
def report(study):
    return study["review"]()


def test_replay_and_reaction_energy_decomposition(study, report):
    baseline.check_equal(report, json.loads((study["DATA"] / "comparison.json").read_text()))
    for row in report["samples"]:
        assert abs(row["baseline_closure_ln_K"]) < 2e-11
        assert sum(row["reaction_delta_G_components_J_mol"].values()) == pytest.approx(
            row["reaction_delta_G_J_mol"], abs=2e-8)
        assert row["reaction_delta_G_components_J_mol"]["anchor"] == pytest.approx(6490.578890420962)
        assert row["printed_1998_reaction_delta_G_J_mol"]-row["reaction_delta_G_J_mol"] == pytest.approx(-17439)
        assert 0 < row["atomic_delta_G_J_mol"] < 110
        assert 1 < row["energy_aligned_janaf_over_nasa"] < 1.06
    assert report["samples"][0]["curated_over_nasa"] is None
    assert report["samples"][1]["curated_over_nasa"] is None


def test_online_formation_columns_have_wrong_reference_signature(report):
    audit = report["formation_reference_audit"]
    rows = audit["online_reference_signature_samples"]
    assert len(rows) == 64
    assert rows[0]["temperature_K"] == 100
    assert rows[-1]["temperature_K"] == 6000
    # Tolerances reflect three-decimal printed H and S, not fit quality.
    assert audit["max_abs_online_minus_1p5_H2_Hf_kJ_mol"] < 0.002
    assert audit["max_abs_online_minus_1p5_H2_Gf_kJ_mol"] < 0.008
    at_298 = next(row for row in rows if row["temperature_K"] == 298.15)
    assert at_298["Gf_with_P_ref_plus_H2_kJ_mol"] == pytest.approx(113.74515575)
    assert at_298["online_Gf_kJ_mol"]-at_298["Gf_with_P_ref_plus_H2_kJ_mol"] > 7
    # The discrepancy changes sign across the phosphorus reference transition.
    assert report["samples"][2]["janaf_internal_difference_dex"] == pytest.approx(-0.10291764284421814)
    assert report["samples"][3]["janaf_internal_difference_dex"] == pytest.approx(1.2276119512802155)


def test_printed_janaf_reference_closes_and_explains_current_fit(report):
    audit = report["formation_reference_audit"]
    assert audit["online_anchor_kJ_mol"] == 125.938
    assert audit["printed_1998_anchor_kJ_mol"] == 108.499
    assert len(audit["printed_1998_reference_checks"]) == 5
    for row in audit["printed_1998_reference_checks"]:
        assert abs(row["printed_minus_P_ref_plus_H2_Hf_kJ_mol"]) < 0.002
        assert abs(row["printed_minus_P_ref_plus_H2_Gf_kJ_mol"]) < 0.003
    assert len(audit["atomic_reference_checks"]) == 10
    for row in audit["atomic_reference_checks"]:
        assert abs(row["online_minus_elemental_reference_Hf_kJ_mol"]) < 0.002
        assert abs(row["online_minus_elemental_reference_Gf_kJ_mol"]) < 0.003
    fit = report["fastchem_fit"]
    assert len(fit["samples"]) == 26
    assert fit["max_abs_printed_1998_HS_residual_ln_K"] < 0.012
    for row in report["samples"]:
        if row["temperature_K"] >= 1500:
            assert row["ln_Kp"]["fastchem_fit"] == pytest.approx(row["ln_Kp"]["curated"], abs=2e-11)
            assert row["printed_1998_janaf_over_nasa"] > 1.6
            assert row["janaf_over_nasa"] < 1


def test_official_janaf_nodes_match_baseline_and_phosphorus_phases(study):
    inputs = baseline.load_inputs()
    for name, identifier in (("PH2", "H2P1"), ("P", "P1"), ("H", "H1")):
        table = study["read_table"](study["DATA"] / study["TABLES"][name])
        original = inputs["janaf_records"][identifier]
        assert table[298.15][4] == original["hf298_kJ_mol"]
        for sample in original["samples"]:
            node = table[sample["temperature_K"]]
            assert node[1] == sample["entropy_J_mol_K"]
            assert node[3] == sample["enthalpy_increment_kJ_mol"]
            assert node[5] == sample["formation_gibbs_kJ_mol"]
    reference = study["phosphorus_reference"]()
    # Phase-boundary rows are deliberately omitted instead of selecting one side.
    assert 195.4 not in reference
    assert 317.3 not in reference
    assert 1180.008 not in reference
    assert reference[1000][3] == 19.092
    assert reference[1500][3] == 93.516


def test_raw_nasa_properties_obey_independent_gibbs_identity(study):
    inputs = baseline.load_inputs()
    r = inputs["gas_constant_J_mol_K"]
    for name in ("PH2", "P", "H"):
        record = inputs["nasa9_records"][name]
        for temperature in (298.15, 500, 1000, 1500, 3000):
            h, s, _ = study["nasa_properties"](record, temperature, r)
            assert (h-temperature*s)/(r*temperature) == pytest.approx(
                baseline.nasa_g(record, temperature), abs=2e-12)
    assert study["nasa_properties"](inputs["nasa9_records"]["PH2"], 298.15, r)[0] == pytest.approx(119552.7889515424)


def test_primary_nasa_coefficients_match_pinned_baseline(study):
    inputs = baseline.load_inputs()
    text = (study["DATA"] / "nasa_selected_records.txt").read_text()
    for block in text.strip().split("\n\n"):
        name = block.split()[0]
        coefficients = [float(value.replace("D", "E"))
                        for value in re.findall(r"[+-]?\d+\.\d+D[+-]\d+", block)]
        expected = [value for interval in inputs["nasa9_records"][name]["intervals"]
                    for value in interval["coefficients"]]
        assert coefficients == expected


def test_receipt_hashes(study):
    receipt = json.loads((study["DATA"] / "receipt.json").read_text())
    assert receipt["execution_kind"] == "fresh_offline_execution"
    for path in study["archived_sources"]():
        assert str(path.relative_to(ROOT)) in receipt["selected_input_sha256"]
    literature = json.loads((study["DATA"] / "literature.json").read_text())
    for name, expected in literature["archived_files"].items():
        relative = str((study["DATA"] / name).relative_to(ROOT))
        assert receipt["selected_input_sha256"][relative] == expected
    for hashes in (receipt["selected_input_sha256"], receipt["output_sha256"]):
        for path, expected in hashes.items():
            assert sha256((ROOT / path).read_bytes()).hexdigest() == expected
