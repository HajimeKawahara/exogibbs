"""Check independent identities, source drift and the C2H energy decomposition."""

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
    return runpy.run_path(str(ROOT / "mceg/reviews/c2h_source_review.py"))


def test_fresh_tables_reproduce_archive_and_expose_thermal_residual(study):
    report = study["review"]()
    saved = json.loads((study["DATA"] / "comparison.json").read_text())
    baseline.check_equal(report, saved)
    for actual, expected in zip(report["samples"], saved["samples"]):
        assert actual["temperature_K"] == expected["temperature_K"]
        for key in ("janaf_over_nasa", "energy_aligned_janaf_over_nasa",
                    "reaction_delta_G_J_mol", "atomic_delta_G_J_mol"):
            assert actual[key] == pytest.approx(expected[key], abs=2e-8, rel=2e-12)
        assert abs(actual["baseline_closure_ln_K"]) < 2e-11
        assert abs(actual["atomic_delta_G_J_mol"]) < 10
        assert actual["energy_aligned_janaf_over_nasa"] < 0.5
    assert report["samples"][0]["curated_over_nasa"] is None
    assert report["samples"][1]["curated_over_nasa"] is None
    assert report["equilibrium_impact"] == "not_run"


def test_properties_obey_gibbs_identity_and_cp_derivative(study):
    inputs = baseline.load_inputs()
    r = inputs["gas_constant_J_mol_K"]
    for name in ("C2H", "C", "H"):
        record = inputs["nasa9_records"][name]
        for temperature in (298.15, 500, 999, 1001, 1500, 3000):
            h, s, cp = study["nasa_properties"](record, temperature, r)
            assert (h-temperature*s)/(r*temperature) == pytest.approx(
                baseline.nasa_g(record, temperature), abs=2e-12)
            step = 0.01
            plus = study["nasa_properties"](record, temperature+step, r)[0]
            minus = study["nasa_properties"](record, temperature-step, r)[0]
            assert (plus-minus)/(2*step) == pytest.approx(cp, rel=2e-8)
    with pytest.raises(ValueError, match="outside"):
        study["nasa_properties"](inputs["nasa9_records"]["C2H"], 6001, r)


def test_official_janaf_nodes_match_all_pinned_properties(study):
    inputs = baseline.load_inputs()
    for name, identifier in (("C2H", "C2H1"), ("C", "C1"), ("H", "H1")):
        table = study["read_table"](study["DATA"] / study["TABLES"][name])
        original = inputs["janaf_records"][identifier]
        assert table[298.15][4] == original["hf298_kJ_mol"]
        for sample in original["samples"]:
            node = table[sample["temperature_K"]]
            assert node[1] == sample["entropy_J_mol_K"]
            assert node[3] == sample["enthalpy_increment_kJ_mol"]
            assert node[5] == sample["formation_gibbs_kJ_mol"]


def test_official_nasa_coefficients_and_receipt_hashes(study):
    inputs = baseline.load_inputs()
    text = (study["DATA"] / "nasa_selected_records.txt").read_text()
    for block in text.strip().split("\n\n"):
        name = block.split()[0]
        coefficients = [float(value.replace("D", "E"))
                        for value in re.findall(r"[+-]?\d+\.\d+D[+-]\d+", block)]
        expected = [value for interval in inputs["nasa9_records"][name]["intervals"]
                    for value in interval["coefficients"]]
        assert coefficients == expected
    receipt = json.loads((study["DATA"] / "receipt.json").read_text())
    for hashes in (receipt["selected_input_sha256"], receipt["output_sha256"]):
        for path, expected in hashes.items():
            assert sha256((ROOT / path).read_bytes()).hexdigest() == expected
