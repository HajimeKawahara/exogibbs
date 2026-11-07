"""Replay the gas source baseline and protect its scientific distinctions."""

from collections import Counter
from copy import deepcopy
import json
import math
import os
from pathlib import Path
import runpy
import subprocess
import sys

import pytest


ROOT = Path(__file__).resolve().parents[2]
SCRIPT = ROOT / "mceg/reviews/gas_source_comparison.py"
BASELINE = ROOT / "tests/data/validation/gas_source_comparison/baseline-001"


@pytest.fixture(scope="module")
def study():
    return runpy.run_path(str(SCRIPT))


@pytest.fixture(scope="module")
def inputs(study):
    return study["load_inputs"]()


@pytest.fixture(scope="module")
def report(study, inputs):
    return study["compare"](inputs)


@pytest.fixture(scope="module")
def rows(study, inputs, report):
    return {row["species_id"]: row for row in study["summary_rows"](report, inputs)}


def sample(report, species_id, temperature):
    return next(
        row for row in report["samples"]
        if row["species_id"] == species_id and row["temperature_K"] == temperature
    )


def test_screening_scope_and_issue_table_are_preserved(inputs, report, rows):
    assert inputs["snapshot_id"] == report["snapshot_id"] == "baseline-001"
    assert len(inputs["species"]) == len(rows) == 93
    assert len(inputs["excluded"]) == 14
    assert sum(row["identity_status"] == "provisional" for row in inputs["species"]) == 70
    assert len(report["samples"]) == 93 * 4
    assert Counter(row["janaf_grade"] for row in rows.values()) == {
        "green": 68, "yellow": 16, "red": 9,
    }
    assert Counter(row["curated_grade"] for row in rows.values()) == {
        "green": 74, "yellow": 14, "red": 5,
    }
    assert sum(row["janaf_major_10pct"] for row in rows.values()) == 10
    assert sum(row["curated_major_10pct"] for row in rows.values()) == 13


def test_all_migrated_rows_reproduce_the_original_issue_table(inputs, rows):
    assert len(inputs["historical_table"]) == len(rows)
    for original in inputs["historical_table"]:
        row = rows[original["curated_species"]]
        assert row["janaf_over_nasa"] == pytest.approx(original["JANAF_over_NASA"], rel=2e-11)
        assert row["curated_over_nasa"] == pytest.approx(original["curated_over_NASA"], rel=2e-11)
        assert row["curated_basis"] == original["curated_basis"]
        assert row["janaf_grade"] == original["JANAF_grade"]
        assert row["curated_grade"] == original["curated_grade"]


@pytest.mark.parametrize("key, value", [("nasa_id", "HNC"), ("identity_status", "provisional")])
def test_historical_migration_rejects_changed_identity(study, inputs, rows, key, value):
    changed = deepcopy(rows)
    changed["C1H1N1_hcn"][key] = value
    with pytest.raises(ValueError, match="Historical mapping or flag changed: C1H1N1_hcn"):
        study["check_historical"](list(changed.values()), inputs)


def test_hcn_difference_is_the_adopted_atct_energy_shift(inputs, rows):
    # The source thermal coefficients agree; only the reaction energy differs.
    nasa_d0_over_r_K = 152093.27789611174
    atct_d0_over_r_K = 152543.95362014536
    expected_ratio = math.exp((atct_d0_over_r_K - nasa_d0_over_r_K) / 1500)
    source = inputs["nasa9_records"]
    hcn_input = next(row for row in inputs["species"] if row["id"] == "C1H1N1_hcn")
    assert hcn_input["nasa_id"] == "HCN"
    assert hcn_input["identity_status"] == "reviewed"

    def coefficients(nasa_id):
        return next(
            interval["coefficients"] for interval in source[nasa_id]["intervals"]
            if interval["bounds_K"][0] <= 1500 <= interval["bounds_K"][1]
        )

    reaction = [
        value - sum(
            count * coefficients(inputs["atomic_references"][element]["nasa_id"])[index]
            for element, count in hcn_input["composition"].items()
        )
        for index, value in enumerate(coefficients("HCN"))
    ]
    curated = json.loads((ROOT / "src/exogibbs/data/curated/gas_v1.json").read_text())
    adopted = curated["nasa9_reactions"]["C1H1N1_hcn"]
    for index in (0, 1, 2, 3, 4, 5, 6, 8):
        assert adopted[index] == pytest.approx(reaction[index], rel=2e-12, abs=0)
    assert reaction[7] - adopted[7] == pytest.approx(atct_d0_over_r_K - nasa_d0_over_r_K)
    hcn = rows["C1H1N1_hcn"]
    assert hcn["curated_over_nasa"] == pytest.approx(expected_ratio, rel=2e-12)
    assert hcn["janaf_over_nasa"] == pytest.approx(0.8485468833264532, rel=2e-11)
    assert "NASA" in hcn["curated_basis"]
    assert "ATcT" in hcn["curated_basis"]
    assert hcn["janaf_major_10pct"] is True
    assert hcn["curated_major_10pct"] is True


def test_large_p4o6_difference_and_s2o_internal_disagreement_remain_distinct(report, rows):
    p4o6 = rows["O6P4"]
    assert p4o6["janaf_over_nasa"] == pytest.approx(3.228101392995405e20, rel=2e-11)
    assert p4o6["curated_over_nasa"] == pytest.approx(0.9729370984891649, rel=2e-11)
    assert p4o6["janaf_grade"] == "red"
    assert p4o6["curated_grade"] == "green"
    assert rows["O1S2"]["janaf_over_nasa"] == pytest.approx(1.0130543827051879, rel=2e-11)
    for species_id, expected_dex in (
        ("H1S1", 2.2343279230705653),
        ("O1S2", 4.468655846141134),
    ):
        assert sample(report, species_id, 1500)["janaf_internal_difference_dex"] == pytest.approx(
            expected_dex, abs=2e-11,
        )


def test_missing_source_nodes_and_curated_range_are_not_extrapolated(report):
    missing = sample(report, "H2O2", 3000)
    assert missing["ln_Kp"]["janaf"] is None
    assert missing["missing"]["janaf"]
    assert missing["ln_Kp"]["nasa"] is not None
    assert missing["ln_Kp"]["curated"] is not None
    below_range = [row for row in report["samples"] if row["temperature_K"] < 1500]
    assert len(below_range) == 186
    assert all(row["ln_Kp"]["curated"] is None for row in below_range)
    assert all(row["missing"]["curated"] for row in below_range)


def test_malformed_janaf_sample_is_not_reported_as_a_missing_temperature(study, inputs):
    changed = deepcopy(inputs)
    node = next(
        row for row in changed["janaf_records"]["C1H1N1"]["samples"]
        if row["temperature_K"] == 1500
    )
    del node["entropy_J_mol_K"]
    with pytest.raises(KeyError, match="entropy_J_mol_K"):
        study["compare"](changed)


def test_selected_input_hash_rejects_drift(study, inputs, tmp_path):
    changed = deepcopy(inputs)
    selected_path = next(iter(changed["selected_input_sha256"]))
    changed["selected_input_sha256"][selected_path] = "0" * 64
    path = tmp_path / "inputs.json"
    path.write_text(json.dumps(changed))
    with pytest.raises(ValueError, match="(?i)hash|sha256"):
        study["load_inputs"](path)


def test_identical_composition_does_not_allow_hnc_to_replace_hcn(study, inputs):
    changed = deepcopy(inputs)
    hcn = next(row for row in changed["species"] if row["id"] == "C1H1N1_hcn")
    changed["nasa9_records"]["HNC"] = deepcopy(changed["nasa9_records"]["HCN"])
    hcn["nasa_id"] = "HNC"
    with pytest.raises(ValueError, match="(?i)identity|HCN|HNC"):
        study["compare"](changed)


def test_composition_mismatch_is_rejected(study, inputs):
    changed = deepcopy(inputs)
    hcn = next(row for row in changed["species"] if row["id"] == "C1H1N1_hcn")
    hcn["composition"]["H"] = 2
    with pytest.raises(ValueError, match="(?i)composition"):
        study["compare"](changed)


@pytest.mark.parametrize(
    "ratio, expected",
    [
        (1.0, "green"),
        (math.nextafter(2.0, 1.0), "green"),
        (math.nextafter(0.5, 1.0), "green"),
        (2.0, "yellow"),
        (0.5, "yellow"),
        (math.nextafter(10.0, 1.0), "yellow"),
        (math.nextafter(0.1, 1.0), "yellow"),
        (10.0, "red"),
        (0.1, "red"),
    ],
)
def test_grades_use_symmetric_factor_boundaries(study, ratio, expected):
    assert study["grade"](ratio) == expected


@pytest.mark.parametrize(
    "ratio, expected",
    [(0.9, True), (1.1, True), (0.90001, False), (1.09999, False), (1 / 1.1, False)],
)
def test_major_marker_uses_ten_percent_relative_to_nasa(study, ratio, expected):
    assert study["major_marker"](ratio, True) is expected
    assert study["major_marker"](ratio, False) is False


def test_archived_samples_replay_with_numerical_tolerance(report):
    saved = json.loads((BASELINE / "comparison.json").read_text())
    assert report["snapshot_id"] == saved["snapshot_id"]
    assert len(report["samples"]) == len(saved["samples"])
    for actual, expected in zip(report["samples"], saved["samples"]):
        for key in ("species_id", "temperature_K", "missing"):
            assert actual[key] == expected[key]
        for source in ("janaf", "nasa", "curated"):
            value = expected["ln_Kp"][source]
            if value is None:
                assert actual["ln_Kp"][source] is None
            else:
                assert actual["ln_Kp"][source] == pytest.approx(value, rel=2e-12, abs=2e-11)
        difference = expected["janaf_internal_difference_dex"]
        if difference is None:
            assert actual["janaf_internal_difference_dex"] is None
        else:
            assert actual["janaf_internal_difference_dex"] == pytest.approx(
                difference, rel=2e-12, abs=2e-11,
            )


def test_offline_cli_exports_views_and_rejects_changed_artifact(study, inputs, report, tmp_path):
    env = dict(os.environ, JAX_PLATFORMS="cpu", JAX_ENABLE_X64="1")
    output = tmp_path / "run"
    subprocess.run(
        [sys.executable, str(SCRIPT), "--output-dir", str(output)],
        cwd=ROOT, env=env, check=True, capture_output=True, text=True,
    )
    for name in ("comparison.json", "table.csv", "table.rst", "table.md", "receipt.json"):
        assert (output / name).stat().st_size > 0
    expected_table = study["render_markdown"](study["summary_rows"](report, inputs))
    assert (output / "table.md").read_text() == expected_table
    rerun = subprocess.run(
        [sys.executable, str(SCRIPT), "--output-dir", str(output)],
        cwd=ROOT, env=env, capture_output=True, text=True,
    )
    assert rerun.returncode != 0
    assert "already exists" in rerun.stderr
    subprocess.run(
        [sys.executable, str(SCRIPT), "--check", "--output-dir", str(output)],
        cwd=ROOT, env=env, check=True, capture_output=True, text=True,
    )
    with (output / "table.md").open("a") as table:
        table.write("Changed after generation.\n")
    changed = subprocess.run(
        [sys.executable, str(SCRIPT), "--check", "--output-dir", str(output)],
        cwd=ROOT, env=env, capture_output=True, text=True,
    )
    assert changed.returncode != 0
