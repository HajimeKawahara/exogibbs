"""Protect S5 source/reference separation without runtime or MCEG imports."""

import json
import math
from pathlib import Path
import sys

import pytest


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "mceg/reviews"))
import s5_source_review as study
from gas_source_comparison import check_equal, verify_hashes


@pytest.fixture(scope="module")
def report():
    return study.review()


def test_s5_official_coefficients_and_frozen_nodes_replay(report):
    expected = json.loads((study.DATA / "comparison.json").read_text())
    check_equal(report, expected)
    receipt = json.loads((study.DATA / "receipt.json").read_text())
    verify_hashes(receipt["input_and_artifact_sha256"])
    assert report["official_nasa_matching_coefficients"] == 18
    assert report["fresh_janaf_frozen_matching_nodes"] == 4
    header = report["nasa_header_check"]
    assert header["H298_J_mol"] == pytest.approx(132993.354, abs=0.001)
    assert header["S298_J_mol_K"] == pytest.approx(354.086, abs=0.0005)
    assert header["Cp298_J_mol_K"] == pytest.approx(87.870, abs=0.0005)
    closures = [r["baseline_closure_ln_K"] for r in report["source_samples"]
                if r["baseline_closure_ln_K"] is not None]
    assert len(closures) == 4
    assert max(abs(value) for value in closures) < 2e-11


def test_s5_entropy_difference_survives_balanced_energy_alignment(report):
    for row in report["source_samples"]:
        terms = row["reaction_delta_G_components_J_mol"]
        delta = row["reaction_delta_G_J_mol"]
        assert delta == pytest.approx(sum(terms.values()), abs=1e-8)
        molecule, atom = row["properties"]["S5"], row["properties"]["S"]
        assert delta == pytest.approx(molecule["delta_G_J_mol"]-5*atom["delta_G_J_mol"], abs=1e-8)
        assert row["atomic_delta_G_J_mol"] == pytest.approx(-5*atom["delta_G_J_mol"], abs=1e-8)
        assert terms["H298_anchor"] == pytest.approx(-22680.494806, abs=1e-6)
        ratio = math.exp(-delta/(report["gas_constant_J_mol_K"]*row["temperature_K"]))
        assert row["janaf_over_nasa"] == pytest.approx(ratio, rel=2e-12)
    rows = {r["temperature_K"]: r for r in report["source_samples"]}
    assert rows[1500]["H298_aligned_janaf_over_nasa"] == pytest.approx(0.004234113721)
    assert rows[3000]["H298_aligned_janaf_over_nasa"] == pytest.approx(0.003902515570)
    assert rows[1500]["curated_over_nasa"] == pytest.approx(0.026509711419)
    assert rows[3000]["curated_over_nasa"] == pytest.approx(0.009547540305)
    assert all(rows[t]["curated_over_nasa"] is None for t in (500, 1000))


def test_s5_unused_function_cell_is_separate_from_formation_constants(report):
    audit = {r["temperature_K"]: r for r in report["janaf_free_energy_function_audit"]}
    assert audit[298.15]["reported_J_mol_K"] == 477.413
    assert audit[298.15]["from_HS_J_mol_K"] == 308.638
    assert audit[298.15]["difference_J_mol_K"] == pytest.approx(168.775)
    assert max(abs(r["difference_J_mol_K"]) for t, r in audit.items() if t != 298.15) < 0.001
    # Printed formation-G and H/S agree at all comparison nodes even though
    # the unused free-energy-function cell is inconsistent at 298.15 K.
    assert max(abs(r["janaf_raw_minus_hs_ln_K"]) for r in report["source_samples"]) < 0.0005


def test_s5_reused_footprint_counts_all_five_sulfur_atoms(report):
    footprint = report["footprint"]
    assert footprint["cases"] == 75
    assert footprint["species_count"] == 537
    assert footprint["all_cases_converged"]
    assert footprint["sulfur_fraction"] == pytest.approx(1.9701161370540327e-24, rel=1e-12, abs=0)
    assert footprint["mole_fraction"] == pytest.approx(8.908621430529851e-30, rel=1e-12, abs=0)
    assert footprint["temperature_K"] == 1500
    assert footprint["pressure_bar"] == 1e-4
    assert footprint["new_equilibrium_substitution"] == "not_run"
    assert report["physical_accuracy"] == "unknown"
    assert report["adoption"] == "none; retain current fit provisionally"
