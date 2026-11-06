"""Check PH3 source conventions, derivatives and archived grid scope."""

from hashlib import sha256
import json
import math
from pathlib import Path
import sys

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "mceg/reviews"))
from curated_adoption_history import historical_source
import ph3_source_review as study


@pytest.fixture(scope="module")
def report():
    return study.review()


def test_exact_source_decomposition_and_frozen_comparison(report):
    frozen = json.loads((study.BASELINE / "comparison.json").read_text())
    for row in report["source_samples"]:
        previous = next(x for x in frozen["samples"] if x["species_id"] == report["species_id"] and x["temperature_K"] == row["temperature_K"])
        delta = sum(row["molecular_JANAF_minus_NASA_components"].values()) + row["atomic_reference_contribution_J_mol"]
        assert row["reaction_JANAF_minus_NASA_G_J_mol"] == pytest.approx(delta, abs=1e-9)
        assert math.log(row["janaf_over_nasa"]) == pytest.approx(previous["ln_Kp"]["janaf"] - previous["ln_Kp"]["nasa"], abs=2e-11)
        assert row["nasa_ln_K"] == pytest.approx(previous["ln_Kp"]["nasa"], abs=2e-11)
        if row["curated_ln_K"] is not None:
            assert row["curated_ln_K"] == pytest.approx(previous["ln_Kp"]["curated"], abs=2e-11)
    with pytest.raises(ValueError, match="outside"):
        study.current_ln_k(1499, report["species_id"])


def test_published_table_attribution_has_quantified_residual(report):
    rows = report["published_table_replay"]
    assert max(abs(x["NASA_minus_published_H_J_mol"]) for x in rows) < 2
    assert max(abs(x["NASA_minus_published_S_J_mol_K"]) for x in rows) < .0044
    assert max(abs(x["NASA_minus_published_Cp_J_mol_K"]) for x in rows) < .0064


def test_partition_derivatives_have_independent_analytic_control():
    ts = np.arange(1000., 2001.)
    synthetic = np.stack([ts, ts**3], axis=1)
    for t in (1500, 2000):
        row = study.partition_calorics(synthetic, t, 2)
        # Backward second differences have O(step**3) truncation: about
        # 5e-7 J/mol/K here, despite the exact analytic power-law target.
        assert row["Cp_J_mol_K"] == pytest.approx(5.5 * study.R, abs=1e-6)
        assert row["H_minus_H0_J_mol"] == pytest.approx(5.5 * study.R * t, abs=2e-5)
        assert row["stencil"] == ("central" if t == 1500 else "backward")
    with pytest.raises(ValueError, match="missing exact"):
        study.partition_calorics(synthetic, 999, 2)


def test_mass_and_nuclear_normalization_are_explicit(report):
    inputs = json.loads((study.DATA / "inputs.json").read_text())
    for row in report["partition_samples"]:
        if "downloaded_partition" not in row:
            continue
        p = row["downloaded_partition"]
        ratio = (inputs["isotopologue_mass_u"] / inputs["nasa_records"][inputs["name"]]["mass_u"])**1.5
        assert p["same_energy_atoms_isotope_mass_K_proxy"] == pytest.approx(p["Q_over_NASA"] * ratio)
        assert p["Cp_rounding_bound_J_mol_K"] > 0
    assert report["production_change"] == "none"
    assert report["equilibrium_substitution"] == "not_run"


def test_archived_inventory_accounts_for_each_network(report):
    inputs = json.loads((study.DATA / "inputs.json").read_text())
    for mode, count in (("ions", 537), ("neutral", 421)):
        f = report["footprint"][mode]
        assert (f["model"], f["cases"], f["species_count"], f["all_converged"]) == ("curated-cos-v1", 75, count, True)
        assert f["maximum_abs_target_h_error"] < 2e-12
        with np.load(study.GRID / (mode + ".npz")) as grid:
            i = grid["species"].tolist().index(report["species_id"])
            e = grid["elements"].tolist().index(inputs["inventory_element"])
            shares = grid["formula_matrix"][e, i] * np.exp(grid["ln_n"][:, i]) / grid["budget"][:, e]
            assert f["maximum_element_fraction"] == pytest.approx(float(shares.max()), rel=1e-12)
            assert f["case_index"] == int(np.argmax(shares))


def test_saved_replay_and_selected_input_hashes(report):
    expected = json.loads((study.DATA / "comparison.json").read_text())
    def equal(a, b, path=()):
        if isinstance(a, dict):
            assert a.keys() == b.keys()
            for k in a:
                equal(a[k], b[k], path + (k,))
        elif isinstance(a, list):
            assert len(a) == len(b)
            for i, (x, y) in enumerate(zip(a, b)):
                equal(x, y, path + (i,))
        elif isinstance(a, float):
            # Rounded-table second derivatives amplify ordinary float roundoff.
            caloric = path and path[-1] in {"Cp_J_mol_K", "Cp_alternate_step_J_mol_K"}
            assert a == pytest.approx(b, rel=2e-11, abs=1e-6 if caloric else 2e-10), path
        else:
            assert a == b
    equal(report, expected)
    receipt = json.loads((study.DATA / "receipt.json").read_text())
    for group in ("selected_input_sha256", "output_sha256"):
        for name, expected_hash in receipt[group].items():
            source = (historical_source(ROOT, name, expected_hash)
                      if group == "selected_input_sha256" else ROOT / name)
            assert sha256(source.read_bytes()).hexdigest() == expected_hash, name


def test_phosphorus_reference_correction_is_not_applied_twice(report):
    inputs = json.loads((study.DATA / "inputs.json").read_text())
    assert inputs["nasa_records"]["PH3"]["hf298_J_mol"] == 5439
    assert inputs["published_hf298_J_mol"] == 5440
    assert study.read_table(study.DATA / "janaf.txt")[298.15][4] == 22.886
    at1500 = next(r for r in report["source_samples"] if r["temperature_K"] == 1500)
    assert .99 < at1500["curated_over_nasa"] < 1.02
    assert at1500["janaf_over_nasa"] < .25


def test_printed_polynomial_mismatch_and_salty_limit_remain_explicit(report):
    audit = report["printed_2014_fit_audit"]
    assert audit["printed_Q296"] == pytest.approx(3216.4739936787482)
    assert abs(audit["relative_mismatch"]) > audit["declared_relative_fit_accuracy"] + audit["coefficient_rounding_relative_bound"]
    assert report["nuclear_spin_divisor"] == 16
    top = next(r for r in report["partition_samples"] if r["temperature_K"] == 3000)["downloaded_partition"]
    assert .99 < top["Q_over_NASA"] < 1.03
    assert top["NASA_Cp_common_R_J_mol_K"] - top["Cp_J_mol_K"] > 25
    assert "downloaded_partition" not in report["partition_samples"][-1]
