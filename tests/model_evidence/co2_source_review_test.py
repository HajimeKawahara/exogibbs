"""Check independent CO2 caloric identities, conventions and retained energies."""

from hashlib import sha256
import json
import math
from pathlib import Path
import sys

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "mceg/reviews"))
import co2_source_review as study
from gas_source_comparison import check_equal


@pytest.fixture(scope="module")
def report():
    return study.review()


@pytest.fixture(scope="module")
def nodes():
    return json.loads((study.DATA / "source_nodes.json").read_text())


def test_adopted_energy_and_same_atomic_reference(report):
    check_equal(report, json.loads((study.DATA / "comparison.json").read_text()))
    assert report["balanced_D0_J_mol"] == pytest.approx(
        {"nasa9": 1597918.774, "janaf": 1597916, "atct": 1598180}, abs=1e-6)
    assert report["balanced_D0_J_mol"]["atct"] == pytest.approx((711.381+2*246.844+393.111)*1000)
    for row in report["source_samples"]:
        t = row["temperature_K"]
        assert t*(row["reconstructed_hybrid_ln_K"]-row["raw_nasa_ln_K"]) == pytest.approx(32.51347227819497, abs=2e-9)
        if 1500 <= t <= 4000:
            assert row["curated_ln_K"] == pytest.approx(row["reconstructed_hybrid_ln_K"], abs=3e-12)
        else:
            assert row["curated_ln_K"] is None and row["curated_over_raw_NASA"] is None
    with pytest.raises(ValueError, match="outside"):
        study.curated_ln_k(1000)


def test_nist_tables_obey_caloric_identities_despite_printed_plus(report):
    for row in report["nist_consistency"]["derivative_checks"]:
        # Independent Q/H/S/Cp columns, not a reapplication of the state sum.
        assert row["Cp_R_from_H_derivative"] == pytest.approx(row["Cp_R_published"], abs=2e-6, rel=0)
        assert row["Cp_R_from_S_derivative"] == pytest.approx(row["Cp_R_published"], abs=2e-6, rel=0)
        assert row["H_RT_from_Q_derivative"] == pytest.approx(row["H_RT_published"], abs=1e-7, rel=0)
        if row["temperature_K"] >= 1500:
            assert row["printed_plus_formula_excess_Cp_R"] > 20


def test_natural_isotope_average_is_not_an_unannounced_entropy_shift(nodes, report):
    result = report["nist_consistency"]
    assert result["sum_isotope_weights"] == pytest.approx(1, abs=1e-15)
    assert all(a < b for a, b in zip(result["max_abs_weighted_minus_natural_Cp_S_H"], [1e-8, 3e-7, 6e-8]))
    assert result["mean_log_nuclear_spin_weight"] == pytest.approx(.008778412028604736)
    isotope_mixing_entropy = -sum(x*math.log(x) for x in result["natural_isotope_weights"].values())
    assert isotope_mixing_entropy > .08  # Much larger than any rounding residual.
    for mono, natural in zip(nodes["nist_isotopologues"]["626"], nodes["nist_natural"]):
        if mono[0] >= 1500:
            assert abs(mono[3]-natural[1]) < .000257
    assert math.exp(study.ln_translation(2000, 44.0095)-study.ln_translation(2000, study.MASS_626)) == pytest.approx(
        report["translational_mass_factor_natural_over_626"], rel=2e-14)


def test_nist_entropy_uses_its_published_mass_and_constants(nodes):
    for row in nodes["nist_isotopologues"]["626"]:
        t, q = row[:2]
        kt = 1.38064852e-23*t
        translation = 1.5*math.log(2*math.pi*study.MASS_626*1.66053886e-27*kt/(6.62607004e-34)**2)+math.log(kt/1e5)
        assert row[5]-row[7] == pytest.approx(math.log(q)+translation, abs=1e-8, rel=0)


def test_current_ucl_pf_is_explained_by_energy_cut_not_J_cut(nodes):
    ucl = nodes["exomol"]["UCL-4000"]
    assert ucl["states"] == 3562798 and ucl["J_max"] == 230
    for moment, pf, cp in zip(ucl["rows"], ucl["published_pf"], ucl["published_cp"]):
        if moment["temperature_K"] >= 1500:
            cut = moment["restricted_states"]["E_le_26000"]
            assert cut["Q"]/pf[1] == pytest.approx(1, abs=1.4e-6, rel=0)
            assert abs(cut["Cp_R"]*study.major.R-cp[1]) < .0036
    hottest = ucl["rows"][-1]
    assert hottest["Q"]/ucl["published_pf"][-1][1] > 1.025
    assert hottest["restricted_states"]["J_le_202"]["Q"]/ucl["published_pf"][-1][1] > 1.02
    # NIST Table6 cites the full UCL calculation, rounded/truncated to integers.
    assert abs(ucl["rows"][-3]["Q"]-114728) < 1
    assert abs(ucl["rows"][-1]["Q"]-376550) < 1


def test_dozen_pf_and_moments_expose_high_state_dependence(nodes, report):
    dozen = nodes["exomol"]["Dozen"]
    assert dozen["states"] == 3646814 and dozen["J_max"] == 250
    assert dozen["energy_min_cm-1"] == 0 and dozen["energy_max_cm-1"] < 36000
    for row in report["source_samples"]:
        check = row["exomol"]["Dozen"]
        assert check["state_over_published_Q"] == pytest.approx(1, abs=1.5e-6, rel=0)
        assert check["Cp_R_states"] > 2.5
        if row["temperature_K"] >= 1500:
            assert abs(row["NIST_constant_rescaling_first_order_delta_ln_Q"]) < 1.5e-6
            assert row["NASA_native_R_minus_common_R_Cp_J_mol_K"] < .0004
    hot = report["source_samples"][-1]
    assert .9956 < hot["exomol"]["Dozen"]["state_over_NIST_626_Q"] < .9957
    assert hot["NIST_626_Cp_R"]-hot["exomol"]["Dozen"]["Cp_R_states"] > .30
    assert dozen["rows"][-1]["energy_above_30000_Q_fraction"] > .0074


def test_archived_current_grid_conserves_both_atoms(report):
    result = report["footprint"]
    assert result["cases"] == 75 and result["species"] == 537 and result["all_converged"]
    assert result["dataset"] == "curated-tio-v1" and result["max_abs_potential_difference"] < 3e-12
    with np.load(study.GRID) as grid:
        n = np.exp(grid["ln_n"])
        col = grid["species"].tolist().index("C1O2")
        for symbol in ("C", "O"):
            element = grid["elements"].tolist().index(symbol)
            assert n @ grid["formula_matrix"][element] == pytest.approx(grid["budget"][:, element], rel=3e-13, abs=0)
            point = result["elements"][symbol]
            assert point["case_index"] == 3
            assert point["mole_fraction"] == pytest.approx(n[3, col]/n[3].sum(), rel=2e-12, abs=0)
    assert report["equilibrium_substitution"] == "not_run" and report["production_change"] == "none"


def test_receipt_pins_evidence_and_actual_inputs():
    receipt = json.loads((study.DATA / "receipt.json").read_text())
    for group in ("selected_input_sha256", "output_sha256"):
        for path, digest in receipt[group].items():
            assert sha256((ROOT/path).read_bytes()).hexdigest() == digest, path
