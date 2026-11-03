"""Independent tables, conventions and conserved budgets for the H2O review."""

from hashlib import sha256
import json
import math
from pathlib import Path
import re
import sys

import numpy as np
import pytest


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "mceg/reviews"))
import h2o_source_review as study
from gas_source_comparison import check_equal


@pytest.fixture(scope="module")
def report():
    return study.review()


def test_balanced_anchor_and_actual_hybrid_are_reproduced(report):
    check_equal(report, json.loads((study.DATA / "comparison.json").read_text()))
    assert report["balanced_atomization_J_mol"] == pytest.approx(
        {"atct": 917819., "nasa9": 917782.492, "janaf": 917782.}, abs=1e-6, rel=0)
    assert report["balanced_atomization_J_mol"]["atct"] == pytest.approx(
        (2 * 216.034 + 246.844 + 238.907) * 1000, abs=1e-8, rel=0)
    for row in report["samples"]:
        t = row["temperature_K"]
        if t >= 1500:
            assert row["curated_ln_K"] == pytest.approx(row["reconstructed_hybrid_ln_K"], abs=4e-12, rel=0)
        else:
            assert row["curated_ln_K"] is None
        assert math.log(row["curated_over_raw_NASA"]) * t == pytest.approx(
            report["adopted_minus_native_NASA_D0_over_k_K"], abs=1e-9, rel=0)
    with pytest.raises(ValueError, match="outside"):
        study.curated_ln_k(1000)


def test_fresh_official_nasa_excerpt_matches_frozen_source():
    reference = json.loads(study.major.REFERENCE_PATH.read_text())
    for block in (study.DATA / "nasa9_excerpt.txt").read_text().strip().split("\n\n"):
        lines = block.splitlines()
        record = reference["nasa9_records"][lines[0].split()[0]]
        assert float(lines[1].split()[-1]) == record["hf298_J_mol"]
        for i, interval in enumerate(record["intervals"]):
            offset = 2 + 3 * i
            assert float(lines[offset].split()[-1]) == record["h298_minus_h0_J_mol"]
            numbers = [float(v.replace("D", "E")) for v in re.findall(
                r"[+-]?\d+\.\d+D[+-]\d+", " ".join(lines[offset+1:offset+3]))]
            assert numbers == interval["coefficients"]


def test_full_level_summaries_close_separate_published_data():
    states = json.loads((study.DATA / "state_samples.json").read_text())
    tables = json.loads((study.DATA / "published_samples.json").read_text())
    assert states["pokazatel"]["levels"] == 810269
    assert states["pw2024"]["levels"] == 810270
    for source in states.values():
        assert source["maximum_J"] == 69
        assert source["maximum_energy_cm-1"] > study.FIRST_THRESHOLD_CM
        assert source["para_ground_cm-1"] == 0
        assert source["ortho_ground_cm-1"] == pytest.approx(23.79436, abs=1e-5, rel=0)
        for row in source["samples"]:
            # The two finite empirical/theoretical level sets approach 3:1;
            # neither is forced to an exactly fixed ortho/para population.
            assert row["ortho_over_para"] == pytest.approx(3, abs=1e-5, rel=0)
            assert row["above_first_threshold_Q_fraction"] < 2.5e-8
            assert abs(row["full_minus_cut_Cp_J_mol_K"]) < 3e-5
    for row in states["pokazatel"]["samples"]:
        table = next((p for p in tables["pokazatel"] if p["temperature_K"] == row["temperature_K"]), None)
        if table is None:
            continue  # The actual ExoMol temperature columns have integer nodes.
        assert row["Q"] / table["Q"] == pytest.approx(1, abs=1.2e-6, rel=0)
        props = study.molecular_properties(row["temperature_K"], row, study.ISOTOPE_MASS_U)
        assert abs(props["Cp_J_mol_K"] - table["Cp_J_mol_K"]) < .0035
    for row in states["pw2024"]["samples"]:
        table = next(p for p in tables["partition_2025"] if p["T/K"] == row["temperature_K"])
        # The evaluated upper-temperature table also treats unbound-state effects;
        # direct sums and the table are not relabeled as identical calculations.
        assert abs(row["Q"] - table["Q"]) < table["U_ULB(Q)"]


def test_2025_table_units_spin_and_caloric_identities():
    tables = json.loads((study.DATA / "published_samples.json").read_text())
    for row in tables["partition_2025"]:
        t, q = row["T/K"], row["Q"]
        thermo = next(r for r in tables["thermo_2025"] if r["T/K"] == t)
        mean = row["Q'"] / q
        h = study.R * t * (2.5 + mean)
        cp = study.R * (2.5 + row['Q"'] / q - mean**2)
        s = study.R * (2.5 + mean + math.log(q / 4)
                       + study.translation_log_q(t, study.ISOTOPE_MASS_U))
        # Independent printed moments lose precision at the upper nodes.
        assert h == pytest.approx(thermo["H/kJ/mol"], abs=1., rel=0)
        assert cp == pytest.approx(thermo["C_p/J/K mol"], abs=.0015, rel=0)
        assert s == pytest.approx(thermo["S/J/K mol"], abs=.001, rel=0)
        if t == 298.15:
            assert h == pytest.approx(9904.040932, abs=1e-5, rel=0)
            assert s == pytest.approx(188.828794409, abs=1e-7, rel=0)
        # A mass change affects translation only, not internal degeneracies.
        delta = study.translation_log_q(t, study.NATURAL_MASS_U) - study.translation_log_q(t, study.ISOTOPE_MASS_U)
        assert delta == pytest.approx(1.5 * math.log(study.NATURAL_MASS_U / study.ISOTOPE_MASS_U), abs=2e-14, rel=0)


def test_scoped_thermal_candidate_and_conserved_current_footprint(report):
    row = next(r for r in report["samples"] if r["temperature_K"] == 4000)
    assert .9820 < row["same_anchor_2025_over_current_K_from_H_S"] < .9822
    assert row["evaluated_minus_NASA_Cp_J_mol_K"] < -.34
    assert .067 < row["evaluated_minus_direct_Cp_J_mol_K"] < .069
    result = report["footprint"]
    assert result["dataset"] == "curated-tio-v1" and result["cases"] == 75 and result["species"] == 537
    assert result["all_converged"] and result["maximum_oxygen_share"]["case_index"] == 4
    assert result["maximum_oxygen_share"]["fraction"] == pytest.approx(.9061158949044955, rel=1e-12)
    with np.load(study.GRID) as grid:
        column = grid["species"].tolist().index("H2O1")
        oxygen = grid["elements"].tolist().index("O")
        n = np.exp(grid["ln_n"])
        np.testing.assert_allclose(n @ grid["formula_matrix"][oxygen], grid["budget"][:, oxygen], rtol=1e-10, atol=0)
        for t, h in zip(grid["temperature_K"], grid["h"][:, column]):
            assert -h == pytest.approx(study.curated_ln_k(t), abs=4e-12, rel=0)
        assert result["maximum_mole_fraction"] == pytest.approx((n[:, column] / n.sum(axis=1)).max(), rel=2e-12, abs=0)
    assert report["production_change"] == "none" and report["equilibrium_substitution"] == "not_run"


def test_evidence_hashes_and_extractor_reject_changed_original(tmp_path):
    receipt = json.loads((study.DATA / "receipt.json").read_text())
    for group in ("selected_input_sha256", "output_sha256"):
        for name, digest in receipt[group].items():
            assert sha256((ROOT / name).read_bytes()).hexdigest() == digest, name
    (tmp_path / "pokazatel.states.bz2").write_bytes(b"changed original")
    with pytest.raises(ValueError, match="Changed original source"):
        study.extract_sources(tmp_path)
