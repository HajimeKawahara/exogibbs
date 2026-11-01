"""Protect the SSO source diagnosis without a runtime or MCEG dependency."""

from copy import deepcopy
import json
import math
from pathlib import Path
import runpy

import pytest

ROOT = Path(__file__).resolve().parents[2]
STUDY = runpy.run_path(str(ROOT / "mceg/reviews/s2o_source_investigation.py"))
SNAPSHOT = STUDY["SNAPSHOT"]


def test_s2o_official_records_replay_and_preserve_baseline():
    result = STUDY["investigate"]()
    STUDY["check_equal"](result, json.loads((SNAPSHOT / "comparison.json").read_text()))
    receipt = json.loads((SNAPSHOT / "receipt.json").read_text())
    STUDY["verify_hashes"](receipt["input_sha256"])
    STUDY["verify_hashes"](receipt["output_sha256"])
    assert result["official_janaf_matching_records"] == 3
    assert result["official_nasa_matching_coefficients"] == 18
    row = result["samples"][2]
    assert row["janaf_internal_difference_dex"] == pytest.approx(4.468655846141134)
    assert row["janaf_over_nasa"] == pytest.approx(1.0130543827051879)
    assert row["curated_over_nasa"] == pytest.approx(0.923283142225359, rel=1e-8)


def test_s2o_anchor_and_thermal_terms_close_separately():
    for row in STUDY["investigate"]()["samples"]:
        terms = row["molecular_delta_g_kJ_mol"]
        assert terms["anchor"] == pytest.approx(1.094)
        assert terms["total"] == pytest.approx(sum(terms[k] for k in ("anchor", "enthalpy_increment", "minus_T_delta_S")))
        assert row["cp_lodders_minus_janaf_J_mol_K"] == 0
        expected = math.exp(-1000 * terms["total"] / (8.31446261815324 * row["temperature_K"]))
        assert row["lodders_with_janaf_atoms_over_janaf"] == pytest.approx(expected, abs=2e-14)
    # Curated values below its declared domain stay missing.
    assert all(s["ln_Kp_curated"] is None for s in STUDY["investigate"]()["samples"][:2])


def test_s2o_atct_energy_uses_both_sulfur_atoms_and_preserves_unknowns():
    result = STUDY["investigate"]()
    assert result["atct_atomization_0K_kJ_mol_from_rounded_values"] == pytest.approx(851.124)
    assert result["atct_minus_lodders_atomization_kJ_mol"] == pytest.approx(1.124)
    assert result["assessment"]["full_physical_accuracy"] == "unknown"
    assert result["assessment"]["equilibrium_abundance_impact"] == "not_run"


def test_s2o_hash_check_rejects_changed_official_record():
    changed = deepcopy(json.loads((SNAPSHOT / "receipt.json").read_text())["input_sha256"])
    key = str((SNAPSHOT / "O-011.txt").relative_to(ROOT))
    changed[key] = "0" * 64
    with pytest.raises(ValueError, match="SHA256 mismatch"):
        STUDY["verify_hashes"](changed)
