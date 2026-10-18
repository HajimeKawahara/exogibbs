"""Regression checks for source adoption, without importing MCEG."""

from collections import Counter
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
from unittest.mock import patch

import numpy as np
import pytest


ROOT = Path(__file__).resolve().parents[3]


def _load_script(name):
    spec = importlib.util.spec_from_file_location(
        name, ROOT / "examples/comparisons" / (name + ".py"))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


audit = _load_script("audit_atomic_ions")
with patch.dict(sys.modules, {"audit_atomic_ions": audit}):
    adoption = _load_script("select_atomic_ions")

REFERENCE = json.loads(audit.REFERENCE_PATH.read_text())
DATA = json.loads(adoption.DATA_PATH.read_text())
NASA = json.loads(audit.NASA_PATH.read_text())["records"]
NASA.update(REFERENCE["nasa9_supplement"])


def test_cds_primary_excerpt_matches_the_audit():
    assert DATA["temperature_K"] == REFERENCE["temperature_K"]
    assert len(DATA["cds_partition_records"]) == 54
    for element, atom in REFERENCE["atoms"].items():
        for stage, field in (("I", "partition_neutral"), ("II", "partition_cation")):
            assert DATA["cds_partition_records"][element + "_" + stage]["values"] == atom[field]
    assert len(DATA["nasa9_enthalpy_records"]) == 75
    for key, energy in DATA["nasa9_enthalpy_records"].items():
        assert energy["source_name"] == NASA[key]["source_name"]
        assert energy["source_line_1based"] == NASA[key]["source_line_1based"]


def test_ground_state_anchor_uses_source_gas_constant_and_both_charge_signs():
    # Constant-Cp H+, H- and electron independently expose NASA's original R.
    # Using the modern R here changes the inferred anchor by about one kelvin.
    for key in ("H1+", "H1-", "e-"):
        energy = DATA["nasa9_enthalpy_records"][key]
        h0 = energy["hf298_J_mol"] - energy["h298_minus_h0_J_mol"]
        assert h0 / adoption.NASA_GAS_CONSTANT == pytest.approx(NASA[key]["coefficients"][7], abs=5e-5, rel=0)
    for charge in (-1, 1):
        species = "H1+" if charge == 1 else "H1-"
        shift = adoption.enthalpy_shift_kelvin("H", charge, REFERENCE, DATA)
        updated = adoption.updated_nasa9_record(NASA[species], shift)
        for t in REFERENCE["temperature_K"]:
            ln_k = (audit.nasa9_g_rt(t, NASA["H"]) - audit.nasa9_g_rt(t, updated)
                    - charge * audit.nasa9_g_rt(t, NASA["e-"]))
            assert abs(ln_k - audit.reference_log_kp(t, "H", charge, REFERENCE)) < .001
    with pytest.raises(ValueError, match="bound-state"):
        adoption.enthalpy_shift_kelvin("N", -1, REFERENCE, DATA)
    with pytest.raises(ValueError, match="singly"):
        adoption.enthalpy_shift_kelvin("H", 2, REFERENCE, DATA)


def test_energy_update_preserves_entropy_and_heat_capacity_between_nodes():
    from jax import config
    from exogibbs.thermo.standard import _minimum_cp, _nasa9

    _, decisions = adoption.compute_adoption()
    keys = [row["species"] for row in decisions if row["selected_family"] == "nasa9_threshold_updated"]
    original = np.array([NASA[key]["coefficients"] for key in keys])
    shifted = np.array([
        adoption.updated_nasa9_record(NASA[row["species"]], row["ion_b1_shift_K"])["coefficients"]
        for row in decisions if row["selected_family"] == "nasa9_threshold_updated"
    ])
    # The production NASA evaluator checks the scientific identity, including
    # off-grid temperatures; this is not a second copy of the update formula.
    previous_x64 = config.jax_enable_x64
    config.update("jax_enable_x64", True)
    try:
        for t in (1000., 1234., 2727., 4567., 6000.):
            g0, s0, cp0 = _nasa9(t, original)
            g1, s1, cp1 = _nasa9(t, shifted)
            np.testing.assert_allclose(g1 - g0, (shifted[:, 7] - original[:, 7]) / t, atol=2e-13)
            np.testing.assert_array_equal(s1, s0)
            np.testing.assert_array_equal(cp1, cp0)
    finally:
        config.update("jax_enable_x64", previous_x64)
    # Analytic extrema over the whole interval, not only a sampled Cp check.
    minima = _minimum_cp(shifted[:, :7], (1000, 6000))
    assert np.all(minima >= 2.49)
    assert keys[int(np.argmin(minima))] == "Ca1+"


def test_all_species_have_a_scoped_choice_without_self_validation():
    rows, decisions = adoption.compute_adoption()
    assert len(decisions) == 49 and len(rows) == 343
    assert Counter(row["selected_family"] for row in decisions) == {
        "nasa9_threshold_updated": 43, "spectroscopic_partition": 4, "exclude_unbound": 2}
    assert {row["species"] for row in decisions if row["selected_family"] == "exclude_unbound"} == {"N1-", "Zn1-"}
    assert {row["species"] for row in decisions if row["selected_family"] == "spectroscopic_partition"} == {"Co1+", "C1-", "Ca1-", "Ge1-"}
    assert {row["species"] for row in decisions if row["condition"] == "high_temperature_density_review"} == {"Na1+", "Na1-", "K1+", "K1-"}
    for row in rows:
        if row["selected_family"] == "spectroscopic_partition":
            assert row["comparison_kind"] == "reference_by_construction"
        if row["selected_family"] == "exclude_unbound":
            assert row["log10_Kp_selected"] is None
            assert row["log10_Kp_threshold_updated"] is None


def test_threshold_updates_resolve_energy_errors_but_not_partition_errors():
    _, decisions = adoption.compute_adoption()
    by_species = {row["species"]: row for row in decisions}
    for species in ("Ni1+", "Cu1+", "Ti1+", "V1+", "Cr1-"):
        assert by_species[species]["threshold_updated_nasa9_max_factor"] < 1.006
    for species in ("C1-", "Ge1-"):
        assert by_species[species]["threshold_updated_nasa9_max_factor"] > 2
    assert by_species["Co1+"]["threshold_updated_nasa9_max_factor"] > 1.06
    ordinary = [row for row in decisions if row["condition"] == "candidate_1000_6000_K"]
    assert len(ordinary) == 39
    assert max(row["threshold_updated_nasa9_max_factor"] for row in ordinary) < 1.014


def test_evidence_identifies_the_evaluated_files_and_scientific_limits():
    directory = ROOT / "results/atomic_ion_adoption"
    receipt = json.loads((directory / "receipt.json").read_text())
    for name, expected in receipt["selected_file_sha256"].items():
        assert hashlib.sha256((ROOT / name).read_bytes()).hexdigest() == expected, name
    for name, expected in receipt["output_sha256"].items():
        assert hashlib.sha256((directory / name).read_bytes()).hexdigest() == expected, name
    contract = (ROOT / "mceg/contracts/atomic_ion_adoption.yaml").read_text()
    script_hash = receipt["selected_file_sha256"]["examples/comparisons/select_atomic_ions.py"]
    assert "revision: sha256:" + script_hash in contract
    assert "kind: physical_validation\n  status: unknown" in contract
    assert "results/atomic_ion_adoption/receipt.json" in contract
    assert "results/atomic_ion_adoption/tests.json" in contract
