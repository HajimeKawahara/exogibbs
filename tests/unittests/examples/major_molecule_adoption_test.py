"""Offline scientific checks of the molecular source-adoption study."""

from collections import Counter
from copy import deepcopy
import hashlib
import importlib.util
import json
import math
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
    study = _load_script("select_major_molecules")

DATA = json.loads(study.REFERENCE_PATH.read_text())
CHOICES = json.loads(study.CHOICES_PATH.read_text())["choices"]


def test_source_excerpts_preserve_species_and_temperature_identity():
    lines = audit.FASTCHEM_PATH.read_text().splitlines()
    for name, record in DATA["species"].items():
        i = record["source_line_1based"] - 1
        assert lines[i].split()[0] == record["fastchem_species"], name
        assert list(map(float, lines[i + 1].split())) == record["coefficients"]
        assert lines[i].split("#", 1)[1].strip() == record["source"]
    assert DATA["species"]["HCN"]["composition"] == DATA["species"]["HNC"]["composition"]
    assert DATA["species"]["HCN"]["fastchem_species"] != DATA["species"]["HNC"]["fastchem_species"]
    assert DATA["atct"]["records"]["HCN"]["atct_id"] != DATA["atct"]["records"]["HNC"]["atct_id"]
    assert "HNC" not in DATA["janaf"]
    assert DATA["janaf"]["AlH"]["table_id"] == "Al-056"
    assert 700 in DATA["temperature_K"] and 750 not in DATA["temperature_K"]
    assert len(DATA["barklem_collet"]) == 24
    assert sum(name in DATA["nasa9_records"] for name in CHOICES) == 26


def test_atct_atomization_is_invariant_under_elemental_enthalpy_gauge():
    shifted = deepcopy(DATA)
    atom_offsets = {"H": 13.1, "C": -29.4, "N": 17.9, "O": 6.2, "S": 51.3, "Si": -9.7}
    for name, record in shifted["atct"]["records"].items():
        composition = DATA["species"][name]["composition"] if name in DATA["species"] else {name: 1}
        record["hf0_kJ_mol"] += sum(n * atom_offsets[el] for el, n in composition.items())
    for name in CHOICES:
        if name in DATA["atct"]["records"]:
            assert study.atomization_kelvin(name, "atct", shifted) == pytest.approx(
                study.atomization_kelvin(name, "atct", DATA), rel=0, abs=1e-9)
    # Ground-state H2 dissociation fixes both the sign and the atom count.
    assert study.atomization_kelvin("H2", "atct", DATA) * study.R == pytest.approx(432068.)


def test_balanced_reaction_energy_closes_across_species():
    # Steam reforming: CH4 + H2O -> CO + 3 H2. Atomic references cancel.
    reaction = {"CH4": -1, "H2O": -1, "CO": 1, "H2": 3}
    atoms = Counter()
    for name, coefficient in reaction.items():
        for element, count in DATA["species"][name]["composition"].items():
            atoms[element] += coefficient * count
    assert all(count == 0 for count in atoms.values())
    delta_h = 1000 * sum(n * DATA["atct"]["records"][name]["hf0_kJ_mol"]
                         for name, n in reaction.items())
    from_d0 = -study.R * sum(n * study.atomization_kelvin(name, "atct", DATA)
                             for name, n in reaction.items())
    assert from_d0 == pytest.approx(delta_h, abs=1e-8)


def test_nasa_update_matches_production_evaluator_and_preserves_calorics():
    from jax import config
    from exogibbs.thermo.standard import _minimum_cp, _nasa9

    previous = config.jax_enable_x64
    config.update("jax_enable_x64", True)
    try:
        for name, choice in CHOICES.items():
            if choice["thermal_source"] != "nasa9":
                continue
            old_energy = study.atomization_kelvin(name, "nasa9", DATA)
            target = study.target_energy(name, choice, DATA)
            for interval in DATA["nasa9_records"][name]["intervals"]:
                original = np.array([interval["coefficients"]])
                updated = original.copy()
                updated[:, 7] += old_energy - target
                assert _minimum_cp(updated[:, :7], tuple(interval["bounds_K"]))[0] > 0
                lo, hi = interval["bounds_K"]
                t = (lo + hi) / 2  # Off the comparison grid in both intervals.
                g0, s0, cp0 = _nasa9(t, original)
                g1, s1, cp1 = _nasa9(t, updated)
                np.testing.assert_array_equal(s0, s1)
                np.testing.assert_array_equal(cp0, cp1)
                np.testing.assert_allclose(g1 - g0, (old_energy - target) / t, atol=1e-12)
                atoms = sum(n * audit.nasa9_g_rt(t, study.nasa_interval(el, t, DATA))
                            for el, n in DATA["species"][name]["composition"].items())
                assert atoms - float(g1[0]) == pytest.approx(
                    study.formation_log_kp(name, t, "nasa9", target, DATA), abs=1e-10)
    finally:
        config.update("jax_enable_x64", previous)


def test_original_nasa_gas_constant_is_required_by_the_energy_convention():
    # H is monatomic with essentially constant Cp; its b1 exposes H(0)/R.
    record = study.nasa_interval("H", 500, DATA)
    h0 = study.assigned_h0("H", "nasa9", DATA)
    assert h0 / study.NASA_R == pytest.approx(record["coefficients"][7], abs=2e-4, rel=0)
    assert abs(h0 / study.R - record["coefficients"][7]) > .1


def test_partition_reconstruction_fixes_pa_units_and_homonuclear_counting():
    errors = []
    for name, record in DATA["barklem_collet"].items():
        for i, t in enumerate(DATA["temperature_K"]):
            from_partition = study.partition_log_kp(name, t, DATA) / audit.LN10
            tabulated = 5 - record["dissociation_log10_pK_Pa"]["values"][i]
            errors.append(abs(from_partition - tabulated))
    # The small residual is recorded, not called exact identity. A wrong
    # pressure unit gives 1 dex; an extra homonuclear factor gives 0.301 dex.
    assert len(errors) == 240 and max(errors) < .006


def test_janaf_formation_columns_cannot_replace_balanced_h_s_for_sh():
    for name in ("H2O", "SH"):
        t = 6000
        composition = DATA["species"][name]["composition"]
        tabulated = 1000 * (sum(n * study.janaf_row(el, t, DATA)[5]
                               for el, n in composition.items())
                           - study.janaf_row(name, t, DATA)[5]) / (study.R * t)
        from_h_s = study.formation_log_kp(
            name, t, "janaf", study.atomization_kelvin(name, "janaf", DATA), DATA, "janaf")
        if name == "H2O":
            assert abs(tabulated - from_h_s) < .001
        else:
            assert abs(tabulated - from_h_s) > 8


def test_common_energy_and_atomic_anchors_isolate_molecular_thermal_difference():
    for name in ("CH4", "HCN", "VO"):
        t = 6000
        energy = study.target_energy(name, CHOICES[name], DATA)
        nasa = study.formation_log_kp(name, t, "nasa9", energy, DATA)
        janaf = study.formation_log_kp(name, t, "janaf", energy, DATA)
        molecular_only = (study.thermal_g_rt(name, t, "janaf", DATA)
                          - study.thermal_g_rt(name, t, "nasa9", DATA))
        assert nasa - janaf == pytest.approx(molecular_only, abs=1e-12)
        assert study.formation_log_kp(name, t, "nasa9", energy + 900, DATA) - nasa == pytest.approx(.15)
    _, decisions = study.compute_adoption()
    by_name = {r["species"]: r for r in decisions}
    assert by_name["HCN"]["nasa9_updated_vs_janaf_updated_max_factor"] < 1.002
    assert by_name["CH4"]["nasa9_updated_vs_janaf_updated_max_factor"] > 2.9
    assert by_name["VO"]["nasa9_updated_vs_janaf_updated_max_factor"] > 1.4


def test_mgh_update_uses_d0_without_claiming_bulk_isotope_precision():
    rows, _ = study.compute_adoption()
    assert DATA["spectroscopy"]["MgH"]["isotopologue"] == "24Mg1H"
    for row in rows:
        if row["species"] == "MgH":
            shift = (1.28517 - 1.271) * study.EV_K / (row["temperature_K"] * audit.LN10)
            assert row["log10_Kp_selected"] - row["log10_Kp_fastchem"] == pytest.approx(shift)
    assert "isotopologue" in DATA["spectroscopy"]["MgH"]["inspection"]
    assert "2013" in CHOICES["MgH"]["follow_up"]


def test_table_nodes_and_tih_window_are_not_silent_extrapolations():
    with pytest.raises(ValueError, match="outside"):
        study.nasa_interval("HCN", 299, DATA)
    with pytest.raises(ValueError, match="outside"):
        study.nasa_interval("H2O", 6001, DATA)
    with pytest.raises(ValueError, match="No JANAF node"):
        study.janaf_row("CH4", 1234, DATA)
    with pytest.raises(ValueError, match="No JANAF node"):
        study.janaf_row("HNC", 1000, DATA)
    rows, decisions = study.compute_adoption()
    assert len(rows) == 350 and len(decisions) == 35
    for row in rows:
        if row["species"] == "TiH":
            assert (row["log10_Kp_selected"] is None) == (not 1200 <= row["temperature_K"] <= 4800)
    assert Counter(r["status"] for r in decisions) == {
        "prefer": 14, "prefer_conditionally": 6, "retain_provisionally": 15}


def test_source_disagreement_is_preserved_instead_of_ranked_away():
    _, decisions = study.compute_adoption()
    by_name = {row["species"]: row for row in decisions}
    assert by_name["CaH"]["fastchem_vs_bc_max_factor"] > 1e9
    assert CHOICES["CaH"]["energy_source"] == "nasa9"
    assert CHOICES["KH"]["status"] == "retain_provisionally"
    assert CHOICES["CH4"]["thermal_source"] == "janaf"
    assert CHOICES["VO"]["energy_source"] == "spectroscopy"
    assert "1.86" in CHOICES["FeH"]["difference_cause"]


def test_receipt_and_contract_identify_scoped_evidence():
    directory = ROOT / "tests/data/validation/major_molecule_adoption"
    receipt = json.loads((directory / "receipt.json").read_text())
    for name, expected in receipt["selected_file_sha256"].items():
        assert hashlib.sha256((ROOT / name).read_bytes()).hexdigest() == expected, name
    for name, expected in receipt["output_sha256"].items():
        assert hashlib.sha256((directory / name).read_bytes()).hexdigest() == expected, name
    contract = (ROOT / "mceg/contracts/major_molecule_adoption.yaml").read_text()
    key = "examples/comparisons/select_major_molecules.py"
    assert "revision: sha256:" + receipt["selected_file_sha256"][key] in contract
    assert "kind: physical_validation\n  status: unknown" in contract
    assert "tests/data/validation/major_molecule_adoption/tests.json" in contract
    assert "No MCEG runtime or test dependency" in contract
