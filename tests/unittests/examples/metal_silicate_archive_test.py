"""Archive chemistry must be rechecked from amounts, including rejected roots."""

from __future__ import annotations

import hashlib
import importlib
import json
from pathlib import Path
import shutil
import sys

import numpy as np
import pytest


ROOT = Path(__file__).resolve().parents[3]
EXAMPLES = ROOT / "examples" / "metal_silicate"
ARCHIVE = ROOT / "results" / "subneptune_taxonomy" / "20260911"


def _load_verifier():
    names = tuple(path.stem for path in EXAMPLES.glob("*.py"))
    previous = {name: sys.modules.get(name) for name in names}
    old_path = sys.path[:]
    try:
        sys.path.insert(0, str(EXAMPLES))
        for name in names:
            sys.modules.pop(name, None)
        return importlib.import_module("revalidate_archive")
    finally:
        sys.path[:] = old_path
        for name, module in previous.items():
            if module is None:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = module


@pytest.fixture(scope="module")
def verifier():
    return _load_verifier()


@pytest.fixture
def archive_copy(tmp_path):
    return shutil.copytree(ARCHIVE, tmp_path / "archive")


def _read(archive, name):
    return json.loads((archive / name).read_text())


def _replace_and_rehash(archive, name, record):
    path = archive / name
    path.write_text(json.dumps(record, indent=2) + "\n")
    receipt = _read(archive, "receipt.json")
    receipt["file_sha256"][name] = hashlib.sha256(path.read_bytes()).hexdigest()
    (archive / "receipt.json").write_text(json.dumps(receipt, indent=2) + "\n")


def _displace(amounts, species, formula, reaction):
    original = np.asarray(amounts)
    changed = original + 0.001 * np.array([reaction.get(name, 0) for name in species])
    assert np.all(changed > 0)
    np.testing.assert_allclose(np.asarray(formula) @ changed,
                               np.asarray(formula) @ original, rtol=1e-14, atol=1e-13)
    return changed


def _update_phase_amounts(state, amounts, species, phases, formula):
    indices = [[species.index(name) for name in phase] for phase in phases.values()]
    state["phase_amounts_mol"] = [float(amounts[items].sum()) for items in indices]
    state["phase_element_amounts_mol"] = [
        (np.asarray(formula)[:, items] @ amounts[items]).tolist() for items in indices
    ]


def test_saved_archive_revalidates_without_optional_provider(monkeypatch):
    # An installed provider must not hide an accidental default dependency.
    monkeypatch.setitem(sys.modules, "exoeos", None)
    verifier = _load_verifier()
    report = verifier.revalidate_archive(ARCHIVE)
    cases = report["cases"]
    assert cases["hydrogen"]["chemistry"] == "recomputed"
    assert cases["hydrogen"]["accepted"]
    for index in range(4):
        assert cases[f"source_{index}"]["chemistry"] == "recomputed"
        assert cases[f"source_{index}"]["accepted"]
    stored = _read(ARCHIVE, "sulfide.json")["states"]
    for index, state in enumerate(stored):
        assert cases[f"sulfide_{index}"]["chemistry"] == "recomputed"
        assert cases[f"sulfide_{index}"]["accepted"] is state["accepted"]
    for branch in ("present", "absent"):
        assert cases[f"melts_{branch}"]["chemistry"] == "not_run"
    assert "melts_phase_comparison" not in report


def test_sulfide_reaction_displacement_fails_after_hash_update(verifier, archive_copy):
    record = _read(archive_copy, "sulfide.json")
    state = next(state for state in record["states"] if state["accepted"])
    # REACTIONS[0]: FeO + H2 = Fe + H2O; saved chemical residuals stay unchanged.
    amounts = _displace(
        state["amounts_mol"], record["species"], record["formula_matrix_element_rows"],
        {"FeO_melt": -1, "H2_gas": -1, "Fe_metal": 1, "H2O_gas": 1},
    )
    state["amounts_mol"] = amounts.tolist()
    _replace_and_rehash(archive_copy, "sulfide.json", record)
    with pytest.raises(AssertionError):
        verifier.revalidate_archive(archive_copy)


@pytest.mark.parametrize("network_name", ["carbon", "sulfur_nitrogen"])
def test_source_reaction_displacement_fails_after_hash_update(
    verifier, archive_copy, network_name,
):
    reference = json.loads((EXAMPLES / "sulfur_reference.json").read_text())
    network = reference["networks"][network_name]
    record = _read(archive_copy, "sulfur_carbon_source.json")
    state = next(case for case in record["cases"] if case["model_id"] == network["model_id"])
    species = state["species"]
    formula = [[network["component_formulas"][name].get(element, 0) for name in species]
               for element in state["elements"]]
    amounts = _displace(
        state["component_amounts_mol"], species, formula,
        {"MgO_silicate": -1, "SiO2_silicate": -1, "MgSiO3_silicate": 1},
    )
    state["component_amounts_mol"] = amounts.tolist()
    _update_phase_amounts(state, amounts, species, network["phases"], formula)
    _replace_and_rehash(archive_copy, "sulfur_carbon_source.json", record)
    with pytest.raises(AssertionError):
        verifier.revalidate_archive(archive_copy)


def test_hydrogen_reaction_displacement_fails_after_hash_update(verifier, archive_copy):
    record = _read(archive_copy, "hydrogen.json")
    ledger, state = record["ledger"], record["state"]
    amounts = _displace(
        state["component_amounts_mol"], ledger["components"], ledger["formula_matrix"],
        {"MgO_silicate": -1, "SiO2_silicate": -1, "MgSiO3_silicate": 1},
    )
    state["component_amounts_mol"] = amounts.tolist()
    _update_phase_amounts(state, amounts, ledger["components"], ledger["phases"],
                          ledger["formula_matrix"])
    _replace_and_rehash(archive_copy, "hydrogen.json", record)
    with pytest.raises(AssertionError):
        verifier.revalidate_archive(archive_copy)


@pytest.mark.parametrize("recorded_acceptance", [False, True])
def test_recomputed_scss_acceptance_must_match_both_saved_statuses(
    verifier, archive_copy, recorded_acceptance,
):
    record = _read(archive_copy, "sulfide.json")
    state = next(state for state in record["states"]
                 if state["accepted"] is recorded_acceptance)
    state["accepted"] = not recorded_acceptance
    _replace_and_rehash(archive_copy, "sulfide.json", record)
    with pytest.raises(AssertionError):
        verifier.revalidate_archive(archive_copy)


def test_unchanged_receipt_detects_tampered_archive(verifier, archive_copy):
    path = archive_copy / "hydrogen.json"
    path.write_text(path.read_text() + "\n")
    with pytest.raises(AssertionError):
        verifier.revalidate_archive(archive_copy)


def test_component_metadata_mismatch_fails_after_hash_update(verifier, archive_copy):
    record = _read(archive_copy, "sulfide.json")
    record["species"][0], record["species"][1] = record["species"][1], record["species"][0]
    _replace_and_rehash(archive_copy, "sulfide.json", record)
    with pytest.raises(AssertionError):
        verifier.revalidate_archive(archive_copy)


@pytest.mark.parametrize("section,key,value", [
    ("ledger", "R_J_mol_K", 1.0),
    ("ledger", "standard_pressure_bar", 1e5),
    ("ledger", "metal_model_id", "unrelated_metal_model"),
    ("ledger", "amount_unit", "kg"),
    ("ledger", "mixing", "ideal metal"),
    ("ledger", "standards", "zero species standards"),
    ("ledger", "phases", {}),
    ("h2_calibration", "model_id", "unrelated_solubility_model"),
    ("h2_calibration", "equation", "x_H2 = 0"),
    ("h2_calibration", "concentration_basis", "mass fraction"),
    ("h2_calibration", "fugacity_basis", "Pa"),
    ("h2_calibration", "pressure_basis", "bar"),
])
def test_hydrogen_model_or_basis_mismatch_fails_after_hash_update(
    verifier, archive_copy, section, key, value,
):
    record = _read(archive_copy, "hydrogen.json")
    record[section][key] = value
    _replace_and_rehash(archive_copy, "hydrogen.json", record)
    with pytest.raises(AssertionError):
        verifier.revalidate_archive(archive_copy)


def test_rejected_sulfide_branches_cannot_be_replaced_by_accepted_copies(
    verifier, archive_copy,
):
    record = _read(archive_copy, "sulfide.json")
    accepted = next(state for state in record["states"] if state["accepted"])
    record["states"] = [accepted] * len(record["states"])
    _replace_and_rehash(archive_copy, "sulfide.json", record)
    with pytest.raises(AssertionError):
        verifier.revalidate_archive(archive_copy)
