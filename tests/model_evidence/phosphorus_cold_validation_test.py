"""Check that cold-network interventions isolate chemistry and preserve bounds."""

import gzip
import json
from pathlib import Path
import runpy

import jax
import numpy as np
import pytest

from ._phosphorus_test_data import synthetic_data


ROOT = Path(__file__).resolve().parents[2]
STUDY = runpy.run_path(str(ROOT / "mceg/reviews/phosphorus_cold_validation_20261031.py"))
RECEIPTS = ROOT / "tests/data/validation/phosphorus_cold_validation"


def test_energy_probes_keep_entropy_heat_capacity_and_all_other_species():
    previous = jax.config.x64_enabled
    jax.config.update("jax_enable_x64", True)
    try:
        cold = STUDY["cold"]
        data = synthetic_data()
        names = cold.networks(data)["CNS_full_P"][0]
        t = np.array([200., 298.15, 999., 1000., 1001., 1500.])
        baseline = cold.thermodynamics(data, names, t)
        for species, offset in (("H3PO4", -21.053888), ("HPO2", 24.2672), ("HPO", -86.224041)):
            changed = STUDY["energy_shift"](data, species, offset)
            evaluated = cold.thermodynamics(changed, names, t)
            np.testing.assert_array_equal(evaluated[1], baseline[1])
            np.testing.assert_array_equal(evaluated[2], baseline[2])
            expected = np.zeros(np.shape(baseline[0]))
            expected[:, names.index(species)] = offset * 1000.
            np.testing.assert_allclose((evaluated[0] - baseline[0]) * t[:, None]
                                       * data["source_gas_constant_J_mol_K"], expected, atol=3e-9, rtol=0)
            assert changed["records"]["P4O6"] == data["records"]["P4O6"]
        assert data == synthetic_data()
    finally:
        jax.config.update("jax_enable_x64", previous)


def test_removals_preserve_background_and_reject_out_of_range_temperatures():
    data = synthetic_data()
    scenarios = STUDY["interventions"](data)
    baseline = scenarios["full"]["species"]
    for scenario in scenarios.values():
        if scenario["kind"] == "species_removal":
            assert scenario["species"] == [s for s in baseline if s not in scenario["removed"]]
            assert scenario["data"] is data
    for bad_t in (199.99, 1500.01, np.nan):
        with pytest.raises(ValueError, match="200-1500 K"):
            STUDY["solve_case"](data, baseline, 1., np.array([bad_t]), np.array([.001]))


def test_pressure_grid_and_source_removal_obey_conservation():
    previous = jax.config.x64_enabled
    jax.config.update("jax_enable_x64", True)
    try:
        data = synthetic_data()
        scenario = STUDY["interventions"](data)["without_HOPO"]
        result = STUDY["solve_case"](data, scenario["species"], 1., np.array([500., 500.]),
                                     np.array([1e-6, 1.]))
        assert result["accepted"] == [True, True]
        assert "HPO2" not in result["P_atom_fractions"]
        np.testing.assert_allclose(np.sum(list(result["P_atom_fractions"].values()), axis=0),
                                   1., atol=1e-5, rtol=0)
        assert np.isfinite(result["element_chemical_potential_RT"]).all()
        assert not np.allclose(result["log_mole_fractions"][0], result["log_mole_fractions"][1])
    finally:
        jax.config.update("jax_enable_x64", previous)


def test_archived_gas_comparisons_preserve_phosphorus_inventory():
    gas = json.loads(gzip.decompress((RECEIPTS / "gas_summary.json.gz").read_bytes()))
    comparison = json.loads(gzip.decompress((RECEIPTS / "ggchem_comparison.json.gz").read_bytes()))
    assert gas["cases_passed"] == gas["cases_total"] == 448
    assert comparison["cases_passed"] == comparison["cases_total"] == 448
    assert gas["cases"].keys() == comparison["cases"].keys()
    total = 0
    for name, case in gas["cases"].items():
        reference = comparison["cases"][name]
        total += len(case["temperature_K"])
        assert all(case["accepted"]) and all(reference["accepted"])
        np.testing.assert_array_equal(case["temperature_K"], reference["temperature_K"])
        np.testing.assert_array_equal(case["pressure_bar"], reference["pressure_bar"])
        fractions = np.asarray(list(case["P_atom_fractions"].values()))
        assert np.isfinite(fractions).all() and np.all(fractions >= 0)
        np.testing.assert_allclose(fractions.sum(axis=0), 1., rtol=0, atol=1e-8)
        for metric, threshold in comparison["gates"].items():
            values = np.asarray(reference["metrics"][metric])
            assert np.isfinite(values).all() and np.all(values < threshold)
    assert total == 448


def test_archived_condensation_comparison_reconstructs_full_element_budgets():
    report = json.loads(gzip.decompress((RECEIPTS / "condensate_comparison.json.gz").read_bytes()))
    assert report["states_passed"] == len(report["states"]) == 31
    matrix = np.asarray(report["gas_formula_matrix"])
    phase_matrix = np.asarray(report["condensate_formula_matrix"])
    budget = np.asarray(report["budget"])
    for state in report["states"]:
        assert state["passed"]
        ln_n = np.asarray(state["gas_log_amounts"])
        amounts = np.asarray(state["condensate_amounts_per_H"])
        assert np.isfinite(ln_n).all() and np.all(amounts >= 0)
        recovered = matrix @ np.exp(ln_n) + phase_matrix @ amounts
        np.testing.assert_allclose(recovered / budget, 1., rtol=0,
                                   atol=report["gates"]["relative_element"])
        logx = ln_n - np.logaddexp.reduce(ln_n)
        np.testing.assert_allclose(logx, state["gas_logx"], rtol=0, atol=1e-11)
        # Native trace densities use the explicitly documented atomic reconstruction.
        np.testing.assert_allclose(logx, state["native_gas_logx_reconstructed"],
                                   rtol=0, atol=report["gates"]["delta_logx"])
        difference = amounts - np.asarray(state["native_condensate_amounts_per_H"])
        assert np.max(np.abs(phase_matrix * difference) / budget[:, None]) < report["gates"]["delta_condensate_element_fraction"]
