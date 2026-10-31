"""Check that cold-network interventions isolate chemistry and preserve bounds."""

from pathlib import Path
import runpy

import jax
import numpy as np
import pytest


ROOT = Path(__file__).resolve().parents[2]
STUDY = runpy.run_path(str(ROOT / "mceg/reviews/phosphorus_cold_validation_20261031.py"))


def test_energy_probes_keep_entropy_heat_capacity_and_all_other_species():
    previous = jax.config.x64_enabled
    jax.config.update("jax_enable_x64", True)
    try:
        cold = STUDY["cold"]
        data = cold.load_data()
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
        assert data == cold.load_data()
    finally:
        jax.config.update("jax_enable_x64", previous)


def test_removals_preserve_background_and_reject_out_of_range_temperatures():
    data = STUDY["cold"].load_data()
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
        data = STUDY["cold"].load_data()
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
