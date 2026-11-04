"""Check the OH candidate against archived sources and thermodynamic identities."""

import json
import math
from pathlib import Path
import sys

import numpy as np
import pytest


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "mceg/reviews"))
import oh_candidate as candidate


@pytest.fixture(scope="module")
def report():
    return candidate.validate()


def test_rebuild_preserves_sources_and_changes_only_the_balanced_energy():
    model = candidate.load_candidate()
    assert candidate.build() == model
    assert model["D0_J_mol"] == 425604.
    assert model["nasa_D0_J_mol"] == pytest.approx(425785.9, abs=1e-9, rel=0)
    assert model["ATcT_minus_nasa_D0_over_k_K"] == pytest.approx(-21.585709716324345, abs=1e-10, rel=0)
    records = json.loads(candidate.INPUTS[0].read_text())["nasa9_records"]
    coefficients = {s: np.array(next(r["coefficients"] for r in records[s]["intervals"]
                    if r["bounds_K"] == [1000., 6000.])) for s in ("OH", "H", "O")}
    original = coefficients["OH"] - coefficients["H"] - coefficients["O"]
    difference = np.asarray(model["formation_nasa9_coefficients"]) - original
    assert np.delete(difference, 7) == pytest.approx(np.zeros(8), abs=0, rel=0)
    assert difference[7] == pytest.approx(21.585709716324345, abs=1e-10, rel=0)
    # Using modern R for the NASA anchor would silently change this offset.
    wrong_shift = (model["D0_J_mol"] - model["nasa_D0_J_mol"]) / candidate.R
    assert abs(wrong_shift - model["ATcT_minus_nasa_D0_over_k_K"]) > .29


def test_dense_independent_reconstruction_and_jax_derivatives(report):
    assert (report["temperature_count"], report["step_K"], report["temperature_range_K"]) == (2501, 1, [1500, 4000])
    assert report["maximum_absolute_formation_h_reconstruction_error"] < 1e-12
    errors = report["maximum_absolute_molecular_property_reconstruction_errors"]
    assert errors["G_relative_J_mol"] < 1e-8
    assert errors["H_relative_J_mol"] < 1e-8
    assert errors["S_J_mol_K"] < 1e-10
    assert errors["Cp_J_mol_K"] < 1e-10
    jax_errors = report["jax_float64_maximum_absolute_errors"]
    assert jax_errors["formation_h"] < 1e-12
    assert jax_errors["reaction_H_J_mol"] < 1e-8
    assert jax_errors["reaction_Cp_J_mol_K"] < 1e-10


def test_archived_validation_replays_with_cancellation_aware_bounds(report):
    frozen = json.loads((candidate.DATA / "validation.json").read_text())
    assert report["mythos_over_nasa_K"] == pytest.approx(frozen["mythos_over_nasa_K"], abs=1e-12, rel=0)
    for actual, expected in zip(report["thermal_comparison"], frozen["thermal_comparison"]):
        for key in ("temperature_K", "nasa_ATcT_over_current_K", "mythos_ATcT_over_current_K", "mythos_over_nasa_K"):
            assert actual[key] == pytest.approx(expected[key], abs=1e-12, rel=0)
        assert actual["mythos_minus_nasa_molecular_properties"] == pytest.approx(
            expected["mythos_minus_nasa_molecular_properties"], abs=1e-8, rel=0)
    assert report["production_change"] == "none"


def test_mythos_control_retains_spin_zero_and_independent_source_calorics(report):
    energy, weights = candidate._states()
    assert len(energy) == 10606 and energy.min() == 0
    # Resolve the 0.0554 cm-1 parity partner when probing the ground weight.
    # This checks source weights, not extrapolation of the bounded candidate.
    assert np.sum(weights * np.exp(-energy / .001)) == pytest.approx(4., rel=1e-10)
    assert candidate.load_candidate()["molecular_mass_u"] == 17.00734
    previous = json.loads((candidate.source.DATA / "comparison.json").read_text())
    for row in previous["source_samples"]:
        t = row["temperature_K"]
        if t < 1500:
            continue
        independent = row["modern"]["MYTHOS"]
        actual = candidate.molecular_properties(t, "mythos")
        assert actual["H_relative_J_mol"] == pytest.approx(independent["H_increment_J_mol"], abs=1e-8, rel=0)
        assert actual["S_J_mol_K"] == pytest.approx(independent["S_natural_mass_J_mol_K"], abs=1e-10, rel=0)
        assert actual["Cp_J_mol_K"] == pytest.approx(independent["Cp_J_mol_K"], abs=1e-10, rel=0)
        assert candidate.formation_h(t, "mythos") == pytest.approx(-independent["ATcT_same_atoms_ln_K"], abs=1e-12, rel=0)
        assert candidate.formation_h(t) == pytest.approx(-row["nasa_ATcT_ln_K"], abs=1e-12, rel=0)
    assert report["mythos_over_nasa_K"]["maximum_absolute_relative_difference"] < .000976
    assert report["mythos_over_nasa_K"]["minimum_temperature_K"] == 4000


def test_both_thermal_paths_insert_D0_once_and_keep_the_same_atomic_gauge():
    reference = json.loads(candidate.source.major.REFERENCE_PATH.read_text())
    d0 = candidate.load_candidate()["D0_over_k_K"]
    for t in (1500., 1637., 2253., 3000., 3871., 4000.):
        atoms = sum(candidate.source.major.thermal_g_rt(s, t, "nasa9", reference) for s in ("H", "O"))
        for thermal in candidate.THERMAL:
            molecular_g = candidate.molecular_properties(t, thermal)["G_relative_J_mol"]
            assert candidate.formation_h(t, thermal) == pytest.approx(
                molecular_g / (candidate.R * t) - atoms - d0 / t, abs=1e-12, rel=0)
        ratio = math.exp(candidate.formation_h(t) - candidate.formation_h(t, "mythos"))
        molecule_ratio = math.exp((candidate.molecular_properties(t)["G_relative_J_mol"]
            - candidate.molecular_properties(t, "mythos")["G_relative_J_mol"]) / (candidate.R * t))
        assert ratio == pytest.approx(molecule_ratio, abs=1e-13, rel=0)


def test_jax_array_vmap_and_mythos_derivatives_match_independent_atom_calorics():
    import jax

    records = json.loads(candidate.INPUTS[0].read_text())["nasa9_records"]
    headers = json.loads(candidate.INPUTS[2].read_text())["nasa_headers"]
    temperatures = np.array([1500., 1637., 2253., 3000., 3871., 4000.])
    with jax.experimental.enable_x64():
        for thermal in candidate.THERMAL:
            f = jax.jit(lambda t: candidate.formation_h_jax(t, thermal))
            expected = [candidate.formation_h(t, thermal) for t in temperatures]
            assert np.asarray(f(temperatures)) == pytest.approx(expected, abs=1e-12, rel=0)
            assert np.asarray(jax.jit(jax.vmap(f))(temperatures)) == pytest.approx(expected, abs=1e-12, rel=0)
            assert float(f(2500)) == pytest.approx(candidate.formation_h(2500, thermal), abs=1e-12, rel=0)
            g = lambda t: candidate.R * t * f(t)
            first = jax.jit(jax.vmap(jax.grad(g)))(temperatures)
            second = jax.jit(jax.vmap(jax.grad(jax.grad(g))))(temperatures)
            for t, slope, curvature in zip(temperatures, first, second):
                atoms = np.zeros(3)
                for s in ("H", "O"):
                    h, entropy, cp = candidate.source.nasa_properties(records[s], t, candidate.source.major.NASA_R)
                    h -= headers[s]["hf298_J_mol"] - headers[s]["h298_minus_h0_J_mol"]
                    atoms += np.array([h, entropy, cp]) * candidate.R / candidate.source.major.NASA_R
                mol = candidate.molecular_properties(t, thermal)
                reaction_h = mol["H_relative_J_mol"] - atoms[0] - candidate.load_candidate()["D0_J_mol"]
                assert float(g(t) - t * slope) == pytest.approx(reaction_h, abs=1e-8, rel=0)
                assert float(-slope) == pytest.approx(mol["S_J_mol_K"] - atoms[1], abs=1e-10, rel=0)
                assert float(-t * curvature) == pytest.approx(mol["Cp_J_mol_K"] - atoms[2], abs=1e-10, rel=0)


def test_declared_domain_and_unknown_thermal_selector():
    import jax

    invalid = [1499., 4001., float("nan"), float("inf"), -float("inf")]
    for thermal in candidate.THERMAL:
        for t in invalid:
            for function in (candidate.formation_h, candidate.molecular_properties):
                with pytest.raises(ValueError, match="outside"):
                    function(t, thermal)
        with jax.experimental.enable_x64():
            f = jax.jit(lambda t: candidate.formation_h_jax(t, thermal))
            assert np.isnan(np.asarray(f(np.array(invalid)))).all()
            assert np.isfinite(np.asarray(f(np.array([1500., 4000.])))).all()
    for function in (candidate.formation_h, candidate.molecular_properties, candidate.formation_h_jax):
        with pytest.raises(ValueError, match="thermal"):
            function(2500., "other")
