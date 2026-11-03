"""Independent table identities and derivative checks for the H2O candidates."""

from hashlib import sha256
import json
import math
from pathlib import Path
import sys

import numpy as np
from numpy.polynomial import Polynomial
import pytest


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "mceg/reviews"))
import h2o_candidate as candidate


def read(name):
    return json.loads((candidate.DATA / name).read_text())


def test_evaluated_fit_resolves_withheld_Cp_and_independent_H_S_Q():
    report = candidate.validate()
    frozen = read("validation.json")
    # Residuals subtract large caloric values. Use dimensional replay bounds
    # rather than assigning source-value precision to cancellation noise.
    for key, value in report["evaluated_max_absolute_table_errors"].items():
        tolerance = 1e-7 if key in ("G_relative_J_mol", "H_relative_J_mol") else 1e-10
        assert value == pytest.approx(frozen["evaluated_max_absolute_table_errors"][key], abs=tolerance, rel=0)
    assert report["maximum_errors_over_published_expanded_U"] == pytest.approx(
        frozen["maximum_errors_over_published_expanded_U"], abs=2e-6, rel=0)
    assert (report["published_rows"], report["fit_Cp_rows"], report["withheld_Cp_rows"]) == (2501, 251, 2250)
    assert report["evaluated_withheld_Cp_max_error_J_mol_K"] < .0012
    assert report["evaluated_max_absolute_table_errors"]["Cp_J_mol_K"] < .0015
    assert report["maximum_errors_over_published_expanded_U"]["H"] < .49
    assert report["maximum_errors_over_published_expanded_U"]["S"] < .24
    assert report["maximum_errors_over_published_expanded_U"]["Cp"] < .055
    assert report["max_ln_K_difference_from_independent_rounded_Q"] < 1.6e-5
    assert report["evaluated_minimum_Cp_J_mol_K"] > 47.14
    model = candidate.load_candidate()["evaluated"]
    cp = Polynomial(model["Cp_over_R_polynomial_u"])
    h_integral = Polynomial(model["enthalpy_integral_u"])
    s_integral = Polynomial(model["entropy_integral_u"])
    assert h_integral.deriv().coef == pytest.approx(cp.coef, rel=0, abs=1e-15)
    # Analytic entropy integration retains the logarithmic remainder exactly.
    recovered = s_integral.deriv() * Polynomial([2750 / 1250, 1]) + model["entropy_log_coefficient"]
    assert recovered.coef == pytest.approx(cp.coef, rel=0, abs=1e-14)


def test_anchor_units_spin_and_translation_match_independent_source():
    published = json.loads((candidate.source.DATA / "published_samples.json").read_text())
    anchor = next(r for r in published["thermo_2025"] if r["T/K"] == 1500)
    moment = next(r for r in published["partition_2025"] if r["T/K"] == 1500)
    result = candidate.molecular_properties(1500)
    mass_shift = 1.5 * candidate.R * math.log(candidate.MASS_U / candidate.source.ISOTOPE_MASS_U)
    assert result["H_relative_J_mol"] == pytest.approx(anchor["H/kJ/mol"], abs=1e-8, rel=0)
    assert result["S_J_mol_K"] == pytest.approx(anchor["S/J/K mol"] + mass_shift, abs=1e-11, rel=0)
    independently_from_Q = candidate.R * (2.5 + moment["Q'"] / moment["Q"]
        + math.log(moment["Q"] / 4) + candidate.source.translation_log_q(1500, candidate.MASS_U))
    assert result["S_J_mol_K"] == pytest.approx(independently_from_Q, abs=4e-7, rel=0)
    assert candidate.load_candidate()["molecular_mass_u"] == 18.01528
    assert candidate.load_candidate()["D0_J_mol"] == 917819.


def test_discrete_interpolant_retains_level_moments_and_C2_joins():
    prior = json.loads((candidate.source.DATA / "state_samples.json").read_text())["pw2024"]
    for row in read("discrete_nodes.json"):
        t = row["temperature_K"]
        independent = next(r for r in prior["samples"] if r["temperature_K"] == t)
        assert row["Q"] == pytest.approx(independent["Q"], rel=2e-14, abs=0)
        assert row["Cp_J_mol_K"] == pytest.approx(candidate.R * (
            2.5 + independent["variance_energy_K2"] / t**2), abs=1e-10, rel=0)
        for key, value in candidate.molecular_properties(t, "discrete").items():
            assert value == pytest.approx(row[key], rel=0, abs=2e-8)
    record = candidate.load_candidate()["discrete"]
    c = np.asarray(record["coefficients"])
    for left, right in zip(c[:-1], c[1:]):
        assert left.sum() == pytest.approx(right[0], abs=2e-10, rel=0)
        assert np.arange(1, 6) @ left[1:] == pytest.approx(right[1], abs=2e-10, rel=0)
        assert (np.arange(2, 6) * np.arange(1, 5)) @ left[2:] == pytest.approx(2 * right[2], abs=2e-10, rel=0)
    errors = read("validation.json")["discrete_maximum_fit_errors"]
    assert errors["ln_K"] < 8.7e-7 and errors["Cp_J_mol_K"] < .0021
    assert read("validation.json")["discrete_off_knot_points"] == 45


def test_formation_reference_and_thermal_scope_are_separate():
    reference = json.loads(candidate.source.major.REFERENCE_PATH.read_text())
    d0 = candidate.source.major.atomization_kelvin("H2O", "atct", reference)
    assert candidate.load_candidate()["D0_over_k_K"] == d0
    for t in (1500., 1637., 2500., 3333., 4000.):
        atoms = (2 * candidate.source.major.thermal_g_rt("H", t, "nasa9", reference)
                 + candidate.source.major.thermal_g_rt("O", t, "nasa9", reference))
        for thermal in candidate.THERMAL:
            g = candidate.molecular_properties(t, thermal)["G_relative_J_mol"]
            assert candidate.formation_h(t, thermal) == pytest.approx(g / (candidate.R * t) - atoms - d0 / t,
                                                                       abs=1e-12, rel=0)
    hottest = read("validation.json")["thermal_scope_comparison"][-1]
    assert .9820 < hottest["evaluated_over_current_K"] < .9822
    assert 4e-5 < math.log(hottest["evaluated_over_discrete_K"]) < 4.4e-5
    assert .067 < hottest["evaluated_minus_discrete_properties"]["Cp_J_mol_K"] < .069
    for t in (1499., 4001., float("nan"), float("inf")):
        with pytest.raises(ValueError, match="outside"):
            candidate.formation_h(t)
    with pytest.raises(ValueError, match="thermal"):
        candidate.formation_h(2500, "other")


def test_jax_vectorization_and_derivatives_preserve_all_caloric_quantities():
    jax = pytest.importorskip("jax")
    previous = jax.config.x64_enabled
    jax.config.update("jax_enable_x64", True)
    try:
        temperatures = np.array([1500., 1637., 2253., 3000., 3871., 4000.])
        for thermal in candidate.THERMAL:
            function = jax.jit(lambda t: candidate.formation_h_jax(t, thermal))
            assert np.asarray(function(temperatures)) == pytest.approx(
                [candidate.formation_h(t, thermal) for t in temperatures], abs=1e-12, rel=0)
            for t in temperatures:
                props = candidate.molecular_properties(t, thermal)
                _, slope, curvature = candidate._nasa(t, candidate.load_candidate()["atomic_background_nasa9"], np)
                g = lambda x: candidate.R * x * function(x)
                assert float(jax.grad(g)(t)) == pytest.approx(-props["S_J_mol_K"] + candidate.R * slope,
                                                            abs=2e-9, rel=0)
                assert float(-t * jax.grad(jax.grad(g))(t)) == pytest.approx(
                    props["Cp_J_mol_K"] - candidate.R * t * curvature, abs=2e-9, rel=0)
            assert np.isnan(np.asarray(function(np.array([1499., 4001., np.nan])))).all()
    finally:
        jax.config.update("jax_enable_x64", previous)


def test_new_receipt_pins_inputs_without_rewriting_historical_evidence():
    receipt = read("receipt.json")
    for group in ("selected_input_sha256", "output_sha256"):
        for name, digest in receipt[group].items():
            assert sha256((ROOT / name).read_bytes()).hexdigest() == digest, name
    old = json.loads((candidate.source.DATA / "receipt.json").read_text())
    for group in ("selected_input_sha256", "output_sha256"):
        for name, digest in old[group].items():
            assert sha256((ROOT / name).read_bytes()).hexdigest() == digest, name
