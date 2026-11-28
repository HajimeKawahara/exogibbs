"""Offline provider comparisons against the imported, fixed Planetlib records."""

from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pytest

pytest.importorskip("mceg.schema_v06")
pytest.importorskip("sympy")

from mceg import load_contract, mathematical_model_digest, to_sympy
from mceg.equation_checks import equation_checks
from mceg.integrity import file_integrity

from exogibbs.solubility.sun_yao2026 import h2o_sun_yao2026
from exogibbs.thermo.ideal_gibbs import R, gibbs_j
from exogibbs.thermo.standard import _nasa9


ROOT = Path(__file__).resolve().parents[2]
CONTRACT = load_contract(ROOT / "mceg/planetlib/contract.yaml")
EQUATIONS = {e.id: to_sympy(e.sympy) for e in CONTRACT.equations}


def value(identifier, **values):
    """Evaluate a stored RHS through MCEG's codec, without a formula registry."""
    expression = EQUATIONS[identifier].rhs
    assert {s.name for s in expression.free_symbols} <= values.keys()
    return float(expression.subs({s: values[s.name] for s in expression.free_symbols}))


def test_native_records_and_local_pins():
    import json

    lock = json.loads((ROOT / "mceg/planetlib/imports.lock.json").read_text())
    for entry in lock["imports"]:
        pin = entry["model"]
        assert mathematical_model_digest(CONTRACT, pin["id"], pin["revision"]) == pin["sha256"]
    assert len(CONTRACT.mathematical_models) == 3
    assert len(CONTRACT.equations) == 9
    assert all(e.sympy is not None for e in CONTRACT.equations)
    assert all(a.status == "unknown" for a in CONTRACT.assessments)
    for collection in (CONTRACT.subjects, CONTRACT.contexts, CONTRACT.claims, CONTRACT.assessments):
        assert all(record.id.startswith("exogibbs.") for record in collection)
    for assessment in CONTRACT.assessments:
        checks = equation_checks(CONTRACT, assessment, root=ROOT)
        assert checks and all(c["status"] == "PASS" for c in checks)
        integrity = file_integrity(CONTRACT, assessment, set(assessment.evidence), ROOT)
        files = integrity["selected_files"] + integrity["artifact_hashes"]
        assert files and all(f["status"] == "PASS" for f in files)


@pytest.mark.parametrize("temperature,total_pressure,water_pressure,na,al", [
    (1200., 1000., 100., .02, .10),
    (1800., 10000., 1000., .05, .15),
    (2100., 20000., 50., .00, .20),
    (1800., 10000., 0., .05, .15),
])
def test_water_fraction_and_distinct_pressures(temperature, total_pressure, water_pressure, na, al):
    expected = value("planetlib.solubility.sun_yao2026.water_mass_fraction",
                     T=temperature, P=total_pressure, p_H2O=water_pressure,
                     x_Na2O=na, x_Al2O3=al)
    actual = h2o_sun_yao2026(water_pressure, temperature, total_pressure,
                            x_na2o=na, x_al2o3=al)
    np.testing.assert_allclose(actual, expected, rtol=2e-13, atol=1e-15)


@pytest.mark.parametrize("temperature", [500., 1750., 4000.])
def test_nasa9_interval_tuple_and_coefficient_order(temperature):
    # Synthetic intervals isolate evaluator algebra from dataset adoption.
    coefficients = np.array([
        [1.2e4, -230., 4.1, 1.2e-3, -2e-7, 3e-11, -1e-15, -1200., 6.2],
        [-3e4, 110., 3.5, -2e-4, 1e-7, -2e-12, 2e-16, 900., -2.3],
    ])
    actual_g, actual_s, actual_cp = _nasa9(temperature, coefficients)
    for index, row in enumerate(coefficients):
        inputs = dict(zip(("a1", "a2", "a3", "a4", "a5", "a6", "a7", "b1", "b2"), row))
        inputs["T"] = temperature
        prefix = "planetlib.thermodynamics.nasa9."
        cp = value(prefix + "heat_capacity", **inputs)
        entropy = value(prefix + "entropy", **inputs)
        enthalpy = value(prefix + "enthalpy", **inputs)
        gibbs = value(prefix + "gibbs", hbar=enthalpy, sbar=entropy)
        np.testing.assert_allclose([actual_g[index], actual_s[index], actual_cp[index]],
                                   [gibbs, entropy, cp], rtol=2e-13, atol=2e-13)
        np.testing.assert_allclose(actual_g[index] + actual_s[index], enthalpy,
                                   rtol=2e-13, atol=2e-13)


@pytest.mark.parametrize("temperature,pressure,amounts", [
    (800., 1e4, [1.2, .7]), (1800., 1e5, [.02, 3.]), (2500., 3e7, [2., 4.]),
])
def test_binary_mixing_normalization_and_amount_derivatives(temperature, pressure, amounts):
    amounts = jnp.asarray(amounts)
    standards = jnp.array([-2.3, 1.4])
    selected_pressure_standards = standards + jnp.log(pressure / 1e5)
    inputs = dict(n1=float(amounts[0]), n2=float(amounts[1]),
                  u01=float(selected_pressure_standards[0]), u02=float(selected_pressure_standards[1]))
    prefix = "planetlib.thermodynamics.ideal_mixing."
    expected_g = value(prefix + "gibbs", **inputs)
    expected_mu = [value(prefix + "potential_1", **inputs), value(prefix + "potential_2", **inputs)]
    actual_g = gibbs_j(temperature, pressure, amounts, standards) / (R * temperature)
    actual_mu = jax.grad(gibbs_j, argnums=2)(temperature, pressure, amounts, standards) / (R * temperature)
    np.testing.assert_allclose(actual_g, expected_g, rtol=2e-13, atol=2e-13)
    np.testing.assert_allclose(actual_mu, expected_mu, rtol=2e-13, atol=2e-13)
    np.testing.assert_allclose(actual_g, value(prefix + "euler", **inputs,
                                             u1=float(actual_mu[0]), u2=float(actual_mu[1])),
                               rtol=2e-13, atol=2e-13)
