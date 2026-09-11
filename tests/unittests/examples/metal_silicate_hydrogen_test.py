"""Common-standard hydrogen closure and integrable host-dilution regressions."""

from __future__ import annotations

from functools import lru_cache
import importlib
import json
from pathlib import Path
import sys

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from exogibbs.solubility import h2_hirschmann2012


EXAMPLE = Path(__file__).resolve().parents[3] / "examples" / "metal_silicate"


def _load_example():
    names = ("local", "reference", "source", "hydrogen")
    previous = {name: sys.modules.get(name) for name in names}
    old_path = sys.path[:]
    try:
        sys.path.insert(0, str(EXAMPLE))
        for name in names:
            sys.modules.pop(name, None)
        return importlib.import_module("hydrogen")
    finally:
        sys.path[:] = old_path
        for name, module in previous.items():
            if module is None:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = module


H = _load_example()
RECORD = H.load_reference()
CASE = next(case for case in RECORD["cases"] if case["T_K"] == 2350.0)
FIXTURE = next(case for case in json.loads((EXAMPLE / "equilibrium_reference.json").read_text())["cases"]
               if case["id"] == "source_full_2350")
SPECIES = tuple(name for names in RECORD["phases"].values() for name in names)
REACTIONS = np.asarray([[item["stoichiometry"].get(name, 0) for name in SPECIES]
                        for item in RECORD["reactions"]])


@pytest.fixture(scope="module")
def provider():
    value = pytest.importorskip("exoeos")
    if not hasattr(value, "MaFeSiOHLiquid"):
        pytest.skip("The optional ExoEOS installation predates MaFeSiOHLiquid.")
    return value


@pytest.mark.parametrize("pressure_bar,phi", [(1e-8, 1.0), (7000.0, 1.0), (7000.0, 1.8), (30000.0, 2.1)])
def test_dissolution_uses_h2_fugacity_and_one_pressure_correction(pressure_bar, phi):
    fugacity = 0.3 * phi * pressure_bar
    liquid = h2_hirschmann2012(fugacity, pressure_bar * 1e-4)
    gas_mu0 = -13.7
    liquid_mu0 = H.dissolved_h2_standard_rt(gas_mu0, H.hirschmann2012_ln_solubility(pressure_bar))
    residual = liquid_mu0 - gas_mu0 + jnp.log(liquid) - jnp.log(fugacity)
    np.testing.assert_allclose(residual, 0, atol=5e-15)
    # The empirical coefficient is against fugacity; replacing it by total P fails.
    assert not np.isclose(float(liquid), float(h2_hirschmann2012(pressure_bar, pressure_bar * 1e-4)), rtol=1e-3, atol=0)
    derivative = jax.jit(jax.grad(lambda p: H.dissolved_h2_standard_rt(gas_mu0, H.hirschmann2012_ln_solubility(p))))(pressure_bar)
    np.testing.assert_allclose(derivative, 0.76e-4, atol=1e-16)


def test_standard_pressure_conversion_and_revised_data_gate():
    reference_pressure, fugacity, coefficient = 3.0, 17.0, 0.004
    liquid_mu0 = H.dissolved_h2_standard_rt(-5.0, np.log(coefficient), standard_pressure_bar=reference_pressure)
    residual = liquid_mu0 + 5.0 + np.log(coefficient * fugacity) - np.log(fugacity / reference_pressure)
    np.testing.assert_allclose(residual, 0, atol=1e-15)
    assert H.H2_CALIBRATION["revised_candidate"]["status"].startswith("missing_")
    for invalid in (-1., np.nan, np.inf):
        assert np.isnan(H.hirschmann2012_ln_solubility(invalid))
    with pytest.raises(ValueError, match="standard_pressure"):
        H.dissolved_h2_standard_rt(0., 0., standard_pressure_bar=0.)


def test_h2_changes_all_reaction_standards_and_preserves_okuchi_absolute_h(provider):
    model = provider.MaFeSiOHLiquid()
    source = H.make_source_standard_potentials_rt(RECORD, CASE)
    common = H.make_common_standard_potentials_rt(RECORD, CASE, model=model)
    pressure = 10000.0
    initial = np.asarray(source(2350., pressure))
    actual = np.asarray(jax.jit(common)(2350., pressure))
    h2_liquid, h2_gas = SPECIES.index("H2_silicate"), SPECIES.index("H2_gas")
    metal_indices = [SPECIES.index(name + "_metal") for name in model.components]
    expected = initial.copy()
    expected[h2_liquid] = initial[h2_gas] + 11.403 + 0.76
    expected[metal_indices] += np.asarray(model.standard_state_shift_RT(2350.))
    np.testing.assert_allclose(actual, expected, atol=1e-14)
    np.testing.assert_allclose(actual[SPECIES.index("H_metal")], initial[SPECIES.index("H_metal")], atol=0)
    assert abs(actual[SPECIES.index("H_metal")]) > 1
    # Hold native metal standards fixed to isolate the H2 replacement in R4/R6/R14.
    native_base = initial.copy()
    native_base[metal_indices] += np.asarray(model.standard_state_shift_RT(2350.))
    delta = actual[h2_liquid] - native_base[h2_liquid]
    for index in (4, 6, 14):
        np.testing.assert_allclose(REACTIONS[index] @ (actual - native_base), REACTIONS[index, h2_liquid] * delta, atol=1e-14)
    # Two gas-mediated exchange paths must agree with direct common potentials.
    reaction_g = REACTIONS @ actual
    h_partition = actual[h2_gas] - 2 * actual[SPECIES.index("H_metal")]
    np.testing.assert_allclose(reaction_g[4] - reaction_g[14], h_partition, atol=1e-14)
    si_water = (actual[SPECIES.index("SiO2_silicate")] + 2 * actual[h2_gas]
                - actual[SPECIES.index("Si_metal")] - 2 * actual[SPECIES.index("H2O_gas")])
    np.testing.assert_allclose(reaction_g[6] - 2 * reaction_g[14] + 2 * reaction_g[15], si_water, atol=3e-14)


def _mixed_host(amounts):
    # Deliberately different backend R, converted before adding common-R dilution.
    ratio = 8.3143 / 8.31446261815324
    standards = jnp.asarray([-4., -7.])
    mu = standards + ratio * jnp.log(amounts / amounts.sum())
    return amounts @ mu, mu


def _diluted(amounts):
    host_g, host_mu = _mixed_host(amounts[:2])
    return H.ideal_host_h2_dilution(host_g, host_mu, amounts[:2], amounts[2], 2.4)


def test_host_dilution_full_potentials_are_extensive_and_reciprocal():
    amounts = jnp.asarray([0.7, 0.3, 0.08])
    state = jax.jit(_diluted)(amounts)
    potentials = np.append(state.host_mu_rt, state.h2_mu_rt)
    derivative = jax.jit(jax.grad(lambda n: _diluted(n).gibbs_rt))(amounts)
    np.testing.assert_allclose(derivative, potentials, atol=2e-15)
    np.testing.assert_allclose(amounts @ potentials, state.gibbs_rt, atol=1e-15)
    hessian = jax.jacfwd(jax.grad(lambda n: _diluted(n).gibbs_rt))(amounts)
    np.testing.assert_allclose(hessian, hessian.T, atol=2e-15)
    np.testing.assert_allclose(hessian @ amounts, 0, atol=1e-15)
    for index in range(3):
        change = np.eye(3)[index] * 1e-5
        finite = (_diluted(amounts + change).gibbs_rt - _diluted(amounts - change).gibbs_rt) / 2e-5
        np.testing.assert_allclose(finite, potentials[index], rtol=3e-8, atol=1e-8)
    scaled = _diluted(amounts * 20)
    np.testing.assert_allclose(scaled.gibbs_rt, state.gibbs_rt * 20, atol=1e-13)
    np.testing.assert_allclose(scaled.host_mu_rt, state.host_mu_rt, atol=1e-15)
    np.testing.assert_allclose(scaled.h2_mu_rt, state.h2_mu_rt, atol=1e-15)
    _, undiluted_mu = _mixed_host(amounts[:2])
    np.testing.assert_allclose(state.host_mu_rt - undiluted_mu, np.log(1 / 1.08), atol=1e-15)


def test_zero_h2_preserves_host_exactly_and_rejects_pure_h2():
    amounts = jnp.asarray([0.7, 0.3, 0.0])
    host_g, host_mu = _mixed_host(amounts[:2])
    state = jax.jit(H.ideal_host_h2_dilution)(host_g, host_mu, amounts[:2], amounts[2], 2.4)
    np.testing.assert_array_equal(state.gibbs_rt, host_g)
    np.testing.assert_array_equal(state.host_mu_rt, host_mu)
    assert np.isneginf(state.h2_mu_rt)
    assert np.isnan(H.ideal_host_h2_dilution(0., jnp.zeros(2), jnp.zeros(2), 1., 0.).gibbs_rt)
    assert np.isnan(H.ideal_host_h2_dilution(0., jnp.zeros(2), jnp.ones(2), -1., 0.).gibbs_rt)
    with pytest.raises(ValueError, match="vector shape"):
        H.ideal_host_h2_dilution(0., jnp.zeros(2), jnp.ones(3), 1., 0.)


@lru_cache(maxsize=None)
def _finite_control(hydrogen_scale, gas_correction=0.):
    budget = np.append(FIXTURE["element_amounts_mol"], 3.)
    budget[4] *= hydrogen_scale
    def pure_phi(t, p, x):
        assert x is None
        return jnp.zeros(11).at[0].set(gas_correction)
    problem = H.build_source_control(RECORD, CASE, budget, pure_lnphi_func=pure_phi)
    initial = np.append(FIXTURE["component_amounts_mol"], 3.)
    excluded = np.ones(26, dtype=bool)
    excluded[problem.species_indices] = False
    initial[excluded] = 0
    result = H.solve(problem, 2350., 1., budget, initial_component_amounts_mol=initial)
    return problem, budget, initial, result


@pytest.mark.parametrize("hydrogen_scale", [0., 1.])
def test_finite_source_host_retains_helium_and_exact_zero_h(provider, hydrogen_scale):
    problem, budget, _, result = _finite_control(hydrogen_scale)
    assert problem.reaction_offset is None
    assert bool(result.root_solution.converged)
    totals = np.asarray(problem.full_formula_matrix) @ np.asarray(result.component_amounts_mol)
    positive = budget > 0
    np.testing.assert_allclose(totals[positive] / budget[positive], 1, atol=1e-9, rtol=0)
    np.testing.assert_array_equal(totals[~positive], 0)
    np.testing.assert_allclose(result.reaction_residual, 0, atol=1e-8)
    np.testing.assert_allclose(result.component_amounts_mol[-1], 3., atol=1e-12)
    if hydrogen_scale == 0:
        hydrogen_components = np.asarray(problem.full_formula_matrix)[4] > 0
        np.testing.assert_array_equal(result.component_amounts_mol[hydrogen_components], 0)


def test_finite_control_fugacity_response_and_budget_rescaling(provider):
    problem, budget, initial, base = _finite_control(1.)
    _, _, _, nonideal = _finite_control(1., np.log(2.))
    h2 = problem.full_species.index("H2_silicate")
    assert abs(float(nonideal.component_amounts_mol[h2] / base.component_amounts_mol[h2]) - 1) > 0.1
    for result, phi in ((base, 1.), (nonideal, 2.)):
        liquid_fraction = result.mole_fractions[h2]
        gas_fraction = result.mole_fractions[problem.full_species.index("H2_gas")]
        np.testing.assert_allclose(liquid_fraction, h2_hirschmann2012(phi * gas_fraction, 1e-4), rtol=1e-10, atol=0)
    scaled = H.solve(problem, 2350., 1., budget * 5., initial_component_amounts_mol=initial * 5.)
    assert bool(scaled.root_solution.converged)
    np.testing.assert_allclose(scaled.component_amounts_mol, base.component_amounts_mol * 5., rtol=1e-9, atol=0)
