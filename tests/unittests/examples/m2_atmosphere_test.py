"""Independent Gibbs-envelope derivatives, cloud amounts, and zero support."""

import importlib
from pathlib import Path
import sys

import jax.numpy as jnp
import numpy as np
import pytest

from exogibbs.api.condensate import build_condensate_chemical_setup
from exogibbs.thermo.models import ChemicalSetup


DIRECTORY = Path(__file__).resolve().parents[3] / "examples" / "metal_silicate"
NAMES = ("local", "full_potential", "common_gibbs", "m1_chemistry", "m2_atmosphere")
previous = {name: sys.modules.get(name) for name in NAMES}
sys.path.insert(0, str(DIRECTORY))
try:
    for name in NAMES:
        sys.modules.pop(name, None)
    LOCAL, FULL, ENERGY, M1, ATM = [importlib.import_module(name) for name in NAMES]
finally:
    sys.path.pop(0)
    for name, module in previous.items():
        if module is None:
            sys.modules.pop(name, None)
        else:
            sys.modules[name] = module


@pytest.fixture(scope="module")
def analytic_setup():
    # An artificial ideal model with two independently saturated pure phases.
    gas = ChemicalSetup(
        formula_matrix=jnp.diag(jnp.array([2., 1., 1.])),
        hvector_func=lambda t: jnp.zeros(3), elements=("H", "He", "Mg"),
        species=("H2", "He1", "Mg1"),
    )
    cloud = ChemicalSetup(
        formula_matrix=jnp.array([[2., 0.], [0., 0.], [0., 1.]]),
        hvector_func=lambda t: jnp.log(jnp.array([.2, .3])),
        elements=gas.elements, species=("H2_test(s)", "Mg_test(s)"),
        temperature_validity_upper=(1500., 1500.),
    )
    return build_condensate_chemical_setup(gas_setup=gas, condensate_setup=cloud)


def test_two_pure_clouds_obey_independent_saturation_without_cloud_mixing(analytic_setup):
    phase = ATM.make_atmosphere_phase(analytic_setup, np.zeros(3))
    report = phase.parcel(1000., 1., np.array([2., 1., 2.]))
    assert report["accepted"]
    np.testing.assert_allclose(report["gas_amounts_mol"], [.4, 1., .6], atol=1e-10)
    np.testing.assert_allclose(report["condensate_amounts_mol"], [.6, 1.4], atol=1e-10)
    expected_mu = np.array([.5 * np.log(.2), np.log(.5), np.log(.3)])
    np.testing.assert_allclose(report["elemental_potentials_rt"], expected_mu, atol=1e-10)
    expected = np.log(.2) + np.log(.5) + 2 * np.log(.3)
    assert report["gibbs_rt"] == pytest.approx(expected, abs=1e-10)


@pytest.mark.parametrize("budget", [[2., 1., 2.], [.1, 1., .1]])
def test_scalar_finite_differences_euler_and_amount_scaling(analytic_setup, budget):
    b = np.array(budget)
    phase = ATM.make_atmosphere_phase(analytic_setup, np.array([.3, -.7, 1.2]))
    state = phase(1000., 1., b)
    estimates = []
    for index in range(len(b)):
        delta = np.zeros_like(b)
        delta[index] = 1e-4 * b[index]
        estimates.append((phase(1000., 1., b + delta).gibbs_rt
                          - phase(1000., 1., b - delta).gibbs_rt) / (2 * delta[index]))
    np.testing.assert_allclose(estimates, state.mu_rt, atol=1e-7)
    assert state.gibbs_rt == pytest.approx(b @ state.mu_rt, abs=1e-10)
    scaled = phase(1000., 1., 7.3 * b)
    assert scaled.gibbs_rt == pytest.approx(7.3 * state.gibbs_rt, abs=1e-9)
    np.testing.assert_allclose(scaled.mu_rt, state.mu_rt, atol=1e-10)


def test_temperature_ineligible_condensates_are_exactly_zero(analytic_setup):
    report = ATM.make_atmosphere_phase(analytic_setup, np.zeros(3)).parcel(2000., 1., [2., 1., 2.])
    assert report["accepted"]
    assert report["condensate_temperature_eligible"] == [False, False]
    assert report["condensate_insertion_rt"] == [None, None]
    np.testing.assert_array_equal(report["condensate_amounts_mol"], np.zeros(2))
    np.testing.assert_allclose(report["gas_amounts_mol"], [1., 1., 2.], atol=1e-10)


def test_missing_elements_and_an_empty_atmosphere_have_no_floors(analytic_setup):
    phase = ATM.make_atmosphere_phase(analytic_setup, np.zeros(3))
    report = phase.parcel(1000., 1., [0., 2., 0.])
    assert report["accepted"]
    np.testing.assert_array_equal(report["gas_amounts_mol"], [0., 2., 0.])
    np.testing.assert_array_equal(report["condensate_amounts_mol"], [0., 0.])
    assert report["condensate_element_supported"] == [False, False]
    np.testing.assert_array_equal(phase(1000., 1., [0., 2., 0.]).mu_rt, [-np.inf, 0., -np.inf])
    empty = phase(1000., 1., np.zeros(3))
    assert empty.gibbs_rt == 0.
    np.testing.assert_array_equal(empty.mu_rt, np.full(3, -np.inf))


def test_independent_audit_detects_omitted_clouds_and_unsupported_atoms(analytic_setup):
    phase = ATM.make_atmosphere_phase(analytic_setup, np.zeros(3))
    report = phase.parcel(1000., 1., [2., 1., 2.])
    invalid = ATM.audit_atmosphere(analytic_setup, 1000., 1., [2., 1., 2.],
                                  report["gas_amounts_mol"], [0., 0.])
    assert not invalid["accepted"]
    assert max(abs(x) for x in invalid["relative_element_residual"]) > .5
    with pytest.raises(ValueError, match="exact-zero"):
        ATM.audit_atmosphere(analytic_setup, 1000., 1., [0., 1., 0.], [1., 1., 0.], [0., 0.])
    # A caller cannot mutate the cached physical state through its report.
    report["gas_amounts_mol"][0] = 999.
    assert phase.parcel(1000., 1., [2., 1., 2.])["gas_amounts_mol"][0] < 1.


@pytest.fixture(scope="module")
def fastchem_setup():
    return M1.build_setups()[1]


# Archived finite-BSE source gas ledger from results/m2_contact/20260922/contact.json,
# reduced by 1e23 mol to keep independent finite differences well scaled.
BSE_ATMOSPHERE = np.array([9.99295840634972, 1., .18823552485434354,
                          .010065287005740377, .049710919365219,
                          .04974492773582125, .06534238286807431])


def test_fastchem_bse_iron_cloud_scalar_derivative_and_elemental_gauge(fastchem_setup):
    setup, b = fastchem_setup, BSE_ATMOSPHERE
    phase = ATM.make_atmosphere_phase(setup, np.zeros(7))
    report = phase.parcel(2173.15, 1., b)
    assert report["accepted"]
    assert report["condensate_amounts_mol"][setup.condensate_species.index("Fe(s,l)")] > .03
    assert report["gas_mass_fraction"] < .95
    state = phase(2173.15, 1., b)
    assert state.gibbs_rt == pytest.approx(b @ state.mu_rt, abs=1e-8)
    for index in (0, 5):
        delta = np.zeros(7)
        delta[index] = 1e-4 * b[index]
        finite_difference = (phase(2173.15, 1., b + delta).gibbs_rt
                             - phase(2173.15, 1., b - delta).gibbs_rt) / (2 * delta[index])
        assert finite_difference == pytest.approx(state.mu_rt[index], abs=2e-6)
    gauge = np.array([.2, -1., 3., 2., -2., 1., .1])
    shifted = ATM.make_atmosphere_phase(setup, lambda t, p: gauge).parcel(2173.15, 1., b)
    np.testing.assert_allclose(shifted["gas_amounts_mol"], report["gas_amounts_mol"], atol=1e-10)
    np.testing.assert_allclose(shifted["condensate_amounts_mol"], report["condensate_amounts_mol"], atol=1e-10)
    assert shifted["gibbs_rt"] - report["gibbs_rt"] == pytest.approx(gauge @ b, abs=1e-9)
    for phase_setup, key in ((setup.gas_setup, "gas_standard_potentials_rt"),
                             (setup.condensate_setup, "condensate_standard_potentials_rt")):
        np.testing.assert_allclose(np.array(shifted[key]) - report[key],
                                   np.asarray(phase_setup.formula_matrix).T @ gauge, atol=1e-12)


def test_fastchem_hydrogen_helium_only_restores_full_catalog_exact_zeros(fastchem_setup):
    phase = ATM.make_atmosphere_phase(fastchem_setup, np.zeros(7))
    report = phase.parcel(2000., 1., [2., .1, 0., 0., 0., 0., 0.])
    assert report["accepted"]
    assert len(report["gas_amounts_mol"]) == 35
    assert len(report["condensate_amounts_mol"]) == 26
    forbidden = np.any(np.asarray(fastchem_setup.gas_setup.formula_matrix)[2:] != 0, axis=0)
    np.testing.assert_array_equal(np.array(report["gas_amounts_mol"])[forbidden], 0.)
    np.testing.assert_array_equal(report["condensate_amounts_mol"], np.zeros(26))
    np.testing.assert_allclose(report["gas_element_amounts_mol"], [2., .1, 0., 0., 0., 0., 0.], atol=1e-10)


def test_nested_solver_failure_is_not_a_thermodynamic_value(analytic_setup, monkeypatch):
    def fail(*args, **kwargs):
        raise RuntimeError("parcel unavailable")
    monkeypatch.setattr(ATM, "solve_condensate", fail)
    phase = ATM.make_atmosphere_phase(analytic_setup, np.zeros(3))
    with pytest.raises(ATM.PhaseEvaluationError, match="parcel unavailable"):
        phase(1000., 1., [2., 1., 2.])
