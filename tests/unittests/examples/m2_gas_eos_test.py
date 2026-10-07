"""Full-basis nonideal parcels, independent scalar derivatives and contact."""

import importlib
from pathlib import Path
import sys
from types import SimpleNamespace

import jax.numpy as jnp
import numpy as np
import pytest

from exogibbs.api.condensate import build_condensate_chemical_setup
from exogibbs.thermo.models import ChemicalSetup


DIRECTORY = Path(__file__).resolve().parents[3] / "examples/metal_silicate"
sys.path.insert(0, str(DIRECTORY))
try:
    ATM = importlib.import_module("m2_atmosphere")
    SOURCE = importlib.import_module("m2_expanded_source")
    ADAPTER = importlib.import_module("m2_gas_eos")
    CONTACT = importlib.import_module("run_m2_contact")
finally:
    sys.path.pop(0)


class FrozenEOS:
    """Analytic test fixture; the production scalar remains provider-owned."""

    species = ("H2", "He1", "H2O1", "Mg1", "H1")

    def __init__(self, factor=1.):
        self.matrix = factor * np.array([
            [.003, .0002, .0001, 0., 0.],
            [.0002, .002, .0005, 0., 0.],
            [.0001, .0005, -.0002, 0., 0.],
            [0., 0., 0., 0., 0.],
            [0., 0., 0., 0., 0.],
        ])

    def state(self, temperature, pressure, x):
        x = jnp.asarray(x)
        b = x @ self.matrix @ x
        rho = 2 * pressure / (8.314 * temperature) / (
            1 + jnp.sqrt(1 + 4 * pressure / (8.314 * temperature) * b))
        z = 1 + rho * b
        return SimpleNamespace(lnphi=2 * rho * (self.matrix @ x) - jnp.log(z),
                               gres_RT=2 * rho * b - jnp.log(z), Z=z)

    def gibbs_residual_rt(self, temperature, pressure, amounts):
        total = jnp.sum(amounts)
        return total * self.state(temperature, pressure, amounts / total).gres_RT

    def parameters(self, temperature, pressure):
        return {"schema": "m2_full_catalog_major_gas_second_virial_v1",
                "species": list(self.species), "coefficients_m3_mol": self.matrix.tolist(),
                "temperature_K": temperature, "pressure_Pa": pressure,
                "gas_constant_J_mol_K": 8.314}


@pytest.fixture(scope="module")
def setup():
    gas = ChemicalSetup(
        formula_matrix=jnp.array([[2., 0., 2., 0., 1.], [0., 1., 0., 0., 0.],
                                  [0., 0., 1., 0., 0.], [0., 0., 0., 1., 0.]]),
        hvector_func=lambda t: jnp.array([0., 0., 0., 0., 2.]),
        elements=("H", "He", "O", "Mg"), species=FrozenEOS.species)
    cloud = ChemicalSetup(
        formula_matrix=jnp.array([[2., 0.], [0., 0.], [1., 0.], [0., 1.]]),
        hvector_func=lambda t: jnp.log(jnp.array([.2, .3])),
        elements=gas.elements, species=("H2O_test(s)", "Mg_test(s)"),
        temperature_validity_upper=(1800., 1800.))
    return build_condensate_chemical_setup(gas_setup=gas, condensate_setup=cloud)


@pytest.mark.scientific
def test_nonideal_cloud_saturation_and_independent_scalar_envelope(setup):
    eos, budget = FrozenEOS(), np.array([6., 1., 2., 2.])
    phase = ATM.make_atmosphere_phase(setup, np.zeros(4), gas_eos=eos)
    report = phase.parcel(1500., 1., budget)
    ideal = ATM.make_atmosphere_phase(setup, np.zeros(4)).parcel(1500., 1., budget)
    assert report["accepted"] and report["gas_eos_convexity"]["accepted"]
    assert report["gas_eos_iteration"]["iterations"] > 1
    assert not np.allclose(report["gas_amounts_mol"], ideal["gas_amounts_mol"], rtol=1e-5)
    x, lnphi = np.array(report["gas_mole_fractions"]), np.array(report["gas_lnphi"])
    np.testing.assert_allclose(x[[2, 3]] * np.exp(lnphi[[2, 3]]), [.2, .3], atol=1e-10)
    assert lnphi[3] == pytest.approx(-np.log(eos.state(1500., 1e5, x).Z), abs=1e-14)
    assert lnphi[3] != 0.
    state = phase(1500., 1., budget)
    value, gradient = phase.energy_value_and_grad_rt(1500., 1., budget)
    assert value == pytest.approx(state.gibbs_rt, abs=1e-11)
    np.testing.assert_allclose(gradient, state.mu_rt, atol=1e-10)
    assert value == pytest.approx(budget @ gradient, abs=1e-10)
    for index in (0, 2):
        delta = np.eye(4)[index] * 1e-5
        numerical = (phase(1500., 1., budget + delta).gibbs_rt
                     - phase(1500., 1., budget - delta).gibbs_rt) / 2e-5
        assert numerical == pytest.approx(gradient[index], abs=1e-7)
    # A density-only or frozen ideal-composition substitution must fail KKT.
    stale = ATM.audit_atmosphere(setup, 1500., 1., budget, ideal["gas_amounts_mol"],
                                ideal["condensate_amounts_mol"], gas_eos=eos)
    assert not stale["accepted"]


@pytest.mark.scientific
def test_zero_coefficients_recover_ideal_and_full_basis_survives_zero_elements(setup):
    budget = np.array([6., 1., 2., 2.])
    ideal = ATM.make_atmosphere_phase(setup, np.zeros(4)).parcel(1500., 1., budget)
    zero = ATM.make_atmosphere_phase(setup, np.zeros(4), gas_eos=FrozenEOS(0.)).parcel(1500., 1., budget)
    for key in ("gas_amounts_mol", "condensate_amounts_mol", "elemental_potentials_rt", "gibbs_rt"):
        np.testing.assert_allclose(zero[key], ideal[key], atol=1e-12)
    phase = ATM.make_atmosphere_phase(setup, np.zeros(4), gas_eos=FrozenEOS())
    reduced = phase.parcel(1500., 1., [2., .1, 0., 0.])
    assert reduced["accepted"] and reduced["gas_species"] == list(FrozenEOS.species)
    np.testing.assert_array_equal(np.array(reduced["gas_amounts_mol"])[[2, 3]], 0.)
    np.testing.assert_array_equal(reduced["condensate_amounts_mol"], [0., 0.])
    _, gradient = phase.energy_value_and_grad_rt(1500., 1., [2., .1, 0., 0.])
    assert np.isfinite(gradient[:2]).all()
    np.testing.assert_array_equal(gradient[2:], -np.inf)
    empty = phase(1500., 1., np.zeros(4))
    assert empty.gibbs_rt == 0.


@pytest.mark.scientific
def test_gauge_invariance_primitive_unpack_and_contact_use_same_residual(setup, monkeypatch):
    monkeypatch.setattr(CONTACT, "build_atmosphere_setup", lambda model: setup)
    budget, gauge = np.array([6., 1., 2., 2.]), np.array([.2, -.3, .4, .1])
    phase = ATM.make_atmosphere_phase(setup, gauge, gas_eos=FrozenEOS())
    unshifted = ATM.make_atmosphere_phase(setup, np.zeros(4), gas_eos=FrozenEOS())
    state = phase(1500., 1., budget)
    assert state.gibbs_rt - unshifted(1500., 1., budget).gibbs_rt == pytest.approx(gauge @ budget)
    names = [element + "_atmosphere_atom" for element in setup.elements]
    record = {"elements": list(setup.elements), "phases": {"atmosphere": names},
              "component_formulas": {name: {element: 1.} for name, element in zip(names, setup.elements)},
              "atmosphere_element_order": list(setup.elements), "reactions": []}
    internal = {"accepted": True, "component_amounts_mol": budget, "gibbs_rt": state.gibbs_rt}
    public, result, callbacks, parcel = SOURCE.unpack_expanded_source(
        record, internal, {"atmosphere": phase}, 1500., 1.)
    assert CONTACT.audit_expanded_contact(public, result, callbacks, 1500., 1., parcel)["accepted"]
    altered = {**parcel, "gas_eos": {**parcel["gas_eos"], "pressure_Pa": 2e5}}
    with pytest.raises(ValueError, match="receipt"):
        CONTACT.audit_expanded_contact(public, result, callbacks, 1500., 1., altered)
    with pytest.raises(ValueError, match="same gas EOS"):
        CONTACT.audit_expanded_contact(public, result, callbacks, 1500., 1., {**parcel, "gas_eos": None})
    gas = np.asarray(parcel["gas_amounts_mol"])
    energy, gradient = callbacks["gas"].energy_value_and_grad_rt(1500., 1., gas)
    primitive = callbacks["gas"](1500., 1., gas)
    np.testing.assert_allclose(gradient, primitive.mu_rt, atol=1e-12)
    assert energy == pytest.approx(primitive.gibbs_rt, abs=1e-12)
    empty_energy, _ = callbacks["gas"].energy_value_and_grad_rt(1500., 1., np.zeros_like(gas))
    assert empty_energy == 0.


def test_nonconverged_or_unproved_eos_never_returns_an_accepted_parcel(setup, monkeypatch):
    unstable = ATM.make_atmosphere_phase(setup, np.zeros(4), gas_eos=FrozenEOS(1e4))
    with pytest.raises(ATM.PhaseEvaluationError, match="curvature"):
        unstable.parcel(1500., 1., [6., 1., 2., 2.])
    eos = FrozenEOS()
    calls = []
    def oscillating_state(*args):
        calls.append(None)
        return SimpleNamespace(lnphi=np.full(5, len(calls) % 2))
    monkeypatch.setattr(eos, "state", oscillating_state)
    monkeypatch.setattr(ATM, "solve_condensate", lambda *a, **k: SimpleNamespace(
        gas_n=np.ones(5), condensate_amounts=np.ones(2), converged=True))
    with pytest.raises(ATM.PhaseEvaluationError, match="did not converge"):
        ATM.make_atmosphere_phase(setup, np.zeros(4), gas_eos=eos).parcel(1500., 1., [6., 1., 2., 2.])
    assert len(calls) == 64


def test_provider_factory_binds_order_options_and_source_receipt(tmp_path, setup):
    exoeos = pytest.importorskip("exoeos")
    checkout = Path(exoeos.__file__).resolve().parents[2]
    if not (checkout / "examples/m2_material/major_gas_eos.py").exists():
        pytest.skip("Requires the declared major-gas EOS provider")
    assert ADAPTER.build_gas_eos(None, FrozenEOS.species, None) is None
    options = {"h2_he_cm3_mol": 10., "water_cross_temperature_policy": "hold_2000",
               "trace_pair_policy": "zero"}
    model = ADAPTER.build_gas_eos(checkout, FrozenEOS.species, options)
    assert model.species == FrozenEOS.species
    assert model.parameters(2173.15, 267e5)["options"] == options
    phase = ATM.make_atmosphere_phase(setup, np.zeros(4), gas_eos=model)
    parcel = phase.parcel(2173.15, 267., [2., .1, 0., 0.])
    energy, gradient = phase.energy_value_and_grad_rt(2173.15, 267., [2., .1, 0., 0.])
    assert parcel["accepted"] and energy == pytest.approx(parcel["gibbs_rt"], abs=1e-11)
    np.testing.assert_allclose(gradient[:2], parcel["elemental_potentials_rt"][:2], atol=1e-10)
    large_energy, large_gradient = phase.energy_value_and_grad_rt(
        2173.15, 267., np.array([2., .1, 0., 0.]) * 1e24)
    assert large_energy / 1e24 == pytest.approx(energy, abs=1e-11)
    np.testing.assert_allclose(large_gradient[:2], gradient[:2], atol=1e-10)
    _, _, callbacks, _, metadata = SOURCE.build_expanded_bse_problem(
        checkout / "examples/m2_material/bse_inventory.json", checkout, tmp_path, sys.executable,
        pressure_bar=267., gas_model="janaf_condensed", gas_eos_options=options,
        helium_solubility_model="guillot2012_olivine")
    assert len(callbacks["atmosphere"].gas_eos.species) == 76
    assert metadata["gas_eos"] == callbacks["atmosphere"].gas_eos.parameters(2173.15, 267e5)
    assert metadata["helium_dissolution"]["gas_fugacity_policy"].startswith("Nonideal retained gas")
    assert "m2_gas_eos.py" in metadata["provenance"]["file_sha256"]
    assert metadata["numerical_execution"]["native_melt_calls"] == 0
    ideal = SOURCE.build_expanded_bse_problem(
        checkout / "examples/m2_material/bse_inventory.json", checkout, tmp_path, sys.executable,
        pressure_bar=267., gas_model="janaf_condensed", helium_solubility_model="guillot2012_olivine")
    assert ideal[-1]["provenance"]["file_sha256"] == metadata["provenance"]["file_sha256"]
