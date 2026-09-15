"""Independent source/parcel contracts for the local milestone 1 diagnostic."""

from copy import deepcopy
from dataclasses import replace
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import jax.numpy as jnp
import numpy as np
import pytest

from exogibbs.thermo.models import ChemicalSetup
from exogibbs.utils.elements import element_mass


PATH = Path(__file__).resolve().parents[3] / "examples" / "metal_silicate" / "m1_chemistry.py"
SPEC = importlib.util.spec_from_file_location("m1_chemistry", PATH)
M1 = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(M1)

PARCELS = json.loads(
    Path(__file__).with_name("data").joinpath("m1_condensate_parcels.json").read_text()
)


def test_named_subset_preserves_atom_order_standards_and_temperature_bounds():
    elements = (*reversed(M1.ELEMENTS), "C")
    species = ("He1", "C1", "H2", "Mg1")
    formulas = ({"He": 1}, {"C": 1}, {"H": 2}, {"Mg": 1})
    setup = ChemicalSetup(
        formula_matrix=jnp.asarray([[f.get(e, 0) for f in formulas] for e in elements]),
        hvector_func=lambda t: jnp.asarray(t)[..., None] * jnp.arange(1, 5),
        elements=elements, species=species,
        temperature_validity_upper=(5000., 6000., 7000., 8000.),
        metadata={"source": "analytic test"},
    )
    selected = M1.subset_setup(setup, ("H2", "He1"))
    assert selected.elements == M1.ELEMENTS
    assert selected.species == ("H2", "He1")
    np.testing.assert_array_equal(selected.formula_matrix, [[2, 0], [0, 1], *([[0, 0]] * 5)])
    np.testing.assert_array_equal(selected.hvector_func(10.), [30., 10.])
    np.testing.assert_array_equal(selected.hvector_func(jnp.array([10., 20.])), [[30., 10.], [60., 20.]])
    assert selected.temperature_validity_upper == (7000., 5000.)
    assert selected.metadata["source"] == "analytic test"
    assert setup.metadata == {"source": "analytic test"}
    with pytest.raises(ValueError, match="excluded element"):
        M1.subset_setup(setup, ("H2", "C1"))
    with pytest.raises(ValueError, match="unique"):
        M1.subset_setup(setup, ("H2", "H2"))


@pytest.mark.parametrize("oxygen_factor", [0.9, 1.1])
def test_source_input_variation_keeps_mass_helium_ratio_and_exact_zero_cns(oxygen_factor):
    network, case, reference = M1.source_inputs()
    _, _, budget = M1.source_inputs(oxygen_factor)
    index = {e: i for i, e in enumerate(network["elements"])}
    weights = np.array([element_mass[e] * 1e-3 for e in network["elements"]])
    assert case["T_K"] == 2350.
    assert budget @ weights == pytest.approx(reference @ weights, rel=1e-14)
    assert budget[index["He"]] / budget[index["H"]] == pytest.approx(0.1)
    np.testing.assert_array_equal(budget[[index[e] for e in ("C", "N", "S")]], 0.)
    assert budget[index["O"]] / budget[index["Si"]] == pytest.approx(
        oxygen_factor * reference[index["O"]] / reference[index["Si"]],
    )
    other = [index[e] for e in ("H", "He", "Mg", "Si", "Fe", "Na")]
    np.testing.assert_allclose(budget[other] / budget[index["Si"]], reference[other] / reference[index["Si"]])


def test_source_gas_inventory_counts_named_gas_only_after_reordering():
    network, _, _ = M1.source_inputs()
    network = deepcopy(network)
    network["elements"].reverse()
    network["phases"] = {phase: list(reversed(names)) for phase, names in reversed(list(network["phases"].items()))}
    species, _, _ = M1.SOURCE.component_matrices(network)
    gas = dict(zip(M1.SHARED_SOURCE_SPECIES, np.arange(1., 10.)))
    amounts = np.array([gas.get(s, 0.) if s in network["phases"]["gas"] else 1000. for s in species])
    expected = [sum(network["component_formulas"][s].get(e, 0) * n for s, n in gas.items()) for e in M1.ELEMENTS]
    result = {"component_amounts_mol": amounts}
    np.testing.assert_array_equal(M1.source_gas_inventory(network, result), expected)
    carbon_gas = next(s for s in network["phases"]["gas"] if network["component_formulas"][s].get("C", 0))
    amounts[species.index(carbon_gas)] = 1e-30
    with pytest.raises(ValueError, match="exact-zero C/N/S"):
        M1.source_gas_inventory(network, result)


@pytest.fixture
def analytic_parcel():
    """An exact ideal H-He-O root with one present and one absent pure phase."""
    a_g = np.array([[2., 0., 2., 0.], [0., 1., 0., 0.], [0., 0., 1., 2.]])
    a_c = np.array([[2., 2.], [0., 0.], [1., 1.]])
    gas, cloud = np.array([2., 1., 3., 4.]), np.array([2., 0.])
    potentials = np.array([0.4, -0.2, 0.7])
    h_g = a_g.T @ potentials - np.log(gas / gas.sum()) - np.log(2.)
    h_c = a_c.T @ potentials + [0., 0.6]
    setup = ChemicalSetup(
        formula_matrix=jnp.asarray(a_g), hvector_func=lambda t: jnp.asarray(h_g),
        elements=("H", "He", "O"), species=("H2", "He1", "H2O1", "O2"),
    )
    condensate = ChemicalSetup(
        formula_matrix=jnp.asarray(a_c), hvector_func=lambda t: jnp.asarray(h_c),
        elements=setup.elements, species=("water_a", "water_b"),
        temperature_validity_upper=(1000., 1000.),
    )
    return setup, condensate, a_g @ gas + a_c @ cloud, gas, cloud


@pytest.mark.parametrize("scale", [1e-12, 1., 1e20])
def test_retained_parcel_uses_gas_pressure_and_total_mass_basis(analytic_parcel, scale):
    setup, condensate, budget, gas, cloud = analytic_parcel
    result = M1.audit_parcel(setup, 500., 2., budget * scale, gas * scale,
                             condensate_setup=condensate, condensate_amounts=cloud * scale)
    assert result["accepted"]
    weights = np.array([element_mass[e] * 1e-3 for e in setup.elements])
    gas_mass = weights @ (np.asarray(setup.formula_matrix) @ gas)
    total_mass = weights @ budget
    assert result["gas_mass_kg"] == pytest.approx(gas_mass * scale)
    assert result["cloud_mass_kg"] == pytest.approx((total_mass - gas_mass) * scale)
    assert result["gas_mass_fraction"] == pytest.approx(gas_mass / total_mass)
    np.testing.assert_allclose(result["total_element_mol_per_kg"], budget / total_mass)
    np.testing.assert_allclose(result["partial_pressures_bar"], [0.4, 0.2, 0.6, 0.8])
    assert result["mean_gas_molar_mass_kg_per_mol"] == pytest.approx(gas_mass / gas.sum())
    assert result["h2o_h2"] == pytest.approx(1.5)
    assert result["condensate_temperature_eligible"] == [True, True]


@pytest.mark.parametrize("failure", ["budget", "gas_stationarity", "present_phase", "absent_phase", "temperature"])
def test_independent_audit_rejects_invalid_equilibrium(analytic_parcel, failure):
    setup, condensate, budget, gas, cloud = analytic_parcel
    if failure == "budget":
        budget = budget * 1.01
    elif failure == "gas_stationarity":
        h = np.asarray(setup.hvector_func(500.)) + [0., 0., 0.1, 0.]
        setup = replace(setup, hvector_func=lambda t: jnp.asarray(h))
    elif failure in ("present_phase", "absent_phase"):
        h = np.asarray(condensate.hvector_func(500.)).copy()
        h[0 if failure == "present_phase" else 1] -= 1.
        condensate = replace(condensate, hvector_func=lambda t: jnp.asarray(h))
    else:
        condensate = replace(condensate, temperature_validity_upper=(400., 1000.))
    result = M1.audit_parcel(setup, 500., 2., budget, gas,
                             condensate_setup=condensate, condensate_amounts=cloud)
    assert not result["accepted"]


def test_ineligible_absent_phase_does_not_fail_insertion_audit(analytic_parcel):
    setup, condensate, budget, gas, cloud = analytic_parcel
    h = np.asarray(condensate.hvector_func(500.)).copy()
    h[1] = -100.
    condensate = replace(condensate, hvector_func=lambda t: jnp.asarray(h),
                        temperature_validity_upper=(1000., 400.))
    result = M1.audit_parcel(setup, 500., 2., budget, gas,
                             condensate_setup=condensate, condensate_amounts=cloud)
    assert result["accepted"]
    assert result["condensate_temperature_eligible"] == [True, False]
    assert result["condensate_insertion_rt"][1] is None


def test_solver_convergence_does_not_override_independent_budget_audit(monkeypatch):
    gas = ChemicalSetup(
        formula_matrix=jnp.eye(7), hvector_func=lambda t: jnp.full(7, np.log(7.)),
        elements=M1.ELEMENTS, species=tuple(e + "1" for e in M1.ELEMENTS),
    )
    cloud = ChemicalSetup(
        formula_matrix=jnp.eye(7)[:, [5]], hvector_func=lambda t: jnp.ones(1),
        elements=M1.ELEMENTS, species=("Fe(s)",),
    )
    setup = M1.build_condensate_chemical_setup(gas_setup=gas, condensate_setup=cloud)
    def fake_solve(setup, temperature, pressure, budget, *, options):
        np.testing.assert_allclose(budget, np.ones(7) / 7.)
        assert not options.rainout
        return SimpleNamespace(gas_n=np.full(7, 2. / 7.), condensate_amounts=np.zeros(1),
                               converged=True, status="converged")
    monkeypatch.setattr(M1, "solve_condensate", fake_solve)
    result = M1.solve_parcel(setup, 500., 1., np.ones(7))
    assert result["solver_converged"]
    assert not result["accepted"]
    np.testing.assert_allclose(result["relative_element_residual"], 1.)


def test_failed_gas_solver_is_rejected_despite_valid_independent_audit(monkeypatch):
    setup = ChemicalSetup(
        formula_matrix=jnp.eye(7), hvector_func=lambda t: jnp.full(7, np.log(7.)),
        elements=M1.ELEMENTS, species=tuple(e + "1" for e in M1.ELEMENTS),
    )
    assert M1.audit_parcel(setup, 500., 1., np.ones(7), np.ones(7))["accepted"]

    def fake_solve(setup, temperature, pressure, budget, *, options, return_diagnostics):
        assert return_diagnostics
        return SimpleNamespace(n=np.full(7, 1. / 7.)), {"converged": False}

    monkeypatch.setattr(M1, "solve_gas", fake_solve)
    result = M1.solve_parcel(setup, 500., 1., np.ones(7))
    assert not result["solver_converged"]
    assert not result["accepted"]
    np.testing.assert_allclose(result["relative_element_residual"], 0., atol=1e-14)


def test_source_failure_preserves_attempted_diagnostic_inputs(monkeypatch):
    network, case, budget = M1.source_inputs(1.1)
    points = ((1400., 0.1), (1000., 0.01))
    monkeypatch.setattr(M1, "provenance", lambda: {})

    def failed_source(network, case, *, element_amounts_mol, pressure_bar):
        np.testing.assert_array_equal(element_amounts_mol, budget)
        assert pressure_bar == 100.
        raise RuntimeError("Controlled source failure")

    monkeypatch.setattr(M1.SOURCE, "solve_reduced_source", failed_source)
    record = M1.run_diagnostic(100., oxygen_factor=1.1, points=points)
    assert record["source"] is None
    assert record["parcels"] == []
    assert record["failures"][0]["stage"] == "source"
    assert record["failures"][0]["reason"] == "Controlled source failure"
    assert record["inputs"] == {
        "source_T_K": 2350., "source_P_bar": 100.,
        "source_model_id": network["model_id"], "source_case_id": case["id"],
        "elements": network["elements"], "element_amounts_mol": budget.tolist(),
        "upper_points_T_K_P_bar": [list(point) for point in points],
    }
    assert record["oxygen_factor"] == 1.1


def test_real_shared_gas_has_separate_source_and_upper_model_residuals():
    network, case, budget = M1.source_inputs()
    source = M1.SOURCE.solve_reduced_source(network, case, element_amounts_mol=budget,
                                           pressure_bar=100.)
    assert source["accepted"]
    assert np.max(np.abs(source["reaction_residual"])) < M1.CHEMICAL_TOLERANCE
    shared, upper = M1.build_setups()
    gas_budget = M1.source_gas_inventory(network, source)
    result = M1.solve_parcel(shared, 2350., source["P_bar"], gas_budget)
    assert result["accepted"]
    np.testing.assert_allclose(np.asarray(shared.formula_matrix) @ result["gas_amounts_mol"], gas_budget, rtol=1e-9, atol=0.)
    comparison = M1.shared_reaction_comparison(network, case, result)
    assert comparison["source_reaction_indices"]
    assert np.max(np.abs(comparison["source_reaction_residual"])) > 1e-3
    assert result["gas_stationarity_max_abs"] < M1.CHEMICAL_TOLERANCE
    expanded = M1.solve_parcel(upper.gas_setup, 1400., 0.1, gas_budget)
    assert expanded["accepted"]
    np.testing.assert_allclose(
        np.asarray(upper.gas_setup.formula_matrix) @ expanded["gas_amounts_mol"],
        gas_budget, rtol=1e-9, atol=0.,
    )


@pytest.fixture(scope="module")
def retained_parcel_setup():
    assert tuple(PARCELS["elements"]) == M1.ELEMENTS
    return M1.build_setups()[1]


@pytest.mark.parametrize("case", PARCELS["cases"], ids=lambda case: case["id"])
def test_native_condensate_accepts_saved_retained_parcels(retained_parcel_setup, case):
    report = M1.solve_parcel(
        retained_parcel_setup, case["temperature_k"], case["pressure_bar"],
        np.asarray(case["element_amounts_mol"]),
    )

    _assert_native_parcel_accepted(report, case["condensate_expected"])


def _assert_native_parcel_accepted(report, condensate_expected):
    assert report["solver_status"] == "converged"
    assert report["solver_converged"]
    assert report["accepted"]
    assert np.max(np.abs(report["relative_element_residual"])) < 1e-9
    assert abs(report["relative_mass_residual"]) < 1e-9
    assert report["gas_stationarity_max_abs"] < 1e-8
    assert report["present_condensate_residual_max_abs"] < 1e-8
    assert report["absent_condensate_violation_max_abs"] < 1e-8
    cloud = np.asarray(report["condensate_amounts_mol"])
    eligible = np.asarray(report["condensate_temperature_eligible"])
    assert np.all(cloud >= 0.)
    assert not np.any((cloud > 0.) & ~eligible)
    assert bool(np.any(cloud > 0.)) == condensate_expected


@pytest.mark.parametrize("pressure_factor, amount_scale", [(0.99, 1.), (1.01, 1.), (1., 1e20)])
@pytest.mark.parametrize("case_id", [
    "coupled_32_layer_18_1000K",
    "fixed_oxygen_0p9_128_layer_63_1000K",
    "fixed_oxygen_1p1_128_layer_86_1000K",
    "contact_oxygen_0p9_128_layer_1_1000K",
])
def test_native_retained_column_parcel_pressure_and_amount_gauge(
    retained_parcel_setup, pressure_factor, amount_scale, case_id,
):
    case = next(case for case in PARCELS["cases"] if case["id"] == case_id)
    budget = np.asarray(case["element_amounts_mol"])
    report = M1.solve_parcel(
        retained_parcel_setup, case["temperature_k"], case["pressure_bar"] * pressure_factor,
        budget * amount_scale,
    )
    _assert_native_parcel_accepted(report, True)
    np.testing.assert_allclose(
        np.asarray(report["gas_element_amounts_mol"]) + report["cloud_element_amounts_mol"],
        budget * amount_scale, rtol=1e-9, atol=0.,
    )
    if amount_scale != 1.:
        reference = M1.solve_parcel(
            retained_parcel_setup, case["temperature_k"], case["pressure_bar"], budget,
        )
        _assert_native_parcel_accepted(reference, True)
        for field in ("gas_amounts_mol", "condensate_amounts_mol"):
            np.testing.assert_allclose(
                np.asarray(report[field]) / amount_scale, reference[field], rtol=1e-8, atol=1e-15,
            )
        assert report["gas_mass_fraction"] == pytest.approx(reference["gas_mass_fraction"], rel=1e-9)


@pytest.mark.parametrize("case", [
    case for case in PARCELS["cases"]
    if case.get("provider_base_commit") == "e529f546e9039e0d35399cb811d25715efae1439"
], ids=lambda case: case["id"])
@pytest.mark.parametrize("perturbation", [
    "temperature_ulp_up", "temperature_ulp_down",
    "pressure_ulp_up", "pressure_ulp_down",
    "pressure_relative_up", "pressure_relative_down",
    "inventory_ulp_up", "inventory_ulp_down", "oxygen_relative", "amount_scale",
])
def test_native_retained_parcel_dual_termination_neighborhood(
    retained_parcel_setup, case, perturbation,
):
    temperature, pressure = case["temperature_k"], case["pressure_bar"]
    budget = np.asarray(case["element_amounts_mol"])
    direction = np.inf if perturbation.endswith("up") else -np.inf
    if perturbation.startswith("temperature"):
        temperature = np.nextafter(temperature, direction)
    elif perturbation.startswith("pressure_ulp"):
        pressure = np.nextafter(pressure, direction)
    elif perturbation.startswith("pressure_relative"):
        pressure *= 1.0 + (1.0e-12 if direction > 0.0 else -1.0e-12)
    elif perturbation.startswith("inventory"):
        budget = np.nextafter(budget, direction)
    elif perturbation == "oxygen_relative":
        budget[M1.ELEMENTS.index("O")] *= 1.0 + 1.0e-12
    else:
        budget *= 1.0e20
    report = M1.solve_parcel(retained_parcel_setup, temperature, pressure, budget)
    _assert_native_parcel_accepted(report, True)
