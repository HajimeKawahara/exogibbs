"""Independent thermodynamic identities and finite-source trace counterexamples."""

import importlib.util
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import numpy as np
import pytest


EXAMPLE = Path(__file__).resolve().parents[3] / "examples/metal_silicate"
sys.path.insert(0, str(EXAMPLE))
try:
    spec = importlib.util.spec_from_file_location("m2_janaf_tested", EXAMPLE / "m2_janaf.py")
    janaf = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(janaf)
finally:
    sys.path.remove(str(EXAMPLE))


def test_atomic_energy_uses_298_formation_enthalpy_not_temperature_formation_g():
    # K(g) is the JANAF reference element at 2200 K: delta-f G(T)=0.
    # Its absolute G in this convention is nevertheless large and negative.
    value = janaf.atomic_standard_audit(2200.)["atomic_standards"]["K"]["value_rt"]
    expected = ((89.000 + 39.627) * 1000. - 2200. * 201.931) / (8.31446261815324 * 2200.)
    assert value == pytest.approx(expected, abs=1e-13)
    assert value < -10.


def test_tabulated_g_function_cross_checks_independently_transcribed_h_and_s():
    report = janaf.atomic_standard_audit(2173.15)
    assert max(r["g_function_reconstruction_max_abs_J_mol"] for r in report["atomic_standards"].values()) <= 2.901
    assert not report["physical_cross_phase_calibration_accepted"]
    assert not report["interpolation_error_bounded"]
    assert not report["empirical_uncertainty_available"]


def test_interpolation_reproduces_g_and_entropy_at_interior_nodes():
    data = json.loads(janaf.DATA_PATH.read_bytes())
    r, t, step = data["common_R_J_mol_K"], 2200., .001
    plus, minus = janaf.atomic_standard_audit(t + step), janaf.atomic_standard_audit(t - step)
    for element, row in data["rows"].items():
        entropy = row["data"][1][1]
        derivative = (plus["atomic_standards"][element]["value_rt"] * r * (t + step)
                      - minus["atomic_standards"][element]["value_rt"] * r * (t - step)) / (2 * step)
        assert derivative == pytest.approx(-entropy, abs=1e-5)


@pytest.mark.parametrize("t", [True, np.nan, np.inf, 298.15, 2099.99, 2400.01])
def test_domain_is_explicit_and_never_extrapolated(t):
    with pytest.raises(ValueError):
        janaf.janaf_atomic_references(t)


def test_pinned_source_bytes_are_checked(monkeypatch, tmp_path):
    changed = tmp_path / "data.json"
    changed.write_bytes(janaf.DATA_PATH.read_bytes() + b" ")
    monkeypatch.setattr(janaf, "DATA_PATH", changed)
    with pytest.raises(ValueError, match="changed"):
        janaf.atomic_standard_audit(2173.15)


def demand(log_x, budget, gas=10.):
    return janaf.fixed_reservoir_demand([[2.], [0.]], [log_x], gas, budget,
                                       elements=["P", "O"], species=["P2"], mole_fraction_target=.01)


def test_small_gas_fraction_can_exceed_a_trace_element_inventory():
    result = demand(np.log(.001), [.001, 1.])
    assert result["sum_mole_fractions"] == pytest.approx(.001)
    assert not result["fraction_sum_at_least_one"]
    assert result["species_trace_target_exceeded"] == []
    assert result["source_budget_exceeded_elements"] == ["P"]
    assert result["element_demands"][0]["demand_over_source_budget"] == pytest.approx(20.)
    assert not result["is_equilibrated_finite_state"]
    assert not result["trace_screen_passed"]
    assert not result["accepted_omission_bound"]


def test_source_amount_scaling_changes_moles_but_not_fractional_demand():
    first, second = demand(np.log(.001), [.001, 1.]), demand(np.log(.001), [.007, 7.], gas=70.)
    assert second["element_demands"][0]["demand_mol"] == pytest.approx(first["element_demands"][0]["demand_mol"] * 7)
    assert second["element_demands"][0]["demand_over_source_budget"] == pytest.approx(first["element_demands"][0]["demand_over_source_budget"])


def test_log_demand_overflow_and_zero_inventory_are_explicit():
    result = demand(1000., [0., 1.])
    row = result["element_demands"][0]
    assert row["log_demand_mol"] == pytest.approx(1000. + np.log(20.))
    assert row["demand_mol"] is None
    assert row["demand_over_source_budget"] is None
    assert row["source_budget_exceeded"]
    assert result["fraction_sum_at_least_one"]
    assert result["species_trace_target_exceeded"] == ["P2"]
    assert len(result["reasons"]) == 3


def test_passing_trace_screen_does_not_promote_to_coupled_error_bound():
    result = demand(-30., [1., 1.])
    assert result["trace_screen_passed"]
    assert not result["accepted_omission_bound"]
    assert not result["is_equilibrated_finite_state"]


@pytest.mark.parametrize("a, log_x, budget", [([[-1.]], [0.], [1.]), ([[1.]], [np.nan], [1.]), ([[1.]], [0.], [-1.])])
def test_invalid_demands_are_rejected(a, log_x, budget):
    with pytest.raises(ValueError):
        janaf.fixed_reservoir_demand(a, log_x, 10., budget,
                                     elements=["P"], species=["P"], mole_fraction_target=.01)


def test_wrapper_uses_the_local_ledger_and_preserves_existing_anchors(monkeypatch):
    from exogibbs.presets import fastchem4_cond
    elements = ["Mg", "Fe", "Na", "P"]
    source = {"temperature_K": 2173.15, "pressure_bar": 200.,
              "source_record": {"phases": {"silicate": ["P_host"], "gas": ["Mg_gas", "Fe_gas"]}, "component_formulas": {"P_host": {"P": 2}, "Mg_gas": {"Mg": 1.}, "Fe_gas": {"Fe": 1.}}, "gas_species_aliases": {"Mg_gas": "Mg1", "Fe_gas": "Fe1"}},
              "source_result": {"accepted": True, "component_amounts_mol": [.01, 60., 40.]},
              "source_atmosphere_parcel": {"accepted": True, "T_K": 2173.15, "P_bar": 200., "gas_amounts_mol": [60., 40.], "gas_species": ["Mg1", "Fe1"]},
              "planetary_inventory": {"P": 1e40}}
    gauge = {e: -7. for e in elements[:3]}
    monkeypatch.setattr(janaf, "screen_source", lambda *a, **k: {"elements": elements,
        "atomic_gauge_rt": dict(gauge), "gas_candidates": [{"species": "P2", "formula": {"P": 2}, "log_mole_fraction": np.log(.001)}]})
    monkeypatch.setattr(fastchem4_cond, "condensate_chemical_setup", lambda **kw:
                        SimpleNamespace(gas_setup=SimpleNamespace(species=[e + "1" for e in [*elements[:3], *janaf.OMITTED_ELEMENTS]], elements=[*elements[:3], *janaf.OMITTED_ELEMENTS], formula_matrix=np.eye(9), hvector_func=lambda t: np.zeros(9))))
    result = janaf.screen_source_janaf(source)
    assert result["atomic_gauge_rt"] == gauge
    row = result["fixed_reservoir_demand"]["element_demands"][-1]
    assert row["source_budget_mol"] == .02
    assert row["demand_over_source_budget"] == pytest.approx(10.)

    source["source_atmosphere_parcel"]["gas_amounts_mol"] = [50., 50.]
    with pytest.raises(ValueError, match="amount scale"):
        janaf.screen_source_janaf(source)
