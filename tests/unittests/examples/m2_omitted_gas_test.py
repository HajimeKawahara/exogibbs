"""Trace-gas mixing entropy and unknown reference energies cannot be suppressed."""

import importlib.util
from pathlib import Path
import sys
from types import SimpleNamespace

import numpy as np
import pytest


path = Path(__file__).resolve().parents[3] / "examples/metal_silicate/m2_omitted_gas.py"
spec = importlib.util.spec_from_file_location("m2_omitted_gas_tested", path)
screen = importlib.util.module_from_spec(spec)
spec.loader.exec_module(screen)


def test_trace_gas_uses_equilibrium_fugacity_and_pressure_not_insertion_sign():
    result = screen.trace_gas_requirements([[1.]], [12.], [2.], [0.], 10., mole_fraction_target=1e-6)[0]
    assert result["log_mole_fraction"] == pytest.approx(-10. - np.log(10.))
    assert result["mole_fraction"] == pytest.approx(np.exp(-10.) / 10.)
    assert result["mole_fraction"] > 0  # Positive standard cost does not imply exact gas absence.
    assert not result["fixed_reservoir_trace_target_passed"]


def test_common_element_gauge_cancels_from_trace_estimate():
    a = np.array([[2., 1.], [1., 2.]])
    lam, gauge, delta = np.array([3., -2.]), np.array([1., 4.]), np.array([14., -37.])
    first = screen.trace_gas_requirements(a, [1., 2.], lam, gauge, 100., mole_fraction_target=1e-8)
    second = screen.trace_gas_requirements(a, [1., 2.], lam + delta, gauge + delta, 100., mole_fraction_target=1e-8)
    assert [row["log_mole_fraction"] for row in first] == pytest.approx([row["log_mole_fraction"] for row in second])


def test_missing_atomic_reference_yields_a_checkable_inequality_not_a_zero_anchor():
    a = [[1.], [2.]]
    result = screen.trace_gas_requirements(a, [2.], [3., 4.], [1., None], 10., mole_fraction_target=1e-5)[0]
    assert result["log_mole_fraction"] is None
    assert result["fixed_reservoir_trace_target_passed"] is None
    assert result["missing_gauge_coefficients"] == [0., 2.]
    critical = result["required_missing_gauge_dot_lower_rt"] / 2
    for delta, passed in ((-1., False), (1., True)):
        explicit = screen.trace_gas_requirements(a, [2.], [3., 4.], [1., critical + delta],
                                                  10., mole_fraction_target=1e-5)[0]
        assert explicit["fixed_reservoir_trace_target_passed"] is passed


def test_log_estimate_survives_overflow_without_clipping_to_a_physical_fraction():
    result = screen.trace_gas_requirements([[1.]], [-1000.], [0.], [0.], 1., mole_fraction_target=.01)[0]
    assert result["log_mole_fraction"] == 1000.
    assert result["mole_fraction"] is None
    assert not result["fixed_reservoir_trace_target_passed"]


@pytest.mark.parametrize("gauge", [[np.nan], [np.inf]])
def test_nonfinite_reference_is_not_an_unknown_reference(gauge):
    with pytest.raises(ValueError):
        screen.trace_gas_requirements([[1.]], [0.], [0.], gauge, 1., mole_fraction_target=.01)


@pytest.mark.parametrize("record_key", ["record", "source_record"])
def test_provider_contact_and_planetary_source_records_are_screened(monkeypatch, record_key):
    from exogibbs.presets import fastchem4_cond

    elements = ["H", "He", "O", "Mg", "Si", "Fe", "Na", *screen.OMITTED_ELEMENTS]
    formula = np.zeros((len(elements), 1))
    formula[elements.index("Al"), 0] = 1.
    catalog = SimpleNamespace(elements=elements, species=["Al1"], formula_matrix=formula,
                              hvector_func=lambda temperature: np.zeros(1))
    monkeypatch.setattr(fastchem4_cond, "condensate_chemical_setup",
                        lambda **kwargs: SimpleNamespace(gas_setup=catalog))
    monkeypatch.setitem(sys.modules, "m1_chemistry", SimpleNamespace(provenance=lambda: {}))
    source = {"temperature_K": 2173.15, "pressure_bar": 200.,
              "source_internal_record": {"elements": elements},
              "source_internal_result": {"accepted": True, "elemental_potentials_rt": [0.] * len(elements)},
              "source_metadata": {"standards": {"evaluation_temperature_K": 2173.15,
                  "common_gas": {"elements": elements[:7], "element_gauge_rt": [0.] * 7}}},
              record_key: {"phases": {"metal": ["Fe_metal"]}},
              "source_result": {"component_amounts_mol": [1.]}}
    result = screen.screen_source(source)
    assert result["gas_candidates"][0]["species"] == "Al1"
    assert result["gas_candidates"][0]["status"] == "missing_atomic_reference"
    assert not result["accepted_omission_bound"]
