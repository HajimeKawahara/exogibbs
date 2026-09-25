"""Trace-gas mixing entropy and unknown reference energies cannot be suppressed."""

import importlib.util
from pathlib import Path

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
