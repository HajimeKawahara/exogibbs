"""Independent references and exact amount bases for the M2 contact audit."""

import importlib
from pathlib import Path
import sys

import numpy as np
import pytest


DIRECTORY = Path(__file__).resolve().parents[3] / "examples" / "metal_silicate"
NAMES = ("local", "reference", "source", "hydrogen", "full_potential", "common_gibbs",
         "melts_coupled", "run_melts_reference", "run_bse_common_gibbs", "m1_chemistry", "m2_standards_audit")
previous = {name: sys.modules.get(name) for name in NAMES}
sys.path.insert(0, str(DIRECTORY))
try:
    for name in NAMES:
        sys.modules.pop(name, None)
    AUDIT = importlib.import_module("m2_standards_audit")
finally:
    sys.path.pop(0)
    for name, module in previous.items():
        if module is None:
            sys.modules.pop(name, None)
        else:
            sys.modules[name] = module


def test_element_gauge_does_not_hide_a_changed_water_formation_energy():
    matrix = np.array([[2., 0., 2.], [0., 2., 1.], [0., 0., 0.]])
    upper = np.array([-2., -3., -7.])
    lower = upper + matrix.T @ np.array([11., -37., 100.])
    kwargs = dict(species=["H2", "O2", "H2O"], elements=["H", "O", "Al"])
    result = AUDIT.audit_shared_standards(matrix, lower, upper, **kwargs)
    assert result["accepted"] and not result["gauge_unique"]
    lower[-1] += .1
    result = AUDIT.audit_shared_standards(matrix, lower, upper, **kwargs)
    assert not result["accepted"]
    reaction = np.array([-1., -.5, 1.])
    assert reaction @ np.asarray(result["nongauge_residual_rt"]) == pytest.approx(.1, abs=2e-14)


@pytest.mark.parametrize("temperature,expected", [
    (2173.15, [-.00278721138156, -.00024310315643, -.09313289842181, -.01576341579684]),
    (2350., [-.00269981279463, -.00133258339282, -.08641962946805, -.02104573125898]),
])
def test_independent_m1_table_exposes_all_four_gas_reaction_differences(temperature, expected):
    result = AUDIT.gas_standard_audit(temperature)
    assert not result["accepted"] and result["independent_reaction_rank"] == 4
    np.testing.assert_allclose([row["difference_rt"] for row in result["independent_reactions"]],
                               expected, atol=2e-11, rtol=0)
    for row in result["independent_reactions"]:
        assert not any(row["element_imbalance"])


def test_partial_pressure_comparison_preserves_exact_absence():
    kwargs = dict(species=["H2", "SiH4", "He"])
    assert AUDIT.audit_contact([1., 0., .1], [1., 0., .1], **kwargs)["accepted"]
    result = AUDIT.audit_contact([1., 0., .1], [1., 1e-300, .1], **kwargs)
    assert not result["accepted"] and result["support_mismatch_species"] == ["SiH4"]
    assert result["log_pressure_residual"][1] is None
    assert result["maximum_residual"] is None
    assert not AUDIT.audit_contact([1., 0., .1], [1.001, 0., .1], **kwargs)["accepted"]


@pytest.mark.parametrize("mass_fraction", [0., 1e-5, .01, .2])
def test_mass_fraction_conversion_preserves_hydrogen_and_total_mass(mass_fraction):
    host_amounts = np.array([.7, .2, .03])
    host_masses = np.array([.06008, .20377, .018015])
    host_mass = host_amounts @ host_masses
    hydrogen, fraction = AUDIT.h2_mass_fraction_to_amount(mass_fraction, host_mass, host_amounts.sum(), .00201588)
    recovered = hydrogen * .00201588 / (host_mass + hydrogen * .00201588)
    assert recovered == pytest.approx(mass_fraction, abs=3e-17)
    assert fraction == pytest.approx(hydrogen / (host_amounts.sum() + hydrogen), abs=1e-16)
    scaled, scaled_fraction = AUDIT.h2_mass_fraction_to_amount(mass_fraction, 17 * host_mass,
                                                            17 * host_amounts.sum(), .00201588)
    assert scaled == pytest.approx(17 * hydrogen)
    assert scaled_fraction == pytest.approx(fraction)


@pytest.mark.parametrize("fraction", [-1., 1., np.nan, np.inf])
def test_invalid_mass_fraction_is_not_clipped(fraction):
    with pytest.raises(ValueError, match="mass_fraction"):
        AUDIT.h2_mass_fraction_to_amount(fraction, .1, .8, .00201588)
