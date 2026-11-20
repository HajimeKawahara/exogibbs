"""Primitive KKT audits reject conservative composition and phase mistakes."""

import jax.numpy as jnp
import numpy as np
import pytest

from exogibbs.api.condensate import audit, build_condensate_chemical_setup
from exogibbs.thermo.models import ChemicalSetup


def setup(cloud_standard=4., upper=2000.):
    gas = ChemicalSetup(formula_matrix=jnp.array([[1., 0., 2.], [0., 1., 1.]]),
        elements=("H", "O"), species=("H", "O", "H2O"),
        hvector_func=lambda t: jnp.array([1., 2., 4.])-jnp.log(jnp.array([2., 3., 5.])))
    cloud = ChemicalSetup(formula_matrix=jnp.array([[2.], [1.]]), elements=gas.elements,
        species=("water",), hvector_func=lambda t: jnp.array([cloud_standard]),
        temperature_validity_upper=(upper,))
    return build_condensate_chemical_setup(gas_setup=gas, condensate_setup=cloud)


def test_fresh_kkt_and_common_element_gauge():
    base = audit(setup(), 1000., 10., [14., 9.], [2., 3., 5.], [1.])
    shifted = audit(setup(), 1000., 10., [14., 9.], [2., 3., 5.], [1.], element_gauge_rt=[11., -7.])
    assert base["accepted"] and shifted["accepted"]
    np.testing.assert_allclose(base["elemental_potentials_rt"], [1., 2.], atol=1e-14)
    np.testing.assert_allclose(shifted["gibbs_rt"]-base["gibbs_rt"], 14*11-9*7, atol=1e-13)
    tampered = audit(setup(), 1000., 10., [14., 9.], [1.8, 2.9, 5.1], [1.])
    assert max(abs(v) for v in tampered["relative_element_residual"]) < 1e-15
    assert not tampered["accepted"]


def test_absent_and_temperature_ineligible_clouds():
    assert audit(setup(5.), 1000., 10., [12., 8.], [2., 3., 5.], [0.])["accepted"]
    assert not audit(setup(3.), 1000., 10., [12., 8.], [2., 3., 5.], [0.])["accepted"]
    assert not audit(setup(4., upper=900.), 1000., 10., [14., 9.], [2., 3., 5.], [1.])["accepted"]
    assert audit(setup(3., upper=900.), 1000., 10., [12., 8.], [2., 3., 5.], [0.])["accepted"]


def test_exact_zero_element_excludes_its_species():
    result = audit(setup(), 1000., 10., [2., 0.], [2., 0., 0.], [0.])
    assert result["accepted"] and result["elemental_potentials_rt"][1] is None
    assert result["condensate_insertion_rt"] == [None]


def trace_setup(trace_standard=800., cloud_standard=1.):
    gas = ChemicalSetup(formula_matrix=jnp.array([[1., 2.]]),
        elements=("H",), species=("H", "H2"),
        hvector_func=lambda t: jnp.array([0., trace_standard]))
    cloud = ChemicalSetup(formula_matrix=jnp.array([[1.]]), elements=gas.elements,
        species=("solid",), hvector_func=lambda t: jnp.array([cloud_standard]))
    return build_condensate_chemical_setup(gas_setup=gas, condensate_setup=cloud)


def test_underflow_logs_preserve_independent_stationarity_and_atom_bound():
    with pytest.raises(ValueError, match="Reachable gas species"):
        audit(trace_setup(), 300., 1., [1.], [1., 0.], [0.])
    result = audit(trace_setup(), 300., 1., [1.], [1., 0.], [0.],
                   gas_log_amounts_mol=[0., -800.])
    assert result["accepted"]
    assert result["gas_underflow_species"] == ["H2"]
    assert result["gas_stationarity_max_abs"] == 0.
    assert result["gas_underflow_element_bound_mol"][0] >= 2*np.nextafter(0., 1.)
    # Finite logs alone are not sufficient: chemical and phase KKT still apply.
    wrong_mu = audit(trace_setup(799.), 300., 1., [1.], [1., 0.], [0.],
                     gas_log_amounts_mol=[0., -800.])
    wrong_phase = audit(trace_setup(cloud_standard=-1.), 300., 1., [1.], [1., 0.], [0.],
                        gas_log_amounts_mol=[0., -800.])
    wrong_budget = audit(trace_setup(), 300., 1., [1.01], [1., 0.], [0.],
                         gas_log_amounts_mol=[0., -800.])
    assert not wrong_mu["accepted"] and not wrong_phase["accepted"] and not wrong_budget["accepted"]


@pytest.mark.parametrize("amounts, logs", [
    ([1., 0.], [0., -500.]),  # Arbitrary truncation is not underflow.
    ([1., 1e-30], [0., -800.]),  # A floor cannot stand in for a trace amount.
    ([1., 0.], [0., -np.inf]),
    ([1., 0.], [0., np.nan]),
    ([1., 0.], [0., np.inf]),
    ([1., 0.], [0.]),
    ([1.1, 0.], [0., -800.]),
    ([1., 0.], [800., -800.]),
])
def test_log_amounts_reject_inconsistent_or_nonfinite_representation(amounts, logs):
    with pytest.raises(ValueError):
        audit(trace_setup(), 300., 1., [1.], amounts, [0.], gas_log_amounts_mol=logs)


def test_normal_log_audit_matches_linear_audit_and_gauge():
    plain = audit(setup(), 1000., 10., [14., 9.], [2., 3., 5.], [1.])
    logged = audit(setup(), 1000., 10., [14., 9.], [2., 3., 5.], [1.],
                   gas_log_amounts_mol=np.log([2., 3., 5.]))
    shifted = audit(setup(), 1000., 10., [14., 9.], [2., 3., 5.], [1.],
                    gas_log_amounts_mol=np.log([2., 3., 5.]), element_gauge_rt=[11., -7.])
    assert plain["accepted"] and logged["accepted"] and shifted["accepted"]
    assert logged["gas_underflow_species"] == []
    assert logged["gas_underflow_element_bound_mol"] == [0., 0.]
    np.testing.assert_allclose(logged["elemental_potentials_rt"], plain["elemental_potentials_rt"], atol=1e-14)
    np.testing.assert_allclose(shifted["gibbs_rt"]-logged["gibbs_rt"], 14*11-9*7, atol=1e-13)


def test_log_exact_exclusion_requires_minus_infinity():
    result = audit(setup(), 1000., 10., [2., 0.], [2., 0., 0.], [0.],
                   gas_log_amounts_mol=[np.log(2.), -np.inf, -np.inf])
    assert result["accepted"] and result["gas_underflow_species"] == []
    with pytest.raises(ValueError, match="excluded gas logs"):
        audit(setup(), 1000., 10., [2., 0.], [2., 0., 0.], [0.],
              gas_log_amounts_mol=[np.log(2.), -800., -800.])


def test_subnormal_rounding_is_distinct_from_underflow_and_truncation():
    logs = np.array([0., -744.])
    amounts = np.exp(logs)
    result = audit(trace_setup(744.), 300., 1., [1.], amounts, [0.], gas_log_amounts_mol=logs)
    assert result["accepted"] and result["gas_underflow_species"] == []
    with pytest.raises(ValueError, match="zeros"):
        audit(trace_setup(744.), 300., 1., [1.], [1., 0.], [0.], gas_log_amounts_mol=logs)


def test_omitted_atom_bound_prevents_unresolved_trace_element_acceptance():
    gas = ChemicalSetup(formula_matrix=jnp.array([[1., 0.], [0., 1.]]),
        elements=("H", "O"), species=("H", "O"),
        hvector_func=lambda t: jnp.array([0., 800.]))
    cloud = ChemicalSetup(formula_matrix=jnp.array([[1.], [0.]]),
        elements=gas.elements, species=("solid",), hvector_func=lambda t: jnp.array([1.]))
    tiny = np.nextafter(0., 1.)
    result = audit(build_condensate_chemical_setup(gas_setup=gas, condensate_setup=cloud),
                   300., 1., [1., tiny], [1., 0.], [0.], gas_log_amounts_mol=[0., -800.])
    assert not result["accepted"]
    assert result["relative_element_residual_upper_bound"][1] > 1.


def test_absolute_log_conversion_recovers_a_normalized_underflow():
    normalized_logs = np.array([0., -800.])
    scale = 1e100
    absolute_logs = normalized_logs+np.log(scale)
    assert np.exp(normalized_logs)[1] == 0.
    assert np.exp(absolute_logs)[1] > 0.
    result = audit(trace_setup(), 300., 1., [scale], np.exp(absolute_logs), [0.],
                   gas_log_amounts_mol=absolute_logs)
    assert result["accepted"] and result["gas_underflow_species"] == []
    with pytest.raises(ValueError, match="zeros"):
        audit(trace_setup(), 300., 1., [scale], np.exp(normalized_logs)*scale, [0.],
              gas_log_amounts_mol=absolute_logs)


def test_relative_omission_bound_rounds_outward_even_below_float_range():
    scale = 1e100
    logs = np.array([np.log(scale), -800.])
    result = audit(trace_setup(800.+np.log(scale)), 300., 1., [scale], np.exp(logs), [0.],
                   gas_log_amounts_mol=logs)
    assert result["accepted"]
    assert result["relative_element_residual_upper_bound"][0] > 0.
