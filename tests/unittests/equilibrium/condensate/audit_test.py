"""Primitive KKT audits reject conservative composition and phase mistakes."""

import jax.numpy as jnp
import numpy as np

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
