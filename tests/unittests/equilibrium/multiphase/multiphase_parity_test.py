"""Reduction to the existing public gas and pure-condensate APIs."""

import jax.numpy as jnp
import numpy as np

from exogibbs.api import condensate, gas, multiphase as mp
from exogibbs.api.chemistry import ChemicalSetup
from exogibbs.equilibrium.multiphase.types import R_SCALE as R


def test_gas_only_matches_existing_api_on_the_same_absolute_basis():
    matrix = jnp.array([[1., 0., 2.], [0., 1., 1.]])
    standard = jnp.array([0., 0., -2.])
    budget = jnp.array([2., 1.])
    setup = ChemicalSetup(formula_matrix=matrix, elements=("H", "O"),
                           species=("H", "O", "H2O"),
                           hvector_func=lambda t: standard)
    old = gas.solve(setup, T=1000., P=3., b=budget,
                    options=gas.EquilibriumOptions(epsilon_crit=1e-12))

    def gibbs(t, p, n, parameters):
        return R * t * (n @ (standard + jnp.log(n / n.sum()) + jnp.log(p / 1e5)))

    problem = mp.Problem(setup.elements, (mp.Phase("gas", setup.species, matrix, gibbs),))
    result = mp.solve(problem, temperature_k=1000., pressure_pa=3e5,
                      element_amounts_mol=budget, init=[1., .5, .5])
    assert result.converged and result.local_kkt_valid and result.minimum_valid
    np.testing.assert_allclose(result.amounts_mol, old.n, rtol=1e-8, atol=1e-10)
    assert abs(float(result.gibbs_j - gibbs(1000., 3e5, old.n, ()))) / (R * 1000 * 3) < 1e-9


def test_gas_plus_pure_condensate_matches_existing_api():
    elements = ("H", "O")
    gas_setup = ChemicalSetup(formula_matrix=jnp.eye(2), elements=elements,
                               species=("H", "O"), hvector_func=lambda t: jnp.zeros(2))
    pure_setup = ChemicalSetup(formula_matrix=jnp.array([[2.], [1.]]), elements=elements,
                                species=("H2O_s",), hvector_func=lambda t: jnp.zeros(1))
    setup = condensate.build_condensate_chemical_setup(
        gas_setup=gas_setup, condensate_setup=pure_setup,
    )
    old = condensate.solve(setup, T=1000., P=10., b=jnp.ones(2))

    def gibbs(t, p, n, parameters):
        return R * t * (n @ jnp.log(n / n.sum() * (p / 1e5)))

    problem = mp.Problem(elements, (
        mp.Phase("gas", gas_setup.species, gas_setup.formula_matrix, gibbs),
        mp.Phase("water", pure_setup.species, pure_setup.formula_matrix,
                 lambda t, p, n, parameters: n[0] * 0.),
    ))
    result = mp.solve(problem, temperature_k=1000., pressure_pa=1e6,
                      element_amounts_mol=[1., 1.], init=[.4, .7, .3])
    assert result.converged and result.local_kkt_valid and result.minimum_valid
    np.testing.assert_allclose(result.amounts_mol[:2], old.gas_n, rtol=1e-7, atol=1e-10)
    np.testing.assert_allclose(result.amounts_mol[2:], old.condensate_amounts, rtol=1e-7, atol=1e-10)
