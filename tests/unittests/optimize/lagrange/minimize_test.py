import pytest
import jax.numpy as jnp
from jax import config
from jax import jacrev
from exogibbs.optimize.lagrange.minimize import minimize_gibbs_core, minimize_gibbs
from exogibbs.optimize.lagrange.core import compute_ln_normalized_pressure
from exogibbs.test.analytic_hsystem import HSystem


@pytest.fixture
def h_system_setup():
    """Setup common test parameters for H system tests."""
    config.update("jax_enable_x64", True)
    
    hsystem = HSystem()
    
    # Common test parameters
    formula_matrix = jnp.array([[1.0, 2.0]])
    temperature = 3500.0
    P = 1.0
    Pref = 1.0
    
    ln_normalized_pressure = compute_ln_normalized_pressure(P, Pref)
    ln_nk = jnp.array([0.0, 0.0])
    ln_ntot = 0.0
    
    def hvector_func(temperature): 
        return jnp.array([hsystem.hv_h(temperature), hsystem.hv_h2(temperature)])
    
    b_element_vector = jnp.array([1.0])
    epsilon_crit = 1e-11
    max_iter = 1000
    
    return {
        'hsystem': hsystem,
        'formula_matrix': formula_matrix,
        'temperature': temperature,
        'P': P,
        'ln_normalized_pressure': ln_normalized_pressure,
        'ln_nk': ln_nk,
        'ln_ntot': ln_ntot,
        'hvector_func': hvector_func,
        'b_element_vector': b_element_vector,
        'epsilon_crit': epsilon_crit,
        'max_iter': max_iter
    }


def test_minimize_gibbs_core_h_system(h_system_setup):
    """Test minimize_gibbs_core against analytical H system solution."""
    setup = h_system_setup
    
    # Run Gibbs minimization
    ln_nk_result, ln_ntot_result, counter = minimize_gibbs_core(
        setup['temperature'],
        setup['ln_normalized_pressure'],
        setup['b_element_vector'],
        setup['ln_nk'],
        setup['ln_ntot'],
        setup['formula_matrix'],
        setup['hvector_func'],
        epsilon_crit=setup['epsilon_crit'],
        max_iter=setup['max_iter'],
    )
    
    # Compare with analytical solution
    k = setup['hsystem'].compute_k(setup['ln_normalized_pressure'], setup['temperature'])
    diff_h = jnp.log(setup['hsystem'].nh(k)) - ln_nk_result[0]
    diff_h2 = jnp.log(setup['hsystem'].nh2(k)) - ln_nk_result[1]
    
    # Validate numerical accuracy
    assert jnp.abs(diff_h) < setup['epsilon_crit']
    assert jnp.abs(diff_h2) < setup['epsilon_crit']
    assert counter < setup['max_iter']


def test_minimize_gibbs_temperature_gradient_h_system(h_system_setup):
    """Test minimize_gibbs temperature gradient against analytical H system."""
    setup = h_system_setup
    
    # Compute temperature gradient using jacrev
    dln_dT = jacrev(lambda temperature_in: minimize_gibbs(
        temperature_in,
        setup['ln_normalized_pressure'],
        setup['b_element_vector'],
        setup['ln_nk'],
        setup['ln_ntot'],
        setup['formula_matrix'],
        setup['hvector_func'],
        epsilon_crit=setup['epsilon_crit'],
        max_iter=setup['max_iter'],
    ))(setup['temperature'])
    
    # Get analytical reference derivatives
    refH = setup['hsystem'].ln_nH_dT(jnp.array([setup['temperature']]), setup['ln_normalized_pressure'])[0]
    refH2 = setup['hsystem'].ln_nH2_dT(jnp.array([setup['temperature']]), setup['ln_normalized_pressure'])[0]
    
    # Test differences are small
    diff_h = refH - dln_dT[0]
    diff_h2 = refH2 - dln_dT[1]
    
    assert jnp.abs(diff_h) < 1e-11, f"H gradient difference too large: {diff_h}"
    assert jnp.abs(diff_h2) < 1e-11, f"H2 gradient difference too large: {diff_h2}"


def test_minimize_gibbs_pressure_gradient_h_system(h_system_setup):
    """Test minimize_gibbs pressure gradient against analytical H system."""
    setup = h_system_setup
    
    # Compute pressure gradient using jacrev
    dln_dlogp = jacrev(lambda ln_normalized_pressure_in: minimize_gibbs(
        setup['temperature'],
        ln_normalized_pressure_in,
        setup['b_element_vector'],
        setup['ln_nk'],
        setup['ln_ntot'],
        setup['formula_matrix'],
        setup['hvector_func'],
        epsilon_crit=setup['epsilon_crit'],
        max_iter=setup['max_iter'],
    ))(setup['ln_normalized_pressure'])
    
    # Get analytical reference pressure derivatives
    refH = setup['hsystem'].ln_nH_dlogp(jnp.array([setup['temperature']]), setup['ln_normalized_pressure'])[0]
    refH2 = setup['hsystem'].ln_nH2_dlogp(jnp.array([setup['temperature']]), setup['ln_normalized_pressure'])[0]
    
    # Test differences are small
    diff_h = refH - dln_dlogp[0]
    diff_h2 = refH2 - dln_dlogp[1]
    
    assert jnp.abs(diff_h) < 1e-11, f"H pressure gradient difference too large: {diff_h}"
    assert jnp.abs(diff_h2) < 1e-11, f"H2 pressure gradient difference too large: {diff_h2}"


if __name__ == "__main__":
    import pytest
    pytest.main([__file__, "-v"])