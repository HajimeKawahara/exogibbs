import pytest
import jax.numpy as jnp
from jax import config
from exogibbs.optimize.lagrange.minimize import minimize_gibbs_core
from exogibbs.test.analytic_hsystem import HSystem


def test_minimize_gibbs_core_h_system():
    """Test minimize_gibbs_core against analytical H system solution."""
    config.update("jax_enable_x64", True)
    
    # Initialize H system
    hsystem = HSystem()
    
    # Test parameters from main section
    formula_matrix = jnp.array([[1.0, 2.0]])
    temperature = 3500.0
    P = 1.0
    
    normalized_pressure = P / hsystem.P_ref
    ln_nk = jnp.array([0.0, 0.0])
    ln_ntot = 0.0
    hvector = jnp.array([hsystem.hv_h(temperature), hsystem.hv_h2(temperature)])
    b_element_vector = jnp.array([1.0])
    
    #set criterions
    epsilon_crit = 1e-11
    max_iter = 1000

    # Run Gibbs minimization
    ln_nk_result, ln_ntot_result, counter = minimize_gibbs_core(
        temperature,
        normalized_pressure,
        b_element_vector,
        ln_nk,
        ln_ntot,
        formula_matrix,
        hvector,
        epsilon_crit=epsilon_crit,
        max_iter=max_iter,
    )
    
    # Compare with analytical solution
    k = hsystem.compute_k(P, temperature)
    diff_h = jnp.log(hsystem.nh(k)) - ln_nk_result[0]
    diff_h2 = jnp.log(hsystem.nh2(k)) - ln_nk_result[1]
    
    # Validate numerical accuracy
    assert jnp.abs(diff_h) < epsilon_crit
    assert jnp.abs(diff_h2) < epsilon_crit
    assert counter < max_iter


if __name__ == "__main__":
    test_minimize_gibbs_core_h_system()
    print("Test passed!")