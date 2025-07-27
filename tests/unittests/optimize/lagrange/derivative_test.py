import pytest
import jax.numpy as jnp
from jax import config
from exogibbs.optimize.lagrange.minimize import minimize_gibbs_core
from exogibbs.optimize.lagrange.derivative import derivative_temperature
from exogibbs.optimize.lagrange.core import compute_ln_normalized_pressure
from exogibbs.test.analytic_hsystem import HSystem
from exogibbs.optimize.lagrange.core import _A_diagn_At 


def test_derivative_temperature_h_system():
    """Test derivative_temperature against analytical H system solution."""
    config.update("jax_enable_x64", True)
    
    # Initialize H system
    hsystem = HSystem()
    
    # Test parameters
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
    
    # Run Gibbs minimization
    ln_nk_result, ln_ntot_result, counter = minimize_gibbs_core(
        temperature,
        ln_normalized_pressure,
        b_element_vector,
        ln_nk,
        ln_ntot,
        formula_matrix,
        hvector_func,
        epsilon_crit=1e-11,
        max_iter=1000,
    )
    
    # Test derivative computation
    hdot = jnp.array([hsystem.dot_hv_h(temperature), hsystem.dot_hv_h2(temperature)])
    nk_result = jnp.exp(ln_nk_result)
    
    # Compute derivatives
    Bmatrix = _A_diagn_At(nk_result, formula_matrix)
    nk_cdot_hdot = jnp.dot(nk_result, hdot)    
    ln_nspecies_dT = derivative_temperature(nk_result, formula_matrix, hdot, nk_cdot_hdot, Bmatrix, b_element_vector)

    # Get reference analytical derivatives
    refH = hsystem.ln_nH_dT(jnp.array([temperature]), ln_normalized_pressure)[0]
    refH2 = hsystem.ln_nH2_dT(jnp.array([temperature]), ln_normalized_pressure)[0]
    
    # Test differences are small
    diff_h = refH - ln_nspecies_dT[0]
    diff_h2 = refH2 - ln_nspecies_dT[1]
    
    #print(f"Difference for H: {diff_h}, Difference for H2: {diff_h2}")
    #Difference for H: -7.51091897011058e-15, Difference for H2: -3.2149847020712663e-15 (July 21th 2025)
    
    assert jnp.abs(diff_h) < 1e-14, f"H derivative difference too large: {diff_h}"
    assert jnp.abs(diff_h2) < 1e-14, f"H2 derivative difference too large: {diff_h2}"

if __name__ == "__main__":
    test_derivative_temperature_h_system()
    print("Test passed!")