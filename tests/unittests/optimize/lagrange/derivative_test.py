import pytest
import jax.numpy as jnp
from jax import config
from exogibbs.optimize.minimize import minimize_gibbs_core
from exogibbs.optimize.derivative import derivative_temperature, derivative_pressure, derivative_element_one
from exogibbs.optimize.core import compute_ln_normalized_pressure
from exogibbs.test.analytic_hsystem import HSystem
from exogibbs.test.analytic_hcosystem import HCOSystem, derivative_dlnnCO_db
from exogibbs.optimize.core import _A_diagn_At 


@pytest.fixture(params=[1.0, 2.0])
def h_system_setup(request):
    """Setup common test parameters for H system derivative tests.
    
    Parameterized fixture that provides different b_element_vector values:
    - 1.0: For temperature and pressure derivative tests
    - 2.0: For element derivative tests
    """
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
    
    b_element_vector = jnp.array([request.param])  # Parameterized value
    epsilon_crit = 1e-11
    max_iter = 1000
    
    # Run Gibbs minimization
    ln_nk_result, ln_ntot_result, _ = minimize_gibbs_core(
        temperature,
        ln_normalized_pressure,
        b_element_vector,
        ln_nk,
        ln_ntot,
        formula_matrix,
        hvector_func,
        epsilon_crit=epsilon_crit,
        max_iter=max_iter,
    )
    
    # Common computed quantities
    nk_result = jnp.exp(ln_nk_result)
    ntot_result = jnp.exp(ln_ntot_result)
    Bmatrix = _A_diagn_At(nk_result, formula_matrix)
    
    return {
        'hsystem': hsystem,
        'formula_matrix': formula_matrix,
        'temperature': temperature,
        'P': P,
        'Pref': Pref,
        'ln_normalized_pressure': ln_normalized_pressure,
        'ln_nk': ln_nk,
        'ln_ntot': ln_ntot,
        'hvector_func': hvector_func,
        'b_element_vector': b_element_vector,
        'epsilon_crit': epsilon_crit,
        'max_iter': max_iter,
        'ln_nk_result': ln_nk_result,
        'ln_ntot_result': ln_ntot_result,
        'nk_result': nk_result,
        'ntot_result': ntot_result,
        'Bmatrix': Bmatrix
    }


@pytest.mark.parametrize("h_system_setup", [1.0], indirect=True)
def test_derivative_temperature_h_system(h_system_setup):
    """Test derivative_temperature against analytical H system solution."""
    setup = h_system_setup
    
    # Test derivative computation
    hdot = jnp.array([setup['hsystem'].dot_hv_h(setup['temperature']), setup['hsystem'].dot_hv_h2(setup['temperature'])])
    
    # Compute derivatives
    nk_cdot_hdot = jnp.dot(setup['nk_result'], hdot)    
    ln_nspecies_dT = derivative_temperature(setup['nk_result'], setup['formula_matrix'], hdot, nk_cdot_hdot, setup['Bmatrix'], setup['b_element_vector'])

    # Get reference analytical derivatives
    refH = setup['hsystem'].ln_nH_dT(jnp.array([setup['temperature']]), setup['ln_normalized_pressure'])[0]
    refH2 = setup['hsystem'].ln_nH2_dT(jnp.array([setup['temperature']]), setup['ln_normalized_pressure'])[0]
    
    # Test differences are small
    diff_h = refH - ln_nspecies_dT[0]
    diff_h2 = refH2 - ln_nspecies_dT[1]
    
    assert jnp.abs(diff_h) < 1e-14, f"H derivative difference too large: {diff_h}"
    assert jnp.abs(diff_h2) < 1e-14, f"H2 derivative difference too large: {diff_h2}"


@pytest.mark.parametrize("h_system_setup", [1.0], indirect=True)
def test_derivative_pressure_h_system(h_system_setup):
    """Test derivative_pressure against analytical H system solution."""
    setup = h_system_setup
    
    # Compute pressure derivatives
    ln_nspecies_dlogp = derivative_pressure(setup['ntot_result'], setup['formula_matrix'], setup['Bmatrix'], setup['b_element_vector'])
    
    # Get reference analytical pressure derivatives
    refH = setup['hsystem'].ln_nH_dlogp(jnp.array([setup['temperature']]), setup['ln_normalized_pressure'])[0]
    refH2 = setup['hsystem'].ln_nH2_dlogp(jnp.array([setup['temperature']]), setup['ln_normalized_pressure'])[0]
    
    # Test differences are small
    diff_h = refH - ln_nspecies_dlogp[0]
    diff_h2 = refH2 - ln_nspecies_dlogp[1]
    
    assert jnp.abs(diff_h) < 1e-11, f"H pressure derivative difference too large: {diff_h}"
    assert jnp.abs(diff_h2) < 1e-11, f"H2 pressure derivative difference too large: {diff_h2}"


@pytest.mark.parametrize("h_system_setup", [2.0], indirect=True)
def test_derivative_element_one_h_system(h_system_setup):
    """Test derivative_element_one against analytical H system solution."""
    setup = h_system_setup
    
    # Compute element derivatives
    ln_nspecies_dbH = derivative_element_one(setup['formula_matrix'], setup['Bmatrix'], setup['b_element_vector'], i_element=0)
    
    # Get reference analytical element derivatives
    refH = setup['hsystem'].ln_nH_dbH(setup['b_element_vector'][0])
    refH2 = setup['hsystem'].ln_nH2_dbH(setup['b_element_vector'][0])

    # Test differences are small
    diff_h = refH - ln_nspecies_dbH[0]
    diff_h2 = refH2 - ln_nspecies_dbH[1]
    
    assert jnp.abs(diff_h) < 1e-11, f"H element derivative difference too large: {diff_h}"
    assert jnp.abs(diff_h2) < 1e-11, f"H2 element derivative difference too large: {diff_h2}"


def test_derivative_element_one_hcosystem():
    """Test derivative_element_one using analytical HCO system solution."""
    config.update("jax_enable_x64", True)
    
    hcosystem = HCOSystem()
    
    # Formula matrix: [H, C, O] x [H2, CO, CH4, H2O]
    formula_matrix = jnp.array(
        [[2.0, 0.0, 0.0], [0.0, 1.0, 1.0], [4.0, 1.0, 0.0], [2.0, 0.0, 1.0]]
    ).T
    
    temperature = 1500.0
    P = 1.5
    Pref = 1.0
    ln_normalized_pressure = compute_ln_normalized_pressure(P, Pref)
    
    ln_nk = jnp.array([0.0, 0.0, 0.0, 0.0])
    ln_ntot = 0.0
    
    def hvector_func(temperature):
        return hcosystem.hv_hco(temperature)
    
    bH = 0.5
    bC = 0.2
    bO = 0.3
    b_element_vector = jnp.array([bH, bC, bO])
    
    epsilon_crit = 1e-11
    max_iter = 1000
    
    # Run Gibbs minimization
    from exogibbs.optimize.minimize import minimize_gibbs
    ln_nk_result = minimize_gibbs(
        temperature,
        ln_normalized_pressure,
        b_element_vector,
        ln_nk,
        ln_ntot,
        formula_matrix,
        hvector_func,
        epsilon_crit=epsilon_crit,
        max_iter=max_iter,
    )
    
    nk_result = jnp.exp(ln_nk_result)
    Bmatrix = _A_diagn_At(nk_result, formula_matrix)
    
    # Compute derivatives
    dlnn_dbH = derivative_element_one(formula_matrix, Bmatrix, b_element_vector, 0)
    dlnn_dbC = derivative_element_one(formula_matrix, Bmatrix, b_element_vector, 1)
    dlnn_dbO = derivative_element_one(formula_matrix, Bmatrix, b_element_vector, 2)
    dlnnCO_db = jnp.array([dlnn_dbH[1], dlnn_dbC[1], dlnn_dbO[1]])
    
    # Analytical derivatives
    k = hcosystem.equilibrium_constant(temperature, P/Pref)
    gradf = derivative_dlnnCO_db(ln_nk_result[1], bC, bH, bO, k)
    
    diff = jnp.abs(dlnnCO_db/gradf - 1.0)
    assert jnp.all(diff < 1.e-5), f"Derivative mismatch: {diff}"


if __name__ == "__main__":
    import pytest
    pytest.main([__file__, "-v"])