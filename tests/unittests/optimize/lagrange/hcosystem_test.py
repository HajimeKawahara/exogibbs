"""test minimize_gibbs_core for HCO system equilibrium validation."""


import pytest
import jax.numpy as jnp
from jax import config
from exogibbs.api.chemistry import ThermoState
from exogibbs.optimize.minimize import minimize_gibbs_core
from exogibbs.optimize.core import compute_ln_normalized_pressure
from exogibbs.test.analytic_hcosystem import HCOSystem, function_equilibrium


@pytest.fixture
def hco_system_setup():
    """Setup common test parameters for HCO system tests."""
    config.update("jax_enable_x64", True)
    
    hcosystem = HCOSystem()
    
    # Formula matrix: [H, C, O] per species [H₂, CO, CH₄, H₂O]
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
    
    thermo_state = ThermoState(temperature, ln_normalized_pressure, b_element_vector)
    
    return {
        'hcosystem': hcosystem,
        'formula_matrix': formula_matrix,
        'temperature': temperature,
        'P': P,
        'Pref': Pref,
        'ln_normalized_pressure': ln_normalized_pressure,
        'ln_nk': ln_nk,
        'ln_ntot': ln_ntot,
        'hvector_func': hvector_func,
        'b_element_vector': b_element_vector,
        'bH': bH,
        'bC': bC,
        'bO': bO,
        'epsilon_crit': epsilon_crit,
        'max_iter': max_iter,
        'thermo_state': thermo_state
    }


def test_minimize_gibbs_core_hco_system(hco_system_setup):
    """Test minimize_gibbs_core against analytical HCO system equilibrium."""
    setup = hco_system_setup
    
    # Run Gibbs minimization
    ln_nk_result, ln_ntot_result, counter = minimize_gibbs_core(
        setup['thermo_state'],
        setup['ln_nk'],
        setup['ln_ntot'],
        setup['formula_matrix'],
        setup['hvector_func'],
        epsilon_crit=setup['epsilon_crit'],
        max_iter=setup['max_iter'],
    )
    
    # Validate against analytical equilibrium function
    k = setup['hcosystem'].equilibrium_constant(setup['temperature'], setup['P']/setup['Pref'])
    n_CO = jnp.exp(ln_nk_result[1]) 
    res = function_equilibrium(n_CO, k, setup['bC'], setup['bH'], setup['bO'])
    
    # Check convergence and equilibrium constraint
    assert jnp.abs(res) < setup['epsilon_crit'] * 10.0
    assert counter < setup['max_iter']




if __name__ == "__main__":
    import pytest
    pytest.main([__file__, "-v"])