import pytest
import jax.numpy as jnp
from jax import config, vmap, jacrev
from exogibbs.optimize.minimize import minimize_gibbs
from exogibbs.optimize.core import compute_ln_normalized_pressure
from exogibbs.test.analytic_hsystem import HSystem


def test_minimize_gibbs_vmap_h_system():
    """Test vectorized minimize_gibbs against analytical H system over temperature range."""
    config.update("jax_enable_x64", True)
    
    # Initialize H system
    hsystem = HSystem()
    
    # Test parameters
    formula_matrix = jnp.array([[1.0, 2.0]])
    P = 1.0
    Pref = 1.0
    ln_normalized_pressure = compute_ln_normalized_pressure(P, Pref)
    ln_nk_init = jnp.array([0.0, 0.0])
    ln_ntot_init = 0.0
    b_element_vector = jnp.array([1.0])
    epsilon_crit = 1e-11
    max_iter = 1000
    
    def hvector_func(temperature):
        return jnp.array([hsystem.hv_h(temperature), hsystem.hv_h2(temperature)])
    
    # Temperature array for vectorized computation
    Tarr = jnp.linspace(1000.0, 4000.0, 10)  # Small range for fast testing
    
    # Vectorized minimize_gibbs
    vmap_minimize_gibbs = vmap(
        minimize_gibbs, in_axes=(0, None, None, None, None, None, None, None, None)
    )
    
    ln_nk_arr = vmap_minimize_gibbs(
        Tarr,
        ln_normalized_pressure,
        b_element_vector,
        ln_nk_init,
        ln_ntot_init,
        formula_matrix,
        hvector_func,
        epsilon_crit,
        max_iter,
    )
    
    # Compute VMRs from results
    n_H = jnp.exp(ln_nk_arr[:, 0])
    n_H2 = jnp.exp(ln_nk_arr[:, 1])
    ntot = n_H + n_H2
    vmrH = n_H / ntot
    vmrH2 = n_H2 / ntot
    
    # Analytical reference
    karr = vmap(hsystem.compute_k, in_axes=(None, 0))(ln_normalized_pressure, Tarr)
    vmrH_ref = vmap(hsystem.vmr_h)(karr)
    vmrH2_ref = vmap(hsystem.vmr_h2)(karr)
    
    # Test differences are small
    diffH = jnp.max(jnp.abs(vmrH - vmrH_ref))
    diffH2 = jnp.max(jnp.abs(vmrH2 - vmrH2_ref))
    print(f"Max difference in VMR for H: {diffH}, H2: {diffH2}")
    #Max difference in dln_dT for H: 3.4612070154427244e-15, H2: 8.360282952035725e-16 July 21 (2025)

    assert diffH < 1e-11, f"H VMR difference too large: {diffH}"
    assert diffH2 < 1e-11, f"H2 VMR difference too large: {diffH2}"


def test_minimize_gibbs_vmap_gradient_h_system():
    """Test vectorized temperature gradients of minimize_gibbs."""
    config.update("jax_enable_x64", True)
    
    # Initialize H system
    hsystem = HSystem()
    
    # Test parameters
    formula_matrix = jnp.array([[1.0, 2.0]])
    P = 1.0
    Pref = 1.0
    ln_normalized_pressure = compute_ln_normalized_pressure(P, Pref)
    ln_nk_init = jnp.array([0.0, 0.0])
    ln_ntot_init = 0.0
    b_element_vector = jnp.array([1.0])
    epsilon_crit = 1e-11
    max_iter = 1000
    
    def hvector_func(temperature):
        return jnp.array([hsystem.hv_h(temperature), hsystem.hv_h2(temperature)])
    
    # Temperature array
    Tarr = jnp.linspace(2000.0, 3000.0, 5)  # Small range for fast testing
    
    # Vectorized temperature gradients
    vmap_minimize_gibbs_dT = vmap(
        jacrev(minimize_gibbs),
        in_axes=(0, None, None, None, None, None, None, None, None),
    )
    
    dln_dT_arr = vmap_minimize_gibbs_dT(
        Tarr,
        ln_normalized_pressure,
        b_element_vector,
        ln_nk_init,
        ln_ntot_init,
        formula_matrix,
        hvector_func,
        epsilon_crit,
        max_iter,
    )
    
    # Analytical reference gradients
    dln_dT_H_ref = hsystem.ln_nH_dT(Tarr, ln_normalized_pressure)
    dln_dT_H2_ref = hsystem.ln_nH2_dT(Tarr, ln_normalized_pressure)
    
    # Test differences are small
    diff_dT_H = jnp.max(jnp.abs(dln_dT_arr[:, 0] - dln_dT_H_ref))
    diff_dT_H2 = jnp.max(jnp.abs(dln_dT_arr[:, 1] - dln_dT_H2_ref))
    print(f"Max difference in dln_dT for H: {diff_dT_H}, H2: {diff_dT_H2}")
    #Max difference in VMR for H: 1.1938228183794308e-12, H2: 1.1937950628038152e-12 July 21 (2025)
    assert diff_dT_H < 1e-11, f"H gradient difference too large: {diff_dT_H}"
    assert diff_dT_H2 < 1e-11, f"H2 gradient difference too large: {diff_dT_H2}"


if __name__ == "__main__":
    test_minimize_gibbs_vmap_gradient_h_system()
    test_minimize_gibbs_vmap_h_system()
    