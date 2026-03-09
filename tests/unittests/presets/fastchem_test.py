from turtle import reset
import pytest

def test_chemsetup():
    from exogibbs.presets.fastchem import chemsetup
    gas = chemsetup()
    
    assert len(gas.elements) == 28
    assert len(gas.species) == 523

def test_run():
    import jax.numpy as jnp
    from jax import config
    config.update("jax_enable_x64", True)
    from exogibbs.api.equilibrium import equilibrium
    from exogibbs.presets.fastchem import chemsetup
    setup = chemsetup()
    T, P = 1500.0, 1.0  # K, bar
    b = setup.element_vector_reference  # or your own jnp.array([...])
    result = equilibrium(setup, T=T, P=P, b=b)
    
    # total mixing ratio should be 1.0
    assert jnp.sum(result.x) == pytest.approx(1.0, rel=1e-15)
    # this is a known value from previous runs (2025/11/7)
    # if this fails, please re-check the consistency with fastchem results
    # print(result.x[0]) #2.1066081741146924e-08
    assert result.x[0] == pytest.approx(2.1066081741146924e-08, rel=1e-15)


def test_hvector_profile_shape_species_last():
    import jax.numpy as jnp
    from exogibbs.presets.fastchem import chemsetup

    setup = chemsetup(silent=True)
    K = setup.formula_matrix.shape[1]
    temperatures = jnp.array([800.0, 1000.0, 1200.0])

    h_profile = setup.hvector_func(temperatures)
    h_profile_vmap = jnp.stack([setup.hvector_func(T) for T in temperatures], axis=0)

    assert h_profile.shape == (temperatures.shape[0], K)
    assert jnp.allclose(h_profile, h_profile_vmap)

if __name__ == "__main__":
    test_chemsetup()
    test_run()
