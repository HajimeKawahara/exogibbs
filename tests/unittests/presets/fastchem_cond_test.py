import numpy as np    
import jax.numpy as jnp
import pytest

from exogibbs.thermo.models import ChemicalSetup


def _logk(temp: float, coeffs) -> float:
    a1, a2, a3, a4, a5 = coeffs
    return a1 / temp + a2 * np.log(temp) + a3 + a4 * temp + a5 * temp * temp

def test_hvector_for_h2o():
    from exogibbs.presets.fastchem_cond import chemsetup as condsetup
    
    from jax import config
    config.update("jax_enable_x64", True)
    
    cond = condsetup()
    cond_species = list(cond.species)
    index_h2o_cond = cond_species.index('H2O(s,l)')  

    T1=273.16 
    coeff_ice = np.array([1.1610444238898961e+05, -3.7180754189807614e+00, -9.9697686374472188e+00, -3.4870743183681364e-02, 5.8028334311999769e-05])
    coeff_water = np.array([  1.1592048349840334e+05, -1.2230111683190726e+01, 2.2781352565814203e+01, 4.4780347912393416e-02, -2.3737039228581724e-05])

    def _hice(T):
        return - _logk(T, coeff_ice)

    def _hwater(T):
        return - _logk(T, coeff_water)

    def _hh2o(T):
        mask = T < T1
        h = np.where(mask, _hice(T), _hwater(T))
        return h
        
    T = np.linspace(100, 700, 100)
    h2o_cond_h_values = cond.hvector_func(T)[:, index_h2o_cond] #mu_c(T)/RT
    
    assert np.all(h2o_cond_h_values/_hh2o(T) - 1 ) == 0.0


def test_chemsetup_piecewise_segments():
    from exogibbs.presets.fastchem_cond import chemsetup

    cond = chemsetup()

    assert cond.species is not None
    species = list(cond.species)
    assert "Al(s)" in species

    target = "AlCl3(s,l)"
    assert target in species
    idx = species.index(target)

    coeff_low = (
        1.6752020623095072e05,
        -7.5508095875449666,
        -2.6523945119240171e01,
        2.2745417658541090e-02,
        -1.0189579863311542e-05,
    )
    coeff_high = (
        1.6529023277388894e05,
        2.4647025638091016,
        -7.5718997854319525e01,
        1.9312847893898392e-03,
        -2.8258846125321814e-07,
    )

    temp_low = 400.0
    temp_high = 600.0

    hv_low = cond.hvector_func(temp_low)
    hv_high = cond.hvector_func(temp_high)

    expected_low = -_logk(temp_low, coeff_low)
    expected_high = -_logk(temp_high, coeff_high)

    assert hv_low[idx].item() == pytest.approx(expected_low, rel=1e-5)
    assert hv_high[idx].item() == pytest.approx(expected_high, rel=1e-5)

    hv_stack = cond.hvector_func(jnp.array([temp_low, temp_high]))
    assert hv_stack.shape == (2, len(species))
    assert hv_stack[0, idx].item() == pytest.approx(expected_low, rel=1e-5)
    assert hv_stack[1, idx].item() == pytest.approx(expected_high, rel=1e-5)


def test_condensate_chemsetup_uses_injected_gas_element_order():
    from exogibbs.presets.fastchem import chemsetup as gas_chemsetup
    from exogibbs.presets.fastchem_cond import chemsetup

    base = gas_chemsetup(silent=True)
    elements = tuple(reversed(base.elements))
    gas_setup = ChemicalSetup(
        formula_matrix=jnp.flip(base.formula_matrix, axis=0),
        hvector_func=base.hvector_func,
        elements=elements,
        species=base.species,
        element_vector_reference=jnp.flip(base.element_vector_reference),
    )

    condensate_setup = chemsetup(gas_setup=gas_setup, silent=True)

    assert condensate_setup.elements == gas_setup.elements
    species = list(condensate_setup.species)
    water_index = species.index("H2O(s,l)")
    assert condensate_setup.formula_matrix[elements.index("H"), water_index] == 2
    assert condensate_setup.formula_matrix[elements.index("O"), water_index] == 1
    assert condensate_setup.element_vector_reference is (
        gas_setup.element_vector_reference
    )


def test_condensate_chemsetup_rejects_missing_gas_elements():
    from exogibbs.presets.fastchem_cond import chemsetup

    gas_setup = ChemicalSetup(
        formula_matrix=jnp.eye(2),
        hvector_func=lambda temperature: jnp.zeros((2,)),
        elements=("H", "O"),
        species=("H", "O"),
        element_vector_reference=jnp.asarray([1.0, 1.0e-3]),
    )

    with pytest.raises(ValueError, match="missing elements"):
        chemsetup(gas_setup=gas_setup, silent=True)


def test_corrected_species_default_elements_keyword_is_additive():
    from exogibbs.presets.fastchem import chemsetup

    setup = chemsetup(species_default_elements=True, silent=True)

    assert setup.metadata["fastchem_species_default_elements"] == "True"

    with pytest.raises(ValueError, match="conflict"):
        chemsetup(
            species_defalt_elements=False,
            species_default_elements=True,
            silent=True,
        )
