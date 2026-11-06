"""Verify measured TiO2 thermal adoption and preserve the cumulative PS base."""

import json
import math
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from exogibbs.presets import curated_ps, curated_tio2, curated_v1


ROOT = Path(__file__).resolve().parents[3]
R = 8.31446261815324


@pytest.fixture(scope="module", autouse=True)
def float64():
    previous = jax.config.x64_enabled
    jax.config.update("jax_enable_x64", True)
    yield
    jax.config.update("jax_enable_x64", previous)


@pytest.fixture(scope="module")
def models(float64):
    return curated_ps.chemsetup(), curated_tio2.chemsetup()


def thermal_difference(t, data):
    """Independent scalar H/S/Cp derivation, including one-bar translation."""
    k, h, c = 1.380649e-23, 6.62607015e-34, 29979245800
    mass = data["mass_u"]*1.66053906660e-27
    theta = [h*b*1e6/k for b in data["rotational_constants_MHz"]]
    logqrot = .5*math.log(math.pi*t**3/math.prod(theta))-math.log(2)
    logqtrans = 1.5*math.log(2*math.pi*mass*k*t/h**2)+math.log(k*t/100000)
    enthalpy, entropy, cp = 4*t, logqtrans+logqrot+4, 4.
    for frequency in data["frequencies_cm-1"]:
        x = h*c*frequency/(k*t)
        occupation = 1/math.expm1(x)
        enthalpy += t*x*occupation
        entropy += x*occupation-math.log1p(-math.exp(-x))
        cp += x*x*math.exp(-x)/(1-math.exp(-x))**2
    a,b,d,e,f,g,j,u,v = data["nasa_molecular_thermal_coefficients"]
    nasa_h = -a/t+b*math.log(t)+d*t+e*t*t/2+f*t**3/3+g*t**4/4+j*t**5/5+u
    nasa_s = -a/(2*t*t)-b/t+d*math.log(t)+e*t+f*t*t/2+g*t**3/3+j*t**4/4+v
    nasa_cp = a/t**2+b/t+d+e*t+f*t*t+g*t**3+j*t**4
    dh, ds, dcp = enthalpy-nasa_h, entropy-nasa_s, cp-nasa_cp
    return R*np.array([dh-t*ds,dh,ds,dcp])


def test_only_tio2_changes_with_fixed_energy_and_measured_inputs(models):
    before, after = models
    assert after.metadata["dataset"] == "curated-tio2-v1"
    assert after.metadata["base_dataset"] == "curated-ps-v1"
    assert after.metadata["relative_charge_convergence"] is True
    assert before.species == after.species and before.elements == after.elements
    np.testing.assert_array_equal(before.formula_matrix, after.formula_matrix)
    np.testing.assert_array_equal(before.element_vector_reference, after.element_vector_reference)
    data = json.loads((ROOT/"src/exogibbs/data/curated/tio2_v1.json").read_text())
    assert data["original_source_D0_J_mol"] == 1267776. and data["delta_D0_J_mol"] == 0
    assert data["symmetry_number"] == 2
    assert data["electronic_degeneracy"] == data["nuclear_spin_weight"] == 1
    assert data["frequencies_cm-1"] == [959,322,944]
    temperatures = np.array([1500.,1637.,2000.,2253.,3000.,3871.,4000.])
    delta = np.asarray(after.hvector_func(temperatures)-before.hvector_func(temperatures))
    column = after.species.index("O2Ti1")
    np.testing.assert_array_equal(np.delete(delta,column,axis=-1),0)
    expected = [thermal_difference(t,data)[0]/(R*t) for t in temperatures]
    np.testing.assert_allclose(delta[:,column],expected,atol=2e-12,rtol=0)
    for name in before.species:
        if name != "O2Ti1":
            assert after.metadata["species_provenance"][name] == before.metadata["species_provenance"][name]
    assert after.metadata["species_provenance"]["O2Ti1"]["status"] == "prefer_conditionally"
    for key in ("cn","ps","tio","cos"):
        assert after.metadata["sources"][key] == before.metadata["sources"][key]


def test_automatic_calorics_equal_independent_thermal_difference(models):
    data = json.loads((ROOT/"src/exogibbs/data/curated/tio2_v1.json").read_text())
    column = models[1].species.index("O2Ti1")
    temperatures = jnp.array([1500.,1637.,2253.,3000.,3871.,4000.])
    properties=[]
    for model in models:
        g = lambda t:R*t*model.hvector_func(t)[column]
        values = np.asarray(jax.jit(jax.vmap(g))(temperatures))
        first = np.asarray(jax.jit(jax.vmap(jax.grad(g)))(temperatures))
        second = np.asarray(jax.jit(jax.vmap(jax.grad(jax.grad(g))))(temperatures))
        properties.append(np.stack((values,values-np.asarray(temperatures)*first,-first,-np.asarray(temperatures)*second),axis=-1))
    expected = np.array([thermal_difference(float(t),data) for t in temperatures])
    np.testing.assert_allclose((properties[1]-properties[0])[:,:2],expected[:,:2],atol=1e-8,rtol=0)
    np.testing.assert_allclose((properties[1]-properties[0])[:,2:],expected[:,2:],atol=1e-10,rtol=0)


def test_batch_domain_and_neutral_projection(models):
    model = models[1]
    temperatures = jnp.array([[1500.,1875.],[3250.,4000.]])
    actual = jax.jit(model.hvector_func)(temperatures)
    assert actual.shape == (2,2,537) and actual.dtype == jnp.float64
    assert model.hvector_func(2500).shape == (537,)
    np.testing.assert_allclose(jax.jit(jax.vmap(model.hvector_func))(temperatures[0]),actual[0],atol=1e-12,rtol=0)
    assert np.isnan(model.hvector_func(jnp.array([-1.,1499.,4001.,np.nan,np.inf]))).all()
    neutral,base = curated_tio2.chemsetup(ions=False),curated_ps.chemsetup(ions=False)
    assert len(neutral.species) == 421 and "e-" not in neutral.elements
    assert neutral.species == base.species and neutral.elements == base.elements
    np.testing.assert_array_equal(neutral.formula_matrix,base.formula_matrix)
    np.testing.assert_array_equal(neutral.element_vector_reference,base.element_vector_reference)
    keep = [model.species.index(name) for name in neutral.species]
    np.testing.assert_allclose(neutral.hvector_func(temperatures),np.asarray(actual)[...,keep],atol=1e-12,rtol=0)
    with pytest.raises(TypeError,match="ions"):
        curated_tio2.chemsetup(ions=1)


def test_modified_packaged_data_are_rejected(monkeypatch,tmp_path):
    real=curated_v1.get_data_filepath
    name="curated/tio2_v1.json"
    changed=tmp_path/"tio2_v1.json"
    changed.write_text(Path(real(name)).read_text()+"\n")
    monkeypatch.setattr(curated_v1,"get_data_filepath",lambda p:str(changed) if p==name else real(p))
    with pytest.raises(ValueError,match="pinned version"):
        curated_tio2.chemsetup()
