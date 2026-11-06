"""Check the compact PS thermal adoption and preserve the cumulative CN base."""

import json
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from exogibbs.presets import curated_cn, curated_ps, curated_v1


ROOT = Path(__file__).resolve().parents[3]
R = 8.31446261815324
REFERENCE = ROOT/"tests/data/validation/curated_ps_adoption/20261106/fit_validation.json"


@pytest.fixture(scope="module", autouse=True)
def float64():
    previous = jax.config.x64_enabled
    jax.config.update("jax_enable_x64", True)
    yield
    jax.config.update("jax_enable_x64", previous)


@pytest.fixture(scope="module")
def models(float64):
    return curated_cn.chemsetup(), curated_ps.chemsetup()


def test_only_ps_changes_and_cn_provenance_survives(models):
    before, after = models
    assert after.metadata["dataset"] == "curated-ps-v1"
    assert after.metadata["base_dataset"] == "curated-cn-v1"
    assert after.metadata["relative_charge_convergence"] is True
    assert before.species == after.species and before.elements == after.elements
    np.testing.assert_array_equal(before.formula_matrix,after.formula_matrix)
    np.testing.assert_array_equal(before.element_vector_reference,after.element_vector_reference)
    data = json.loads((ROOT/"src/exogibbs/data/curated/ps_v1.json").read_text())
    assert data["delta_D0_J_mol"] == 0 and data["original_source_D0_J_mol"] == 438000
    assert data["log_partition_representation"]["degree"] == 18
    assert len(data["log_partition_representation"]["coefficients"]) == 19
    probes = json.loads(REFERENCE.read_text())["reference_probes"]
    t = jnp.array([row["temperature_K"] for row in probes])
    delta = np.asarray(after.hvector_func(t)-before.hvector_func(t))
    target = after.species.index("P1S1")
    np.testing.assert_array_equal(np.delete(delta,target,axis=-1),0)
    np.testing.assert_allclose(np.asarray(after.hvector_func(t))[:,target],
                               [row["formation_h"] for row in probes],atol=2e-12,rtol=0)
    assert abs(delta[-1,target]) > .01 and abs(delta[0,target]) > 1e-4
    for name in before.species:
        if name != "P1S1":
            assert after.metadata["species_provenance"][name] == before.metadata["species_provenance"][name]
    assert after.metadata["sources"]["cn"] == before.metadata["sources"]["cn"]
    assert after.metadata["species_provenance"]["P1S1"]["status"] == "prefer_conditionally"


def test_compact_runtime_retains_independent_state_moment_calorics(models):
    after = models[1]
    target = after.species.index("P1S1")
    probes = json.loads(REFERENCE.read_text())["reference_probes"]
    temperatures = np.array([row["temperature_K"] for row in probes])
    g = lambda t:R*t*after.hvector_func(t)[target]
    values = np.asarray(jax.jit(jax.vmap(g))(temperatures))
    first = np.asarray(jax.jit(jax.vmap(jax.grad(g)))(temperatures))
    second = np.asarray(jax.jit(jax.vmap(jax.grad(jax.grad(g))))(temperatures))
    actual = np.column_stack((values,values-temperatures*first,-first,-temperatures*second))
    expected = np.array([list(row["reaction_properties"].values()) for row in probes])
    for index,tolerance in enumerate((5e-8,1e-6,1e-9,1e-7)):
        np.testing.assert_allclose(actual[:,index],expected[:,index],atol=tolerance,rtol=0)


def test_scalar_batch_domain_and_neutral_projection(models):
    model = models[1]
    temperatures = jnp.array([[1500.,1831.25],[3679.125,4000.]])
    actual = jax.jit(model.hvector_func)(temperatures)
    assert actual.shape == (2,2,537) and actual.dtype == jnp.float64
    assert model.hvector_func(2500).shape == (537,)
    np.testing.assert_allclose(jax.jit(jax.vmap(model.hvector_func))(temperatures[0]),actual[0],atol=1e-12,rtol=0)
    assert np.isnan(model.hvector_func(jnp.array([-1.,1499.,4001.,np.nan,np.inf]))).all()
    neutral,base = curated_ps.chemsetup(ions=False),curated_cn.chemsetup(ions=False)
    assert len(neutral.species) == 421 and "e-" not in neutral.elements
    assert neutral.species == base.species and neutral.elements == base.elements
    np.testing.assert_array_equal(neutral.formula_matrix,base.formula_matrix)
    np.testing.assert_array_equal(neutral.element_vector_reference,base.element_vector_reference)
    keep = [model.species.index(name) for name in neutral.species]
    np.testing.assert_allclose(neutral.hvector_func(temperatures),np.asarray(actual)[...,keep],atol=1e-12,rtol=0)
    with pytest.raises(TypeError,match="ions"):
        curated_ps.chemsetup(ions=1)


def test_modified_packaged_ps_data_are_rejected(monkeypatch,tmp_path):
    real = curated_v1.get_data_filepath
    name = "curated/ps_v1.json"
    changed = tmp_path/"ps_v1.json"
    changed.write_text(Path(real(name)).read_text()+"\n")
    monkeypatch.setattr(curated_v1,"get_data_filepath",lambda p:str(changed) if p==name else real(p))
    with pytest.raises(ValueError,match="pinned version"):
        curated_ps.chemsetup()
