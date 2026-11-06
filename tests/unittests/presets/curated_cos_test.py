"""Check the COS energy adoption and preserve the cumulative HO2 base."""

import json
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from exogibbs.presets import curated_ho2, curated_cos, curated_v1


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
    return curated_ho2.chemsetup(), curated_cos.chemsetup()


def test_only_cos_changes_with_the_balanced_same_version_energy(models):
    before, after = models
    assert after.metadata["dataset"] == "curated-cos-v1"
    assert after.metadata["base_dataset"] == "curated-ho2-v1"
    assert after.metadata["relative_charge_convergence"] is True
    assert before.species == after.species and before.elements == after.elements
    np.testing.assert_array_equal(before.formula_matrix, after.formula_matrix)
    np.testing.assert_array_equal(before.element_vector_reference, after.element_vector_reference)
    inputs = json.loads((ROOT/"tests/data/validation/cos_source_review/20261105/inputs.json").read_text())
    energy = inputs["atct_1.228"]
    d0 = 1000*(energy["C"]+energy["O"]+energy["S"]-energy["COS"])
    assert d0 == pytest.approx(1376853., abs=1e-8, rel=0)
    data = json.loads((ROOT/"src/exogibbs/data/curated/cos_v1.json").read_text())
    assert data["D0_J_mol"] == d0 and data["base_D0_J_mol"] == pytest.approx(1371241.)
    t = jnp.array([1500., 1637., 2000., 2253., 3000., 3379.9, 3380.1, 3871., 4000.])
    delta = np.asarray(after.hvector_func(t)-before.hvector_func(t))
    cos = after.species.index("C1O1S1")
    np.testing.assert_array_equal(np.delete(delta, cos, axis=-1), 0)
    np.testing.assert_allclose(delta[:, cos], -(d0-1371241.)/(R*np.asarray(t)), atol=3e-14, rtol=0)
    for name in before.species:
        if name != "C1O1S1":
            assert after.metadata["species_provenance"][name] == before.metadata["species_provenance"][name]
    assert after.metadata["species_provenance"]["C1O1S1"]["status"] == "prefer_conditionally"
    assert after.metadata["sources"]["kh"] == before.metadata["sources"]["kh"]
    assert after.metadata["sources"]["ho2"] == before.metadata["sources"]["ho2"]


def test_constant_reaction_energy_preserves_entropy_heat_capacity_and_knots(models):
    before, after = models
    cos = after.species.index("C1O1S1")
    t = jnp.array([1500., 1937., 2000., 2253., 3000., 3379.9, 3380., 3380.1, 4000.])
    properties = []
    for model in models:
        g = lambda temp: R*temp*model.hvector_func(temp)[cos]
        values = np.asarray(jax.jit(jax.vmap(g))(t))
        first = np.asarray(jax.jit(jax.vmap(jax.grad(g)))(t))
        second = np.asarray(jax.jit(jax.vmap(jax.grad(jax.grad(g))))(t))
        properties.append(np.stack([values, values-np.asarray(t)*first, -first, -np.asarray(t)*second], axis=-1))
    expected = np.zeros_like(properties[0]); expected[:, :2] = -5612.
    np.testing.assert_allclose((properties[1]-properties[0])[:, :2], expected[:, :2], atol=1e-8, rtol=0)
    np.testing.assert_allclose((properties[1]-properties[0])[:, 2:], 0, atol=1e-10, rtol=0)
    # Each side retains the inherited C2 knot behavior after the same constant shift.
    for knot in (2000., 3000.):
        left = float(after.hvector_func(knot-1e-5)[cos])
        right = float(after.hvector_func(knot+1e-5)[cos])
        assert abs(left-right) < 1e-6


def test_batch_domain_and_neutral_projection(models):
    model = models[1]
    temperatures = jnp.array([[1500., 1875.], [3250., 4000.]])
    actual = jax.jit(model.hvector_func)(temperatures)
    assert actual.shape == (2, 2, 537) and actual.dtype == jnp.float64
    assert model.hvector_func(2500).shape == (537,)
    np.testing.assert_allclose(jax.jit(jax.vmap(model.hvector_func))(temperatures[0]), actual[0], atol=1e-12, rtol=0)
    assert np.isnan(model.hvector_func(jnp.array([-1., 1499., 4001., np.nan, np.inf]))).all()
    neutral, base = curated_cos.chemsetup(ions=False), curated_ho2.chemsetup(ions=False)
    assert len(neutral.species) == 421 and "e-" not in neutral.elements
    assert neutral.species == base.species and neutral.elements == base.elements
    np.testing.assert_array_equal(neutral.formula_matrix, base.formula_matrix)
    np.testing.assert_array_equal(neutral.element_vector_reference, base.element_vector_reference)
    keep = [model.species.index(name) for name in neutral.species]
    np.testing.assert_allclose(neutral.hvector_func(temperatures), np.asarray(actual)[..., keep], atol=1e-12, rtol=0)
    with pytest.raises(TypeError, match="ions"):
        curated_cos.chemsetup(ions=1)


def test_modified_packaged_cos_data_are_rejected(monkeypatch, tmp_path):
    real = curated_v1.get_data_filepath
    name = "curated/cos_v1.json"
    changed = tmp_path/"cos_v1.json"
    changed.write_text(Path(real(name)).read_text()+"\n")
    monkeypatch.setattr(curated_v1, "get_data_filepath", lambda p: str(changed) if p == name else real(p))
    with pytest.raises(ValueError, match="pinned version"):
        curated_cos.chemsetup()
