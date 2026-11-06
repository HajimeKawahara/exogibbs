"""Check the isolated CH2 energy adoption without changing the thermal model."""

import json
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from exogibbs.presets import curated_ch2, curated_c2h2, curated_v1


ROOT = Path(__file__).resolve().parents[3]
R = 8.31446261815324


@pytest.fixture(scope="module", autouse=True)
def float64():
    previous = jax.config.x64_enabled
    jax.config.update("jax_enable_x64", True)
    yield
    jax.config.update("jax_enable_x64", previous)


def test_only_ch2_changes_by_the_balanced_physical_source_energy():
    base, model = curated_c2h2.chemsetup(), curated_ch2.chemsetup()
    assert model.metadata["dataset"] == "curated-ch2-v1"
    assert model.metadata["base_dataset"] == "curated-c2h2-v1"
    assert model.species == base.species and model.elements == base.elements
    np.testing.assert_array_equal(model.formula_matrix, base.formula_matrix)
    np.testing.assert_array_equal(model.element_vector_reference, base.element_vector_reference)
    data = json.loads((ROOT/"src/exogibbs/data/curated/ch2_v1.json").read_text())
    hf = data["candidate_hf0_kJ_mol"]
    physical = 1000*(hf["C"]+2*hf["H"]-hf["CH2"])
    assert data["D0_J_mol"] == pytest.approx(physical, abs=1e-8, rel=0)
    assert data["base_D0_J_mol"] == 757340 and data["delta_D0_J_mol"] == -4948
    assert data["formal_fit_intercept_Ra_J_mol"] != data["base_D0_J_mol"]
    t = jnp.array([1500., 1637., 2000., 2253., 3000., 3871., 4000.])
    column = model.species.index("C1H2")
    delta = np.asarray(model.hvector_func(t)-base.hvector_func(t))
    np.testing.assert_array_equal(np.delete(delta, column, axis=-1), 0)
    np.testing.assert_allclose(delta[:, column], 4948/(R*np.asarray(t)), atol=3e-14, rtol=0)
    for species in model.species:
        if species != "C1H2":
            assert model.metadata["species_provenance"][species] == base.metadata["species_provenance"][species]


def test_entropy_and_heat_capacity_are_unchanged():
    base, model = curated_c2h2.chemsetup(), curated_ch2.chemsetup()
    column = model.species.index("C1H2")
    def delta_g(t):
        return R*t*(model.hvector_func(t)[column]-base.hvector_func(t)[column])
    for t in (1500., 1937., 2500., 3379.9, 4000.):
        g = float(delta_g(t))
        first = float(jax.grad(delta_g)(t))
        second = float(jax.grad(jax.grad(delta_g))(t))
        assert g == pytest.approx(4948, abs=1e-8, rel=0)
        assert g-t*first == pytest.approx(4948, abs=1e-8, rel=0)
        assert abs(first) < 1e-10 and abs(t*second) < 1e-10


def test_batch_domain_and_neutral_projection():
    model = curated_ch2.chemsetup()
    t = jnp.array([[1500., 1875.], [3250., 4000.]])
    actual = jax.jit(model.hvector_func)(t)
    assert actual.shape == (2, 2, 537) and actual.dtype == jnp.float64
    assert model.hvector_func(2500).shape == (537,)
    np.testing.assert_allclose(jax.jit(jax.vmap(model.hvector_func))(t[0]), actual[0], atol=1e-12, rtol=0)
    assert np.isnan(model.hvector_func(jnp.array([-1., 1499., 4001., np.nan, np.inf]))).all()
    neutral, base = curated_ch2.chemsetup(ions=False), curated_c2h2.chemsetup(ions=False)
    assert len(neutral.species) == 421 and "e-" not in neutral.elements
    assert neutral.species == base.species and neutral.elements == base.elements
    indices = [model.species.index(s) for s in neutral.species]
    np.testing.assert_allclose(neutral.hvector_func(t), np.asarray(actual)[..., indices], atol=1e-12, rtol=0)
    with pytest.raises(TypeError, match="ions"):
        curated_ch2.chemsetup(ions=1)


def test_modified_packaged_data_are_rejected(monkeypatch, tmp_path):
    real = curated_v1.get_data_filepath
    name = "curated/ch2_v1.json"
    changed = tmp_path/"ch2_v1.json"
    changed.write_text(Path(real(name)).read_text()+"\n")
    monkeypatch.setattr(curated_v1, "get_data_filepath", lambda p: str(changed) if p == name else real(p))
    with pytest.raises(ValueError, match="pinned version"):
        curated_ch2.chemsetup()
