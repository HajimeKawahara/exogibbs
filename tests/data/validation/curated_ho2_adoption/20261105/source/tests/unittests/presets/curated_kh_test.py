"""Check the adopted KH energy against spectroscopy and unchanged calorics."""

import json
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from exogibbs.presets import curated, curated_kh, curated_oh, curated_v1


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
    return curated_oh.chemsetup(), curated.chemsetup()


def test_adoption_changes_only_kh_and_preserves_the_oh_base(models):
    before, after = models
    assert curated.chemsetup is curated_kh.chemsetup
    assert after.metadata["dataset"] == "curated-kh-v1"
    assert after.metadata["base_dataset"] == "curated-oh-v1"
    assert after.metadata["relative_charge_convergence"] is True
    assert before.species == after.species and before.elements == after.elements
    np.testing.assert_array_equal(before.formula_matrix, after.formula_matrix)
    np.testing.assert_array_equal(before.element_vector_reference, after.element_vector_reference)
    t = jnp.array([1500., 1637., 2253., 3000., 3871., 4000.])
    delta = np.asarray(after.hvector_func(t) - before.hvector_func(t))
    kh = after.species.index("H1K1")
    np.testing.assert_array_equal(np.delete(delta, kh, axis=-1), 0)
    assert np.all(delta[:, kh] > 0)
    for name in before.species:
        if name != "H1K1":
            assert after.metadata["species_provenance"][name] == before.metadata["species_provenance"][name]
    assert after.metadata["species_provenance"]["H1K1"]["status"] == "prefer_conditionally"
    data = json.loads((ROOT / "src/exogibbs/data/curated/kh_v1.json").read_text())
    candidate = json.loads((ROOT / "tests/data/validation/kh_candidate/20261105/candidate.json").read_text())
    for key in ("candidate_logK_coefficients", "delta_D0_over_R_K", "source_limit"):
        assert data[key] == candidate[key]


def test_spectroscopic_energy_changes_enthalpy_but_not_entropy_or_heat_capacity(models):
    before, after = models
    kh = after.species.index("H1K1")
    # Independent exact SI conversion, with the inspected balanced JANAF anchor.
    spectroscopic_d0 = 14282.6 * 100 * 299792458 * 6.62607015e-34 * 6.02214076e23
    energy_difference = 180390. - spectroscopic_d0
    t = jnp.array([1500., 1637., 2253., 3000., 3871., 4000.])
    functions = [lambda value, setup=setup: R * value * setup.hvector_func(value)[kh]
                 for setup in models]
    values, first, second = [np.asarray([jax.jit(jax.vmap(function))(t) for function in group])
                             for group in (functions, [jax.grad(f) for f in functions],
                                           [jax.grad(jax.grad(f)) for f in functions])]
    np.testing.assert_allclose(values[1]-values[0], energy_difference, atol=1e-8, rtol=0)
    np.testing.assert_allclose((values-t*first)[1]-(values-t*first)[0], energy_difference, atol=1e-8, rtol=0)
    np.testing.assert_allclose(first[1], first[0], atol=1e-10, rtol=0)
    np.testing.assert_allclose((-t*second)[1], (-t*second)[0], atol=1e-10, rtol=0)
    np.testing.assert_allclose(np.exp(before.hvector_func(t)[:, kh]-after.hvector_func(t)[:, kh]),
                               np.exp(-energy_difference/(R*t)), atol=1e-13, rtol=0)


def test_batch_domain_gradients_and_neutral_projection(models):
    model = models[1]
    kh = model.species.index("H1K1")
    temperatures = jnp.array([[1500., 1875.], [3250., 4000.]])
    actual = jax.jit(model.hvector_func)(temperatures)
    assert actual.shape == (2, 2, 537) and actual.dtype == jnp.float64
    assert model.hvector_func(2500).shape == (537,)
    gradient = jax.jit(jax.vmap(jax.grad(lambda t: model.hvector_func(t)[kh])))
    edges, step = jnp.array([1500., 4000.]), jnp.array([.01, -.01])
    finite = (-3*model.hvector_func(edges)[:, kh] + 4*model.hvector_func(edges+step)[:, kh]
              - model.hvector_func(edges+2*step)[:, kh])/(2*step)
    np.testing.assert_allclose(gradient(edges), finite, atol=2e-9, rtol=2e-8)
    assert np.isnan(model.hvector_func(jnp.array([-1., 1499., 4001., np.nan, np.inf]))).all()
    neutral, base = curated.chemsetup(ions=False), curated_oh.chemsetup(ions=False)
    assert len(neutral.species) == 421 and "e-" not in neutral.elements
    assert neutral.species == base.species and neutral.elements == base.elements
    np.testing.assert_array_equal(neutral.formula_matrix, base.formula_matrix)
    np.testing.assert_array_equal(neutral.element_vector_reference, base.element_vector_reference)
    keep = [model.species.index(name) for name in neutral.species]
    np.testing.assert_allclose(neutral.hvector_func(temperatures), np.asarray(actual)[..., keep], atol=1e-12, rtol=0)
    with pytest.raises(TypeError, match="ions"):
        curated.chemsetup(ions=1)


def test_modified_packaged_kh_data_are_rejected(monkeypatch, tmp_path):
    real = curated_v1.get_data_filepath
    name = "curated/kh_v1.json"
    changed = tmp_path / "kh_v1.json"
    changed.write_text(Path(real(name)).read_text()+"\n")
    monkeypatch.setattr(curated_v1, "get_data_filepath", lambda p: str(changed) if p == name else real(p))
    with pytest.raises(ValueError, match="pinned version"):
        curated.chemsetup()
