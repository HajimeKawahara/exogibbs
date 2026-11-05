"""Check the adopted OH reaction against independent NASA and ATcT inputs."""

import json
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from exogibbs.presets import curated, curated_h2o, curated_oh, curated_v1


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
    return curated_h2o.chemsetup(), curated.chemsetup()


def test_adoption_changes_only_oh_and_preserves_the_h2o_base(models):
    before, after = models
    assert curated.chemsetup is curated_oh.chemsetup
    assert after.metadata["dataset"] == "curated-oh-v1"
    assert after.metadata["base_dataset"] == "curated-h2o-v1"
    assert after.metadata["relative_charge_convergence"] is True
    assert before.species == after.species and before.elements == after.elements
    np.testing.assert_array_equal(before.formula_matrix, after.formula_matrix)
    np.testing.assert_array_equal(before.element_vector_reference, after.element_vector_reference)
    t = jnp.array([1500., 1637., 2253., 3000., 3871., 4000.])
    delta = np.asarray(after.hvector_func(t) - before.hvector_func(t))
    oh = after.species.index("H1O1")
    np.testing.assert_array_equal(np.delete(delta, oh, axis=-1), 0)
    assert np.all(delta[:, oh] < 0)
    for name in before.species:
        if name != "H1O1":
            assert after.metadata["species_provenance"][name] == before.metadata["species_provenance"][name]
    assert after.metadata["species_provenance"]["H1O1"]["status"] == "prefer_conditionally"
    data = json.loads((ROOT / "src/exogibbs/data/curated/oh_v1.json").read_text())
    candidate = json.loads((ROOT / "tests/data/validation/oh_candidate/20261105/candidate.json").read_text())
    for key in ("formation_nasa9_coefficients", "D0_over_k_K", "thermal_convention"):
        assert data[key] == candidate[key]


def test_reaction_free_energy_and_caloric_derivatives_use_the_original_atomic_gauge(models):
    model = models[1]
    oh = model.species.index("H1O1")
    records = json.loads((ROOT / "tests/data/validation/gas_source_comparison/baseline-001/inputs.json").read_text())["nasa9_records"]
    literature = json.loads((ROOT / "tests/data/validation/oh_source_review/20261105/literature.json").read_text())
    reference = json.loads((ROOT / "examples/comparisons/data/major_molecule_reference.json").read_text())
    d0 = 1000 * (sum(reference["atct"]["records"][s]["hf0_kJ_mol"] for s in ("H", "O"))
                 - literature["atct"]["OH_hf0_kJ_mol"])
    h0 = {s: h["hf298_J_mol"] - h["h298_minus_h0_J_mol"] for s, h in literature["nasa_headers"].items()}
    energy_shift = d0 / R - (h0["H"] + h0["O"] - h0["OH"]) / 8.314510
    temperatures = np.array([1500., 1637., 2253., 3000., 3871., 4000.])
    reaction_g, reaction_h, reaction_cp = np.zeros((3, len(temperatures)))
    t = temperatures
    for species, sign in (("OH", 1), ("H", -1), ("O", -1)):
        a, b, c, d, e, f, g, h, s = next(row["coefficients"] for row in records[species]["intervals"]
                                        if row["bounds_K"] == [1000., 6000.])
        enthalpy = -a / t + b * np.log(t) + c * t + d * t**2 / 2 + e * t**3 / 3 + f * t**4 / 4 + g * t**5 / 5 + h
        entropy = -a / (2 * t**2) - b / t + c * np.log(t) + d * t + e * t**2 / 2 + f * t**3 / 3 + g * t**4 / 4 + s
        cp = a / t**2 + b / t + c + d * t + e * t**2 + f * t**3 + g * t**4
        reaction_g += sign * (enthalpy - t * entropy)
        reaction_h += sign * enthalpy
        reaction_cp += sign * cp
    reaction_g -= energy_shift
    reaction_h -= energy_shift
    free_energy = lambda t: R * t * model.hvector_func(t)[oh]
    values = np.asarray(jax.jit(jax.vmap(free_energy))(t))
    first = np.asarray(jax.jit(jax.vmap(jax.grad(free_energy)))(t))
    second = np.asarray(jax.jit(jax.vmap(jax.grad(jax.grad(free_energy))))(t))
    np.testing.assert_allclose(values, R * reaction_g, atol=1e-8, rtol=0)
    np.testing.assert_allclose(values - t * first, R * reaction_h, atol=1e-8, rtol=0)
    np.testing.assert_allclose(-t * second, R * reaction_cp, atol=1e-10, rtol=0)


def test_batch_domain_gradients_and_neutral_projection(models):
    model = models[1]
    oh = model.species.index("H1O1")
    temperatures = jnp.array([[1500., 1875.], [3250., 4000.]])
    actual = jax.jit(model.hvector_func)(temperatures)
    assert actual.shape == (2, 2, 537) and actual.dtype == jnp.float64
    assert model.hvector_func(2500).shape == (537,)
    gradient = jax.jit(jax.vmap(jax.grad(lambda t: model.hvector_func(t)[oh])))
    edges, step = jnp.array([1500., 4000.]), jnp.array([.01, -.01])
    finite = (-3 * model.hvector_func(edges)[:, oh] + 4 * model.hvector_func(edges + step)[:, oh]
              - model.hvector_func(edges + 2 * step)[:, oh]) / (2 * step)
    np.testing.assert_allclose(gradient(edges), finite, atol=2e-9, rtol=2e-8)
    assert np.isnan(model.hvector_func(jnp.array([-1., 1499., 4001., np.nan, np.inf]))).all()
    neutral, base = curated.chemsetup(ions=False), curated_h2o.chemsetup(ions=False)
    assert len(neutral.species) == 421 and "e-" not in neutral.elements
    assert neutral.species == base.species and neutral.elements == base.elements
    np.testing.assert_array_equal(neutral.formula_matrix, base.formula_matrix)
    np.testing.assert_array_equal(neutral.element_vector_reference, base.element_vector_reference)
    keep = [model.species.index(name) for name in neutral.species]
    np.testing.assert_allclose(neutral.hvector_func(temperatures), np.asarray(actual)[..., keep], atol=1e-12, rtol=0)
    with pytest.raises(TypeError, match="ions"):
        curated.chemsetup(ions=1)


def test_modified_packaged_oh_data_are_rejected(monkeypatch, tmp_path):
    real = curated_v1.get_data_filepath
    name = "curated/oh_v1.json"
    changed = tmp_path / "oh_v1.json"
    changed.write_text(Path(real(name)).read_text() + "\n")
    monkeypatch.setattr(curated_v1, "get_data_filepath", lambda p: str(changed) if p == name else real(p))
    with pytest.raises(ValueError, match="pinned version"):
        curated.chemsetup()
