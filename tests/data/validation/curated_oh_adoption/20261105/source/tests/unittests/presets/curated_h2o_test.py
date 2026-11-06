"""Verify adopted H2O thermodynamics using independent published source values."""

import json
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from exogibbs.presets import curated, curated_co2, curated_h2o, curated_v1


ROOT = Path(__file__).resolve().parents[3]
CANDIDATE = ROOT / "tests/data/validation/h2o_candidate/20261104"
R = 8.31446261815324


@pytest.fixture(scope="module", autouse=True)
def float64():
    previous = jax.config.x64_enabled
    jax.config.update("jax_enable_x64", True)
    yield
    jax.config.update("jax_enable_x64", previous)


@pytest.fixture(scope="module")
def models(float64):
    return curated_co2.chemsetup(), curated_h2o.chemsetup()


def test_adoption_changes_only_water_and_preserves_the_co2_base(models):
    before, after = models
    assert curated.chemsetup is curated_h2o.chemsetup
    assert after.metadata["dataset"] == "curated-h2o-v1"
    assert after.metadata["base_dataset"] == "curated-co2-v1"
    assert after.metadata["relative_charge_convergence"] is True
    assert before.species == after.species and before.elements == after.elements
    np.testing.assert_array_equal(before.formula_matrix, after.formula_matrix)
    np.testing.assert_array_equal(before.element_vector_reference, after.element_vector_reference)
    t = jnp.array([1500., 1687.5, 2000., 2687.5, 3500., 3937.5, 4000.])
    delta = np.asarray(after.hvector_func(t) - before.hvector_func(t))
    water = after.species.index("H2O1")
    np.testing.assert_allclose(np.delete(delta, water, axis=-1), 0, atol=1e-12, rtol=0)
    assert np.all(delta[:, water] > 0)
    for species in ("C1O2", "O1Ti1", "C2H1"):
        assert after.metadata["species_provenance"][species] == before.metadata["species_provenance"][species]
    assert after.metadata["species_provenance"]["H2O1"] != before.metadata["species_provenance"]["H2O1"]
    payload = json.loads((ROOT / "src/exogibbs/data/curated/h2o_v1.json").read_text())
    candidate = json.loads((CANDIDATE / "candidate.json").read_text())
    assert payload["thermal"] == candidate["evaluated"]
    for key in ("atomic_background_nasa9", "D0_over_k_K", "molecular_mass_u"):
        assert payload[key] == candidate[key]


def test_source_tables_and_caloric_derivatives_in_the_original_atomic_gauge(models):
    model = models[1]
    water = model.species.index("H2O1")
    reference = json.loads((ROOT / "examples/comparisons/data/major_molecule_reference.json").read_text())
    atoms = [(reference["nasa9_records"][name], count) for name, count in (("H", 2), ("O", 1))]

    def molecular_free_energy_over_r(t):
        atom_thermal = 0.
        for record, count in atoms:
            interval = next(r for r in record["intervals"] if r["bounds_K"] == [1000., 6000.])
            a, b, c, d, e, f, g, h, s = interval["coefficients"]
            g_rt = (-a / (2 * t**2) + b * (1 + jnp.log(t)) / t + c * (1 - jnp.log(t))
                    - d * t / 2 - e * t**2 / 6 - f * t**3 / 12 - g * t**4 / 20 + h / t - s)
            h0_k = (record["hf298_J_mol"] - record["h298_minus_h0_J_mol"]) / 8.314510
            atom_thermal += count * (g_rt - h0_k / t)
        return t * (model.hvector_func(t)[water] + atom_thermal) + 917819. / R

    first = jax.grad(molecular_free_energy_over_r)
    second = jax.grad(first)
    evaluate = jax.jit(jax.vmap(lambda t: (molecular_free_energy_over_r(t), first(t), second(t))))
    # Include fit nodes, withheld points, and extrema of source residuals.
    temperatures = np.array([1500., 1637., 2000., 2500., 3000., 3500., 3650., 3734., 3871., 3990., 4000.])
    g, slope, curvature = [np.asarray(v) for v in evaluate(jnp.asarray(temperatures))]
    with np.load(CANDIDATE / "published_tables.npz") as source:
        indices = np.searchsorted(source["temperature_K"], temperatures)
        np.testing.assert_array_equal(source["temperature_K"][indices], temperatures)
        expected_h = source["H_relative_J_mol"][indices]
        expected_s = source["S_J_mol_K"][indices]
        expected_cp = source["Cp_J_mol_K"][indices]
        expected_g = expected_h - temperatures * expected_s
    np.testing.assert_allclose(R * g, expected_g, atol=.53, rtol=0)
    np.testing.assert_allclose(R * (g - temperatures * slope), expected_h, atol=.57, rtol=0)
    np.testing.assert_allclose(-R * slope, expected_s, atol=2.4e-5, rtol=0)
    np.testing.assert_allclose(-R * temperatures * curvature, expected_cp, atol=.0015, rtol=0)
    assert np.all(-R * temperatures * curvature > 47)


def test_batch_domain_gradients_and_neutral_projection(models):
    model = models[1]
    water = model.species.index("H2O1")
    temperatures = jnp.array([[1500., 1875.], [3250., 4000.]])
    actual = jax.jit(model.hvector_func)(temperatures)
    assert actual.shape == (2, 2, 537) and actual.dtype == jnp.float64
    gradient = jax.jit(jax.vmap(jax.grad(lambda t: model.hvector_func(t)[water])))
    edges = jnp.array([1500., 4000.])
    step = jnp.array([.01, -.01])
    finite = (-3 * model.hvector_func(edges)[:, water] + 4 * model.hvector_func(edges + step)[:, water]
              - model.hvector_func(edges + 2 * step)[:, water]) / (2 * step)
    np.testing.assert_allclose(gradient(edges), finite, atol=2e-9, rtol=2e-8)
    assert np.isnan(model.hvector_func(jnp.array([-1., 1499., 4001., np.nan, np.inf]))).all()
    neutral, base = curated_h2o.chemsetup(ions=False), curated_co2.chemsetup(ions=False)
    assert len(neutral.species) == 421 and "e-" not in neutral.elements and "e1-" not in neutral.species
    assert neutral.species == base.species and neutral.elements == base.elements
    np.testing.assert_array_equal(neutral.formula_matrix, base.formula_matrix)
    np.testing.assert_array_equal(neutral.element_vector_reference, base.element_vector_reference)
    keep = [model.species.index(name) for name in neutral.species]
    np.testing.assert_allclose(neutral.hvector_func(temperatures), np.asarray(actual)[..., keep], atol=1e-12, rtol=0)
    with pytest.raises(TypeError, match="ions"):
        curated_h2o.chemsetup(ions=1)


def test_modified_packaged_water_data_are_rejected(monkeypatch, tmp_path):
    real = curated_v1.get_data_filepath
    name = "curated/h2o_v1.json"
    changed = tmp_path / "h2o_v1.json"
    changed.write_text(Path(real(name)).read_text() + "\n")
    monkeypatch.setattr(curated_v1, "get_data_filepath", lambda p: str(changed) if p == name else real(p))
    with pytest.raises(ValueError, match="pinned version"):
        curated_h2o.chemsetup()
