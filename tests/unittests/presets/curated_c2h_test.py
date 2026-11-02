"""The C2H correction is balanced, scoped and independent of evidence tooling."""

import json
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from exogibbs.api.gas import EquilibriumOptions, solve
from exogibbs.presets import curated, curated_c2h
from exogibbs.thermo.standard import _nasa9


ROOT = Path(__file__).resolve().parents[3]


@pytest.fixture(scope="module", autouse=True)
def float64():
    previous = jax.config.x64_enabled
    jax.config.update("jax_enable_x64", True)
    yield
    jax.config.update("jax_enable_x64", previous)


@pytest.fixture(scope="module")
def models():
    return curated.chemsetup(), curated_c2h.chemsetup()


def test_only_c2h_changes_with_identical_network_and_budget(models):
    old, new = models
    assert old.species == new.species and old.elements == new.elements
    np.testing.assert_array_equal(old.formula_matrix, new.formula_matrix)
    np.testing.assert_array_equal(old.element_vector_reference, new.element_vector_reference)
    temperatures = jnp.array([1500., 1735., 2000., 2517., 3000., 3973., 4000.])
    index = old.species.index("C2H1")
    delta = np.asarray(new.hvector_func(temperatures)-old.hvector_func(temperatures))
    np.testing.assert_allclose(np.delete(delta, index, axis=-1), 0, rtol=0, atol=1e-12)
    assert np.all(delta[:, index] > 0)
    assert new.metadata["dataset"] == "curated-c2h-v1"
    assert new.metadata["species_provenance"]["C2H1"]["status"] == "prefer_conditionally"
    assert old.metadata["species_provenance"]["C2H1"]["status"] == "inherited_unreviewed"
    assert "C2H1" not in json.loads((ROOT / "src/exogibbs/data/curated/gas_v1.json").read_text())["decisions"]


def test_energy_anchor_is_balanced_and_does_not_change_thermal_coefficients(models):
    data = json.loads((ROOT / "src/exogibbs/data/curated/c2h_v1.json").read_text())
    assert data["atct_atomization_0K_kJ_mol"] == pytest.approx(1075.066)
    assert data["energy_shift_K"] < 0
    raw, updated = np.array(data["raw_nasa9_reaction_coefficients"]), np.array(data["nasa9_reaction_coefficients"])
    np.testing.assert_array_equal(np.delete(raw, 7), np.delete(updated, 7))
    assert updated[7]-raw[7] == pytest.approx(-data["energy_shift_K"])
    t = jnp.array([1500., 2500., 4000.])
    raw_h = _nasa9(t, jnp.asarray([raw]))[0][:, 0]
    actual = models[1].hvector_func(t)[:, models[1].species.index("C2H1")]
    np.testing.assert_allclose(actual-raw_h, -data["energy_shift_K"]/t, atol=2e-12, rtol=0)


def test_batch_gradients_domain_and_neutral_network(models):
    model = models[1]
    i = model.species.index("C2H1")
    t = jnp.array([[1500., 1850.], [3000., 4000.]])
    assert jax.jit(model.hvector_func)(t).shape == (2, 2, 537)
    gradient = jax.grad(lambda value: model.hvector_func(value)[i])
    for value in (1500., 2500., 4000.):
        assert np.isfinite(gradient(value))
    step = .01
    finite_difference = (model.hvector_func(2500+step)[i]-model.hvector_func(2500-step)[i])/(2*step)
    assert gradient(2500.) == pytest.approx(float(finite_difference), rel=1e-8)
    assert np.isnan(model.hvector_func(jnp.array([-1., 1499., 4001., np.nan, np.inf]))).all()
    neutral = curated_c2h.chemsetup(ions=False)
    assert len(neutral.species) == 421 and "e-" not in neutral.elements
    keep = [model.species.index(name) for name in neutral.species]
    np.testing.assert_allclose(neutral.hvector_func(t), model.hvector_func(t)[..., keep], atol=1e-12, rtol=0)
    with pytest.raises(TypeError, match="ions"):
        curated_c2h.chemsetup(ions=1)


def test_correction_rejects_changed_packaged_data(monkeypatch, tmp_path):
    real = curated.get_data_filepath
    changed = tmp_path / "c2h_v1.json"
    changed.write_text((ROOT / "src/exogibbs/data/curated/c2h_v1.json").read_text()+"\n")
    monkeypatch.setattr(curated, "get_data_filepath", lambda p: str(changed) if p == "curated/c2h_v1.json" else real(p))
    with pytest.raises(ValueError, match="pinned version"):
        curated_c2h.chemsetup()


def test_carbon_rich_regression_changes_hcn_and_preserves_constraints(models):
    results = []
    for model in models:
        matrix = np.asarray(model.formula_matrix)
        budget = np.array(model.element_vector_reference)
        budget[model.elements.index("C")] = 1.5*budget[model.elements.index("O")]
        budget /= budget.sum()
        result, info = solve(model, 2000., 1e-6, jnp.asarray(budget),
                             options=EquilibriumOptions(epsilon_crit=5e-14), return_diagnostics=True)
        assert bool(info["converged"])
        amounts = np.exp(result.ln_n)
        positive = budget > 0
        assert np.max(abs(matrix@amounts-budget)[positive]/budget[positive]) < 1e-5
        charge = matrix[model.elements.index("e-")]
        assert abs(charge@amounts)/(abs(charge)@amounts) < 1e-12
        results.append(np.asarray(result.x))
    old, new = results
    i = models[0].species.index("C1H1N1_hcn")
    assert old[i] == pytest.approx(1.61822785e-5, rel=1e-6)
    assert new[i] == pytest.approx(2.38135747e-5, rel=1e-6)
    assert new[i]/old[i] == pytest.approx(1.471584, rel=1e-5)
