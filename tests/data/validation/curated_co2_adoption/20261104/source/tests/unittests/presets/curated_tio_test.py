"""Verify the adopted TiO model independently of offline review tooling."""

import json
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from scipy.special import logsumexp

from exogibbs.api.gas import EquilibriumOptions, solve
from exogibbs.presets import curated, curated_c2h, curated_tio, curated_v1


ROOT = Path(__file__).resolve().parents[3]
REFERENCE = ROOT / "tests/data/validation/tio_candidate/20261103"
IMPACT = ROOT / "tests/data/validation/tio_equilibrium_impact/20261103/equilibria.npz"
R = 8.31446261815324
EV_K = 1.602176634e-19/1.380649e-23


@pytest.fixture(scope="module", autouse=True)
def float64():
    previous = jax.config.x64_enabled
    jax.config.update("jax_enable_x64", True)
    yield
    jax.config.update("jax_enable_x64", previous)


@pytest.fixture(scope="module")
def models(float64):
    return curated_c2h.chemsetup(), curated_tio.chemsetup()


def test_default_changes_only_tio_and_preserves_c2h_network_and_budget(models):
    before, after = models
    assert curated.chemsetup is curated_tio.chemsetup
    assert after.metadata["dataset"] == "curated-tio-v1"
    assert after.metadata["base_dataset"] == "curated-c2h-v1"
    assert after.metadata["relative_charge_convergence"] is True
    assert before.species == after.species and before.elements == after.elements
    np.testing.assert_array_equal(before.formula_matrix, after.formula_matrix)
    np.testing.assert_array_equal(before.element_vector_reference, after.element_vector_reference)
    t = jnp.array([1500., 1687.5, 2000., 2687.5, 3500., 3937.5, 4000.])
    delta = np.asarray(after.hvector_func(t)-before.hvector_func(t))
    tio, c2h = [after.species.index(name) for name in ("O1Ti1", "C2H1")]
    np.testing.assert_allclose(np.delete(delta, tio, axis=-1), 0, atol=1e-12, rtol=0)
    np.testing.assert_allclose(delta[:, c2h], 0, atol=1e-12, rtol=0)
    assert np.all(delta[:, tio] > 0)
    assert after.metadata["species_provenance"]["C2H1"] == before.metadata["species_provenance"]["C2H1"]
    assert after.metadata["species_provenance"]["O1Ti1"] != before.metadata["species_provenance"]["O1Ti1"]


def test_independent_off_knot_state_sums_and_caloric_derivatives(models):
    model = models[1]
    index = model.species.index("O1Ti1")
    reference = json.loads((ROOT / "examples/comparisons/data/major_molecule_reference.json").read_text())
    atoms = [reference["nasa9_records"][name] for name in ("Ti", "O")]
    # Molecular properties are recovered using the original atomic source,
    # rather than the production model's background or interpolation helper.
    def molecular_free_energy_over_r(t):
        atom_thermal = 0.
        for record in atoms:
            interval = next(row for row in record["intervals"] if row["bounds_K"] == [1000., 6000.])
            a, b, c, d, e, f, g, h, s = interval["coefficients"]
            g_rt = (-a/(2*t*t)+b*(1+jnp.log(t))/t+c*(1-jnp.log(t))
                    -d*t/2-e*t*t/6-f*t**3/12-g*t**4/20+h/t-s)
            h0_k = (record["hf298_J_mol"]-record["h298_minus_h0_J_mol"])/8.314510
            atom_thermal += g_rt-h0_k/t
        return t*(model.hvector_func(t)[index]+atom_thermal)+6.824*EV_K

    first = jax.grad(molecular_free_energy_over_r)
    second = jax.grad(first)
    evaluate = jax.jit(jax.vmap(lambda t: (molecular_free_energy_over_r(t), first(t), second(t))))
    rows = json.loads((REFERENCE / "off_knot_reference.json").read_text())
    t = np.array([row["temperature_K"] for row in rows])
    f, slope, curvature = [np.asarray(x) for x in evaluate(jnp.asarray(t))]
    expected = {name: np.array([row[name] for row in rows]) for name in
                ("G_relative_J_mol", "H_relative_J_mol", "S_J_mol_K", "Cp_J_mol_K")}
    np.testing.assert_allclose(f/t, expected["G_relative_J_mol"]/(R*t), atol=1e-5, rtol=0)
    np.testing.assert_allclose(R*(f-t*slope), expected["H_relative_J_mol"], atol=.25, rtol=0)
    np.testing.assert_allclose(-R*slope, expected["S_J_mol_K"], atol=.0002, rtol=0)
    np.testing.assert_allclose(-R*t*curvature, expected["Cp_J_mol_K"], atol=.01, rtol=0)
    assert np.all(-R*t*curvature > 0)


def test_batch_gradients_domain_and_neutral_projection(models):
    model = models[1]
    index = model.species.index("O1Ti1")
    t = jnp.array([[1500., 1875.], [3250., 4000.]])
    actual = jax.jit(model.hvector_func)(t)
    assert actual.shape == (2, 2, 537) and actual.dtype == jnp.float64
    gradient = jax.jit(jax.vmap(jax.grad(lambda value: model.hvector_func(value)[index])))
    edges = jnp.array([1500., 4000.])
    direction = jnp.array([1., -1.])
    step = .01*direction
    finite = (-3*model.hvector_func(edges)[:, index]
              +4*model.hvector_func(edges+step)[:, index]
              -model.hvector_func(edges+2*step)[:, index])/(2*step)
    np.testing.assert_allclose(gradient(edges), finite, atol=2e-9, rtol=2e-8)
    assert np.isnan(model.hvector_func(jnp.array([-1., 1499., 4001., np.nan, np.inf]))).all()
    neutral = curated_tio.chemsetup(ions=False)
    old_neutral = curated_c2h.chemsetup(ions=False)
    assert len(neutral.species) == 421 and "e-" not in neutral.elements and "e1-" not in neutral.species
    assert neutral.species == old_neutral.species and neutral.elements == old_neutral.elements
    np.testing.assert_array_equal(neutral.formula_matrix, old_neutral.formula_matrix)
    np.testing.assert_array_equal(neutral.element_vector_reference, old_neutral.element_vector_reference)
    keep = [model.species.index(name) for name in neutral.species]
    np.testing.assert_allclose(neutral.hvector_func(t), np.asarray(actual)[..., keep], atol=1e-12, rtol=0)
    with pytest.raises(TypeError, match="ions"):
        curated_tio.chemsetup(ions=1)


def test_changed_tio_source_is_rejected(monkeypatch, tmp_path):
    real = curated_v1.get_data_filepath
    filename = "curated/tio_v1.json"
    changed = tmp_path / "tio_v1.json"
    changed.write_text(Path(real(filename)).read_text()+"\n")
    monkeypatch.setattr(curated_v1, "get_data_filepath", lambda name: str(changed) if name == filename else real(name))
    with pytest.raises(ValueError, match="pinned version"):
        curated_tio.chemsetup()


def test_production_endpoint_matches_archived_tio_substitution(models):
    model = models[1]
    with np.load(IMPACT) as archive:
        case = 4
        assert archive["temperature_K"][case] == 1500 and archive["pressure_bar"][case] == 100
        budget = archive["budget"][case]
        expected = archive["ln_n"][1, case]
        np.testing.assert_array_equal(model.species, archive["species"])
        np.testing.assert_array_equal(model.formula_matrix, archive["formula_matrix"])
        np.testing.assert_allclose(model.hvector_func(1500.), archive["h"][1, case], atol=2e-12, rtol=0)
    result, info = solve(model, T=1500., P=100., b=jnp.asarray(budget),
                         options=EquilibriumOptions(epsilon_crit=5e-14), return_diagnostics=True)
    assert bool(info["converged"])
    logs = np.asarray(result.ln_n)
    assert np.isfinite(logs).all()
    logx = logs-logsumexp(logs)
    np.testing.assert_allclose(logx, expected-logsumexp(expected), atol=1e-4, rtol=0)
    n, matrix = np.exp(logs), np.asarray(model.formula_matrix)
    positive = budget > 0
    assert np.max(abs(matrix @ n-budget)[positive]/budget[positive]) < 1e-5
    charge = matrix[model.elements.index("e-")]
    assert abs(charge @ n)/(abs(charge) @ n) < 1e-12
    references = [model.species.index("e1-" if element == "e-" else element+"1")
                  for element in model.elements]
    logp = logx+np.log(100.)
    assert np.max(abs(logp-logp[references] @ matrix+model.hvector_func(1500.))) < 1e-8
    tio, titanium = model.species.index("O1Ti1"), model.elements.index("Ti")
    assert n[tio]/budget[titanium] == pytest.approx(.4895723641233141, rel=1e-8)
