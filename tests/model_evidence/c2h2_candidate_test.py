"""Verify acetylene anchor balance, continuous calorics and retained residuals."""

from copy import deepcopy
from hashlib import sha256
import json
import math
from pathlib import Path
import sys

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "mceg/reviews"))
from curated_adoption_history import historical_source
import c2h2_candidate as candidate


@pytest.fixture(scope="module")
def validation():
    return candidate.validate()


def compare_samples(actual, expected, key="samples"):
    """Use dimensional absolute bounds for cancelling thermodynamic quantities."""
    if isinstance(expected, dict):
        assert actual.keys() == expected.keys()
        for name in expected:
            compare_samples(actual[name], expected[name], name)
    elif isinstance(expected, list):
        assert len(actual) == len(expected)
        for left, right in zip(actual, expected):
            compare_samples(left, right, key)
    elif isinstance(expected, (int, float)):
        tolerance = 1e-8 if key in ("G_J_mol", "H_J_mol") else 2e-11
        assert actual == pytest.approx(expected, rel=0 if key in ("G_J_mol", "H_J_mol") else 2e-12, abs=tolerance)
    else:
        assert actual == expected


def test_balanced_anchors_keep_fit_and_nasa_gas_constants_separate():
    model = candidate.load_candidate()
    assert model == candidate.build()
    assert model["D0_J_mol"] == 1626018
    assert model["retained_source_D0_J_mol"] == 1627153
    assert model["delta_D0_J_mol"] == -1135
    assert model["nasa_D0_J_mol"] == pytest.approx(1625696.849, abs=1e-8)
    assert model["nasa_energy_shift_over_R_K"] == pytest.approx(
        model["D0_J_mol"]/candidate.R-model["nasa_D0_J_mol"]/candidate.NASA_R, abs=1e-12)
    naive = (model["D0_J_mol"]-model["nasa_D0_J_mol"])/candidate.R
    assert abs(naive-model["nasa_energy_shift_over_R_K"]) > 1
    assert abs(model["formal_fit_intercept_Ra_J_mol"]-model["retained_source_D0_J_mol"]) > 1
    assert "Ethyne" in model["original_record"]


def test_dense_offgrid_independent_reconstruction_and_jax_derivatives(validation):
    assert validation["temperature_count"] == 2752
    assert validation["integer_grid_count"] == 2501 and validation["offgrid_count"] == 251
    assert validation["maximum_abs_current_h_error"] < 1e-12
    for section in ("independent_property_errors", "jax_property_errors"):
        for values in validation[section].values():
            for key, value in values.items():
                assert value < (1e-8 if key in ("G_J_mol", "H_J_mol") else 2e-11)
    archived = json.loads((candidate.DATA/"validation.json").read_text())
    compare_samples(validation["samples"], archived["samples"])
    assert max(abs(row["error_ln_K"]) for row in validation["source_fit_residuals"]) < 1e-12
    assert validation["production_change"] == "none"


def test_energy_only_changes_g_and_h_without_changing_entropy_or_cp():
    from exogibbs.presets import curated_tio2
    import jax
    jax.config.update("jax_enable_x64", True)
    setup = curated_tio2.chemsetup(ions=True)
    j = setup.species.index("C2H2")
    old_g = lambda t: candidate.R*t*setup.hvector_func(t)[j]
    for t in (1500., 1837.23, 2750.47, 3999.9, 4000.):
        g, first, second = float(old_g(t)), float(jax.grad(old_g)(t)), float(jax.grad(jax.grad(old_g))(t))
        current = np.array([g, g-t*first, -first, -t*second])
        change = np.array(list(candidate.reaction_properties(t).values()))-current
        np.testing.assert_allclose(change, [1135., 1135., 0., 0.], atol=1e-8, rtol=0)


def test_domain_jit_vmap_and_nonphysical_temperature_guards():
    import jax
    import jax.numpy as jnp
    jax.config.update("jax_enable_x64", True)
    ts = jnp.array([1500., 1789.3, 2500., 4000.])
    for thermal in candidate.THERMAL:
        f = lambda t: candidate.formation_h_jax(t, thermal)
        expected = [candidate.formation_h(float(t), thermal) for t in ts]
        np.testing.assert_allclose(jax.jit(f)(ts), expected, atol=2e-12, rtol=0)
        np.testing.assert_allclose(jax.jit(jax.vmap(f))(ts), expected, atol=2e-12, rtol=0)
        assert np.isnan(np.asarray(jax.jit(f)(jnp.array([0., 1499., 4001., np.nan, np.inf])))).all()
        for t in (0., 1499., 4001., np.nan, np.inf):
            with pytest.raises(ValueError, match="outside"):
                candidate.formation_h(t, thermal)
    with pytest.raises(ValueError, match="thermal"):
        candidate.formation_h(2000., "acetylene_finite_pf")


def test_dimensional_replay_tolerance_rejects_meaningful_changes(validation):
    altered = deepcopy(validation["samples"])
    altered[0]["retained_reaction"]["G_J_mol"] += 1e-3
    with pytest.raises(AssertionError):
        compare_samples(altered, validation["samples"])


def test_receipt_preserves_source_inputs_and_direct_candidate_outputs():
    receipt = json.loads((candidate.DATA/"receipt.json").read_text())
    for name, expected in receipt["selected_file_sha256"].items():
        assert sha256(historical_source(ROOT,name,expected).read_bytes()).hexdigest() == expected, name
    for name, expected in receipt["output_sha256"].items():
        assert sha256((candidate.DATA/name).read_bytes()).hexdigest() == expected, name
