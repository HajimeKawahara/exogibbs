"""Guard review-only energy signs, NASA anchoring and multiple-target isolation."""

from dataclasses import dataclass
import math
from pathlib import Path
import sys
from types import ModuleType

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT/"mceg/reviews"))
import pending_candidate as candidate


def test_retained_physical_energy_shift_keeps_entropy_and_heat_capacity():
    old = {"kind": "retained_energy", "coefficients": [91000., -3.7, 1., .001, -5e-8], "delta_GH_J_mol": 0.}
    new = {**old, "delta_GH_J_mol": 4948.}
    for t in (1500., 1987.25, 4000.):
        before, after = (candidate.reaction_properties(t, s) for s in (old, new))
        for key in ("G_J_mol", "H_J_mol"):
            assert after[key]-before[key] == pytest.approx(4948., abs=1e-8, rel=0)
        for key in ("S_J_mol_K", "Cp_J_mol_K"):
            assert after[key] == before[key]
        assert math.exp(candidate.formation_h(t, old)-candidate.formation_h(t, new)) == pytest.approx(math.exp(-4948/(candidate.R*t)), rel=1e-13)
    # A scientifically distinct dimensional perturbation cannot hide in rounding.
    assert not math.isclose(4948.+1e-5, 4948., rel_tol=0, abs_tol=1e-8)


def test_nasa_native_energy_is_aligned_by_caller_exactly_once():
    native_r = 8.314510
    old_energy, selected_energy = 1625696.849, 1626018.
    cp_over_r, entropy_offset = 2.5, 7.
    coefficients = [0., 0., cp_over_r, 0., 0., 0., 0., -old_energy/native_r, entropy_offset]
    coefficients[7] -= selected_energy/candidate.R-old_energy/native_r
    spec = {"kind": "nasa9", "coefficients": coefficients}
    for t in (1500., 2718.25, 4000.):
        props = candidate.reaction_properties(t, spec)
        expected_h = candidate.R*cp_over_r*t-selected_energy
        expected_s = candidate.R*(cp_over_r*math.log(t)+entropy_offset)
        assert props["H_J_mol"] == pytest.approx(expected_h, abs=1e-8, rel=0)
        assert props["G_J_mol"] == pytest.approx(expected_h-t*expected_s, abs=1e-8, rel=0)
        assert props["Cp_J_mol_K"] == pytest.approx(candidate.R*cp_over_r, abs=1e-12, rel=0)


def test_multi_target_substitution_keeps_other_columns_and_respects_fraction():
    import jax
    import jax.numpy as jnp

    jax.config.update("jax_enable_x64", True)

    @dataclass
    class Setup:
        species: list
        hvector_func: object
        metadata: dict

    base = Setup(["A", "B", "C"], lambda t: jnp.zeros(jnp.shape(t)+(3,)), {"dataset": "test-base"})
    spec = {"kind": "retained_energy", "coefficients": [0., 0., 0., 0., 0.], "delta_GH_J_mol": 1135.}
    model = candidate.candidate_setup(base, {"A": spec, "C": {**spec, "delta_GH_J_mol": -2004.}}, fraction=.5)
    ts = np.array([1500., 2314.15, 4000.])
    actual = np.asarray(model.hvector_func(ts))
    np.testing.assert_array_equal(actual[:, 1], 0.)
    np.testing.assert_allclose(actual[:, 0], 1135/(2*candidate.R*ts), atol=1e-15, rtol=0)
    np.testing.assert_allclose(actual[:, 2], -2004/(2*candidate.R*ts), atol=1e-15, rtol=0)
    assert model.metadata["replaced_species"] == ["A", "C"]
    assert base.metadata == {"dataset": "test-base"}


def test_scalar_and_traced_domains_and_derivatives():
    import jax

    jax.config.update("jax_enable_x64", True)
    spec = {"kind": "retained_energy", "coefficients": [91000., -3.7, 1., .001, -5e-8], "delta_GH_J_mol": 4948.}
    for t in (1499., 4001., np.nan, np.inf):
        with pytest.raises(ValueError):
            candidate.formation_h(t, spec)
    assert np.isnan(np.asarray(candidate.formation_h_jax(np.array([1499., 4001., np.nan]), spec))).all()
    report = candidate.validate_definition({"variants": [{"name": "energy", "replacements": {"test": spec}}]})
    assert report["temperature_count"] == 2752
    assert report["offgrid_count"] == 251


def test_custom_model_contract_does_not_add_an_implicit_energy(monkeypatch):
    module = ModuleType("review_test_custom_thermals")
    # G/R = E/R - T*S/R has H=E, S=S, Cp=0 exactly.
    module.formation_g = lambda t, p, xp: (p["E"]/candidate.R-t*p["S"]/candidate.R,
                                          -p["S"]/candidate.R+xp.zeros_like(t), xp.zeros_like(t))
    monkeypatch.setitem(sys.modules, module.__name__, module)
    spec = {"kind": "custom", "module": module.__name__, "parameters": {"E": 123456., "S": 123.}}
    result = candidate.reaction_properties(2314.15, spec)
    assert result["H_J_mol"] == pytest.approx(123456., abs=1e-8, rel=0)
    assert result["S_J_mol_K"] == pytest.approx(123., abs=1e-12, rel=0)
    assert result["Cp_J_mol_K"] == 0.
