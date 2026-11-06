"""Verify independent HCP energy anchors, unchanged thermals and bounded replay."""

from hashlib import sha256
import json
from pathlib import Path
import sys

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "mceg/reviews"))
import hcp_candidate as candidate
from curated_adoption_history import historical_source


@pytest.fixture(scope="module")
def report():
    import jax
    previous = jax.config.x64_enabled
    try:
        jax.config.update("jax_enable_x64", False)
        result = candidate.validate()
        assert not jax.config.x64_enabled
        return result
    finally:
        jax.config.update("jax_enable_x64", previous)


def check_energy_difference(actual, expected):
    # Subtracting two ~MJ/mol potentials incurs platform-dependent cancellation.
    # This dimensional allowance does not relax dimensionless or source checks.
    assert actual == pytest.approx(expected, abs=1e-8, rel=0)


def test_balanced_anchors_preserve_white_phosphorus_reference_and_fit_residual():
    model = candidate.load_candidate()
    assert model == candidate.build()
    assert model["base_dataset"] == "curated-tio2-v1"
    assert model["composition"] == {"C": 1, "H": 1, "P": 1}
    assert model["base_Hf0_J_mol"] == 151192
    assert model["base_D0_J_mol"] == 1091582
    check_energy_difference(model["white_phosphorus_elemental_heat_J_mol"], 5360)
    assert model["theory_own_atomic_TAE0_J_mol"] == 1018000
    assert abs(model["formal_fit_intercept_Ra_J_mol"]-model["base_D0_J_mol"]) > 1
    for name, hf0, delta, width, d0 in [
        ("cccbdb", 213500, 62308, 20000, 1029274),
        ("theory_2001", 225000, 73808, 8000, 1017774),
    ]:
        anchor = model["anchors"][name]
        assert (anchor["Hf0_J_mol"], anchor["delta_Hf0_J_mol"],
                anchor["reported_uncertainty_J_mol"], anchor["common_JANAF_D0_J_mol"]) == (hf0, delta, width, d0)
        assert candidate.energy_shift(name) == delta
        assert candidate.energy_shift(name, -width) == delta-width
    assert candidate.energy_shift(None) == 0
    assert "not independent Gaussian" in model["uncertainty_scope"]
    assert "does not supply states or Q" in model["thermal_scope"]
    for path, digest in model["source_sha256"].items():
        assert sha256(historical_source(ROOT, path, digest).read_bytes()).hexdigest() == digest


def test_dense_continuous_calorics_and_source_fit_residual(report):
    assert (report["temperature_count"], report["range_K"], report["step_K"]) == (2501, [1500, 4000], 1)
    assert report["maximum_abs_current_h_replay_error"] < 1e-12
    for anchor, result in report["anchors"].items():
        assert result["maximum_abs_independent_h_error"] < 1e-12
        assert result["maximum_abs_energy_shift_h_error"] < 1e-12
        for field in ("maximum_abs_energy_only_property_errors", "maximum_abs_jax_property_errors"):
            errors = result[field]
            assert max(errors["G_J_mol"], errors["H_J_mol"]) < 1e-8
            assert max(errors["S_J_mol_K"], errors["Cp_J_mol_K"]) < 1e-10
        for row in result["source_fit_residuals"]:
            assert abs(row["preservation_error_ln_K"]) < 1e-12
        for t in (1500., 1937., 2000., 2799., 3000., 4000.):
            old, new = [candidate.reaction_properties(t, a) for a in (None, anchor)]
            for key in ("G_J_mol", "H_J_mol"):
                check_energy_difference(new[key]-old[key], candidate.energy_shift(anchor))
            for key in ("S_J_mol_K", "Cp_J_mol_K"):
                assert old[key] == new[key]
            check_energy_difference(new["G_J_mol"], new["H_J_mol"]-t*new["S_J_mol_K"])
    with pytest.raises(AssertionError):
        check_energy_difference(62308.+1e-5, 62308.)


def test_jax_scalar_array_jit_vmap_and_domain():
    import jax
    previous = jax.config.x64_enabled
    jax.config.update("jax_enable_x64", True)
    try:
        t = np.array([1500., 1855., 2000., 2897., 3000., 4000.])
        for anchor in (None, *candidate.ANCHORS):
            f = jax.jit(lambda value: candidate.formation_h_jax(value, anchor))
            expected = np.array([candidate.formation_h(value, anchor) for value in t])
            np.testing.assert_allclose(f(t), expected, rtol=0, atol=1e-12)
            np.testing.assert_allclose(jax.jit(jax.vmap(f))(t), expected, rtol=0, atol=1e-12)
            invalid = np.array([1499., 4001., 0., np.nan, np.inf, -np.inf])
            assert np.isnan(f(invalid)).all()
            for value in invalid:
                with pytest.raises(ValueError, match="outside"):
                    candidate.formation_h(value, anchor)
        for anchor, offset in [(None, 1), ("missing", 0), ("cccbdb", np.nan)]:
            with pytest.raises(ValueError):
                candidate.energy_shift(anchor, offset)
    finally:
        jax.config.update("jax_enable_x64", previous)


def test_execution_pins_are_separate_from_later_test_verification():
    receipt = json.loads((candidate.DATA / "receipt.json").read_text())
    assert receipt["result"] == "passed" and receipt["dtype"] == "float64"
    assert receipt["command"][-2:] == ["mceg/reviews/hcp_candidate.py", "--build"]
    for name, digest in receipt["selected_file_sha256"].items():
        assert sha256(historical_source(ROOT, name, digest).read_bytes()).hexdigest() == digest, name
    for name, digest in receipt["output_sha256"].items():
        assert sha256((candidate.DATA / name).read_bytes()).hexdigest() == digest, name
