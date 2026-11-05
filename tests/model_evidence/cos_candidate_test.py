"""Verify the COS energy-only candidate against the actual curated representation."""

from hashlib import sha256
import json
import math
from pathlib import Path
import sys

import numpy as np
import pytest


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "mceg/reviews"))
import cos_candidate as candidate


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


def test_balanced_same_version_energy_and_unchanged_production_thermal_record():
    model = candidate.load_candidate()
    assert candidate.build() == model
    assert model["atct_version"] == "1.228"
    assert model["same_version_hf0_kJ_mol"] == {"C": 711.370, "O": 246.843, "S": 276.85, "COS": -141.79}
    assert model["base_D0_J_mol"] == pytest.approx(1371241., abs=1e-8, rel=0)
    assert model["D0_J_mol"] == pytest.approx(1376853., abs=1e-8, rel=0)
    assert model["delta_D0_J_mol"] == pytest.approx(5612., abs=1e-8, rel=0)
    production = json.loads(candidate.INPUTS[0].read_text())
    assert model["thermal_record"] == production["janaf_reactions"]["C1O1S1"]
    for path, digest in model["source_sha256"].items():
        assert sha256((ROOT/path).read_bytes()).hexdigest() == digest


def test_dense_replay_derivatives_and_piecewise_knot_continuity(report):
    assert (report["temperature_count"], report["step_K"], report["range_K"]) == (2501, 1, [1500, 4000])
    assert report["maximum_abs_current_h_replay_error"] < 1e-12
    assert report["maximum_abs_energy_shift_h_error"] < 1e-12
    for group in ("maximum_abs_energy_only_property_errors", "maximum_abs_jax_property_errors"):
        errors = report[group]
        assert errors["G_J_mol"] < 1e-8 and errors["H_J_mol"] < 1e-8
        assert errors["S_J_mol_K"] < 1e-10 and errors["Cp_J_mol_K"] < 1e-10
    for row in report["internal_knot_jumps"]:
        assert row["temperature_K"] in (2000., 3000.)
        assert abs(row["G_jump_J_mol"]) < 1e-8
        assert abs(row["S_jump_J_mol_K"]) < 1e-10
        assert abs(row["Cp_jump_J_mol_K"]) < 1e-10
    frozen = json.loads((candidate.DATA/"validation.json").read_text())
    for actual, expected in zip(report["samples"], frozen["samples"]):
        assert actual["candidate_over_current_K"] == pytest.approx(expected["candidate_over_current_K"], abs=1e-12, rel=0)
        assert actual["candidate_over_current_K"] == pytest.approx(math.exp(5612/(candidate.R*actual["temperature_K"])), abs=1e-12, rel=0)
        assert actual["reaction_properties"] == pytest.approx(expected["reaction_properties"], abs=1e-8, rel=0)


def test_jax_scalar_array_vmap_and_no_switch_at_external_partition_table_join():
    import jax

    previous = jax.config.x64_enabled
    jax.config.update("jax_enable_x64", True)
    try:
        # Include both production knots, off-grid points and the TIPS join.
        t = np.array([1500., 1937., 2000., 2873., 3000., 3379.9, 3380., 3380.1, 4000.])
        for updated in (False, True):
            f = jax.jit(lambda value: candidate.formation_h_jax(value, updated))
            expected = [candidate.formation_h(value, updated) for value in t]
            np.testing.assert_allclose(f(t), expected, atol=1e-12, rtol=0)
            np.testing.assert_allclose(jax.jit(jax.vmap(f))(t), expected, atol=1e-12, rtol=0)
            assert float(f(2500)) == pytest.approx(candidate.formation_h(2500, updated), abs=1e-12, rel=0)
        for value in t:
            before, after = candidate.reaction_properties(value, False), candidate.reaction_properties(value)
            assert after["S_J_mol_K"] == before["S_J_mol_K"]
            assert after["Cp_J_mol_K"] == before["Cp_J_mol_K"]
            for key in ("G_J_mol", "H_J_mol"):
                assert after[key]-before[key] == pytest.approx(-5612., abs=1e-8, rel=0)
    finally:
        jax.config.update("jax_enable_x64", previous)


def test_scalar_and_jax_domain_guards():
    import jax

    invalid = [1499., 4001., 0., float("nan"), float("inf"), -float("inf")]
    for updated in (False, True):
        for t in invalid:
            for function in (candidate.formation_h, candidate.reaction_properties):
                with pytest.raises(ValueError, match="outside"):
                    function(t, updated)
        f = jax.jit(lambda t: candidate.formation_h_jax(t, updated))
        assert np.isnan(np.asarray(f(np.array(invalid)))).all()
        assert np.isfinite(np.asarray(f(np.array([1500., 4000.])))).all()


def test_candidate_receipt_pins_the_execution_inputs_and_outputs():
    from curated_adoption_history import historical_source

    receipt = json.loads((candidate.DATA/"receipt.json").read_text())
    assert receipt["result"] == "passed" and receipt["dtype"] == "float64"
    assert receipt["command"][-2:] == ["mceg/reviews/cos_candidate.py", "--build"]
    for path, digest in receipt["selected_file_sha256"].items():
        assert sha256(historical_source(ROOT, path, digest).read_bytes()).hexdigest() == digest, path
    for path, digest in receipt["output_sha256"].items():
        assert sha256((candidate.DATA/path).read_bytes()).hexdigest() == digest, path
