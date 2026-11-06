"""Protect the PS retained-energy thermal difference and finite-state conventions."""

from hashlib import sha256
import json
import math
from pathlib import Path
import sys

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/"mceg/reviews"))
import ps_candidate as candidate


@pytest.fixture(scope="module")
def report():
    import jax
    previous = jax.config.x64_enabled
    try:
        jax.config.update("jax_enable_x64",False)
        result = candidate.validate()
        assert not jax.config.x64_enabled
        return result
    finally:
        jax.config.update("jax_enable_x64",previous)


def test_literal_current_fit_and_physical_anchor_are_retained():
    model = candidate.load_candidate()
    assert model == candidate.build()
    assert model["original_source_D0_J_mol"] == 438000.
    assert model["delta_D0_J_mol"] == 0
    assert model["D0_evaluation_margin_J_mol"] == 10000.
    assert model["bound_cutoff_cm-1"] == pytest.approx(36613.94086350959,abs=1e-8,rel=0)
    assert model["NASA_molecular_H0_gauge_J_mol"] == pytest.approx(140815.2,abs=1e-8,rel=0)
    assert model["composition"] == {"P":1,"S":1}
    assert model["baseline_dataset"] == "curated-cos-v1"
    assert "fit/source residual" in model["thermal_rule"]
    for path,digest in model["source_sha256"].items():
        assert sha256((ROOT/path).read_bytes()).hexdigest() == digest,path


def test_dense_replay_and_independent_moment_derivatives(report):
    assert (report["temperature_count"],report["range_K"],report["step_K"]) == (2501,[1500,4000],1)
    for key in ("maximum_abs_current_h_replay_error","maximum_abs_scalar_jax_h_error","maximum_abs_independent_moment_h_error"):
        assert report[key] < 1e-12
    for errors in report["jax_float64_derivative_errors"].values():
        assert max(errors["G_J_mol"],errors["H_J_mol"]) < 1e-8
        assert max(errors["S_J_mol_K"],errors["Cp_J_mol_K"]) < 1e-10
    for t in (1753.,3557.):
        hplus = candidate.reaction_properties(t+.1)["H_J_mol"]
        hminus = candidate.reaction_properties(t-.1)["H_J_mol"]
        assert (hplus-hminus)/.2 == pytest.approx(candidate.reaction_properties(t)["Cp_J_mol_K"],abs=2e-7,rel=0)


def test_nuclear_spin_and_fixed_cutoff_are_applied_once(report):
    full,central = candidate.states("full"),candidate.states()
    assert len(full)==225997 and len(central)==223645
    assert report["omitted_above_central_D0_levels"] == 2352
    assert np.all(full[:,1] == 2*(2*full[:,2]+1))
    assert full[:,0].min() == 0
    assert central[:,0].max() <= candidate.load_candidate()["bound_cutoff_cm-1"]
    assert report["D0_plus_margin_cutoff_keeps_all_distributed_levels"]
    for row in report["samples"]:
        assert abs(row["full_state_sum_over_distributed_pf"]-1) < 1e-6
        assert sum(row["electronic_Q_fractions"].values()) == pytest.approx(1,abs=1e-14,rel=0)
        t = row["temperature_K"]
        ratio = row["K_over_current"]["full"]/row["K_over_current"]["pops"]
        assert ratio == pytest.approx(row["sensitivity"]["distributed_full"]["Q_over_primary"],rel=1e-12)
        shifted = central.copy();shifted[:,0] += 100
        initial = candidate.direct_thermal(t,central)
        changed = candidate.direct_thermal(t,shifted)
        assert changed["Q"]/initial["Q"] == pytest.approx(math.exp(-100*candidate.C2/t),rel=1e-12)
        assert changed["H_J_mol"]-initial["H_J_mol"] == pytest.approx(100*candidate.R*candidate.C2,abs=1e-8,rel=0)
        assert changed["Cp_J_mol_K"] == pytest.approx(initial["Cp_J_mol_K"],abs=1e-10,rel=0)


def test_electronic_effect_and_boundary_sensitivity_are_distinct(report):
    row = report["samples"][-1]
    assert row["temperature_K"] == 4000
    assert row["K_over_current"]["pops"] == pytest.approx(1.014492963809801,rel=1e-12)
    assert row["K_over_current"]["ground"] < 1
    assert row["electronic_Q_fractions"]["a4Pi"] > .015
    assert row["electronic_Q_fractions"]["B2Pi"] < .0005
    cp_delta = row["reaction_properties"]["pops"]["Cp_J_mol_K"]-row["reaction_properties"]["current"]["Cp_J_mol_K"]
    assert cp_delta == pytest.approx(4.205054660804443,abs=1e-8,rel=0)
    assert row["sensitivity"]["distributed_full"]["delta_Cp_J_mol_K"] < .01
    assert abs(row["sensitivity"]["D0_minus_margin_cutoff"]["delta_Cp_J_mol_K"]) < .056
    assert report["isotope_translation_K_factor"] == pytest.approx(.9977895953394582,rel=1e-12)
    assert "no quantified global bound" in report["completeness"]


def test_jit_vectorization_and_domain_guards():
    import jax
    previous = jax.config.x64_enabled
    jax.config.update("jax_enable_x64",True)
    try:
        t = np.array([1500.,1831.,2500.,3781.,4000.])
        f = jax.jit(candidate.formation_h_jax)
        expected = np.array([candidate.formation_h(value) for value in t])
        np.testing.assert_allclose(f(t),expected,rtol=0,atol=1e-12)
        np.testing.assert_allclose(jax.jit(jax.vmap(f))(t),expected,rtol=0,atol=1e-12)
        invalid = np.array([1499.,4001.,0.,np.nan,np.inf,-np.inf])
        assert np.isnan(np.asarray(f(invalid))).all()
        for value in invalid:
            for fn in (candidate.formation_h,candidate.reaction_properties):
                with pytest.raises(ValueError,match="outside"):
                    fn(value)
    finally:
        jax.config.update("jax_enable_x64",previous)


def test_candidate_receipt_pins_execution():
    receipt = json.loads((candidate.DATA/"receipt.json").read_text())
    assert receipt["result"] == "passed" and receipt["dtype"] == "float64"
    assert receipt["command"][-2:] == ["mceg/reviews/ps_candidate.py","--build"]
    for name,digest in receipt["selected_file_sha256"].items():
        assert sha256((ROOT/name).read_bytes()).hexdigest() == digest,name
    for name,digest in receipt["output_sha256"].items():
        assert sha256((candidate.DATA/name).read_bytes()).hexdigest() == digest,name
