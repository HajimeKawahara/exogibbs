"""Protect the CN balanced ATcT anchor, single energy correction and retained thermals."""

from hashlib import sha256
import json
import math
from pathlib import Path
import sys

import numpy as np
import pytest

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/"mceg/reviews"))
import cn_candidate as candidate


@pytest.fixture(scope="module")
def report():
    import jax
    previous=jax.config.x64_enabled
    try:
        jax.config.update("jax_enable_x64",False)
        result=candidate.validate()
        assert not jax.config.x64_enabled
        return result
    finally:
        jax.config.update("jax_enable_x64",previous)


def test_same_version_energy_balance_preserves_fit_intercept_and_residual():
    model=candidate.load_candidate()
    assert model==candidate.build()
    assert model["base_D0_J_mol"]==750156
    assert model["D0_J_mol"]==745355
    assert model["delta_D0_J_mol"]==-4801
    assert model["base_dataset"]=="curated-cos-v1"
    assert model["candidate_ATcT_version"]=="1.228"
    assert model["baseline_energy_source"]=="JANAF 1998"
    assert model["composition"]=={"C":1,"N":1}
    values=model["candidate_hf0_kJ_mol"]
    assert 1000*(values["C"]+values["N"]-values["CN"])==pytest.approx(745355,abs=1e-8,rel=0)
    assert model["formal_fit_intercept_Ra_J_mol"]!=pytest.approx(model["base_D0_J_mol"],abs=1,rel=0)
    assert model["formal_fit_intercept_Ra_J_mol"]==pytest.approx(750234.1411869854,abs=1e-8,rel=0)
    assert "not independent uncertainties" in model["uncertainty_scope"]
    assert "already includes ZPE" in model["anchor_scope"]
    for path,digest in model["source_sha256"].items():
        assert sha256((ROOT/path).read_bytes()).hexdigest()==digest


def test_dense_runtime_reconstruction_and_unchanged_entropy_heat_capacity(report):
    assert (report["temperature_count"],report["range_K"],report["step_K"])==(2501,[1500,4000],1)
    assert report["maximum_abs_current_h_replay_error"]<1e-12
    assert report["maximum_abs_energy_shift_h_error"]<1e-12
    for key in ("maximum_abs_energy_only_property_errors","maximum_abs_jax_property_errors"):
        e=report[key]
        assert max(e["G_J_mol"],e["H_J_mol"])<1e-8
        assert max(e["S_J_mol_K"],e["Cp_J_mol_K"])<1e-10
    for t in (1500.,1937.,2000.,2799.,3000.,4000.):
        before,after=[candidate.reaction_properties(t,v) for v in (False,True)]
        assert before["S_J_mol_K"]==after["S_J_mol_K"]
        assert before["Cp_J_mol_K"]==after["Cp_J_mol_K"]
        for key in ("H_J_mol","G_J_mol"):
            assert after[key]-before[key]==pytest.approx(4801,abs=1e-8,rel=0)
        assert candidate.formation_h(t)-candidate.formation_h(t,False)==pytest.approx(4801/(candidate.R*t),abs=1e-12,rel=0)
    assert report["maximum_abs_independent_fit_h_error"] < 1e-12
    for row in report["source_fit_residuals"]:
        assert abs(row["residual_preservation_error_ln_K"]) < 1e-12
        assert abs(row["current_minus_JANAF_ln_K"]) < 0.003
    from carbon_radical_source_review import curated_properties
    for t in (1500., 1937., 2799., 4000.):
        old = curated_properties(candidate.SPECIES, t)
        actual = candidate.reaction_properties(t, False)
        for key in ("H_J_mol", "S_J_mol_K", "Cp_J_mol_K"):
            assert actual[key] == pytest.approx(old[key], rel=0, abs=1e-8)


def test_scalar_jit_vectorization_and_no_out_of_domain_evaluation():
    import jax
    previous=jax.config.x64_enabled;jax.config.update("jax_enable_x64",True)
    try:
        t=np.array([1500.,1855.,2000.,2897.,3000.,4000.])
        for updated in (False,True):
            f=jax.jit(lambda value:candidate.formation_h_jax(value,updated))
            expected=np.array([candidate.formation_h(value,updated) for value in t])
            np.testing.assert_allclose(f(t),expected,rtol=0,atol=1e-12)
            np.testing.assert_allclose(jax.jit(jax.vmap(f))(t),expected,rtol=0,atol=1e-12)
            invalid=np.array([1499,4001,0,np.nan,np.inf,-np.inf])
            assert np.isnan(np.asarray(f(invalid))).all()
            for value in invalid:
                for fn in (candidate.formation_h,candidate.reaction_properties):
                    with pytest.raises(ValueError,match="outside"):
                        fn(value,updated)
    finally:
        jax.config.update("jax_enable_x64",previous)


def test_candidate_receipt_pins_actual_execution_inputs_and_outputs():
    receipt=json.loads((candidate.DATA/"receipt.json").read_text())
    assert receipt["result"]=="passed" and receipt["dtype"]=="float64"
    assert receipt["command"][-2:]==["mceg/reviews/cn_candidate.py","--build"]
    for name,digest in receipt["selected_file_sha256"].items():
        assert sha256((ROOT/name).read_bytes()).hexdigest()==digest,name
    for name,digest in receipt["output_sha256"].items():
        assert sha256((candidate.DATA/name).read_bytes()).hexdigest()==digest,name
