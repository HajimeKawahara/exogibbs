"""Protect the CS2 balanced ATcT anchor, single energy correction and retained thermals."""

from hashlib import sha256
import json
import math
from pathlib import Path
import sys

import numpy as np
import pytest

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/"mceg/reviews"))
import cs2_candidate as candidate


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


def test_same_version_atoms_and_molecule_balance_without_double_alignment():
    model=candidate.load_candidate()
    assert model==candidate.build()
    assert model["base_D0_J_mol"]==pytest.approx(1149161,abs=1e-8,rel=0)
    assert model["D0_J_mol"]==pytest.approx(1149900,abs=1e-8,rel=0)
    assert model["delta_D0_J_mol"]==pytest.approx(739,abs=1e-8,rel=0)
    assert model["base_dataset"]=="curated-cos-v1"
    assert model["candidate_ATcT_version"]=="1.228"
    assert model["baseline_ATcT_version"]=="1.172"
    assert model["composition"]=={"C":1,"S":2}
    for key,energy in (("baseline_hf0_kJ_mol",1149161),("candidate_hf0_kJ_mol",1149900)):
        values=model[key]
        assert 1000*(values["C"]+2*values["S"]-values["CS2"])==pytest.approx(energy,abs=1e-8,rel=0)
    production=json.loads((ROOT/"src/exogibbs/data/curated/gas_v1.json").read_text())
    assert model["original_reaction_coefficients"]==production["nasa9_reactions"]["C1S2"]
    assert "not an independent measurement" in model["uncertainty_scope"]
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
            assert after[key]-before[key]==pytest.approx(-739,abs=1e-8,rel=0)
        assert candidate.formation_h(t)-candidate.formation_h(t,False)==pytest.approx(-739/(candidate.R*t),abs=1e-12,rel=0)
    assert report["maximum_abs_independent_native_source_h_error"] < 1e-12


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
    assert receipt["command"][-2:]==["mceg/reviews/cs2_candidate.py","--build"]
    for name,digest in receipt["selected_file_sha256"].items():
        assert sha256((ROOT/name).read_bytes()).hexdigest()==digest,name
    for name,digest in receipt["output_sha256"].items():
        assert sha256((candidate.DATA/name).read_bytes()).hexdigest()==digest,name
