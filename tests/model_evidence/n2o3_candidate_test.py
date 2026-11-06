"""Verify N2O3 candidate construction and the actual finite-substitution archive."""

from hashlib import sha256
import json
import math
from pathlib import Path
import sys

import numpy as np
import pytest

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/"mceg/reviews"))
from curated_adoption_history import historical_source
import n2o3_candidate as candidate
import n2o3_equilibrium_impact as impact

DATA=ROOT/"tests/data/validation/n2o3_equilibrium_impact/20261105"


def test_balanced_energy_and_actual_production_input():
    model=candidate.load_candidate()
    assert model==candidate.build()
    assert model["D0_J_mol"]==pytest.approx(1590989,abs=1e-8,rel=0)
    assert model["delta_D0_J_mol"]==pytest.approx(-1404,abs=1e-8,rel=0)
    assert model["atct_version"]=="1.228" and model["base_dataset"]=="curated-cos-v1"
    assert "src/exogibbs/data/FastChem4/logK/logK.dat" in model["source_sha256"]
    for p,h in model["source_sha256"].items():assert sha256((ROOT/p).read_bytes()).hexdigest()==h


def test_dense_independent_source_reconstruction_and_caloric_identities():
    import jax
    previous=jax.config.x64_enabled
    try:
        jax.config.update("jax_enable_x64",False)
        report=candidate.validate()
        assert not jax.config.x64_enabled
    finally:jax.config.update("jax_enable_x64",previous)
    assert (report["temperature_count"],report["range_K"],report["internal_joins"])==(2501,[1500,4000],[])
    assert report["maximum_current_h_replay_error"]<2e-12
    for row in report["variants"].values():
        assert row["maximum_h_reconstruction_error"]<2e-12
        for key in ["maximum_property_reconstruction_errors","maximum_autodiff_errors"]:
            for name,error in row[key].items():assert error<(1e-7 if name in ["G_J_mol","H_J_mol"] else 1e-9)
    model=candidate.load_candidate()
    for t in [1500.,1876.4,2751.,4000.]:
        old=candidate.reaction_properties(t,"current");new=candidate.reaction_properties(t,"energy_only")
        assert new["S_J_mol_K"]==old["S_J_mol_K"] and new["Cp_J_mol_K"]==old["Cp_J_mol_K"]
        assert new["H_J_mol"]-old["H_J_mol"]==pytest.approx(-model["delta_D0_J_mol"],abs=1e-8,rel=0)


def test_traced_scalar_vector_domain_and_only_target_substitution():
    import jax
    previous=jax.config.x64_enabled;jax.config.update("jax_enable_x64",True)
    try:
        base=impact.curated.chemsetup(ions=True);col=base.species.index(candidate.source.SPECIES)
        t=np.array([1500.,1937.2,2500.,3179.1,4000.])
        for variant in candidate.VARIANTS:
            f=jax.jit(lambda temp:candidate.formation_h_jax(temp,variant))
            np.testing.assert_allclose(f(t),[candidate.formation_h(x,variant) for x in t],atol=2e-12,rtol=0)
            np.testing.assert_allclose(jax.jit(jax.vmap(f))(t),f(t),atol=2e-12,rtol=0)
            assert np.isnan(f(np.array([1499.,4001.,0.,np.nan,np.inf]))).all()
            setup=impact.candidate_setup(base,variant)
            np.testing.assert_array_equal(np.delete(np.asarray(setup.hvector_func(t)-base.hvector_func(t)),col,axis=-1),0)
            assert setup.species==base.species and setup.elements==base.elements
        with pytest.raises(ValueError,match="outside"):candidate.formation_h(1499)
        with pytest.raises(ValueError,match="Unknown"):candidate.formation_h(2500,"bad")
    finally:jax.config.update("jax_enable_x64",previous)


def test_low_temperature_cycle_uses_all_three_energies_without_domain_extension():
    report=candidate.low_temperature_dissociation()
    records=json.loads((candidate.source.DATA/"nasa_records.json").read_text())
    stoichiometry={"N2O3":-1,"NO":1,"NO2":1}
    temperature=300.
    native_g=0.
    native_h0=0.
    for name,sign in stoichiometry.items():
        record=records[name]
        interval=next(row for row in record["intervals"] if row["bounds_K"][0]<=temperature<=row["bounds_K"][1])
        a,b,c,d,e,f,g,h,s=interval["coefficients"]
        t=temperature
        enthalpy=candidate.NASA_R*(-a/t+b*math.log(t)+c*t+d*t*t/2+e*t**3/3+f*t**4/4+g*t**5/5+h)
        entropy=candidate.NASA_R*(-a/(2*t*t)-b/t+c*math.log(t)+d*t+e*t*t/2+f*t**3/3+g*t**4/4+s)
        native_g+=sign*(enthalpy-t*entropy)
        native_h0+=sign*(record["hf298_J_mol"]-record["h298_minus_h0_J_mol"])
    aligned_h0=90687+36926-90850
    expected=(native_g-native_h0)*candidate.R/candidate.NASA_R+aligned_h0
    assert report["same_version_ATcT_reaction_H0_J_mol"]==36763
    assert report["NASA_thermal_ATcT_delta_G_J_mol"]==pytest.approx(expected,abs=1e-8,rel=0)
    assert report["NASA_thermal_ATcT_delta_G_J_mol"]==pytest.approx(-1825.069972,abs=1e-6,rel=0)
    assert abs(report["native_NASA_residual_J_mol"])>70
    assert abs(report["NASA_thermal_ATcT_residual_J_mol"])<70
    assert report["ATcT_version"]=="1.228"
    assert report["network_input"]["independent_validation"] is False
    assert candidate.load_candidate()["temperature_range_K"]==[1500,4000]
    with pytest.raises(ValueError,match="outside"):candidate.formation_h(temperature)


def test_actual_endpoints_constraints_mass_action_and_gibbs_limits():
    summary=json.loads((DATA/"summary.json").read_text())
    assert summary["reused_baseline_cases"]==150
    assert summary["new_endpoint_solves"]==150*len(candidate.VARIANTS)
    diagnostics=json.loads((DATA/"diagnostics.json").read_text())
    assert len(diagnostics)==summary["new_endpoint_solves"] and all(r["converged"] for r in diagnostics)
    assert summary["new_tighter_cold_solves"]>0 and summary["new_finite_path_solves"]==10
    for mode,count in [("ions",537),("neutral",421)]:
        with np.load(DATA/f"{mode}.npz") as g:
            names=g["variants"].tolist();species=g["species"].tolist();elements=g["elements"].tolist();col=species.index(candidate.source.SPECIES)
            assert names==["current",*candidate.VARIANTS]
            assert g["ln_n"].shape==(1+len(candidate.VARIANTS),75,count)
            np.testing.assert_array_equal(np.delete(g["h"][1:]-g["h"][0],col,axis=-1),0)
            checks=impact.constraints(g["formula_matrix"],elements,species,g["budget"],g["ln_n"],g["h"],g["pressure_bar"])
            assert checks["mass_action_max_abs_ln"].max()<1e-8
            lx=impact.fractions(g["ln_n"])
            for vi in range(1,len(names)):
                np.testing.assert_allclose(g["h"][vi,:,col],[candidate.formation_h(float(t),names[vi]) for t in g["temperature_K"]],atol=2e-12,rtol=0)
                expected=-(g["h"][vi,:,col]-g["h"][0,:,col])+sum(n*(lx[vi,:,species.index(e+"1")]-lx[0,:,species.index(e+"1")]) for e,n in candidate.source.COMPOSITION.items())
                np.testing.assert_allclose(lx[vi,:,col]-lx[0,:,col],expected,atol=1e-8,rtol=0)
                raw,corrected=impact.gibbs_ordering(g["ln_n"][[0,vi]],g["h"][[0,vi]],g["pressure_bar"],g["formula_matrix"],elements,species)
                np.testing.assert_allclose(corrected,g["delta_g_corrected"][vi-1],atol=1e-18,rtol=1e-8)
            highlight=int(g["highlighted_case_index"]);path=impact.fractions(g["lambda_ln_n"])
            delta=path[:,col]-path[0,col]
            assert np.all(np.diff(delta)>0) if candidate.PRIMARY=="nasa_atct" else np.all(np.diff(delta)<0)
            np.testing.assert_allclose(path[[0,-1],col],lx[[0,names.index(candidate.PRIMARY)],highlight,col],atol=1e-8,rtol=0)


def test_tighter_controls_and_independent_solver_are_bounded():
    summary=json.loads((DATA/"summary.json").read_text())
    for network in summary["networks"]:
        assert network["tolerance_stability"]["target_delta_ln_x"]<1e-8
        assert network["tolerance_stability"]["absolute_delta_x"]<1e-9
        assert network["maximum_target_elements_relative_residual"]<1e-9
        for variant in network["variants"]:
            assert variant["finite_mass_action_closure_ln"]<1e-8
            assert variant["gibbs_corrected_current_min"]>=-5e-14
            assert variant["gibbs_corrected_candidate_max"]<=5e-14
    assert isinstance(summary["independent_solver"],list) and summary["independent_solver"]
    for case in summary["independent_solver"]:
        assert len(case["comparisons"])==1+len(candidate.VARIANTS)
        for row in case["comparisons"]:assert row["max_abs_delta_log10_x"]<1e-4


def test_candidate_and_impact_execution_receipts():
    for data in [candidate.DATA,DATA]:
        receipt=json.loads((data/"receipt.json").read_text())
        assert receipt["result"]=="passed" and receipt["dtype"]=="float64"
        for p,h in receipt["selected_file_sha256"].items():assert sha256(historical_source(ROOT,p,h).read_bytes()).hexdigest()==h,p
        for p,h in receipt["output_sha256"].items():assert sha256((data/p).read_bytes()).hexdigest()==h,p
    assert len(json.loads((DATA/"receipt.json").read_text())["fastchem_binary_sha256"])==64
