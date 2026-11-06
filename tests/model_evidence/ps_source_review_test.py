"""Offline phosphorus-source balance, convention and caloric checks."""
from copy import deepcopy
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
import phosphorus_diatomic_source_review as study
from gas_source_comparison import check_equal
SLUG="ps"
DATA=ROOT/f"tests/data/validation/{SLUG}_source_review/20261106"
@pytest.fixture(scope="module")
def report(): return study.review(SLUG)

def assert_replay_equal(actual, expected):
    """Allow cancellation noise only in the dimensional thermal-H difference."""
    actual, expected = deepcopy(actual), deepcopy(expected)
    for left, right in zip(actual["modern"]["samples"], expected["modern"]["samples"]):
        # This subtracts large enthalpies after a state-moment BLAS reduction.
        # Platform-dependent last bits can exceed the generic absolute tolerance
        # near cancellation; 1e-8 J/mol still rejects physically relevant changes.
        assert left.pop("delta_Hthermal_J_mol") == pytest.approx(
            right.pop("delta_Hthermal_J_mol"), rel=0, abs=1e-8)
    check_equal(actual, expected)


def test_pinned_replay(report):
    assert_replay_equal(report,json.loads((DATA/"comparison.json").read_text()))
    receipt=json.loads((DATA/"receipt.json").read_text())
    for path,expected in receipt["selected_input_sha256"].items():
        assert sha256(historical_source(ROOT,path,expected).read_bytes()).hexdigest()==expected,path
    assert report["JANAF_frozen_nodes_matched"]==4
    assert report["curated_change"]=="none" and report["new_equilibrium_solves"]==0
    assert report["physical_accuracy"]=="unknown"

def test_neutral_atom_balance_and_energy_alignment(report):
    inputs=json.loads((DATA/"inputs.json").read_text())
    for row in report["samples"]:
        t=row["temperature_K"];p=row["properties"]
        def delta(s):
            v=p[s]
            return v["janaf_H_J_mol"]-v["nasa_H_J_mol"]-t*(v["janaf_S_J_mol_K"]-v["nasa_S_J_mol_K"])
        balanced=delta(inputs["name"])-math.fsum(n*delta(s) for s,n in inputs["composition"].items())
        assert row["reaction_delta_G_J_mol"]==pytest.approx(balanced,abs=2e-9)
        assert row["H0_aligned_janaf_over_nasa"]==pytest.approx(row["janaf_over_nasa"]*math.exp(row["delta_H0_J_mol"]/(study.R*t)),rel=1e-12)
    for t in [1499,4001]:
        with pytest.raises(ValueError):study.current_ln_k(t,inputs["species_id"])

def test_archived_allocation_is_not_a_substitution(report):
    fp=report["footprint"];inputs=json.loads((DATA/"inputs.json").read_text())
    with np.load(study.GRID) as grid:
        col=grid["species"].tolist().index(inputs["species_id"]);p=grid["elements"].tolist().index("P")
        shares=inputs["composition"]["P"]*np.exp(grid["ln_x"][:,col])*np.exp(grid["ln_n"]).sum(axis=1)/grid["budget"][:,p]
        assert fp["maximum_phosphorus_fraction"]==pytest.approx(max(shares),rel=2e-12)
    assert fp["cases"]==75 and fp["species"]==537 and fp["all_converged"]
    assert fp["maximum_abs_target_h_error"]<1e-11

def test_state_moments_and_spin_convention(report):
    inp=json.loads((DATA/"inputs.json").read_text())
    with np.load(DATA/"states.npz") as packet: states=packet["states"]
    assert np.all(states[:,1]==inp["nuclear_spin_factor"]*(2*states[:,2]+1))
    for row in report["modern"]["samples"]:
        t=row["temperature_K"]
        assert abs(row["state_sum_over_distributed_pf"]-1)<1e-6
        assert math.fsum(row["electronic_fractions"].values())==pytest.approx(1,abs=3e-14)
        hplus=study.moments(states,t+.1)["H_minus_H0_J_mol"]
        hminus=study.moments(states,t-.1)["H_minus_H0_J_mol"]
        assert (hplus-hminus)/.2==pytest.approx(row["Cp_J_mol_K"],abs=2e-7)
    shifted=states.copy();shifted[:,0]+=100
    t=2000; original=study.moments(states,t); change=study.moments(shifted,t)
    assert change["Q"]/original["Q"]==pytest.approx(math.exp(-100*study.C2/t),rel=1e-12)
    assert change["Cp_J_mol_K"]==pytest.approx(original["Cp_J_mol_K"],abs=1e-12)


def test_thermal_enthalpy_tolerance_rejects_a_changed_result(report):
    altered = deepcopy(report)
    altered["modern"]["samples"][0]["delta_Hthermal_J_mol"] += 1e-5
    with pytest.raises(AssertionError):
        assert_replay_equal(altered, report)
