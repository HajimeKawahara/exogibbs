"""Independent balanced-source, caloric, archive and pinned replay checks."""
from copy import deepcopy
from hashlib import sha256
import json
import math
from pathlib import Path
import sys
import numpy as np
import pytest
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT / "mceg/reviews"))
from curated_adoption_history import historical_source
from gas_source_comparison import check_equal
import cp_source_review as study

@pytest.fixture(scope="module")
def report():
    return study.review()


def test_pinned_replay_and_source_hashes(report):
    saved=json.loads((study.DATA/"comparison.json").read_text())
    # State moment reductions may change the last bits after cancellation.
    actual, expected=deepcopy(report),deepcopy(saved)
    for left,right in zip(actual["modern"],expected["modern"]):
        assert left.pop("delta_Hthermal_J_mol")==pytest.approx(right.pop("delta_Hthermal_J_mol"),rel=0,abs=1e-8)
    check_equal(actual,expected)
    receipt=json.loads((study.DATA/"receipt.json").read_text())
    for path,digest in receipt["selected_input_sha256"].items():
        assert sha256(historical_source(ROOT,path,digest).read_bytes()).hexdigest()==digest,path
    for path,digest in receipt["output_sha256"].items():
        assert sha256((ROOT/path).read_bytes()).hexdigest()==digest,path


def test_balanced_neutral_atoms_and_energy_alignment(report):
    inp=json.loads((study.DATA/"inputs.json").read_text())
    for row in report["samples"]:
        t=row["temperature_K"]; p=row["properties"]
        for src in ("nasa","janaf"):
            for key in ("G_J_mol","H_J_mol","S_J_mol_K","Cp_J_mol_K"):
                balanced=p[inp["name"]][src][key]-math.fsum(n*p[s][src][key] for s,n in inp["composition"].items())
                assert row["reaction"][src][key]==pytest.approx(balanced,abs=2e-9)
        assert row["reaction_delta_G_J_mol"]==pytest.approx(row["delta_H0_J_mol"]+row["delta_Hthermal_J_mol"]+row["minus_T_delta_S_J_mol"],abs=3e-9)
        assert row["H0_aligned_janaf_over_nasa"]==pytest.approx(row["janaf_over_nasa"]*math.exp(row["delta_H0_J_mol"]/(study.R*t)),rel=2e-12)


def test_literal_current_calorics_and_domain(report):
    species=report["species_id"]
    for t in (1800.,2500.,3500.):
        c=study.current_properties(t,species)
        plus=study.current_properties(t+.1,species);minus=study.current_properties(t-.1,species)
        assert c["H_J_mol"]==pytest.approx(c["G_J_mol"]+t*c["S_J_mol_K"],abs=1e-8)
        assert (plus["H_J_mol"]-minus["H_J_mol"])/.2==pytest.approx(c["Cp_J_mol_K"],abs=3e-7)
    for t in (1499,4001):
        with pytest.raises(ValueError):study.current_properties(t,species)


def test_final_baseline_is_saved_allocation_not_substitution(report):
    fp=report["footprint"]
    assert fp["cases"]==75 and fp["species"]==537 and fp["all_converged"]
    assert fp["maximum_abs_target_h_error"]<1e-11
    with np.load(study.GRID) as grid:
        col=grid["species"].tolist().index(report["species_id"])
        assert fp["maximum_mole_fraction"]==pytest.approx(float(np.exp(grid["ln_x"][:,col]).max()),rel=2e-12)
        e=grid["elements"].tolist().index(fp["maximum_element"])
        # Independent conversion from mole fractions back to absolute amounts.
        shares=grid["formula_matrix"][e,col]*np.exp(grid["ln_x"][:,col])*np.exp(grid["ln_n"]).sum(axis=1)/grid["budget"][:,e]
        assert fp["maximum_element_fraction"]==pytest.approx(float(shares.max()),rel=2e-12)
    assert report["curated_change"]=="none" and report["new_equilibrium_solves"]==0
