"""Check CS2 balanced energies, partition conventions and saved-grid scope."""
import json
import math
from hashlib import sha256
from pathlib import Path
import sys

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "mceg/reviews"))
from curated_adoption_history import historical_source
import cs2_source_review as study


@pytest.fixture(scope="module")
def report():
    return study.review()


def test_same_version_energy_and_curated_reconstruction(report):
    assert report["atct_1.228_balanced_D0_J_mol"] == pytest.approx(1149900)
    assert report["balanced_D0_J_mol"]["atct"] == pytest.approx(1149161)
    for r in report["source_samples"]:
        t=r["temperature_K"]
        assert r["new_ATcT_energy_only_K_ratio"] == pytest.approx(math.exp((1149900-1149161)/(study.major.R*t)))
        if t>=1500 and "reconstructed_adopted_ln_K" in r:
            assert r["curated_ln_K"] == pytest.approx(r["reconstructed_adopted_ln_K"],abs=2e-11)
    with pytest.raises(ValueError,match="outside"):
        study.curated_ln_k(1000)


def test_partition_stencil_calorics_rounding_and_join(report):
    t=np.arange(1000.,2001.,10.)
    # Independent analytic power-law partition has Cp/R=2.5+3.
    synthetic=np.stack([t,t**3],axis=1)
    value=study.table_calorics(synthetic,1500,10)
    assert value["Cp_J_mol_K"] == pytest.approx(5.5*study.major.R,abs=2e-6)
    assert 1.009 < report["tips_join_over_previous_log_slope"] < 1.012
    for r in report["source_samples"]:
        p=r["tips2025"]
        assert p["isotope_translation_K_proxy"] == pytest.approx(p["Q_over_nasa"]*(75.944142/76.1407)**1.5)
        assert p["Cp_rounding_bound_J_mol_K"] > 0
    top=report["source_samples"][-1]["tips2025"]
    assert top["Q_over_nasa"] == pytest.approx(0.9471072993893781)
    assert abs(top["Cp_J_mol_K"]-top["Cp_step20_J_mol_K"]) < top["Cp_rounding_bound_J_mol_K"]


def test_archived_sulfur_share_accounts_for_stoichiometry(report):
    f=report["footprint"]
    assert (f["model"],f["cases"],f["species"],f["all_converged"]) == ("curated-oh-v1",75,537,True)
    assert f["maximum_abs_h_error"] < 2e-12
    assert f["maximum_relative_sulfur_conservation_residual"] < 3e-12
    assert f["maximum_sulfur_fraction"] == pytest.approx(5.845228937199336e-06)
    assert f["case_index"] == 68
    with np.load(study.GRID) as g:
        i=g["species"].tolist().index("C1S2");s=g["elements"].tolist().index("S")
        assert g["formula_matrix"][s,i] == 2
        assert 2*np.exp(g["ln_n"][68,i])/g["budget"][68,s] == pytest.approx(f["maximum_sulfur_fraction"])
    assert report["production_change"] == "none"
    assert report["equilibrium_substitution"] == "not_run"


def test_report_and_actual_input_receipts(report):
    expected=json.loads((study.DATA/"comparison.json").read_text())
    def equal(a,b,path=()):
        if isinstance(a,dict):
            assert a.keys()==b.keys()
            for k in a:equal(a[k],b[k],path+(k,))
        elif isinstance(a,list):
            assert len(a)==len(b)
            for i,(x,y) in enumerate(zip(a,b)):equal(x,y,path+(i,))
        elif isinstance(a,float):
            # Cp second differences amplify roundoff by R * T**2 / step**2.
            # 1e-7 J/mol/K is far below even the table-rounding contribution
            # (roughly 0.1 J/mol/K for TIPS); it is not a physical tolerance.
            finite_difference_cp = (
                len(path)>=2 and path[-2] in {"tips2025"}
                and path[-1] in {"Cp_J_mol_K", "Cp_step20_J_mol_K", "delta_Cp_J_mol_K"}
            )
            tolerance = 1e-7 if finite_difference_cp else 2e-10
            assert a==pytest.approx(b,rel=2e-11,abs=tolerance),path
        else:assert a==b
    # The observed CI cancellation is accepted only in partition-derived Cp.
    ci_value, saved_value = -1.6409908880750876, -1.6409908875110233
    equal(ci_value,saved_value,("source_samples",3,"tips2025","delta_Cp_J_mol_K"))
    for path in (("source_samples",3,"curated_ln_K"), ("balanced_D0_J_mol","atct")):
        with pytest.raises(AssertionError):
            equal(ci_value,saved_value,path)
    equal(report,expected)
    receipt=json.loads((study.DATA/"receipt.json").read_text())
    for group in ("selected_input_sha256","output_sha256"):
        for p,digest in receipt[group].items():
            assert sha256(historical_source(ROOT,p,digest).read_bytes()).hexdigest()==digest,p
