"""Guard the neutral NCO identity, source gauges and fixed current inventory."""

from copy import deepcopy
from hashlib import sha256
import json
import math
from pathlib import Path
import re
import sys

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "mceg/reviews"))
import nco_source_review as study
from curated_adoption_history import historical_source
from gas_source_comparison import check_equal


@pytest.fixture(scope="module")
def report():
    return study.review()


def assert_replay_equal(actual, expected):
    """Bound cancellation noise only in the dimensional curated-minus-JANAF G."""
    actual, expected = deepcopy(actual), deepcopy(expected)
    assert len(actual["source_samples"]) == len(expected["source_samples"])
    for left, right in zip(actual["source_samples"], expected["source_samples"]):
        # Subtracting reaction Gibbs energies of about 7.7e5 J/mol at 2000 K
        # shows 2.33e-10 J/mol sequential-sum noise. A 1e-8 J/mol absolute
        # bound is below 1e-12 in ln(K) here; all other fields stay unchanged.
        assert left["curated_minus_janaf_reaction"].pop("G_J_mol") == pytest.approx(
            right["curated_minus_janaf_reaction"].pop("G_J_mol"), rel=0, abs=1e-8)
    check_equal(actual, expected)


def test_offline_replay_and_balanced_source_decomposition(report):
    assert_replay_equal(report, json.loads((study.DATA/"comparison.json").read_text()))
    for row in report["source_samples"]:
        assert abs(row["decomposition_closure_ln_K"]) < 5e-13
        terms = row["janaf_minus_nasa_ln_K_components"]
        assert row["janaf_over_nasa"] == pytest.approx(math.exp(sum(terms.values())),rel=2e-13)
        assert row["energy_aligned_janaf_over_nasa"] == pytest.approx(
            math.exp(terms["molecular_thermal"]+terms["atomic_thermal"]),rel=2e-13)
        for reaction in row["native_reactions"].values():
            assert reaction["G_J_mol"] == pytest.approx(
                reaction["H_J_mol"]-row["temperature_K"]*reaction["S_J_mol_K"],abs=1e-8)
    assert report["production_change"] == "none"
    assert report["equilibrium_substitution"] == "not_run"


def test_original_nasa_coefficients_and_neutral_composition():
    nasa = json.loads((study.DATA/"nasa_records.json").read_text())
    frozen = json.loads((ROOT/"tests/data/validation/gas_source_comparison/baseline-001/inputs.json").read_text())
    raw = (study.DATA/"nasa_selected_records.txt").read_text().split("\n\n")
    assert nasa[study.inputs()["nasa_record_key"]]["composition"] == study.inputs()["composition"]
    for name,record in nasa.items():
        lines = next(b.splitlines() for b in raw if b.startswith(name+" "))
        composition = {}
        for pos in range(10,50,8):
            field = lines[1][pos:pos+8]
            element = field[:2].strip()
            if element:
                composition[element] = int(float(field[2:]))
        assert composition == record["composition"]
        assert int(lines[1][50:52]) == 0
        coefficients = [a for row in record["intervals"] for a in row["coefficients"]]
        printed = [float(v.replace("D","E")) for i in range(len(record["intervals"]))
                   for v in re.findall(r"[-+]?\d+\.\d+D[-+]\d+", "\n".join(lines[3+3*i:5+3*i]))]
        assert coefficients == printed
        if name in frozen["nasa9_records"]:
            assert coefficients == [a for row in frozen["nasa9_records"][name]["intervals"] for a in row["coefficients"]]
        assert record["hf298_J_mol"] == float(lines[1][65:])
        assert record["h298_minus_h0_J_mol"] == float(lines[2][65:])
    assert study.inputs()["atct"][study.NAME]["atct_id"] == study.inputs()["exact_atct_id"]


def test_literal_fit_matches_final_named_runtime_and_caloric_derivatives():
    import jax
    from exogibbs.presets import curated_tio2
    jax.config.update("jax_enable_x64",True)
    setup = curated_tio2.chemsetup(ions=True)
    index = setup.species.index(study.inputs()["species_id"])
    assert setup.metadata["species_provenance"][study.inputs()["species_id"]]["record"]["coefficients"] == study.inputs()["fastchem_record"]["coefficients"]
    for t in (1500.,2000.,3000.,4000.):
        p = study.curated_properties(t)
        assert p["ln_K"] == pytest.approx(-float(setup.hvector_func(t)[index]),abs=2e-12)
        assert p["G_J_mol"] == pytest.approx(p["H_J_mol"]-t*p["S_J_mol_K"],abs=1e-8)
    t,dt = 2500.,.05
    left,mid,right = [study.curated_properties(v) for v in (t-dt,t,t+dt)]
    assert (right["H_J_mol"]-left["H_J_mol"])/(2*dt) == pytest.approx(mid["Cp_J_mol_K"],abs=2e-7)
    assert (right["ln_K"]-left["ln_K"])/(2*dt) == pytest.approx(mid["H_J_mol"]/(study.R*t*t),abs=3e-10)
    with pytest.raises(ValueError,match="outside"):
        study.curated_properties(1000)


def test_current_inventory_uses_each_conserved_element(report):
    f = report["footprint"]
    assert (f["model"],f["cases"],f["species_count"],f["all_converged"]) == ("curated-tio2-v1",75,537,True)
    assert f["maximum_abs_h_error"] < 3e-12
    with np.load(study.GRID) as g:
        col = g["species"].tolist().index(study.inputs()["species_id"])
        for element,row in f["element_shares"].items():
            e,i = g["elements"].tolist().index(element),row["case_index"]
            expected = study.inputs()["composition"][element]*np.exp(g["ln_n"][i,col])/g["budget"][i,e]
            assert row["maximum_share"] == pytest.approx(expected,rel=2e-12,abs=0)
            assert row["maximum_budget_residual"] < 2e-8


def test_native_energy_and_fitting_residual_are_separate(report):
    cfg = study.inputs()
    actual = 1000*(sum(n*cfg["atct"][e]["hf0_kJ_mol"] for e,n in cfg["composition"].items())-cfg["atct"][study.NAME]["hf0_kJ_mol"])
    assert report["balanced_D0_J_mol"]["atct_1.228"] == pytest.approx(actual,abs=1e-8)
    if report["balanced_D0_J_mol"]["janaf"] is not None:
        assert abs(study.R*cfg["fastchem_record"]["coefficients"][0]-report["balanced_D0_J_mol"]["janaf"]) > 1
        for row in report["source_samples"]:
            assert row["ATcT_energy_only_over_current"] == pytest.approx(
                math.exp(report["ATcT_minus_JANAF_D0_J_mol"]/(study.R*row["temperature_K"])),rel=2e-13)
    else:
        assert report["energy_alignment_reference_K"] == 298.15
        assert all(row["ATcT_energy_only_over_current"] is None for row in report["source_samples"])


def test_pinned_inputs_resolve_history_and_outputs_are_direct():
    receipt = json.loads((study.DATA/"receipt.json").read_text())
    for name,digest in receipt["selected_input_sha256"].items():
        if name == Path(__file__).relative_to(ROOT).as_posix():
            # Preserve the original execution's test bytes after this CI amendment.
            path = study.DATA.parent/"ci-revalidation-20261106/source"/Path(name).name
        else:
            path = historical_source(ROOT,name,digest)
        assert sha256(path.read_bytes()).hexdigest() == digest,name
    for name,digest in receipt["output_sha256"].items():
        assert sha256((ROOT/name).read_bytes()).hexdigest() == digest,name
    revalidation = json.loads((study.DATA.parent/"ci-revalidation-20261106/revalidation.json").read_text())
    assert sha256(Path(__file__).read_bytes()).hexdigest() == revalidation["current_test_sha256"]
    assert sha256((study.DATA/"receipt.json").read_bytes()).hexdigest() == revalidation["original_receipt_sha256"]


def test_exact_molecular_identity_precedes_numerical_comparison():
    cfg = study.inputs()
    header = (study.DATA/cfg["tables"][study.NAME]).read_text().splitlines()[0]
    expected = json.loads((study.DATA/"identity.json").read_text())
    assert header == expected["janaf_table_header"]
    nasa = json.loads((study.DATA/"nasa_records.json").read_text())[cfg["nasa_record_key"]]
    assert nasa["reference"] == expected["nasa_reference"]
    assert cfg["nasa_same_identity"] == expected["nasa_same_identity"]
    assert cfg["exact_atct_id"] == expected["atct_id"]


def test_gibbs_difference_accepts_sum_roundoff_but_rejects_changed_energy(report):
    expected, altered = deepcopy(report), deepcopy(report)
    expected["source_samples"][1]["curated_minus_janaf_reaction"]["G_J_mol"] = 27.068383472738788
    altered["source_samples"][1]["curated_minus_janaf_reaction"]["G_J_mol"] = 27.068383472505957
    with pytest.raises(ValueError, match="curated_minus_janaf_reaction.G_J_mol"):
        check_equal(altered, expected)
    assert_replay_equal(altered, expected)
    altered["source_samples"][1]["curated_minus_janaf_reaction"]["G_J_mol"] = 27.068383472738788 + 1e-5
    with pytest.raises(AssertionError):
        assert_replay_equal(altered, expected)
