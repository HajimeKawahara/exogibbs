"""Verify O2 source anchors, moments, inherited fit and immutable evidence."""

from copy import deepcopy
from hashlib import sha256
import json
from pathlib import Path
import re
import sys

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "mceg/reviews"))
import o2_source_review as study
from curated_adoption_history import historical_source


def compare(actual, expected, path=""):
    if isinstance(expected, dict):
        assert actual.keys() == expected.keys(), path
        for key in expected:
            compare(actual[key], expected[key], path + "." + key)
    elif isinstance(expected, list):
        assert len(actual) == len(expected), path
        for i, value in enumerate(expected):
            compare(actual[i], value, path + f"[{i}]")
    elif isinstance(expected, float):
        assert actual == pytest.approx(expected, rel=2e-12,
            abs=1e-8 if "_J_mol" in path else 2e-11), path
    else:
        assert actual == expected, path


def test_source_replay_and_dimensional_regression():
    expected = json.loads((study.DATA / "comparison.json").read_text())
    compare(study.review(), expected)
    changed = deepcopy(expected)
    changed["source_samples"][0]["modern"]["natural_2025_minus_NASA_thermal_H_J_mol"] += 1e-4
    with pytest.raises(AssertionError):
        compare(changed, expected)
    for row in expected["source_samples"]:
        source = row["source"]
        assert abs(source["closure_ln_K"]) < 5e-13
        assert source["native_janaf_D0_J_mol"] == pytest.approx(493579)
        assert row["modern"]["balanced_ATcT_D0_J_mol"] == 493686
        assert row["modern"]["ATcT_minus_JANAF_D0_J_mol"] == 107
        assert 1.005 < row["modern"]["natural_2025_ATcT_NASA_atoms_over_curated_K"] < 1.009


def test_printed_nasa_and_janaf_inputs():
    config = study.inputs()
    raw = (study.DATA / "nasa-selected-records.txt").read_text().split("\n\n")
    packaged = np.load(ROOT / "src/exogibbs/data/ykb4/gibbs_matrices.npz", allow_pickle=True)["arr_0"].item()
    for name, record in config["nasa9"].items():
        lines = next(b.splitlines() for b in raw if b.startswith(name + " "))
        printed = [float(v.replace("D", "E")) for i in range(len(record["intervals"]))
                   for v in re.findall(r"[-+]?\d+\.\d+D[-+]\d+", "\n".join(lines[3+3*i:5+3*i]))]
        assert printed == [v for interval in record["intervals"] for v in interval["coefficients"]]
        assert record["hf298_J_mol"] == float(lines[1][65:])
        assert record["h298_minus_h0_J_mol"] == float(lines[2][65:])
        table = study.read_table(study.DATA / config["tables"][name])
        frame = packaged["O2" if name == "O2" else "O1"]
        for t in [0., 298.15, 1500., 2000., 3000., 4000.]:
            actual = frame[frame["T(K)"] == t].iloc[0].to_numpy(dtype=float)[1:]
            np.testing.assert_array_equal(actual, table[t])


def test_actual_named_factory_and_analytic_calorics():
    import jax
    from exogibbs.presets import curated_tio2
    jax.config.update("jax_enable_x64", True)
    setup = curated_tio2.chemsetup(ions=True)
    j = setup.species.index(study.SPECIES)
    for t in [1500., 2000., 3000., 4000.]:
        p = study.curated_properties(t)
        assert p["ln_K"] == pytest.approx(-float(setup.hvector_func(t)[j]), abs=3e-12)
        assert p["G_J_mol"] == pytest.approx(p["H_J_mol"]-t*p["S_J_mol_K"], abs=1e-8)
    t, dt = 2500., .05
    left, mid, right = [study.curated_properties(x) for x in [t-dt,t,t+dt]]
    assert (right["H_J_mol"]-left["H_J_mol"])/(2*dt) == pytest.approx(mid["Cp_J_mol_K"], abs=2e-7)
    assert (right["S_J_mol_K"]-left["S_J_mol_K"])/(2*dt) == pytest.approx(mid["Cp_J_mol_K"]/t, abs=2e-10)
    with pytest.raises(ValueError, match="outside"):
        study.curated_properties(1000)


def test_2025_spin_isotope_and_energy_zero_conventions():
    rows = study.review()["source_samples"]
    for row in rows:
        modern = row["modern"]
        # Published high-T nodes are rounded. Their independent Q moments close
        # inside the stated uncertainties; this is not a numerical fit residual.
        assert abs(modern["pure16_Q_moment_H_error_J_mol"]) < 3
        assert abs(modern["pure16_Q_moment_S_error_J_mol_K"]) < .002
        assert abs(modern["pure16_Q_moment_Cp_error_J_mol_K"]) < .003
        assert .003 < modern["natural_minus_pure16_S_J_mol_K"] < .005
        assert modern["published_natural_expanded_U_Cp_J_mol_K"] > 0
    config = study.inputs()
    assert "No extra factor of three" in config["identity"]["spin"]
    assert config["atct"]["O_hf0_kJ_mol"] == 246.843


def test_current_footprint_scope_and_receipt():
    result = study.review()
    f = result["footprint"]
    assert (f["dataset"],f["cases"],f["species"],f["all_converged"]) == ("curated-tio2-v1",75,537,True)
    assert f["maximum_abs_h_error"] < 3e-12
    assert f["element_shares"]["O"]["maximum_share"] == pytest.approx(4.894079029962129e-5)
    assert result["new_equilibrium_substitution"] == "not_run" and result["curated_change"] == "none"
    receipt = json.loads((study.DATA / "receipt.json").read_text())
    for name, expected in receipt["selected_input_sha256"].items():
        assert sha256(historical_source(ROOT, name, expected).read_bytes()).hexdigest() == expected, name
    for name, expected in receipt["output_sha256"].items():
        assert sha256((study.DATA / name).read_bytes()).hexdigest() == expected, name
