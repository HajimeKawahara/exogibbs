"""Verify NaO source separation, caloric derivatives and archived scope offline."""

import json
import math
from pathlib import Path
import sys

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "mceg/reviews"))
import nao_source_review as study


@pytest.fixture(scope="module")
def inputs():
    return json.loads((study.DATA / "inputs.json").read_text())


@pytest.fixture(scope="module")
def report():
    return study.review()


def test_energy_balance_uses_ground_increments_and_common_reference(inputs, report):
    mol = inputs["molecule"]
    direct_janaf = 1000*(sum(n*inputs["janaf"][a]["0.0"][4] for a, n in inputs["composition"].items())
                         -inputs["janaf"][mol]["0.0"][4])
    # Published rounding is 1 J/mol; formation reference changes cancel in a balanced reaction.
    assert report["balanced_D0_J_mol"]["janaf"] == pytest.approx(direct_janaf, abs=2.1)
    assert report["balanced_D0_J_mol"]["nasa9"] == pytest.approx(246999.972, abs=1e-6)
    assert report["standard_pressure_Pa"] == 100000
    for row in report["source_samples"]:
        t = row["temperature_K"]
        tables = inputs["janaf"]
        def g(name):
            table = tables[name]
            knot = table[str(float(t))]
            return 1000*(table["298.15"][4]+knot[3])/(study.R*t)-knot[1]/study.R
        direct = sum(n*g(a) for a, n in inputs["composition"].items())-g(mol)
        j = row["janaf"]
        assert j["ln_K"] == pytest.approx(direct, abs=1e-13)
        assert row["nasa_ln_K"]+sum(j[k] for k in
            ("energy_ln_K_shift", "molecular_thermal_ln_K_shift", "atomic_thermal_ln_K_shift")) == pytest.approx(direct, abs=1e-13)


def test_caloric_derivatives_are_checked_independently(inputs):
    t, dt = 2253., .1
    record = inputs["nasa9"][inputs["molecule"]]
    assert (study.nasa(record, t+dt)["H_J_mol"]-study.nasa(record, t-dt)["H_J_mol"])/(2*dt) == pytest.approx(study.nasa(record, t)["Cp_J_mol_K"], abs=1e-6)
    def logk(temp):
        return study.curated(temp, inputs)[0]
    cp = study.R*(2*t*(logk(t+dt)-logk(t-dt))/(2*dt)
        +t*t*(logk(t+dt)-2*logk(t)+logk(t-dt))/dt**2)
    assert cp == pytest.approx(study.curated(t, inputs)[1], abs=8e-5)
    with pytest.raises(ValueError, match="outside curated"):
        study.curated(1000., inputs)
    with pytest.raises(ValueError, match="outside NASA"):
        study.nasa(record, 7000.)


def test_current_footprint_is_conserved_and_not_a_substitution(inputs, report):
    f = report["footprint"]
    assert (f["model"], f["cases"], f["species"], f["all_converged"]) == ("curated-oh-v1", 75, 537, True)
    assert f["maximum_abs_h_error"] < 2e-12
    assert f["maximum_relative_element_residual"] < 1e-9
    assert f["maximum_element_fraction"] == pytest.approx(9.104318031430397e-06, rel=1e-12)
    with np.load(study.GRID) as grid:
        j = grid["species"].tolist().index(inputs["species_id"])
        e = grid["elements"].tolist().index(inputs["element"])
        i = f["case_index"]
        assert np.exp(grid["ln_n"][i, j])/grid["budget"][i, e] == f["maximum_element_fraction"]
    assert report["production_change"] == "none"
    assert report["equilibrium_substitution"] == "not_run"


def test_frozen_output_and_receipt(report):
    def same(a, b):
        if isinstance(a, dict):
            assert a.keys() == b.keys()
            for key in a:
                same(a[key], b[key])
        elif isinstance(a, list):
            assert len(a) == len(b)
            for x, y in zip(a, b):
                same(x, y)
        elif isinstance(a, float):
            assert a == pytest.approx(b, rel=2e-12, abs=3e-10)
        else:
            assert a == b
    same(report, json.loads((study.DATA / "comparison.json").read_text()))
    receipt = json.loads((study.DATA / "receipt.json").read_text())
    for group in ("selected_input_sha256", "output_sha256"):
        for name, digest in receipt[group].items():
            assert study.digest(ROOT / name) == digest, name


def test_state_conventions_and_moment_calorics(inputs, report):
    with np.load(study.DATA / "states.npz") as archive:
        states = archive["states"]
    assert len(states) == 36120
    assert states[:, 0].min() == 0
    assert np.all(states[:, 1] == inputs["nuclear_spin_product"]*(2*states[:, 2]+1))
    t, dt = 2253., .1
    hplus = study.moments(states, t+dt, inputs["nuclear_spin_product"])["H_increment_J_mol"]
    hminus = study.moments(states, t-dt, inputs["nuclear_spin_product"])["H_increment_J_mol"]
    assert (hplus-hminus)/(2*dt) == pytest.approx(study.moments(states, t, inputs["nuclear_spin_product"])["Cp_J_mol_K"], abs=1e-7)
    for row in report["source_samples"]:
        x, t = row["finite_states"], row["temperature_K"]
        assert -(x["delta_H_J_mol"]-t*x["delta_S_J_mol_K"])/(study.R*t) == pytest.approx(math.log(x["Q_over_nasa"]), abs=2e-14)
        assert sum(x["electronic_Q_fractions"].values()) == pytest.approx(1, abs=5e-16)


def test_extraction_rejects_unreviewed_download(tmp_path, inputs):
    (tmp_path / (inputs["molecule"]+".states.bz2")).write_bytes(b"unreviewed data")
    with pytest.raises(ValueError, match="Unreviewed original"):
        study.extract(tmp_path, tmp_path / "out")


def test_current_distribution_is_not_the_paper_table_or_full_hot_range(report):
    assert report["literature_diagnostics"]["paper_over_current_state_Q_at_300_K"] == pytest.approx(1.0595484304, abs=1e-9)
    for row in report["source_samples"]:
        pf = row["finite_states"]["published_pf"]
        if row["temperature_K"] <= 2500:
            assert abs(pf["states_over_published_at_tabulated_T"]-1) < 8e-7
        else:
            assert pf is None
    assert report["source_samples"][-1]["finite_states"]["top_J_10_percent_Q_fraction"] > .008
