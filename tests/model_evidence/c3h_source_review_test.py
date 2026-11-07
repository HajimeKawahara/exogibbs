"""Guard C3H reaction conventions, source signs and archived carbon accounting."""

from hashlib import sha256
import json
from pathlib import Path
import runpy
import sys

import numpy as np
import pytest


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "mceg/reviews"))
import gas_source_comparison as baseline


@pytest.fixture(scope="module")
def study():
    return runpy.run_path(str(ROOT / "mceg/reviews/c3h_source_review.py"))


def test_source_replay_and_caloric_disagreement(study):
    actual = study["review"]()
    saved = json.loads((study["DATA"] / "comparison.json").read_text())
    baseline.check_equal(actual, saved)
    assert actual["max_abs_source_residual_ln_K"] < 0.025
    for sample in actual["samples"]:
        assert sample["current_minus_burcat_reaction_H_J_mol"] < -170000
        assert sample["burcat_Cp_J_mol_K"] > sample["current_with_nasa_atoms_Cp_J_mol_K"]
    assert actual["equilibrium_substitution"] == "not_run"
    assert actual["production_change"] == "none"


def test_tsuji_pressure_reaction_sign_and_printed_numbers(study):
    inputs = json.loads((study["DATA"] / "inputs.json").read_text())
    coefficients = inputs["tsuji1973"]["coefficients"]
    # Independent evaluation of the visually inspected Table4 row, at theta=1.68.
    kd_cgs = 40.791 - 21.762 * 1.68 + .93377 * 1.68**2 - .13863 * 1.68**3 + .0074549 * 1.68**4
    formation = np.exp(study["tsuji_ln_k"](3000, coefficients))
    assert formation == pytest.approx(1e18 / 10**kd_cgs, rel=2e-14)
    assert study["tsuji_ln_k"](3000, coefficients) == pytest.approx(27.013087169610515)
    with pytest.raises(ValueError, match="outside"):
        study["tsuji_ln_k"](298.15, coefficients)


def test_fit_reaction_calorics_follow_vant_hoff(study):
    coefficients = study["fastchem_coefficients"]()
    r = study["review"]()["gas_constant_J_mol_K"]
    for t in (1600, 2400, 3900):
        step = .02
        _, enthalpy, cp = study["fit_properties"](t, coefficients, r)
        plus = study["fit_properties"](t + step, coefficients, r)
        minus = study["fit_properties"](t - step, coefficients, r)
        assert r * t**2 * (plus[0] - minus[0]) / (2 * step) == pytest.approx(enthalpy, rel=3e-9)
        assert (plus[1] - minus[1]) / (2 * step) == pytest.approx(cp, rel=3e-9)
    with pytest.raises(ValueError, match="outside"):
        study["fit_properties"](298.15, coefficients, r)


def test_burcat_printed_sign_and_cp_derivative(study):
    record = study["burcat_record"]()
    # Guard the visually inspected minus sign in the author-hosted PDF.
    assert record["intervals"][-1]["coefficients"][6] == -5.784616800e-17
    r = study["review"]()["gas_constant_J_mol_K"]
    for t in (298.15, 1500, 3000, 4000):
        h, s, cp = study["nasa_properties"](record, t, r)
        assert (h - t * s) / (r * t) == pytest.approx(baseline.nasa_g(record, t), abs=2e-12)
        plus = study["nasa_properties"](record, t + .01, r)[0]
        minus = study["nasa_properties"](record, t - .01, r)[0]
        assert (plus - minus) / .02 == pytest.approx(cp, rel=2e-8)


def test_balanced_atomization_and_distinct_isomers(study):
    result = study["review"]()
    assert result["atct_balanced_reactions"]["linear"]["reaction_H0_kJ_mol"] == pytest.approx(-1631.427)
    assert result["atct_balanced_reactions"]["linear"]["reaction_H298_kJ_mol"] == pytest.approx(-1645.496)
    assert result["linear_minus_cyclic_H0_kJ_mol"] == pytest.approx(7.51)
    assert result["atct_balanced_reactions"]["cyclic"]["reaction_H0_kJ_mol"] == pytest.approx(-1638.937)


def test_footprint_uses_solved_carbon_budget_and_inputs_are_pinned(study):
    result = study["footprint"]()
    assert result["cases"] == 75 and result["species_count"] == 537
    point = result["maximum_carbon_share"]
    assert point["case_index"] == 55
    assert point["carbon_fraction"] == pytest.approx(.11643053741695293)
    with np.load(study["GRID"]) as grid:
        c = grid["elements"].tolist().index("C")
        species_index = grid["species"].tolist().index("C3H1")
        n = np.exp(grid["ln_n"][55])
        assert grid["formula_matrix"][c] @ n == pytest.approx(grid["budget"][55, c], abs=2e-13)
        assert point["mole_fraction"] == pytest.approx(n[species_index] / n.sum())
        # Establish that the reviewed inherited coefficients actually supplied
        # this archived curated model's C3H potential, without importing JAX.
        for temperature, h in zip(grid["temperature_K"], grid["h"][:, species_index]):
            ln_k = study["fit_properties"](temperature, study["fastchem_coefficients"](), 8.31446261815324)[0]
            assert -h == pytest.approx(ln_k, abs=3e-12)
    receipt = json.loads((study["DATA"] / "receipt.json").read_text())
    for group in ("selected_input_sha256", "output_sha256"):
        for path, digest in receipt[group].items():
            assert sha256((ROOT / path).read_bytes()).hexdigest() == digest, path
