"""Verify TiO2 balanced references and a deliberately restricted RRHO diagnostic."""

from hashlib import sha256
import json
import math
from pathlib import Path
import sys

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "mceg/reviews"))
import tio2_source_review as study
from gas_source_comparison import check_equal


@pytest.fixture(scope="module")
def report():
    return study.review()


def test_pinned_sources_and_replay(report):
    check_equal(report, json.loads((study.DATA / "comparison.json").read_text()))
    receipt = json.loads((study.DATA / "receipt.json").read_text())
    for path, digest in receipt["input_and_artifact_sha256"].items():
        assert sha256((ROOT / path).read_bytes()).hexdigest() == digest, path
    assert report["official_nasa_matching_coefficients"] == 18
    assert report["official_janaf_frozen_matching_nodes"] == 4
    assert report["nasa_header_check"]["H298_J_mol"] == pytest.approx(-305430, abs=0.001)


def test_reaction_balances_both_oxygen_atoms(report):
    for row in report["source_samples"]:
        t = row["temperature_K"]
        prop = row["properties"]
        delta = prop["TiO2"]["delta_G_J_mol"]-prop["Ti"]["delta_G_J_mol"]-2*prop["O"]["delta_G_J_mol"]
        assert row["reaction_delta_G_J_mol"] == pytest.approx(delta, abs=1e-8)
        assert row["janaf_over_nasa"] == pytest.approx(math.exp(-delta/(study.R*t)), rel=2e-12)
        assert abs(row["janaf_raw_minus_hs_ln_K"]) < 4*(1.5+0.0005*t)/(study.R*t)
    for t in (1499, 4001):
        with pytest.raises(ValueError, match="curated range"):
            study.curated_ln_k(t)


def test_rrho_caloric_derivatives_and_high_temperature_limit():
    inputs = json.loads((study.DATA / "inputs.json").read_text())
    for t in (500, 1500, 3000, 4000):
        step = 0.01
        low, middle, high = [study.rrho(temp, inputs) for temp in (t-step, t, t+step)]
        cp_h = (high["H_minus_H0_J_mol"]-low["H_minus_H0_J_mol"])/(2*step)
        cp_s = t*(high["S_J_mol_K"]-low["S_J_mol_K"])/(2*step)
        assert cp_h == pytest.approx(middle["Cp_J_mol_K"], rel=2e-8)
        assert cp_s == pytest.approx(cp_h, rel=2e-8)
    # Three vibrations plus translation/rotation: a formal RRHO limit, not a gas validity range.
    assert study.rrho(1e6, inputs)["Cp_J_mol_K"] == pytest.approx(7*study.R, rel=1e-6)


def test_energy_zero_and_thermal_scope(report):
    modern = report["modern_diagnostic"]
    assert modern["JANAF_D0_kJ_mol"] == pytest.approx(1267.776)
    assert modern["JANAF_D298_kJ_mol"] == pytest.approx(1277.407)
    assert modern["Hildenbrand_minus_JANAF_D0_kJ_mol"] == pytest.approx(4.124)
    row = next(r for r in modern["source_samples"] if r["temperature_K"] == 1500)
    assert row["H298_aligned_RRHO_over_JANAF_K"] == pytest.approx(0.8884389439679563)
    assert "no anharmonicity" in modern["scope"]
    assert report["curated_change"] == "none"
    assert report["physical_accuracy"] == "unknown"


def test_archived_titanium_allocation(report):
    with np.load(study.GRID) as grid:
        col = grid["species"].tolist().index("O2Ti1")
        elem = grid["elements"].tolist().index("Ti")
        share = np.exp(grid["ln_x"][:, col])*np.exp(grid["ln_n"]).sum(axis=1)/grid["budget"][:, elem]
    footprint = report["footprint"]
    assert footprint["titanium_fraction"] == pytest.approx(float(max(share)), rel=2e-12)
    assert footprint["titanium_fraction"] == pytest.approx(0.5008719050496409)
    assert footprint["cases"] == 75 and footprint["species_count"] == 537
    assert footprint["all_cases_converged"]
    assert footprint["maximum_abs_target_h_error"] < 1e-11
    assert footprint["maximum_titanium_conservation_residual"] < 1e-9
    assert report["new_equilibrium_substitution"] == "not_run"
