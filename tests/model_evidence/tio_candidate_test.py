"""Check TiO candidate derivatives, independent state references and gauge closure."""

from hashlib import sha256
import json
from pathlib import Path
import sys

import numpy as np
import pytest


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT/"mceg/reviews"))
import tio_candidate as candidate


def load(name):
    return json.loads((candidate.DATA/name).read_text())


def test_fit_preserves_state_moments_and_meets_withheld_temperature_targets():
    nodes = load("nodes.json")
    record = candidate.load_candidate()
    assert np.asarray(candidate.hermite_coefficients(nodes)) == pytest.approx(np.asarray(record["coefficients"]), abs=1e-10)
    for row in nodes:
        t = row["temperature_K"]
        result = candidate.molecular_properties(t)
        assert result["H_relative_J_mol"] == pytest.approx(candidate.R*(2.5*t+row["mean_energy_K"]), abs=1e-7)
        assert result["Cp_J_mol_K"] == pytest.approx(candidate.R*(2.5+row["variance_energy_K2"]/t**2), abs=1e-10)
        for key, value in result.items():
            assert value == pytest.approx(row[key], rel=1e-13, abs=1e-9)
    report = candidate.validate()
    assert report == load("validation.json")
    assert report["off_knot_points"] == 45
    assert all(row["temperature_K"] not in record["knots_K"] for row in load("off_knot_reference.json"))
    assert report["maximum_absolute_fit_errors"]["ln_K"] < 1.1e-6
    assert report["maximum_absolute_fit_errors"]["Cp_J_mol_K"] < .0025
    # Cp extrema include the real stationary points of each quartic, not just a grid.
    assert report["minimum_molecular_Cp_J_mol_K"] > 38.5


def test_single_gibbs_polynomial_is_c2_and_defines_all_caloric_properties():
    record = candidate.load_candidate()
    coefficients = np.asarray(record["coefficients"])
    widths = np.diff(record["knots_K"])
    for left, right, width, next_width in zip(coefficients[:-1], coefficients[1:], widths[:-1], widths[1:]):
        assert left.sum() == pytest.approx(right[0], rel=0, abs=2e-10)
        assert np.arange(1, 6) @ left[1:]/width == pytest.approx(right[1]/next_width, rel=0, abs=2e-12)
        assert (np.arange(2, 6)*np.arange(1, 5)) @ left[2:]/width**2 == pytest.approx(
            2*right[2]/next_width**2, rel=0, abs=1e-14)
    for t in (1511., 2222., 2991., 3873.):
        step = .05
        left = candidate.molecular_properties(t-step)
        here = candidate.molecular_properties(t)
        right = candidate.molecular_properties(t+step)
        assert (right["G_relative_J_mol"]-left["G_relative_J_mol"])/(2*step) == pytest.approx(-here["S_J_mol_K"], rel=1e-9)
        assert (right["H_relative_J_mol"]-left["H_relative_J_mol"])/(2*step) == pytest.approx(here["Cp_J_mol_K"], rel=1e-8)
        assert here["H_relative_J_mol"] == pytest.approx(here["G_relative_J_mol"]+t*here["S_J_mol_K"], abs=1e-8)


def test_formation_gauge_nasa_proxy_and_energy_only_variants():
    reference = json.loads(candidate.REFERENCE.read_text())
    for t in (1500., 1750., 2500., 3333., 4000.):
        expected = -candidate.major.formation_log_kp("TiO", t, "nasa9", candidate.D0_EV*candidate.EV_K, reference)
        assert candidate.formation_h(t, "nasa") == pytest.approx(expected, abs=1e-12)
        for thermal in ("states", "nasa"):
            h = candidate.formation_h(t, thermal)
            shifted = candidate.formation_h(t, thermal, candidate.D0_EV+.010)
            assert shifted-h == pytest.approx(-.010*candidate.EV_K/t, rel=1e-11)
    # The original fixed-mass diagnostic needs the explicit translation and current-state changes.
    old = json.loads((candidate.SOURCE_REVIEW/"comparison.json").read_text())
    for node in load("nodes.json"):
        t = node["temperature_K"]
        row = next((r for r in old["samples"] if r["temperature_K"] == t), None)
        if row is None:
            continue
        delta_ln_k = (np.log(node["Q"]/row["inferred_Q_janaf"])
                      + 1.5*np.log(candidate.MASS_U/old["mass_u"]["janaf"])
                      + (candidate.D0_EV-old["current_D0_eV"])*candidate.EV_K/t)
        assert -candidate.formation_h(t)-row["curated_ln_K"] == pytest.approx(delta_ln_k, abs=2e-12)
    for t in (1499., 4001., float("nan"), float("inf")):
        with pytest.raises(ValueError, match="outside"):
            candidate.formation_h(t)
    with pytest.raises(ValueError, match="thermal"):
        candidate.formation_h(2500, "janaf")


def test_jax_api_matches_scalar_and_thermodynamic_derivatives():
    jax = pytest.importorskip("jax")
    previous = jax.config.x64_enabled
    jax.config.update("jax_enable_x64", True)
    try:
        temperatures = np.array([1500., 1637., 2253., 3000., 3871., 4000.])
        for thermal in ("states", "nasa"):
            function = jax.jit(lambda t: candidate.formation_h_jax(t, thermal))
            assert np.asarray(function(temperatures)) == pytest.approx(
                [candidate.formation_h(t, thermal) for t in temperatures], abs=1e-12)
            assert float(function(2500)) == pytest.approx(candidate.formation_h(2500, thermal), abs=1e-12)
            for t in (1637., 3000., 3871.):
                molecular = candidate.molecular_properties(t, thermal)
                background, slope, curvature = candidate._nasa(t, candidate.load_candidate()["atomic_background_nasa9"], np)
                entropy = molecular["S_J_mol_K"]-candidate.R*slope
                cp = molecular["Cp_J_mol_K"]-candidate.R*t*curvature
                gibbs = lambda x: candidate.R*x*function(x)
                assert float(jax.grad(gibbs)(t)) == pytest.approx(-entropy, abs=1e-9)
                assert float(-t*jax.grad(jax.grad(gibbs))(t)) == pytest.approx(cp, abs=1e-9)
            assert np.isnan(np.asarray(function(np.array([1499., 4001., np.nan])))).all()
    finally:
        jax.config.update("jax_enable_x64", previous)


def test_bound_cutoff_spin_and_completeness_are_explicit():
    record = candidate.load_candidate()
    assert record["molecular_mass_u"] == 63.8664
    assert record["nuclear_spin_factor"] == 1
    assert record["cutoff"]["retained_states"] == 300670
    assert record["cutoff"]["excluded_states"] == 575
    assert record["cutoff"]["energy_cm-1"] == pytest.approx(55039.27182847103)
    cutoff = load("cutoff_sensitivity.json")
    assert max(row["excluded_Q_fraction"] for row in cutoff) < 3.84e-9
    assert max(abs(row["full_minus_bound_Cp_J_mol_K"]) for row in cutoff) < 1e-5
    assert max(abs(row["full_states_Q_over_published"]-1) for row in cutoff) < 4e-5
    scope = load("scientific_scope.json")
    assert scope["electronic_states"] == 16 and scope["five_approximate_state_count"] == 64937
    reviewed = json.loads((candidate.SOURCE_REVIEW/"exomol_nodes.json").read_text())
    fractions = scope["ciaaw_titanium"]["fractions"]
    assert sum(fractions.values()) == pytest.approx(1)
    for row in scope["rows"]:
        t = str(int(row["temperature_K"]))
        if t not in reviewed["isotopologues"]["48Ti-16O"]["partition"]:
            continue
        normalized = {a: reviewed["isotopologues"][a+"Ti-16O"]["partition"][t]
                      /reviewed["isotopologues"][a+"Ti-16O"]["nuclear_spin_weight"] for a in fractions}
        ratio = sum(fractions[a]*normalized[a] for a in fractions)/normalized["48"]
        assert row["titanium_fraction_weighted_internal_Q_over_48"] == pytest.approx(ratio)
    assert scope["rows"][-1]["omit_five_approximate_states_Q_over_full"] == pytest.approx(.9927939877059252)


def test_extraction_receipt_pins_model_and_independent_references():
    receipt = load("receipt.json")
    for group in ("selected_input_sha256", "output_sha256"):
        for path, digest in receipt[group].items():
            assert sha256((ROOT/path).read_bytes()).hexdigest() == digest, path
    for path, digest in candidate.load_candidate()["source_sha256"].items():
        assert sha256((ROOT/path).read_bytes()).hexdigest() == digest, path
