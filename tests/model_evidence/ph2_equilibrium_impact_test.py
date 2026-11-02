"""Check a PH2-only sensitivity probe without executing FastChem or MCEG."""

import json
from pathlib import Path
import sys

import jax
import numpy as np
import pytest
from scipy.special import logsumexp

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "mceg/reviews"))
from gas_source_comparison import file_hash, load_inputs, nasa_g
from ph2_equilibrium_impact import models

DATA = ROOT / "tests/data/validation/ph2_equilibrium_impact/20261102"


def test_probe_is_only_ph2_and_has_a_distinct_diagnostic_identity():
    previous = jax.config.x64_enabled
    jax.config.update("jax_enable_x64", True)
    try:
        base, probe, _ = models()
        assert base.metadata["dataset"] == "curated-c2h-v1"
        assert probe.metadata["dataset"] == "curated-c2h-v1+raw-nasa-ph2-diagnostic"
        assert "not a scientific adoption" in probe.metadata["interpretation"]
        assert base.species == probe.species and base.elements == probe.elements
        np.testing.assert_array_equal(base.formula_matrix, probe.formula_matrix)
        np.testing.assert_array_equal(base.element_vector_reference, probe.element_vector_reference)
        t = np.array([1500., 2000., 2500., 3000., 4000.])
        before, after = np.asarray(base.hvector_func(t)), np.asarray(probe.hvector_func(t))
        i = base.species.index("H2P1")
        np.testing.assert_array_equal(np.delete(before, i, axis=-1), np.delete(after, i, axis=-1))
        records = load_inputs()["nasa9_records"]
        expected = [nasa_g(records["PH2"], ti)-nasa_g(records["P"], ti)-2*nasa_g(records["H"], ti) for ti in t]
        np.testing.assert_allclose(after[:, i], expected, atol=1e-13, rtol=0)
        # This is a source perturbation, not an energy adjustment or NASA adoption.
        np.testing.assert_allclose(np.exp(after[[0, 3], i]-before[[0, 3], i]),
                                   [2.535338, 1.625158], rtol=1e-6)
        assert np.isnan(np.asarray(probe.hvector_func(np.array([1499., 4001., np.nan])))[:, i]).all()
    finally:
        jax.config.update("jax_enable_x64", previous)


def test_archived_execution_hashes_and_all_convergence_checks():
    receipt = json.loads((DATA / "receipt.json").read_text())
    assert receipt["result"] == "passed" and receipt["dtype"] == "float64"
    assert receipt["datasets"][1].endswith("raw-nasa-ph2-diagnostic")
    assert "not adoption" in receipt["scope"]
    for name, digest in receipt["selected_file_sha256"].items():
        assert file_hash(ROOT / name) == digest, name
    for name, digest in receipt["output_sha256"].items():
        assert file_hash(DATA / name) == digest, name
    diagnostics = json.loads((DATA / "diagnostics.json").read_text())
    assert len(diagnostics) == 300 and all(r["converged"] for r in diagnostics)
    summary = json.loads((DATA / "summary.json").read_text())
    assert summary["paired_cases"] == 75
    assert summary["endpoint_solves"] == summary["tighter_cold_solves"] == 150
    assert summary["tolerance_stability"]["max_abs_delta_x"] < 1e-12
    assert summary["tolerance_stability"]["max_abs_delta_ln_x"] < 1e-4
    assert summary["independent_solver_case"]["temperature_K"] == 3000.
    assert len(summary["independent_solver"]) == 2
    assert max(r["max_abs_delta_log10_x"] for r in summary["independent_solver"]) < 1e-6


def test_full_archive_preserves_amounts_budgets_and_mass_action():
    with np.load(DATA / "equilibria.npz") as saved:
        assert saved["ln_n"].shape == saved["h"].shape == (2, 75, 537)
        assert np.isfinite(saved["ln_n"]).all()
        np.testing.assert_allclose(saved["n"], np.exp(saved["ln_n"]), atol=0, rtol=1e-14)
        logx = saved["ln_n"]-logsumexp(saved["ln_n"], axis=-1, keepdims=True)
        np.testing.assert_allclose(saved["ln_x"], logx, atol=1e-13, rtol=0)
        # Keep logsumexp roundoff separate from the exponential check.
        np.testing.assert_allclose(saved["x"], np.exp(saved["ln_x"]), atol=0, rtol=1e-14)
        np.testing.assert_allclose(saved["delta_log10_x"], (logx[1]-logx[0])/np.log(10), atol=1e-13)
        index = list(saved["species"]).index("H2P1")
        np.testing.assert_allclose(np.delete(saved["h"][1]-saved["h"][0], index, axis=-1), 0, atol=1e-12)
        c, o, hydrogen = [list(saved["elements"]).index(e) for e in ("C", "O", "H")]
        budget = saved["budget"]
        np.testing.assert_allclose(budget[:, c]/budget[:, o], saved["C_over_O"])
        relative = budget/budget[:, hydrogen, None]
        np.testing.assert_allclose(np.delete(relative, c, axis=1),
                                   np.broadcast_to(np.delete(relative[0], c), (75, 27)), atol=1e-15)
        for values in (saved["ln_n"], saved["strict_ln_n"]):
            amounts, matrix = np.exp(values), saved["formula_matrix"]
            positive = budget[0] > 0
            assert np.max(abs(amounts@matrix.T-budget)[..., positive]/budget[:, positive]) < 1e-5
            charge = matrix[list(saved["elements"]).index("e-")]
            assert np.max(abs(amounts@charge)/(amounts@abs(charge))) < 1e-12
            refs = [list(saved["species"]).index("e1-" if e == "e-" else e+"1") for e in saved["elements"]]
            logp = values-logsumexp(values, axis=-1, keepdims=True)+np.log(saved["pressure_bar"])[None, :, None]
            assert np.max(abs(logp-logp[..., refs]@matrix+saved["h"])) < 1e-8
        for element in ("P", "C", "N"):
            i = list(saved["elements"]).index(element)
            expected = saved["n"]*saved["formula_matrix"][i]/budget[None, :, i, None]
            np.testing.assert_allclose(saved[element+"_fraction"], expected)
            np.testing.assert_allclose(saved[element+"_fraction"].sum(axis=-1), 1, rtol=1e-5)


def test_phosphorus_redistribution_and_current_curated_baseline():
    summary = json.loads((DATA / "summary.json").read_text())
    assert summary["highlighted_case"]["temperature_K"] == 2500.
    assert summary["highlighted_case"]["pressure_bar"] == 100.
    partitions = summary["highlighted_phosphorus_partition"]
    np.testing.assert_allclose(np.sum([r["element_fractions"]["P"] for r in partitions], axis=0), 1, rtol=1e-5)
    values = {r["species"]: r for r in summary["highlighted_species"]}
    assert values["H2P1"]["element_fractions"]["P"] == pytest.approx([.6425093541, .5036316707], rel=1e-8)
    assert values["H3P1"]["element_fractions"]["P"] == pytest.approx([.1489177919, .2066480383], rel=1e-8)
    assert values["P1"]["ratio"] > 1
    # The starting point is the already adopted C2H-corrected curated model.
    prior = ROOT / "tests/data/validation/curated_default_adoption/20261102/ions.npz"
    with np.load(DATA / "equilibria.npz") as saved, np.load(prior) as base:
        for key in ("species", "elements", "formula_matrix", "temperature_K", "pressure_bar", "C_over_O"):
            np.testing.assert_array_equal(saved[key], base[key])
        np.testing.assert_allclose(saved["budget"], base["budget"], atol=1e-15)
        np.testing.assert_allclose(saved["h"][0], base["h"], atol=1e-12)
        np.testing.assert_allclose(saved["ln_x"][0], base["ln_x"], atol=1e-4, rtol=0)
