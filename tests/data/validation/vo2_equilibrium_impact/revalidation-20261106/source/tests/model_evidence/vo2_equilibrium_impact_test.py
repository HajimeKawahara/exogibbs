"""Check frozen VO2 diagnostics without rerunning equilibrium or FastChem."""

import json
from pathlib import Path
import sys

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from scipy.special import logsumexp

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "mceg/reviews"))
from gas_source_comparison import file_hash
from vo2_equilibrium_impact import DATA, INPUT, KINDS, PRIOR, correction, models


@pytest.fixture
def prepared():
    previous = jax.config.x64_enabled
    jax.config.update("jax_enable_x64", True)
    try:
        yield models()
    finally:
        jax.config.update("jax_enable_x64", previous)


def test_only_vo2_changes_and_electronic_weight_is_not_counted_twice(prepared):
    setups, inputs = prepared
    base = setups[0]
    index = base.species.index("O2V1")
    temperatures = np.array([1500., 2000., 2500., 3000., 4000.])
    start = np.asarray(base.hvector_func(temperatures))
    shifts = []
    for kind, setup in zip(KINDS[1:], setups[1:]):
        assert setup.species == base.species and setup.elements == base.elements
        np.testing.assert_array_equal(setup.formula_matrix, base.formula_matrix)
        np.testing.assert_array_equal(setup.element_vector_reference, base.element_vector_reference)
        assert setup.metadata["dataset"].endswith(kind+"-diagnostic")
        assert "not adoption" in setup.metadata["interpretation"]
        after = np.asarray(setup.hvector_func(temperatures))
        np.testing.assert_array_equal(np.delete(after, index, axis=-1), np.delete(start, index, axis=-1))
        shifts.append(after[:, index]-start[:, index])
        np.testing.assert_allclose(shifts[-1], correction(temperatures, kind, inputs)["G_J_mol"] /
                                   (inputs["gas_constant_J_mol_K"]*temperatures), atol=2e-14)
        assert np.isnan(np.asarray(setup.hvector_func(np.array([1499., 4001., np.nan])))[:, index]).all()
    np.testing.assert_allclose(shifts[2], shifts[0]+shifts[1], atol=2e-14)
    # The three doublets replace the existing ground doublet, not multiply it again.
    levels = np.array(inputs["electronic_levels_eV"]) * inputs["electronvolt_J_mol"]
    expected = np.sum(np.exp(-levels/(inputs["gas_constant_J_mol_K"]*1500)))
    assert correction(1500., "electronic", inputs)["formation_K_ratio"] == pytest.approx(expected)
    assert 1.24 < expected < 1.25


def test_energy_and_electronic_caloric_terms_follow_one_gibbs_model(prepared):
    _, inputs = prepared
    assert inputs["adoption"] == "none"
    assert inputs["energy_shift_J_mol"] == pytest.approx(
        1000*(inputs["janaf_atomization_D0_kJ_mol"]-inputs["balducci_selected_atomization_D0_kJ_mol"]))
    for kind in KINDS[1:]:
        gibbs = lambda t: correction(t, kind, inputs)["G_J_mol"]
        for temperature in (1500., 2234., 4000.):
            t = jnp.asarray(temperature)
            values = correction(t, kind, inputs)
            derivative = jax.grad(gibbs)(t)
            second = jax.grad(jax.grad(gibbs))(t)
            assert values["S_J_mol_K"] == pytest.approx(-float(derivative), abs=1e-11)
            assert values["H_J_mol"] == pytest.approx(float(gibbs(t)-t*derivative), abs=1e-9)
            assert values["Cp_J_mol_K"] == pytest.approx(-float(t*second), abs=1e-11)
            if kind == "energy":
                assert values["H_J_mol"] == 59342
                assert values["S_J_mol_K"] == values["Cp_J_mol_K"] == 0
            else:
                assert values["Cp_J_mol_K"] > 0


def test_execution_receipt_and_all_cold_and_independent_checks():
    receipt = json.loads((DATA / "receipt.json").read_text())
    assert receipt["result"] == "passed" and receipt["dtype"] == "float64"
    for name, digest in receipt["selected_file_sha256"].items():
        assert file_hash(ROOT / name) == digest, name
    for name, digest in receipt["output_sha256"].items():
        assert file_hash(DATA / name) == digest, name
    summary = json.loads((DATA / "summary.json").read_text())
    assert summary["input_sha256"] == file_hash(INPUT)
    assert summary["diagnostics"] == list(KINDS)
    assert summary["paired_cases"] == 75
    assert summary["endpoint_solves"] == summary["tighter_cold_solves"] == 300
    rows = json.loads((DATA / "diagnostics.json").read_text())
    assert len(rows) == 600 and all(row["converged"] for row in rows)
    assert summary["tolerance_stability"]["max_abs_delta_x"] < 1e-9
    assert summary["tolerance_stability"]["max_abs_delta_ln_x"] < 1e-4
    assert len(summary["independent_solver"]) == 4
    assert max(row["max_abs_delta_log10_x"] for row in summary["independent_solver"]) < 1e-4


def test_complete_archive_preserves_budgets_mass_action_and_allocation():
    with np.load(DATA / "equilibria.npz") as saved:
        assert saved["ln_n"].shape == saved["h"].shape == (4, 75, 537)
        assert saved["diagnostics"].tolist() == list(KINDS)
        np.testing.assert_allclose(saved["n"], np.exp(saved["ln_n"]), atol=0, rtol=1e-14)
        logx = saved["ln_n"]-logsumexp(saved["ln_n"], axis=-1, keepdims=True)
        np.testing.assert_allclose(saved["ln_x"], logx, atol=1e-13, rtol=0)
        np.testing.assert_allclose(saved["x"], np.exp(saved["ln_x"]), atol=0, rtol=1e-14)
        np.testing.assert_allclose(saved["delta_log10_x"], (logx[1:]-logx[0])/np.log(10), atol=1e-13)
        species, elements = saved["species"].tolist(), saved["elements"].tolist()
        index = species.index("O2V1")
        np.testing.assert_allclose(np.delete(saved["h"][1:]-saved["h"][0], index, axis=-1), 0, atol=1e-12)
        budget, matrix = saved["budget"], saved["formula_matrix"]
        c, o, hydrogen = [elements.index(e) for e in ("C", "O", "H")]
        np.testing.assert_allclose(budget[:, c]/budget[:, o], saved["C_over_O"])
        relative = budget/budget[:, hydrogen, None]
        np.testing.assert_allclose(np.delete(relative, c, axis=1),
                                   np.broadcast_to(np.delete(relative[0], c), (75, 27)), atol=1e-15)
        for values in (saved["ln_n"], saved["strict_ln_n"]):
            assert np.isfinite(values).all()
            amounts = np.exp(values)
            positive = budget[0] > 0
            assert np.max(abs(amounts@matrix.T-budget)[..., positive]/budget[:, positive]) < 1e-5
            charge = matrix[elements.index("e-")]
            assert np.max(abs(amounts@charge)/(amounts@abs(charge))) < 1e-12
            refs = [species.index("e1-" if e == "e-" else e+"1") for e in elements]
            logp = values-logsumexp(values, axis=-1, keepdims=True)+np.log(saved["pressure_bar"])[None, :, None]
            assert np.max(abs(logp-logp[..., refs]@matrix+saved["h"])) < 1e-8
        for element in ("V", "C", "O", "N"):
            i = elements.index(element)
            expected = saved["n"]*matrix[i]/budget[None, :, i, None]
            np.testing.assert_allclose(saved[element+"_fraction"], expected)
            np.testing.assert_allclose(saved[element+"_fraction"].sum(axis=-1), 1, rtol=1e-5)


def test_baseline_is_current_curated_and_energy_weakening_reduces_vo2():
    with np.load(DATA / "equilibria.npz") as saved, np.load(PRIOR) as prior:
        for key in ("species", "elements", "formula_matrix", "temperature_K", "pressure_bar", "C_over_O"):
            np.testing.assert_array_equal(saved[key], prior[key])
        np.testing.assert_allclose(saved["budget"], prior["budget"], atol=1e-15)
        np.testing.assert_allclose(saved["h"][0], prior["h"], atol=1e-12)
        np.testing.assert_allclose(saved["ln_x"][0], prior["ln_x"], atol=1e-4, rtol=0)
        index = saved["species"].tolist().index("O2V1")
        assert np.all(saved["ln_x"][1, :, index] < saved["ln_x"][0, :, index])
        assert np.all(saved["ln_x"][2, :, index] > saved["ln_x"][0, :, index])
        assert np.all(saved["ln_x"][3, :, index] < saved["ln_x"][0, :, index])
        assert np.all(saved["ln_x"][3, :, index] > saved["ln_x"][1, :, index])
        summary = json.loads((DATA / "summary.json").read_text())
        for block in summary["highlighted_allocations"]:
            np.testing.assert_allclose(np.sum([r["element_fractions"]["V"] for r in block["vanadium"]], axis=0),
                                       1, rtol=1e-5)
