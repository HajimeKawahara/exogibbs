"""Offline regression checks for the fixed-network thermochemistry study."""

import hashlib
import importlib.util
import json
from pathlib import Path
import sys
from unittest.mock import patch

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from scipy.optimize import brentq

from exogibbs.api.gas import EquilibriumOptions, solve
from exogibbs.presets.fastchem import _parse_fastchem_coeffs, logk
from exogibbs.thermo.models import ChemicalSetup


ROOT = Path(__file__).resolve().parents[3]
RESULTS = ROOT / "results/gas_thermochemistry_impact"


def _load(name):
    spec = importlib.util.spec_from_file_location(name, ROOT / "examples/comparisons" / (name + ".py"))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


audit = _load("audit_atomic_ions")
with patch.dict(sys.modules, {"audit_atomic_ions": audit}):
    atomic = _load("select_atomic_ions")
    molecular = _load("select_major_molecules")
with patch.dict(sys.modules, {"audit_atomic_ions": audit, "select_atomic_ions": atomic,
                              "select_major_molecules": molecular}):
    study = _load("compare_gas_thermochemistry")


@pytest.fixture(scope="module")
def inputs():
    previous = jax.config.jax_enable_x64
    jax.config.update("jax_enable_x64", True)
    try:
        yield study.build_study()
    finally:
        jax.config.update("jax_enable_x64", previous)


def test_fixed_catalog_budget_and_source_scope(inputs):
    setup, budget, tables, checks = inputs
    original = study.chemsetup(path="FastChem4/logK/logK.dat", silent=True)
    keep = [i for i, s in enumerate(original.species) if s not in study.EXCLUDED]
    assert len(setup.species) == 537
    assert setup.species == tuple(original.species[i] for i in keep)
    np.testing.assert_array_equal(setup.formula_matrix, original.formula_matrix[:, keep])
    np.testing.assert_array_equal(tables[0], original.hvector_func(study.TEMPERATURES_K)[:, keep])
    assert tables.shape == (5, 4, 537)
    assert len(checks) == 328  # 47 bound ions plus 35 neutral molecules, at four nodes.
    assert budget.sum() == pytest.approx(1.)
    assert budget[setup.elements.index("e-")] == 0
    assert budget[setup.elements.index("C")] / budget[setup.elements.index("O")] == pytest.approx(10**(8.46-8.69))
    assert budget[setup.elements.index("Ge")] / budget[setup.elements.index("H")] == pytest.approx(10**(3.62-12))
    changed = [set(np.flatnonzero(np.any(np.abs(h-tables[0]) > 1.e-12, axis=0))) for h in tables]
    assert len(changed[1]) == 47
    assert len(changed[2]) == 23
    assert not changed[1] & changed[2]
    assert changed[3] == changed[1] | changed[2]
    # The fifth model changes ionic constants only, with the same molecular choices.
    delta = set(np.flatnonzero(np.any(np.abs(tables[4]-tables[3]) > 1.e-12, axis=0)))
    assert delta <= changed[1]


def test_exact_nodes_and_zero_reference_gauge(inputs):
    setup, _, tables, _ = inputs
    references = [setup.species.index("e1-" if el == "e-" else el + "1") for el in setup.elements]
    np.testing.assert_array_equal(tables[:, :, references], 0.)
    model = study.node_setup(setup, tables[3])
    np.testing.assert_array_equal(model.hvector_func(study.TEMPERATURES_K), tables[3])
    for t in (1000., 1600., 4800., 5000., np.nan, np.inf, np.nextafter(1500., 1600.)):
        assert np.isnan(model.hvector_func(t)).all()
    with pytest.raises(ValueError):
        study.node_setup(setup, tables[3, :, :-1])


def test_fastchem_export_preserves_ions_isomers_and_pressure_convention(inputs):
    setup, _, tables, _ = inputs
    for mi in (0, 3):
        text = study.fastchem_node_text(setup, tables[mi, 0], 1500.)
        coefficients, components = _parse_fastchem_coeffs(text)
        assert len(coefficients) == 509
        assert not set(study.EXCLUDED) & set(coefficients)
        assert components["Ni1+"] == {"Ni": 1, "e-": -1}
        assert components["H1-"] == {"H": 1, "e-": 1}
        assert components["C1H1N1_hcn"] == components["C1H1N1_hnc"]
        assert coefficients["C1H1N1_hcn"] != coefficients["C1H1N1_hnc"]
        for name, coeff in coefficients.items():
            assert float(logk(1500., jnp.array(coeff))) == pytest.approx(
                -tables[mi, 0, setup.species.index(name)], abs=1.e-12)
    with pytest.raises(ValueError):
        study.fastchem_node_text(setup, tables[0, 0], 1600.)


@pytest.mark.parametrize("pressure", [0.01, 10.])
def test_both_charge_signs_against_independent_scalar_mass_action(inputs, pressure):
    # Synthetic constants avoid an ill-conditioned near-zero electron budget.
    # The reference solves charge neutrality in partial pressure, independent of Gibbs minimization.
    kp_plus, kp_minus = 1.e-4, 0.2
    matrix = jnp.array([[1., 0., 1., 1.], [0., 1., -1., 1.]])
    h = jnp.array([0., 0., -np.log(kp_plus), -np.log(kp_minus)])
    setup = ChemicalSetup(matrix, lambda t: h, ("H", "e-"), ("H1", "e1-", "H1+", "H1-"))

    def partial_pressures(electron):
        atom = (pressure-electron) / (1+kp_plus/electron+kp_minus*electron)
        return np.array([atom, electron, kp_plus*atom/electron, kp_minus*atom*electron])

    electron = brentq(lambda e: partial_pressures(e)[2]-partial_pressures(e)[3]-e,
                     1.e-30, pressure, xtol=1.e-15)
    result, diagnostics = solve(setup, T=1500., P=pressure, b=jnp.array([1., 0.]),
                                options=EquilibriumOptions(epsilon_crit=study.SOLVER_TOLERANCE),
                                return_diagnostics=True)
    assert bool(diagnostics["converged"])
    np.testing.assert_allclose(result.x, partial_pressures(electron)/pressure, rtol=1.e-8)


def test_relative_charge_gate_rejects_false_positive_convergence():
    row = {"converged": True, "solver_residual": 1.e-14, "element_relative_residual": 1.e-8,
           "charge_relative_residual": 0.7, "mass_action_max_abs_ln": 1.e-12}
    with pytest.raises(RuntimeError, match="verification failed"):
        study.validate_diagnostics([row])
    row["charge_relative_residual"] = float("nan")
    with pytest.raises(RuntimeError):
        study.validate_diagnostics([row])


def test_archived_all_species_equilibria_close_constraints_and_mass_action(inputs):
    setup, budget, tables, _ = inputs
    data = np.load(RESULTS / "equilibria.npz", allow_pickle=False)
    np.testing.assert_array_equal(data["species"], setup.species)
    np.testing.assert_array_equal(data["h"], tables)
    np.testing.assert_array_equal(data["budget"], budget)
    assert data["ln_n"].shape == (5, 20, 537)
    for values, table in zip(data["ln_n"], tables):
        metrics = study.equilibrium_metrics(setup, budget, values, np.repeat(table, 5, axis=0),
                                            np.tile(study.PRESSURES_BAR, 4))
        assert max(metrics["element_relative_residual"]) <= study.BUDGET_TOLERANCE
        assert max(metrics["charge_relative_residual"]) <= study.BUDGET_TOLERANCE
        assert max(metrics["mass_action_max_abs_ln"]) <= study.MASS_ACTION_TOLERANCE


def test_archived_fastchem_run_verifies_independent_solutions_offline(inputs):
    setup, budget, tables, _ = inputs
    data = np.load(RESULTS / "equilibria.npz", allow_pickle=False)
    receipt = json.loads((RESULTS / "receipt.json").read_text())
    build = json.loads((RESULTS / "fastchem_build.json").read_text())
    assert receipt["fastchem"]["status"] == "passed"
    assert receipt["fastchem"]["cases"] == 100
    assert build["source_revision"] == "ae67cbd559bc64a3233a1cee6030b8e6b50520de"
    assert build["source_clean"] is True
    assert build["executable_sha256"] == receipt["fastchem"]["executable_sha256"]
    assert all(r["byte_identical_to_packaged"] for r in build["shared_inputs"].values())
    assert data["fastchem_ln_n"].shape == (5, 20, 537)
    for mi, table in enumerate(tables):
        for ti, temperature in enumerate(study.TEMPERATURES_K):
            payload = study.fastchem_node_text(setup, table[ti], temperature)
            key = f"{study.MODELS[mi]}/{temperature:g}K"
            assert hashlib.sha256(payload.encode()).hexdigest() == receipt["fastchem"]["single_temperature_input_sha256"][key]
        fc_ln_n = data["fastchem_ln_n"][mi]
        metrics = study.equilibrium_metrics(setup, budget, fc_ln_n, np.repeat(table, 5, axis=0),
                                            np.tile(study.PRESSURES_BAR, 4))
        assert max(metrics["element_relative_residual"]) <= study.BUDGET_TOLERANCE
        assert max(metrics["charge_relative_residual"]) <= study.BUDGET_TOLERANCE
        assert max(metrics["mass_action_max_abs_ln"]) <= study.MASS_ACTION_TOLERANCE
        ex_ln_n = data["ln_n"][mi]
        ex_logx = ex_ln_n - np.log(np.exp(ex_ln_n).sum(axis=-1))[:, None]
        fc_logx = fc_ln_n - np.log(np.exp(fc_ln_n).sum(axis=-1))[:, None]
        delta = (ex_logx - fc_logx) / np.log(10.)
        # This archived run also agrees for all trace species, beyond the reporting gate.
        assert np.max(np.abs(delta)) <= study.PARITY_TOLERANCE_DEX


def test_factorial_attribution_and_trace_reporting(inputs):
    setup, _, _, _ = inputs
    ln_n = np.load(RESULTS / "equilibria.npz")["ln_n"]
    rows, summaries = study.abundance_tables(setup, ln_n)
    for row in rows:
        assert (row["atoms_only_dex"] + row["molecules_only_dex"] + row["interaction_dex"]
                == pytest.approx(np.log10(row["adopted_over_baseline"]), abs=1.e-13))
    electron = next(r for r in rows if r["species"] == "e-" and r["temperature_K"] == 4000 and r["pressure_bar"] == 1)
    assert electron["adopted_over_baseline"] == pytest.approx(0.893302, abs=1.e-5)
    assert abs(electron["atoms_only_dex"]) > 100 * abs(electron["molecules_only_dex"])
    # SO3 remains in all-species results despite falling below the reporting threshold.
    trace = next(r for r in summaries if r["species"] == "SO3")
    assert trace["significant_points"] == 0 and trace["adopted_over_baseline"] is None
    assert trace["ratio_min_all_points"] > 0


def test_reference_evidence_does_not_treat_inherited_inputs_as_independent(inputs):
    _, _, _, checks = inputs
    assert sum(r["reference_role"] == "reference_by_construction" for r in checks) == 16
    assert sum(r["reference_role"] == "source_comparison" for r in checks) == 172
    assert sum(r["reference_role"] == "retained_BC_thermal_fit_check" for r in checks) == 40
    assert all(r["log10_Kp_reference"] is None for r in checks if r["reference_role"] == "no_BC_reference")


def test_run_receipt_pins_inputs_and_outputs():
    receipt = json.loads((RESULTS / "receipt.json").read_text())
    assert receipt["network"]["species_count"] == 537
    assert receipt["equilibrium_cases"] == 100
    for path, digest in receipt["selected_file_sha256"].items():
        assert hashlib.sha256((ROOT / path).read_bytes()).hexdigest() == digest, path
    for name, digest in receipt["output_sha256"].items():
        assert hashlib.sha256((RESULTS / name).read_bytes()).hexdigest() == digest, name
