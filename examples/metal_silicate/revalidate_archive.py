"""Revalidate archived finite chemistry
=======================================

Recompute the pinned archive chemistry from absolute component amounts.

The default path needs no ExoEOS or MELTS installation. Fresh MELTS chemistry
requires an explicit evaluator/runtime; balance checks alone are labelled so.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import jax
import numpy as np
from scipy.linalg import null_space

from generate_equilibrium_reference import completed_metal_ln_gamma
from reference import component_matrices, load_reference, metal_standard_shift, source_standard_potentials
import sulfide
import sulfur_source


DATA_FILES = ("hydrogen.json", "sulfur_carbon_source.json", "sulfide.json",
              "melts_present.json", "melts_absent.json")


def _balance(formula: np.ndarray, amounts: Any, budgets: Any) -> float:
    n, b = np.asarray(amounts), np.asarray(budgets)
    assert n.shape == (formula.shape[1],) and b.shape == (formula.shape[0],)
    assert np.all(np.isfinite(n)) and np.all(n >= 0)
    assert np.all(np.isfinite(b)) and np.all(b >= 0) and np.any(b > 0)
    total = formula @ n
    assert np.all(total[b == 0] == 0)
    error = float(np.max(np.abs(total[b > 0] / b[b > 0] - 1)))
    assert error < 1e-9, f"Element balance failed: {error}"
    return error


def _chemical_result(element_error: float, residual: Any, saved_accepted: bool) -> dict:
    residual = np.asarray(residual)
    assert residual.size and np.all(np.isfinite(residual))
    error = float(np.max(np.abs(residual)))
    accepted = element_error < 1e-9 and error < 1e-8
    assert accepted == saved_accepted, f"Chemical acceptance differs: residual={error}"
    return {"chemistry": "recomputed", "accepted": accepted,
            "max_relative_element_residual": element_error,
            "max_abs_reaction_residual_RT": error}


def _hydrogen(saved: dict) -> dict:
    assert saved["model_id"] == "source_host_ma_fe_si_o_h_hirschmann2012_pressure_control_v1"
    source = load_reference()
    species, formula, _ = component_matrices(source)
    species = (*species, "He_gas")
    formula = np.pad(formula, ((0, 1), (0, 1)))
    formula[-1, -1] = 1
    ledger, state = saved["ledger"], saved["state"]
    assert ledger["components"] == list(species)
    assert ledger["elements"] == source["elements"] + ["He"]
    np.testing.assert_array_equal(ledger["formula_matrix"], formula)
    assert ledger["phases"] == {
        phase: names + (["He_gas"] if phase == "gas" else [])
        for phase, names in source["phases"].items()
    }
    # These recorded conventions must match the independent equations below.
    expected_ledger = {
        "R_J_mol_K": source["source"]["gas_constant_J_mol_K"],
        "standard_pressure_bar": 1.0,
        "metal_model_id": "ma2001_fe_si_o_h_no_h_interaction_v1",
        "amount_unit": "mol of listed component; metal H atomic, dissolved/gas H2 molecular",
        "mixing": "ideal source silicate; four-component native alloy; ideal gas",
        "standards": "source Shomate/exchange fits including Okuchi H; native dry shift; pressure-dependent H2 standard; inert He elemental gauge zero",
    }
    for key, expected in expected_ledger.items():
        assert ledger[key] == expected, f"Hydrogen ledger differs: {key}"
    expected_h2 = {
        "model_id": "hirschmann2012_seo2024_mole_fraction_sensitivity",
        "equation": "x_H2 = (f_H2 / bar) exp(-11.403 - 0.76 P_melt/GPa)",
        "concentration_basis": "H2 molecular mole fraction in the full liquid component basis",
        "fugacity_basis": "H2 fugacity in bar, not total fluid pressure",
        "pressure_basis": "total melt pressure in GPa; pressure correction occurs once in mu0_H2",
    }
    for key, expected in expected_h2.items():
        assert saved["h2_calibration"][key] == expected, f"Hydrogen calibration differs: {key}"
    n, b = np.asarray(state["component_amounts_mol"]), np.asarray(state["element_amounts_mol"])
    element_error = _balance(formula, n, b)
    temperature, pressure = state["T_K"], state["P_bar"]
    assert np.isfinite(pressure) and pressure > 0
    case = next(case for case in source["cases"] if case["T_K"] == temperature)
    mu = np.append(source_standard_potentials(source, case)
                   / (source["source"]["gas_constant_J_mol_K"] * temperature), 0.)
    mu[species.index("H2_silicate")] = mu[species.index("H2_gas")] + 11.403 + 0.76e-4 * pressure
    metal = [species.index(name + "_metal") for name in ("Fe", "Si", "O", "H")]
    mu[metal[:3]] += metal_standard_shift(temperature)
    supported = np.all(formula[b == 0] == 0, axis=0)
    assert np.all(n[supported] > 0)
    for phase, names in source["phases"].items():
        names = names + (["He_gas"] if phase == "gas" else [])
        indices = [species.index(name) for name in names if supported[species.index(name)]]
        if not indices:
            continue
        mu[indices] += np.log(n[indices] / n[indices].sum())
        if phase == "gas":
            mu[indices] += np.log(pressure)
    # Native Fe-Si-O-H extension: dry excess derivatives and ideal H mixing.
    dry = n[metal[:3]]
    mu[metal[:3]] += completed_metal_ln_gamma(temperature, dry / dry.sum())
    reactions = null_space(formula[b > 0][:, supported]).T
    return _chemical_result(element_error, reactions @ mu[supported], saved["accepted"])


def _source_cases(saved: dict) -> dict:
    networks = {network["model_id"]: network
                for network in sulfur_source.load_reference()["networks"].values()}
    assert len(saved["cases"]) == 4
    output = {}
    seen = set()
    for index, state in enumerate(saved["cases"]):
        network = networks[state["model_id"]]
        species, formula, _ = sulfur_source.component_matrices(network)
        assert state["species"] == list(species) and state["elements"] == network["elements"]
        case = next(case for case in network["cases"] if case["T_K"] == state["T_K"])
        identity = (state["model_id"], state["T_K"])
        assert identity not in seen
        seen.add(identity)
        n = np.asarray(state["component_amounts_mol"])
        element_error = _balance(formula, n, state["element_amounts_mol"])
        assert np.all(n > 0) and np.isfinite(state["P_bar"]) and state["P_bar"] > 0
        ln_x = np.empty_like(n)
        for names in network["phases"].values():
            indices = [species.index(name) for name in names]
            ln_x[indices] = np.log(n[indices] / n[indices].sum())
        residual = sulfur_source.make_reaction_residual(network, case)(ln_x, state["P_bar"])
        output[f"source_{index}"] = _chemical_result(element_error, residual, state["accepted"])
    return output


def _sulfide_cases(saved: dict) -> dict:
    assert saved["model_id"] == "finite_sulfur_empirical_scss_analytic_control_v1"
    assert saved["elements"] == list(sulfide.ELEMENTS) and saved["species"] == list(sulfide.SPECIES)
    np.testing.assert_array_equal(saved["formula_matrix_element_rows"], sulfide.FORMULA)
    budget, known, partition, saturation = sulfide.analytic_control()
    np.testing.assert_array_equal(saved["element_amounts_mol"], budget)
    assert saved["T_K"] == 1873. and saved["P_bar"] == 1.
    assert saved["scss"] == saturation(1873., 1., known).__dict__
    assert [state["branch"] for state in saved["states"]] == ["absent", "present"] * 2
    output = {}
    for index, state in enumerate(saved["states"]):
        element_error = _balance(sulfide.FORMULA, state["amounts_mol"], budget)
        fresh = sulfide.evaluate(np.asarray(state["amounts_mol"]), budget, 1873., 1.,
                                 partition, saturation, branch=state["branch"])
        assert fresh.accepted == state["accepted"], f"SCSS acceptance differs for state {index}"
        output[f"sulfide_{index}"] = {
            "chemistry": "recomputed", "accepted": fresh.accepted, "branch": fresh.branch,
            "max_relative_element_residual": element_error,
            "max_abs_reaction_residual_RT": float(np.max(np.abs(fresh.reaction_residual))),
            "saturation_log_ratio": fresh.saturation_log_ratio,
        }
    return output


def revalidate_archive(
    archive: Path, *, evaluator: Any = None, runtime: Any = None, worker_python: Any = None,
) -> dict:
    """Check file identities and recompute acceptance without rerunning roots.

    Saved residual and convergence fields never establish chemical acceptance.
    MELTS is explicitly ``not_run`` unless all provider inputs are supplied.
    """
    if not jax.config.x64_enabled:
        raise RuntimeError("Archive tolerances require JAX_ENABLE_X64=1.")
    if (evaluator is None) != (runtime is None) or (evaluator is None) != (worker_python is None):
        raise ValueError("Supply evaluator, runtime and worker_python together.")
    archive = Path(archive)
    receipt = json.loads((archive / "receipt.json").read_text())
    assert set(DATA_FILES) <= set(receipt["file_sha256"])
    for name, expected in receipt["file_sha256"].items():
        assert hashlib.sha256((archive / name).read_bytes()).hexdigest() == expected, name
    data = {Path(name).stem: json.loads((archive / name).read_text()) for name in DATA_FILES}
    cases = {"hydrogen": _hydrogen(data["hydrogen"])}
    cases.update(_source_cases(data["sulfur_carbon_source"]))
    cases.update(_sulfide_cases(data["sulfide"]))
    melts = {name: data[name] for name in ("melts_present", "melts_absent")}
    for name, saved in melts.items():
        assert saved["model_id"] == "melts_ma_fe_si_o_h_finite_conditional_v1"
        matrix = np.asarray(saved["formula_matrix"])
        error = _balance(matrix, saved["result"]["component_amounts_mol"], saved["element_amounts_mol"])
        cases[name] = {"chemistry": "not_run", "max_relative_element_residual": error,
                       "reason": "Fresh chemistry requires the external MELTS evaluator and runtime."}
    report = {"method": "Reevaluate saved absolute amounts; no equilibrium solve.", "cases": cases,
              "input_sha256": {name: receipt["file_sha256"][name] for name in DATA_FILES}}
    if evaluator is not None:
        from revalidate_melts_archive import revalidate_melts_states
        fresh = revalidate_melts_states(melts, evaluator=evaluator, runtime=runtime, worker_python=worker_python)
        cases.update(fresh.pop("cases"))
        report.update(fresh)
    return report


def main(default_archive: Path | None = None) -> None:
    """Run default controls or explicitly request fresh MELTS potentials."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive", type=Path, default=default_archive, required=default_archive is None)
    parser.add_argument("--exoeos-checkout", type=Path)
    parser.add_argument("--runtime", type=Path)
    parser.add_argument("--worker-python", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    evaluator = None
    if args.exoeos_checkout is not None:
        import exoeos
        from exoeos import MaFeSiOHLiquid
        from melts_coupled import load_melts_evaluator
        assert Path(exoeos.__file__).resolve() == (args.exoeos_checkout / "src/exoeos/__init__.py").resolve()
        assert MaFeSiOHLiquid().components == ("Fe", "Si", "O", "H")
        evaluator = load_melts_evaluator(args.exoeos_checkout)
    report = revalidate_archive(args.archive, evaluator=evaluator, runtime=args.runtime, worker_python=args.worker_python)
    rendered = json.dumps(report, indent=2, allow_nan=False) + "\n"
    if args.output:
        args.output.write_text(rendered)
    print(rendered, end="")


if __name__ == "__main__":
    main()
