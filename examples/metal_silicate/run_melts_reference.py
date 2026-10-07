"""Conditional finite MELTS--metal--gas reference
===================================================

Run with --help. The liquid-only branch is a mechanism comparison, pending
cross-phase standard calibration and final-composition competing-phase checks.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import sys
from typing import Any

import jax
import numpy as np

from full_potential import PhaseState, ideal_phase, solve_full_potentials
from hydrogen import _checkout_provenance, dissolved_h2_standard_rt, hirschmann2012_ln_solubility
from local import build_problem
from melts_coupled import COMMON_R, load_melts_evaluator, make_melts_h2_phase, provider_ledger
from reference import load_reference
from source import make_source_standard_potentials_rt


MODEL_ID = "melts_ma_fe_si_o_h_finite_conditional_v1"


def reduced_gas_standards_rt(temperature_k: float) -> dict[str, float]:
    """Evaluate pinned NASA7 H/He/OH/SiH4 standards on the common-R scale."""
    record = json.loads(Path(__file__).with_name("reduced_gas_reference.json").read_text())
    if not 200 < temperature_k < 6000:
        raise ValueError("The pinned reduced-gas formulation requires 200 < T < 6000 K.")
    t = temperature_k
    result = {}
    for name, item in record["species"].items():
        a, b, c, d, e, f, g = item["low" if t <= 1000 else "high"]
        h_rt = a + b*t/2 + c*t*t/3 + d*t**3/4 + e*t**4/5 + f/t
        s_r = a*np.log(t) + b*t + c*t*t/2 + d*t**3/3 + e*t**4/4 + g
        result[name] = float((h_rt - s_r) * record["source"]["gas_constant_J_mol_K"] / COMMON_R)
    return result


def run_reference(
    exoeos_checkout: Path, runtime: Path, python_executable: str,
    *, temperature_k: float = 2350., pressure_bar: float = 1., max_nfev: int = 100,
    metal_present: bool = True,
) -> dict[str, Any]:
    """Solve a finite reference, retaining every background host element.

    The dry host uses the first MELTS liquid fixture's composition, evaluated
    anew at supplied T/P, with finite molecular H2 and inert He added. Source
    gas/alloy standards use the 2350 K Shomate branches; changing temperature
    is an explicit formal extrapolation, not automatic branch selection.
    """
    import exoeos
    from exoeos import MaFeSiOHLiquid, solution_state

    root = Path(exoeos_checkout).resolve()
    if Path(exoeos.__file__).resolve().parent != root / "src" / "exoeos":
        raise RuntimeError("Set PYTHONPATH to the selected ExoEOS checkout before importing the alloy.")
    evaluator = load_melts_evaluator(root)
    fixture = evaluator.REFERENCE
    liquid = fixture["states"][0]["liquid"]
    host_amounts = np.asarray(liquid["component_moles"], dtype=float)
    indices = np.flatnonzero(host_amounts > 0)
    host_names = [evaluator.COMPONENTS[i] for i in indices]
    source = load_reference()
    source_species = [name for values in source["phases"].values() for name in values]
    source_mu = np.asarray(make_source_standard_potentials_rt(source, source["cases"][0])(temperature_k, pressure_bar))
    # The source's R differs only in its final recorded digits; convert full
    # energies explicitly rather than silently identifying the constants.
    source_mu = source_mu * source["source"]["gas_constant_J_mol_K"] / COMMON_R
    standards = dict(zip(source_species, source_mu))
    extra = reduced_gas_standards_rt(temperature_k)
    gas_names = source["phases"]["gas"] + [name + "_gas" for name in extra]
    metal_names = source["phases"]["metal"]
    formulas = dict(source["component_formulas"])
    extra_record = json.loads(Path(__file__).with_name("reduced_gas_reference.json").read_text())
    for name, value in extra.items():
        standards[name + "_gas"] = value
        formulas[name + "_gas"] = extra_record["species"][name]["formula"]
    melt_names = [name + "_melts" for name in host_names] + ["H2_dissolved"]
    for i, name in zip(indices, melt_names[:-1]):
        formulas[name] = {e: float(n) for e, n in zip(evaluator.ELEMENTS, evaluator.FORMULA_MATRIX[i]) if n}
    formulas["H2_dissolved"] = {"H": 2}
    elements = sorted(set().union(*(formulas[name] for name in melt_names + metal_names + gas_names)))
    phases = {"silicate": melt_names, "metal": metal_names, "gas": gas_names}
    names = [name for values in phases.values() for name in values]
    formula = np.array([[formulas[name].get(e, 0) for name in names] for e in elements])
    initial = np.zeros(len(names))
    initial[:len(indices)] = host_amounts[indices]
    initial[len(indices)] = 1e-5
    initial[len(melt_names):len(melt_names)+4] = [.1, .001, .001, .001]
    for name, value in {"H2_gas": .02, "He_gas": .002}.items():
        initial[names.index(name)] = value
    budgets = formula @ initial
    record = {"elements": elements, "phases": phases, "component_formulas": formulas, "reactions": []}
    selected = ("silicate", "metal", "gas") if metal_present else ("silicate", "gas")
    problem = build_problem(record, budgets, lambda t, p: np.zeros(len(names)), phases=selected)
    supported_gas = [n for n in problem.species if n in gas_names]
    gas_mu = np.array([standards[name] for name in supported_gas])
    gas_phase = ideal_phase(lambda t, p: gas_mu, gas=True)
    h2_standard = float(dissolved_h2_standard_rt(standards["H2_gas"], hirschmann2012_ln_solubility(pressure_bar)))
    melt_phase = make_melts_h2_phase(evaluator, host_names, lambda t, p: h2_standard,
                                     runtime=runtime, python_executable=python_executable)
    alloy = MaFeSiOHLiquid()
    metal_mu = np.array([standards[name] for name in metal_names]) + np.asarray(alloy.standard_state_shift_RT(temperature_k))
    alloy_activity = jax.jit(lambda x: solution_state(alloy, temperature_k, pressure_bar * 1e5, x).lngamma)

    def metal_phase(t, p, n):
        x = n / n.sum()
        alloy.validate_state(t, p * 1e5, x)
        mu = metal_mu + np.log(x) + np.asarray(alloy_activity(x))
        return PhaseState(mu, float(n @ mu))

    callbacks = {"silicate": melt_phase, "gas": gas_phase}
    if metal_present:
        callbacks["metal"] = metal_phase
    # Initial traces seed permitted numerical unknowns only; they are not
    # elemental inventory. Absent phases and zero-budget species remain zero.
    active_initial = np.zeros(len(names))
    active_initial[problem.species_indices] = np.maximum(initial[problem.species_indices], budgets.sum() * 1e-10)
    result = solve_full_potentials(problem, temperature_k, pressure_bar, budgets, callbacks,
                                  initial_component_amounts_mol=active_initial, max_nfev=max_nfev)
    checkouts = {}
    for label, path in (("exogibbs", Path(__file__).resolve().parents[2]), ("exoeos", root)):
        checkouts[label] = _checkout_provenance(path)
    gas_amounts = np.array([result.component_amounts_mol[names.index(name)] for name in supported_gas])
    gas_fractions = dict(zip(supported_gas, (gas_amounts / gas_amounts.sum()).tolist()))
    return {
        "model_id": MODEL_ID, "evidence_level": "conditional_mechanism",
        "temperature_k": temperature_k, "pressure_bar": pressure_bar,
        "elements": elements, "component_order": names, "formula_matrix": formula.tolist(),
        "element_amounts_mol": budgets.tolist(), "active_phases": list(problem.phases),
        "result": {key: value.tolist() if isinstance(value, np.ndarray) else value for key, value in asdict(result).items()},
        "host_ledger": provider_ledger(evaluator, host_names),
        "gas_components": supported_gas, "gas_nonideality": "ideal gas control",
        "gas_mole_fractions": gas_fractions,
        "H2O_H2_gas_ratio": gas_fractions["H2O_gas"] / gas_fractions["H2_gas"],
        "SiO_SiH4_gas_ratio": gas_fractions["SiO_gas"] / gas_fractions["SiH4_gas"],
        "standards": "MELTS full endmember potentials; source absolute gas/Okuchi metal standards plus native alloy shifts; NASA7 H/He/OH/SiH4",
        "phase_acceptance": "unchecked; a liquid-only root is not stable-phase acceptance",
        "missing_acceptance": ["independent exchange-standard alignment", "host-specific revised H2 calibration", "final-composition solid and absent-alloy stability", "gas-network abundance and observable convergence", "alloy pressure response"],
        "provenance": {"checkouts": checkouts, "exogibbs_file": str(Path(__file__).resolve()), "exoeos_file": str(Path(exoeos.__file__).resolve()),
                       "file_sha256": {name: hashlib.sha256(Path(__file__).with_name(name).read_bytes()).hexdigest()
                                       for name in ("run_melts_reference.py", "melts_coupled.py", "full_potential.py", "hydrogen.py", "source.py", "reference.json", "reduced_gas_reference.json")},
                       "jax_version": jax.__version__, "numpy_version": np.__version__, "dtype": "float64", "command": [sys.executable, *sys.argv]},
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--exoeos-checkout", type=Path, required=True)
    parser.add_argument("--runtime", type=Path, required=True)
    parser.add_argument("--python", required=True, help="Python executable for the optional native worker")
    parser.add_argument("--temperature", type=float, default=2350.)
    parser.add_argument("--pressure", type=float, default=1.)
    parser.add_argument("--max-nfev", type=int, default=100)
    parser.add_argument("--metal-absent", action="store_true")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = run_reference(args.exoeos_checkout, args.runtime, args.python,
                           temperature_k=args.temperature, pressure_bar=args.pressure,
                           max_nfev=args.max_nfev, metal_present=not args.metal_absent)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    if not report["result"]["accepted"]:
        raise SystemExit("The conditional branch failed the independent local residual gates.")


if __name__ == "__main__":
    main()
