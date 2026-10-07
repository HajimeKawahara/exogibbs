"""Fresh external MELTS archive replay
======================================

Fresh optional MELTS chemistry replay at archived amounts, without solving."""

from __future__ import annotations

import hashlib
import inspect
import json
from pathlib import Path
from typing import Any

import numpy as np
from scipy.linalg import null_space

from hydrogen import _checkout_provenance, dissolved_h2_standard_rt, hirschmann2012_ln_solubility
from melts_coupled import COMMON_R, make_melts_h2_phase, provider_ledger
from reference import load_reference
from run_melts_reference import MODEL_ID, reduced_gas_standards_rt
from source import make_source_standard_potentials_rt


def revalidate_melts_states(
    saved_states: dict[str, dict], *, evaluator: Any, runtime: Path,
    worker_python: str,
) -> dict[str, Any]:
    """Recompute both declared branches and require saved acceptance to agree.

    The insertion trial can disprove alloy absence within the formal model.
    Local residual acceptance does not establish calibrated phase stability.
    """
    import exoeos
    from exoeos import MaFeSiOHLiquid, solution_state

    labels = ("melts_present", "melts_absent")
    if set(saved_states) != set(labels):
        raise AssertionError("Supply both melts_present and melts_absent states.")
    source = load_reference()
    source_names = [name for phase in source["phases"].values() for name in phase]
    gas_record = json.loads(Path(__file__).with_name("reduced_gas_reference.json").read_text())
    formulas = dict(source["component_formulas"])
    for name, row in zip(evaluator.COMPONENTS, evaluator.FORMULA_MATRIX):
        formulas[name + "_melts"] = {
            element: float(count) for element, count in zip(evaluator.ELEMENTS, row) if count
        }
    formulas["H2_dissolved"] = {"H": 2}
    formulas.update({name + "_gas": item["formula"] for name, item in gas_record["species"].items()})
    alloy = MaFeSiOHLiquid()
    cases, internal = {}, {}
    for label in labels:
        saved = saved_states[label]
        temperature, pressure = saved["temperature_k"], saved["pressure_bar"]
        if saved["model_id"] != MODEL_ID or not all(np.isfinite(v) and v > 0 for v in (temperature, pressure)):
            raise AssertionError(f"{label}: invalid model or thermodynamic conditions.")
        names, elements = saved["component_order"], saved["elements"]
        phases = saved["active_phases"]
        expected_phases = ["silicate", "metal", "gas"] if label == "melts_present" else ["silicate", "gas"]
        if phases != expected_phases or len(set(names)) != len(names) or len(set(elements)) != len(elements):
            raise AssertionError(f"{label}: inconsistent declared phase or component basis.")
        host_names = [name[:-6] for name in names if name.endswith("_melts")]
        gas_names = source["phases"]["gas"] + [name + "_gas" for name in gas_record["species"]]
        phase_names = {
            "silicate": [name + "_melts" for name in host_names] + ["H2_dissolved"],
            "metal": [name + "_metal" for name in alloy.components], "gas": gas_names,
        }
        if set(names) != set(name for group in phase_names.values() for name in group):
            raise AssertionError(f"{label}: incomplete component basis.")
        if set(elements) != set().union(*(formulas[name] for name in names)):
            raise AssertionError(f"{label}: incomplete elemental basis.")
        matrix = np.asarray([[formulas[name].get(e, 0) for name in names] for e in elements], dtype=float)
        if not np.array_equal(matrix, saved["formula_matrix"]):
            raise AssertionError(f"{label}: saved formula matrix differs from independent formulas.")
        n = np.asarray(saved["result"]["component_amounts_mol"], dtype=float)
        budget = np.asarray(saved["element_amounts_mol"], dtype=float)
        if (n.shape != (len(names),) or budget.shape != (len(elements),)
                or not np.all(np.isfinite(n)) or not np.all(np.isfinite(budget))
                or np.any(n < 0) or np.any(budget < 0) or not np.any(budget > 0)):
            raise AssertionError(f"{label}: invalid finite component amounts or element budget.")
        positive = budget > 0
        declared = set(name for phase in phases for name in phase_names[phase])
        active = np.asarray([name in declared for name in names]) & ~np.any(matrix[~positive] != 0, axis=0)
        if np.any(n[active] <= 0) or np.any(n[~active] != 0):
            raise AssertionError(f"{label}: amounts must be positive on declared support and zero elsewhere.")
        indices = {phase: np.asarray([names.index(name) for name in group]) for phase, group in phase_names.items()}
        standards_rt = np.asarray(make_source_standard_potentials_rt(source, source["cases"][0])(temperature, pressure))
        standards = dict(zip(source_names, standards_rt * source["source"]["gas_constant_J_mol_K"] / COMMON_R))
        standards.update({name + "_gas": value for name, value in reduced_gas_standards_rt(temperature).items()})
        h2_standard = float(dissolved_h2_standard_rt(standards["H2_gas"], hirschmann2012_ln_solubility(pressure)))
        melt = make_melts_h2_phase(evaluator, host_names, lambda t, p: h2_standard,
                                   runtime=runtime, python_executable=worker_python)
        liquid = melt(temperature, pressure, n[indices["silicate"]])
        mu = np.full(len(names), np.nan)
        mu[indices["silicate"]] = liquid.mu_rt
        gas_indices = indices["gas"][active[indices["gas"]]]
        gas_n = n[gas_indices]
        mu[gas_indices] = [standards[names[i]] for i in gas_indices] + np.log(gas_n / gas_n.sum()) + np.log(pressure)
        metal_mu0 = np.asarray([standards[name] for name in phase_names["metal"]]) + np.asarray(alloy.standard_state_shift_RT(temperature))

        def metal_mu(x, *, mu0=metal_mu0, t=temperature, p=pressure):
            alloy.validate_state(t, p * 1e5, x)
            with np.errstate(divide="ignore"):
                return mu0 + np.log(x) + np.asarray(solution_state(alloy, t, p * 1e5, x).lngamma)

        metal_n = n[indices["metal"]]
        if "metal" in phases:
            mu[indices["metal"]] = metal_mu(metal_n / metal_n.sum())
        if not np.all(np.isfinite(mu[active])):
            raise AssertionError(f"{label}: fresh potentials are unavailable on declared support.")
        host_active = active[indices["silicate"]]
        host_euler = n[indices["silicate"]][host_active] @ np.asarray(liquid.mu_rt)[host_active]
        if not np.isclose(liquid.gibbs_rt, host_euler, rtol=5e-9, atol=1e-10 * budget.sum()):
            raise AssertionError(f"{label}: fresh host energy violates Euler's identity.")
        totals = matrix @ n
        element_error = float(np.max(np.abs(totals[positive] / budget[positive] - 1)))
        active_matrix = matrix[positive][:, active]
        reaction_error = float(np.max(np.abs(null_space(active_matrix).T @ mu[active]), initial=0.))
        elemental = np.zeros(len(elements))
        elemental[positive] = np.linalg.lstsq(active_matrix.T, mu[active], rcond=None)[0]
        accepted = bool(element_error < 1e-9 and np.all(totals[~positive] == 0) and reaction_error < 1e-8)
        if not isinstance(saved["result"]["accepted"], bool) or accepted != saved["result"]["accepted"]:
            raise AssertionError(f"{label}: recomputed chemical acceptance differs from recorded status.")
        cases[label] = {
            "chemistry": "recomputed", "accepted": accepted,
            "temperature_k": temperature, "pressure_bar": pressure,
            "max_relative_element_residual": element_error,
            "max_abs_reaction_residual_RT": reaction_error,
            "max_abs_stationarity_residual_RT": float(np.max(np.abs(mu[active] - active_matrix.T @ elemental[positive]))),
            "gibbs_RT": float(n[active] @ mu[active]),
            "mu_RT_active": mu[active].tolist(), "active_components": np.asarray(names)[active].tolist(),
            "elemental_potentials_RT": elemental.tolist(),
            "phase_amounts_mol": {phase: float(n[indices[phase]].sum()) for phase in phases},
            "phase_interpretation": "conditional local root" if "metal" in phases else "phase-suppressed diagnostic",
            "host_provider": provider_ledger(evaluator, host_names),
        }
        internal[label] = (metal_n, matrix[:, indices["metal"]], elemental, metal_mu)
    present, absent = (saved_states[label] for label in labels)
    if (present["elements"] != absent["elements"]
            or not np.array_equal(present["element_amounts_mol"], absent["element_amounts_mol"])
            or any(present[key] != absent[key] for key in ("temperature_k", "pressure_bar"))):
        raise AssertionError("MELTS phase comparison requires identical T, P, and elemental budgets.")
    trial_n = internal["melts_present"][0]
    trial_x = trial_n / trial_n.sum()
    _, metal_formula, absent_elemental, absent_metal_mu = internal["melts_absent"]
    trial_active = trial_x > 0
    insertion = float(trial_x[trial_active] @ absent_metal_mu(trial_x)[trial_active]
                      - (metal_formula @ trial_x) @ absent_elemental)
    comparison = {
        "local_roots_accepted": all(case["accepted"] for case in cases.values()),
        "same_budget_gibbs_present_minus_absent_RT": cases["melts_present"]["gibbs_RT"] - cases["melts_absent"]["gibbs_RT"],
        "absent_alloy_trial": {"components": list(alloy.components), "mole_fractions": trial_x.tolist(),
                               "insertion_RT": insertion,
                               "disproves_absence": bool(cases["melts_absent"]["accepted"] and insertion < 0)},
        "interpretation": "At an accepted absent root, a negative trial disproves alloy absence within these formal potentials; one trial cannot establish phase stability.",
        "evidence": "Conditional potentials; cross-phase calibration and competing host phases remain unchecked.",
    }
    alloy_path = Path(inspect.getfile(MaFeSiOHLiquid)).resolve()
    evaluator_path = Path(evaluator.__file__).resolve()
    return {
        "cases": cases, "melts_phase_comparison": comparison,
        "provider_provenance": {
            "exoeos_import_path": str(Path(exoeos.__file__).resolve()),
            "alloy_path": str(alloy_path), "alloy_sha256": hashlib.sha256(alloy_path.read_bytes()).hexdigest(),
            "evaluator_path": str(evaluator_path),
            "exoeos_checkout": _checkout_provenance(evaluator_path.parents[1]),
            "runtime": str(Path(runtime).resolve()), "worker_python": str(Path(worker_python).absolute()),
        },
    }
