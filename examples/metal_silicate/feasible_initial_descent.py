"""
Conserved Gibbs descent for an initial ledger
=============================================

Prepare a conserved lower-G initial ledger; never accept an equilibrium.
"""

from time import perf_counter
from typing import Callable, Mapping, NamedTuple, Optional

import numpy as np


class FeasibleInitialDescentResult(NamedTuple):
    """An initial guess for the unchanged scalar minimizer and its audits."""

    component_amounts_mol: np.ndarray
    adopted: bool
    diagnostics: dict


def _maximum(values):
    return float(np.max(np.abs(values), initial=0.))


def _null_basis(rows):
    """Normalize constraint rows before SVD without changing their null space."""
    norms = np.linalg.norm(rows, axis=1)
    rows = rows[norms > 0] / norms[norms > 0, None]
    _, singular, right = np.linalg.svd(rows, full_matrices=True)
    tolerance = max(rows.shape) * np.finfo(float).eps * singular[0] if len(singular) else 0.
    rank = int(np.sum(singular > tolerance))
    condition = float(singular[0] / singular[rank - 1]) if rank else 1.
    basis = right[rank:].T
    # Refine annihilation so isolated trace-element rows do not accumulate
    # relative inventory drift from an O(eps) dense SVD basis coefficient.
    if rank and basis.shape[1]:
        basis -= np.linalg.lstsq(rows, rows @ basis, rcond=None)[0]
    return basis, rank, condition


def descend_initial_amounts(
    problem: "LocalProblem", temperature_k: float, pressure_bar: float,
    element_amounts_mol: np.ndarray, phase_callbacks: Mapping[str, "PhaseCallback"],
    *, initial_component_amounts_mol: np.ndarray,
    phase_composition_bounds: Optional[Mapping[str, tuple[np.ndarray, np.ndarray]]] = None,
    allow_interior: bool = False,
    update_active_faces: bool = False,
    max_iterations: int = 256, max_evaluations: int = 1024,
    max_seconds: float = 300., progress: Optional[Callable[[dict], None]] = None,
) -> FeasibleInitialDescentResult:
    """Descend current-pressure G on the donor's atom and explicit active face.

    By default an explicit active metal composition face is required. With
    ``allow_interior=True``, a feasible donor containing metal may also descend
    without that face. Only the supplied current bounds define active rows;
    with no active rows the tangent contains atom constraints alone.
    ``update_active_faces=True`` adds newly approached current composition rows
    at the existing active-slack threshold, preserving their current linear
    value. These extra tangent rows constrain this initializer only.

    All trials retain positive support and every declared composition bound.
    The amount-space direction uses the null space of C diag(sqrt(n)), with
    normalized rows, and an Armijo line search. At most 256 iterations, 1024
    complete or partial callback bundles and 1800 seconds may be requested.
    Deadlines are checked between callbacks; a caller must separately bound a
    provider call that can block. No finite differences or physical solves are
    introduced here beyond the supplied callbacks' own evaluations.

    Only a feasible, freshly evaluated, strictly lower-G ledger can be adopted.
    Tangent stationarity is diagnostic, not an equilibrium acceptance gate.
    The original scalar minimizer and all its independent audits must still run.
    """
    # Resolve provider example imports only after the caller binds its checkout.
    from common_gibbs import _composition_constraints
    from local import _budgets

    if (problem.reaction_offset is not None or set(phase_callbacks) != set(problem.phases)
            or not all(np.isfinite(v) and v > 0 for v in (temperature_k, pressure_bar))
            or type(allow_interior) is not bool
            or type(update_active_faces) is not bool
            or type(max_iterations) is not int or not 1 <= max_iterations <= 256
            or type(max_evaluations) is not int or not 1 <= max_evaluations <= 1024
            or not np.isfinite(max_seconds) or not 0 < max_seconds <= 1800
            or (progress is not None and not callable(progress))):
        raise ValueError("Require common callbacks, positive T/P, boolean allow_interior and update_active_faces, and bounded iteration/evaluation/time limits.")
    budget = _budgets(element_amounts_mol, len(problem.elements))
    positive = budget > 0
    if not np.array_equal(np.flatnonzero(positive), problem.element_indices):
        raise ValueError("The donor changed the declared element support.")
    indices = np.asarray(problem.species_indices)
    excluded = np.ones(len(problem.full_species), dtype=bool)
    excluded[indices] = False
    if not np.isrealobj(initial_component_amounts_mol):
        raise ValueError("The initial ledger must be real.")
    original = np.array(initial_component_amounts_mol, dtype=float, copy=True)
    if (original.shape != excluded.shape or np.any(~np.isfinite(original))
            or np.any(original[indices] <= 0) or np.any(original[excluded] != 0)):
        raise ValueError("The initial ledger must be positive on active support and zero elsewhere.")
    scale = float(budget.sum())
    if not np.isfinite(scale) or scale <= 0:
        raise ValueError("The total atom scale must be positive and finite.")
    initial = original[indices] / scale
    formula = np.asarray(problem.formula_matrix, dtype=float)
    rows = formula / (budget[positive] / scale)[:, None]
    domain, labels, sections = _composition_constraints(problem, phase_composition_bounds)

    def slacks(amounts):
        return np.asarray([value / amounts[section].sum()
                           for value, section in zip(domain @ amounts, sections)])

    initial_slack = slacks(initial)
    if (np.any(initial <= 0) or _maximum(rows @ initial - 1.) >= 1e-9
            or np.any(initial_slack < -1e-12)
            or np.any(domain @ initial < -1e-13 * initial.sum())):
        raise ValueError("The initial ledger must already conserve atoms and satisfy every composition bound.")
    active = initial_slack < 1e-8
    active_rows = domain[active]
    active_sections = [section for section, present in zip(sections, active) if present]
    active_values = active_rows @ initial
    tangent = np.vstack((rows, np.asarray([row / initial[section].sum()
                        for row, section in zip(active_rows, active_sections)]).reshape((-1, len(initial)))))
    reaction, rank, _ = _null_basis(tangent)
    started = perf_counter()
    report = {
        "adopted": False, "stage": "input_validation", "iterations": 0,
        "allow_interior": allow_interior,
        "update_active_faces": update_active_faces, "activated_constraints": [],
        "evaluations": 0, "callback_evaluations": 0, "accepted_steps": 0,
        "max_iterations": max_iterations, "max_evaluations": max_evaluations,
        "max_seconds": float(max_seconds), "elapsed_seconds": 0.,
        "active_constraints": [label for label, present in zip(labels, active) if present],
        "initial_active_constraints": [label for label, present in zip(labels, active) if present],
        "tangent_rank": rank, "tangent_reaction_count": reaction.shape[1],
        "initial_max_relative_atom_residual": _maximum(rows @ initial - 1.),
        "initial_relative_composition_slacks": initial_slack.tolist(),
        "initial_gibbs_rt": None, "corrected_gibbs_rt": None, "final_gibbs_rt": None,
        "invalid_callbacks": [], "history": [], "rejected_trials": 0,
        "scope": "Feasible lower-G initial amounts only; the unchanged scalar minimizer and its independent audits remain mandatory."}

    def event(kind, **details):
        report["elapsed_seconds"] = perf_counter() - started
        if progress is not None:
            progress({"event": kind, "stage": report["stage"], "iterations": report["iterations"],
                      "evaluations": report["evaluations"], "callback_evaluations": report["callback_evaluations"],
                      "elapsed_seconds": report["elapsed_seconds"], **details})

    def feasibility(amounts):
        if np.any(~np.isfinite(amounts)) or np.any(amounts <= 0):
            return False, {}
        slack = slacks(amounts)
        drift = np.asarray([(value - original_value) / amounts[section].sum()
                            for value, original_value, section in zip(active_rows @ amounts, active_values, active_sections)])
        atom = _maximum(rows @ amounts - 1.)
        face = _maximum(slack[active])
        valid = (atom < 1e-9 and face < 1e-8 and _maximum(drift) < 1e-10
                 and np.all(slack >= -1e-12) and np.all(domain @ amounts >= -1e-13 * amounts.sum()))
        return bool(valid), {"max_relative_atom_residual": atom, "max_relative_active_face_residual": face,
                             "max_relative_active_face_drift": _maximum(drift),
                             "minimum_component_amount_mol": float(scale * amounts.min()),
                             "relative_composition_slacks": slack.tolist()}

    class LimitReached(Exception):
        pass

    def check_budget():
        if perf_counter() - started >= max_seconds:
            raise LimitReached("elapsed-time budget reached between callbacks")

    def evaluate(amounts):
        if report["evaluations"] >= max_evaluations:
            raise LimitReached("callback-bundle evaluation budget reached")
        check_budget()
        report["evaluations"] += 1
        mu = np.empty(len(initial))
        energy = 0.
        for phase, section in zip(problem.phases, problem.phase_slices):
            check_budget()
            report["callback_evaluations"] += 1
            physical = scale * amounts[section]
            try:
                state = phase_callbacks[phase](temperature_k, pressure_bar, physical)
                phase_mu = np.asarray(state.mu_rt, dtype=float)
                if (phase_mu.shape != physical.shape or np.any(~np.isfinite(phase_mu))
                        or not np.isfinite(state.gibbs_rt)
                        or not np.isclose(state.gibbs_rt, float(physical @ phase_mu),
                                          rtol=5e-9, atol=1e-10 * scale)):
                    raise ValueError("The callback omitted finite full potentials or violated Euler's identity.")
                mu[section] = phase_mu
                energy += float(state.gibbs_rt) / scale
            except (ValueError, TypeError, AttributeError, RuntimeError, FloatingPointError, np.linalg.LinAlgError) as error:
                report["invalid_callbacks"].append({"stage": report["stage"], "phase": phase,
                    "evaluation": report["evaluations"], "phase_component_amounts_mol": physical.tolist(),
                    "error": type(error).__name__ + ": " + str(error)})
                raise
        if not np.isfinite(energy):
            raise ValueError("The total current-pressure Gibbs energy is unavailable.")
        return energy, mu

    def finish(amounts, energy, reason):
        valid, details = feasibility(amounts)
        adopted = bool(valid and energy is not None and initial_energy is not None and energy < initial_energy)
        report.update(details, adopted=adopted, stage="adopted" if adopted else "unchanged", reason=reason,
                      corrected_gibbs_rt=None if energy is None else float(scale * energy),
                      final_gibbs_rt=None if energy is None else float(scale * energy),
                      final_max_relative_atom_residual=details.get("max_relative_atom_residual"),
                      final_stationarity_max_abs=None if mu is None else _maximum(reaction.T @ mu),
                      inequalities_passed=valid, energy_strictly_decreased=adopted,
                      gibbs_change_rt_per_inventory_atom=None if energy is None or initial_energy is None else float(energy - initial_energy))
        result = original.copy()
        if adopted:
            result[indices] = scale * amounts
        event("complete", adopted=adopted, reason=reason)
        return FeasibleInitialDescentResult(result, adopted, report)

    initial_energy = None
    if ("metal" not in problem.phases or (not allow_interior and not any(
            label["phase"] == "metal" for label in report["active_constraints"]))):
        report["stage"] = "skipped"
        report["reason"] = "No explicit active metal composition face."
        event("complete", adopted=False, reason=report["reason"])
        return FeasibleInitialDescentResult(original.copy(), False, report)
    amounts = initial.copy()
    energy = None
    mu = None
    try:
        report["stage"] = "initial_energy"
        initial_energy, mu = evaluate(amounts)
        energy = initial_energy
        report["initial_gibbs_rt"] = float(scale * initial_energy)
        report["initial_stationarity_max_abs"] = _maximum(reaction.T @ mu)
        event("initial", gibbs_rt_per_inventory_atom=energy,
              stationarity_max_abs=report["initial_stationarity_max_abs"])
        reason = "iteration budget reached"
        for iteration in range(1, max_iterations + 1):
            check_budget()
            report.update(stage="descent", iterations=iteration)
            if update_active_faces:
                current_slack = slacks(amounts)
                reached = (current_slack < 1e-8) & ~active
                if np.any(reached):
                    held_values = np.zeros(len(domain))
                    held_values[active] = active_values
                    held_values[reached] = (domain @ amounts)[reached]
                    for index in np.flatnonzero(reached):
                        report["activated_constraints"].append({
                            "iteration": iteration, "constraint_index": int(index),
                            "constraint": labels[index], "relative_slack": float(current_slack[index]),
                            "phase_amount_mol": float(scale * amounts[sections[index]].sum()),
                            "gibbs_rt_per_inventory_atom": float(energy),
                            "held_linear_value_per_inventory_atom": float(held_values[index])})
                    active |= reached
                    active_rows = domain[active]
                    active_sections = [section for section, present in zip(sections, active) if present]
                    active_values = held_values[active]
                    tangent = np.vstack((rows, np.asarray([row / amounts[section].sum()
                        for row, section in zip(active_rows, active_sections)]).reshape((-1, len(initial)))))
                    reaction, rank, _ = _null_basis(tangent)
                    report.update(active_constraints=[label for label, present in zip(labels, active) if present],
                                  tangent_rank=rank, tangent_reaction_count=reaction.shape[1])
                    event("active_faces_updated", activated_indices=np.flatnonzero(reached).tolist())
            root = np.sqrt(amounts)
            basis, current_rank, condition = _null_basis(tangent * root)
            if current_rank != rank:
                reason = "weighted constraint rank changed; retain the best feasible ledger"
                break
            weighted = root[:, None] * basis
            gradient = weighted.T @ mu
            direction = -weighted @ gradient
            slope = -float(gradient @ gradient)
            negative = direction < 0
            boundary = float(np.min(-amounts[negative] / direction[negative], initial=np.inf))
            domain_direction = domain @ direction
            approaching = (domain_direction < 0) & ~active
            if np.any(approaching):
                # Roundoff-sized initial negative slacks may not be made worse.
                boundary = min(boundary, float(np.min(np.maximum(domain @ amounts, 0.)[approaching]
                                                      / -domain_direction[approaching])))
            step = min(1., .9 * boundary)
            if not np.isfinite(slope) or slope >= 0 or step <= 0:
                reason = "no finite feasible descent direction"
                break
            if -step * slope <= 8 * np.finfo(float).eps * max(1., abs(energy)):
                reason = "predicted energy decrease is below binary64 resolution"
                break
            accepted = False
            for backtrack in range(24):
                check_budget()
                candidate = amounts + step * direction
                valid, details = feasibility(candidate)
                if valid:
                    try:
                        candidate_energy, candidate_mu = evaluate(candidate)
                    except (ValueError, TypeError, AttributeError, RuntimeError, FloatingPointError, np.linalg.LinAlgError):
                        candidate_energy = None
                    if (candidate_energy is not None and candidate_energy < energy
                            and candidate_energy <= energy + 1e-4 * step * slope):
                        amounts, energy, mu = candidate, candidate_energy, candidate_mu
                        accepted = True
                        report["accepted_steps"] += 1
                        entry = dict(iteration=iteration, evaluation=report["evaluations"], step=step,
                                     backtracks=backtrack, gibbs_rt_per_inventory_atom=energy,
                                     gibbs_change_rt_per_inventory_atom=float(energy - initial_energy),
                                     stationarity_max_abs=_maximum(reaction.T @ mu),
                                     weighted_gradient_norm=float(np.linalg.norm(gradient)),
                                     weighted_constraint_condition_number=condition, **details)
                        report["history"].append(entry)
                        event("iteration", **entry)
                        break
                report["rejected_trials"] += 1
                step *= .5
                if -step * slope <= 8 * np.finfo(float).eps * max(1., abs(energy)):
                    break
            if not accepted:
                reason = "line search exhausted representable feasible decrease"
                break
        report["final_stationarity_max_abs"] = _maximum(reaction.T @ mu)
        return finish(amounts, energy, reason)
    except LimitReached as error:
        return finish(amounts, energy, str(error))
    except (ValueError, TypeError, AttributeError, RuntimeError, FloatingPointError, np.linalg.LinAlgError) as error:
        return finish(amounts, energy, type(error).__name__ + ": " + str(error))
