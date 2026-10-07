"""Bounded correction of a conserved initial ledger on an explicit active face.

This example helper supplies only an initial guess to the unchanged scalar
minimizer. Its result is neither an equilibrium acceptance nor a phase proof.
"""

from typing import Mapping, NamedTuple, Optional

import numpy as np
from scipy.linalg import null_space
from scipy.optimize import least_squares



class InitialCorrectionResult(NamedTuple):
    """A full amount ledger and numerical diagnostics, never a physical result."""

    component_amounts_mol: np.ndarray
    adopted: bool
    diagnostics: dict


def correct_initial_amounts(
    problem: "LocalProblem",
    temperature_k: float,
    pressure_bar: float,
    element_amounts_mol: np.ndarray,
    phase_callbacks: Mapping[str, "PhaseCallback"],
    *,
    initial_component_amounts_mol: np.ndarray,
    phase_composition_bounds: Optional[Mapping[str, tuple[np.ndarray, np.ndarray]]] = None,
    max_nfev: int = 12,
) -> InitialCorrectionResult:
    """Try constrained stationarity before the caller's original scalar solve.

    The caller must bind the accepted donor and unchanged physical recipe. The
    supplied full ledger must already conserve the current atom inventory and
    obey every declared phase bound. Only explicitly imposed composition rows
    with relative slack below 1e-8 select an active face. Tiny positive species
    are never removed or converted into active simplex bounds.

    ``max_nfev`` counts SciPy solver residual evaluations, excluding finite-
    difference Jacobian calls; diagnostics count all residual/provider calls.
    Failed corrections return a copy of the original valid ledger. Adoption
    additionally requires positive support, relative atom closure below 1e-9,
    face/stationarity residuals below 1e-8, all relative composition slacks at
    least -1e-12, and no increase in freshly evaluated current-pressure G.
    The original scalar minimizer and all its acceptance audits must still run.
    """
    from common_gibbs import _composition_constraints
    from local import _budgets

    if (problem.reaction_offset is not None or set(phase_callbacks) != set(problem.phases)
            or not all(np.isfinite(value) and value > 0 for value in (temperature_k, pressure_bar))
            or type(max_nfev) is not int or not 1 <= max_nfev <= 12):
        raise ValueError("Require common callbacks, positive T/P and max_nfev from 1 to 12 (excluding finite-difference Jacobian calls).")
    budgets = _budgets(element_amounts_mol, len(problem.elements))
    positive = budgets > 0
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
    scale = float(budgets.sum())
    if not np.isfinite(scale) or scale <= 0:
        raise ValueError("The total atom scale must be positive and finite.")
    formula = np.asarray(problem.formula_matrix, dtype=float)
    target = budgets[positive] / scale
    initial = original[indices] / scale
    row_matrix = formula / target[:, None]
    domain, labels, sections = _composition_constraints(problem, phase_composition_bounds)

    def slacks(amounts):
        return np.asarray([value / amounts[section].sum()
                           for value, section in zip(domain @ amounts, sections)])

    initial_atoms = row_matrix @ initial - 1.
    initial_slack = slacks(initial)
    if (np.max(np.abs(initial_atoms), initial=0.) >= 1e-9
            or np.any(initial_slack < -1e-12)
            or np.any(domain @ initial < -1e-13 * initial.sum())):
        raise ValueError("The initial ledger must already conserve atoms and satisfy every composition bound.")
    active = initial_slack < 1e-8
    active_rows = domain[active]
    active_sections = [section for section, present in zip(sections, active) if present]
    diagnostics = {
        "adopted": False, "stage": "input_validation", "max_nfev": max_nfev,
        "nfev": 0, "solver_result_returned": False, "residual_evaluations": 0, "callback_evaluations": 0,
        "invalid_callbacks": [], "active_constraints": [label for label, present in zip(labels, active) if present],
        "initial_max_relative_atom_residual": float(np.max(np.abs(initial_atoms), initial=0.)),
        "initial_relative_composition_slacks": initial_slack.tolist(),
        "initial_gibbs_rt": None, "corrected_gibbs_rt": None,
        "equations_closed": False, "inequalities_passed": False, "energy_nonincreasing": False,
        "scope": "Initial amounts only; the unchanged scalar minimizer and its independent audits remain mandatory."}

    def fallback(reason):
        diagnostics["reason"] = reason
        return InitialCorrectionResult(original.copy(), False, diagnostics)

    if "metal" not in problem.phases or not any(label["phase"] == "metal" for label in diagnostics["active_constraints"]):
        diagnostics["stage"] = "skipped"
        return fallback("No explicit active metal composition face; keep the original initial ledger.")
    tangent = np.vstack((formula, active_rows))
    reaction = null_space(tangent).T
    diagnostics["tangent_reaction_count"] = len(reaction)

    def evaluate(normalized):
        if (normalized.shape != initial.shape or np.any(~np.isfinite(normalized))
                or np.any(normalized <= 0)):
            raise ValueError("A log-amount correction left finite positive support.")
        amounts = scale * normalized
        mu = np.empty(len(indices))
        energy = 0.
        for phase, section in zip(problem.phases, problem.phase_slices):
            diagnostics["callback_evaluations"] += 1
            try:
                state = phase_callbacks[phase](temperature_k, pressure_bar, amounts[section])
                phase_mu = np.asarray(state.mu_rt, dtype=float)
                if (phase_mu.shape != amounts[section].shape or np.any(~np.isfinite(phase_mu))
                        or not np.isfinite(state.gibbs_rt)
                        or not np.isclose(state.gibbs_rt, float(amounts[section] @ phase_mu),
                                          rtol=5e-9, atol=1e-10 * scale)):
                    raise ValueError("The current callback omitted finite full potentials or violated Euler's identity.")
            except (ValueError, RuntimeError, FloatingPointError, np.linalg.LinAlgError) as error:
                diagnostics["invalid_callbacks"].append({"stage": diagnostics["stage"], "phase": phase,
                    "residual_evaluation": diagnostics["residual_evaluations"],
                    "error": type(error).__name__ + ": " + str(error)})
                raise
            mu[section] = phase_mu
            energy += float(state.gibbs_rt)
        if not np.isfinite(energy):
            raise ValueError("The current total Gibbs energy is unavailable.")
        return energy, mu

    def residual(log_amounts):
        diagnostics["residual_evaluations"] += 1
        with np.errstate(over="raise", under="raise", invalid="raise"):
            candidate = np.exp(log_amounts)
        _, mu = evaluate(candidate)
        face = np.asarray([value / candidate[section].sum()
                           for value, section in zip(active_rows @ candidate, active_sections)])
        return np.r_[row_matrix @ candidate - 1., face, reaction @ mu]

    try:
        diagnostics["stage"] = "initial_energy"
        initial_energy, _ = evaluate(initial)
        diagnostics["initial_gibbs_rt"] = initial_energy
        diagnostics["stage"] = "correction"
        optimized = least_squares(residual, np.log(initial), xtol=1e-13, ftol=1e-13, gtol=1e-13,
                                  max_nfev=max_nfev)
        diagnostics.update(nfev=int(optimized.nfev), solver_result_returned=True, solver_success=bool(optimized.success),
                           solver_message=str(optimized.message))
        diagnostics["stage"] = "final_validation"
        with np.errstate(over="raise", under="raise", invalid="raise"):
            candidate = np.exp(optimized.x)
        final_residual = residual(optimized.x)
        energy, _ = evaluate(candidate)
        atom_error = float(np.max(np.abs(row_matrix @ candidate - 1.), initial=0.))
        equation_error = float(np.max(np.abs(final_residual), initial=0.))
        final_slack = slacks(candidate)
        equations_closed = atom_error < 1e-9 and equation_error < 1e-8
        inequalities_passed = bool(np.all(final_slack >= -1e-12)
                                   and np.all(domain @ candidate >= -1e-13 * candidate.sum()))
        nonincreasing = bool(energy <= initial_energy)
        diagnostics.update(corrected_gibbs_rt=energy, final_max_relative_atom_residual=atom_error,
            final_max_equation_residual=equation_error, final_relative_composition_slacks=final_slack.tolist(),
            equations_closed=equations_closed, inequalities_passed=inequalities_passed,
            energy_nonincreasing=nonincreasing, gibbs_change_rt_per_inventory_atom=(energy - initial_energy) / scale)
        if not (equations_closed and inequalities_passed and nonincreasing):
            return fallback("The correction failed an atom, stationarity, domain or current-energy gate.")
        corrected = np.zeros_like(original)
        corrected[indices] = scale * candidate
        diagnostics.update(adopted=True, stage="adopted", reason="Use the gated ledger only as the original scalar's initial guess.")
        return InitialCorrectionResult(corrected, True, diagnostics)
    except (ValueError, RuntimeError, FloatingPointError, np.linalg.LinAlgError) as error:
        return fallback(type(error).__name__ + ": " + str(error))
