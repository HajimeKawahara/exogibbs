"""Inspect an initial stationarity Jacobian without solving equilibrium.

All probes keep the supplied temperature, pressure, physical callbacks and
positive component support. Perturbations need not conserve atoms or satisfy
composition bounds; they are derivative probes, never accepted states.
"""

from time import perf_counter
from typing import Callable, Mapping, Optional

import numpy as np
from scipy.linalg import null_space


def _maximum(values):
    return float(np.max(np.abs(values), initial=0.))


def _finite(value):
    return float(value) if np.isfinite(value) else None


def _matrix_report(matrix, names):
    """Return dense derivatives and explicitly identified weak directions."""
    _, singular, right = np.linalg.svd(matrix, full_matrices=False)
    tolerance = singular[0] * max(matrix.shape) * np.finfo(float).eps
    rank = int(np.sum(singular > tolerance))
    weak = []
    for index in range(len(singular) - 1, max(-1, len(singular) - 4), -1):
        vector = right[index]
        selected = np.argsort(-np.abs(vector), kind="stable")[:6]
        weak.append({"singular_value": float(singular[index]), "log_amount_direction": vector.tolist(),
                     "largest_components": [{"component": names[j], "coefficient": float(vector[j])}
                                             for j in selected]})
    return {"jacobian": matrix.tolist(), "singular_values": singular.tolist(), "rank": rank,
            "rank_tolerance": float(tolerance),
            "condition_number": _finite(singular[0] / singular[-1]) if singular[-1] > 0 else None,
            "column_norms": np.linalg.norm(matrix, axis=0).tolist(), "weak_directions": weak}


def diagnose_initialization(
    problem: "LocalProblem", temperature_k: float, pressure_bar: float,
    element_amounts_mol: np.ndarray, phase_callbacks: Mapping[str, "PhaseCallback"],
    *, initial_component_amounts_mol: np.ndarray,
    phase_composition_bounds: Optional[Mapping[str, tuple[np.ndarray, np.ndarray]]] = None,
    progress: Optional[Callable[[dict], None]] = None,
) -> dict:
    """Compare three forward-difference scales at one unchanged initial ledger.

    No minimizer or equilibrium acceptance is performed. Provider callbacks may
    themselves solve their existing internal thermodynamic problems. The caller
    owns explicit authorization for those evaluations and callback construction.
    Initial validation matches ``correct_initial_amounts``; malformed input raises
    before any provider call. A provider failure returns the partial diagnostic,
    including the actual failing phase amounts. ``progress`` receives JSON-ready
    events after each real residual evaluation, with cumulative elapsed time.

    The grids use SciPy's default binary64 two-point log step, then absolute log
    step magnitudes 1e-4 and 1e-3 with the same signs. With N active components the complete diagnostic uses
    exactly 1 + 3 * (N + 1) residual evaluations. Each grid ends with a base
    reevaluation after perturbing the final phase; this evicts a one-entry
    composition cache. Consecutive cached base calls are not repeatability tests.
    """
    # Preserve physical-checkout ownership when loaded before source._load().
    from common_gibbs import _composition_constraints
    from local import _budgets

    if (problem.reaction_offset is not None or set(phase_callbacks) != set(problem.phases)
            or not all(np.isfinite(value) and value > 0 for value in (temperature_k, pressure_bar))):
        raise ValueError("Require common callbacks and positive finite T/P.")
    if progress is not None and not callable(progress):
        raise ValueError("progress must be callable or None.")
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
    formula = np.asarray(problem.formula_matrix, dtype=float)
    target = budget[positive] / scale
    initial = original[indices] / scale
    rows = formula / target[:, None]
    domain, labels, sections = _composition_constraints(problem, phase_composition_bounds)

    def slacks(amounts):
        return np.asarray([value / amounts[section].sum()
                           for value, section in zip(domain @ amounts, sections)])

    atoms = rows @ initial - 1.
    slack = slacks(initial)
    if (_maximum(atoms) >= 1e-9 or np.any(slack < -1e-12)
            or np.any(domain @ initial < -1e-13 * initial.sum())):
        raise ValueError("The initial ledger must already conserve atoms and satisfy every composition bound.")
    active = slack < 1e-8
    active_rows = domain[active]
    active_sections = [section for section, present in zip(sections, active) if present]
    active_labels = [label for label, present in zip(labels, active) if present]
    names = list(problem.species)
    result = {"completed": False, "stage": "input_validation", "temperature_K": float(temperature_k),
              "pressure_bar": float(pressure_bar), "component_order": names,
              "element_order": [problem.elements[i] for i in np.flatnonzero(positive)],
              "initial_component_amounts_mol": original.tolist(), "amount_scale_mol": scale,
              "initial_relative_composition_slacks": slack.tolist(), "composition_constraints": labels,
              "active_constraints": active_labels, "residual_evaluations": 0, "callback_evaluations": 0,
              "phase_evaluations": {phase: {"calls": 0, "elapsed_seconds": 0.} for phase in problem.phases},
              "elapsed_seconds": 0., "invalid_callbacks": [], "base_repeats": [], "grids": [],
              "comparisons": [],
              "scope": "Derivative probes only; no corrected ledger, equilibrium result, accepted state or donor is produced."}
    started = perf_counter()

    def event(kind, **details):
        result["elapsed_seconds"] = perf_counter() - started
        if progress is not None:
            progress({"event": kind, "stage": result["stage"],
                      "residual_evaluations": result["residual_evaluations"],
                      "callback_evaluations": result["callback_evaluations"],
                      "elapsed_seconds": result["elapsed_seconds"], **details})

    if "metal" not in problem.phases or not any(label["phase"] == "metal" for label in active_labels):
        result.update(stage="skipped", reason="No explicit active metal composition face.")
        event("complete")
        return result
    tangent = np.vstack((formula, active_rows))
    reaction = null_space(tangent).T
    atom_count, face_count = len(target), len(active_rows)
    result.update(tangent_rank=int(np.linalg.matrix_rank(tangent)), tangent_reaction_count=len(reaction),
                  reaction_basis=reaction.tolist(),
                  tangent_element_annihilation_max_abs=_maximum(reaction @ formula.T),
                  tangent_face_annihilation_max_abs=_maximum(reaction @ active_rows.T))
    log_initial = np.log(initial)
    # Analytic amount/face derivatives are independent of the provider physics.
    analytic = [rows * initial]
    face_derivatives = []
    for row, section in zip(active_rows, active_sections):
        total = initial[section].sum()
        derivative = row * initial / total
        derivative[section] -= float(row @ initial) * initial[section] / total**2
        face_derivatives.append(derivative)
    analytic.append(np.asarray(face_derivatives).reshape((face_count, len(initial))))
    analytic = np.vstack(analytic)
    result["analytic_atom_face_jacobian"] = analytic.tolist()

    def evaluate(log_amounts):
        result["residual_evaluations"] += 1
        with np.errstate(over="raise", under="raise", invalid="raise"):
            candidate = np.exp(log_amounts)
            amounts = scale * candidate
        if np.any(~np.isfinite(amounts)) or np.any(amounts <= 0):
            raise ValueError("A derivative probe left finite positive support.")
        mu = np.empty(len(indices))
        energies = []
        durations = []
        for phase, section in zip(problem.phases, problem.phase_slices):
            result["callback_evaluations"] += 1
            result["phase_evaluations"][phase]["calls"] += 1
            before = perf_counter()
            try:
                state = phase_callbacks[phase](temperature_k, pressure_bar, amounts[section])
                phase_mu = np.asarray(state.mu_rt, dtype=float)
                if (phase_mu.shape != amounts[section].shape or np.any(~np.isfinite(phase_mu))
                        or not np.isfinite(state.gibbs_rt)
                        or not np.isclose(state.gibbs_rt, float(amounts[section] @ phase_mu),
                                          rtol=5e-9, atol=1e-10 * scale)):
                    raise ValueError("The callback omitted finite full potentials or violated Euler's identity.")
            except (ValueError, TypeError, AttributeError, RuntimeError, FloatingPointError, np.linalg.LinAlgError) as error:
                result["invalid_callbacks"].append({"stage": result["stage"], "phase": phase,
                    "residual_evaluation": result["residual_evaluations"],
                    "phase_component_amounts_mol": amounts[section].tolist(),
                    "active_component_amounts_mol": amounts.tolist(),
                    "elapsed_seconds": perf_counter() - before,
                    "error": type(error).__name__ + ": " + str(error)})
                raise
            finally:
                result["phase_evaluations"][phase]["elapsed_seconds"] += perf_counter() - before
            mu[section] = phase_mu
            energies.append(float(state.gibbs_rt))
            durations.append(perf_counter() - before)
        face = np.asarray([value / candidate[section].sum()
                           for value, section in zip(active_rows @ candidate, active_sections)])
        residual = np.r_[rows @ candidate - 1., face, reaction @ mu]
        if np.any(~np.isfinite(residual)) or not np.isfinite(sum(energies)):
            raise ValueError("A derivative probe returned nonfinite residuals or energy.")
        return residual, mu, energies, durations

    def base_record(evaluated, after):
        residual, mu, energies, durations = evaluated
        report = {"after": after, "residual": residual.tolist(), "mu_rt": mu.tolist(),
                  "gibbs_rt": float(sum(energies)), "phase_gibbs_rt": energies,
                  "phase_callback_seconds": durations,
                  "atom_max_abs": _maximum(residual[:atom_count]),
                  "face_max_abs": _maximum(residual[atom_count:atom_count + face_count]),
                  "stationarity_max_abs": _maximum(residual[atom_count + face_count:])}
        if result["base_repeats"]:
            reference = result["base_repeats"][0]
            report.update(mu_drift_max_abs=_maximum(mu - reference["mu_rt"]),
                          residual_drift_max_abs=_maximum(residual - reference["residual"]),
                          stationarity_drift_max_abs=_maximum(
                              residual[atom_count + face_count:] - np.asarray(reference["residual"])[atom_count + face_count:]),
                          gibbs_drift_rt_per_inventory_atom=(sum(energies) - reference["gibbs_rt"]) / scale)
        result["base_repeats"].append(report)
        event("base", **report)

    try:
        result["stage"] = "initial_base"
        base = evaluate(log_initial)
        base_record(base, "initial")
        signs = np.where(log_initial >= 0, 1., -1.)
        default = np.sqrt(np.finfo(float).eps) * signs * np.maximum(1., np.abs(log_initial))
        grids = (("scipy_default", default), ("absolute_log_1e-4", signs * 1e-4),
                 ("absolute_log_1e-3", signs * 1e-3))
        matrices = []
        for name, steps in grids:
            result["stage"] = name
            grid = {"name": name, "completed": False, "completed_columns": 0, "requested_log_steps": steps.tolist()}
            result["grids"].append(grid)
            event("grid_start", grid=name, columns=len(initial))
            matrix = np.empty((len(base[0]), len(initial)))
            actual_steps = []
            for column, step in enumerate(steps):
                displaced = log_initial.copy()
                displaced[column] += step
                actual_step = displaced[column] - log_initial[column]
                perturbed = evaluate(displaced)
                matrix[:, column] = (perturbed[0] - base[0]) / actual_step
                actual_steps.append(float(actual_step))
                grid["completed_columns"] += 1
                event("column", grid=name, column=column, component=names[column],
                      columns=len(initial), phase_callback_seconds=perturbed[3])
            base_after = evaluate(log_initial)
            base_record(base_after, name)
            grid.update(_matrix_report(matrix, names))
            constraint_error = matrix[:atom_count + face_count] - analytic
            expected_scale = np.r_[rows @ initial, np.zeros(len(base[0]) - atom_count)]
            scale_error = matrix @ np.ones(len(initial)) - expected_scale
            direction = np.linalg.lstsq(matrix, -base[0], rcond=None)[0]
            grid.update(completed=True, actual_log_steps=actual_steps,
                        reference_residual=base[0].tolist(),
                        analytic_atom_face_error_max_abs=_maximum(constraint_error),
                        analytic_atom_face_column_error_norms=np.linalg.norm(constraint_error, axis=0).tolist(),
                        common_scale_derivative_error=scale_error.tolist(),
                        common_scale_derivative_error_max_abs=_maximum(scale_error),
                        linear_newton_probe={"log_amount_direction": direction.tolist(),
                            "direction_norm": _finite(np.linalg.norm(direction)),
                            "direction_max_abs": _maximum(direction),
                            "predicted_residual_max_abs": _maximum(base[0] + matrix @ direction),
                            "scope": "Linear algebra only; this direction has not been evaluated or accepted."})
            matrices.append(matrix)
            base = base_after
            event("grid_complete", grid=name, rank=grid["rank"], condition_number=grid["condition_number"])
        for left, right in ((0, 1), (1, 2), (0, 2)):
            delta = matrices[left] - matrices[right]
            norms = np.linalg.norm(delta, axis=0)
            reference = np.linalg.norm(matrices[right], axis=0)
            result["comparisons"].append({"left": grids[left][0], "right": grids[right][0],
                "difference_max_abs": _maximum(delta), "difference_frobenius_norm": float(np.linalg.norm(delta)),
                "column_difference_norms": norms.tolist(),
                "column_relative_difference_norms": [_finite(a / b) if b > 0 else None for a, b in zip(norms, reference)],
                "relative_difference_frobenius_norm": _finite(np.linalg.norm(delta) / np.linalg.norm(matrices[right]))
                    if np.linalg.norm(matrices[right]) > 0 else None})
        result.update(completed=True, stage="complete")
        event("complete")
    except (ValueError, TypeError, AttributeError, RuntimeError, FloatingPointError, np.linalg.LinAlgError) as error:
        result["reason"] = type(error).__name__ + ": " + str(error)
        event("failure", reason=result["reason"])
    return result
