"""Bounded negative-witness searches using the unchanged extensive host model.

Search termination or nonnegative samples never certify a global minimum.
ExoEOS supplies every native energy; this module owns the chemical trials.
"""

from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np
from scipy.linalg import null_space
from scipy.optimize import minimize
from scipy.special import xlogy

from m2_host_stability import assess_host_candidates
from melts_coupled import COMMON_R, PROVIDER_MODEL_ID


def _search(objective, seeds, bounds, *, max_evaluations, tolerance_rt, simplex=False, initial_points=()):
    if type(max_evaluations) is not int or max_evaluations < 2:
        raise ValueError("max_evaluations must be an integer of at least two.")
    if isinstance(tolerance_rt, (bool, np.bool_)) or not np.isfinite(tolerance_rt) or tolerance_rt <= 0:
        raise ValueError("tolerance_rt must be positive and finite.")
    trials, failures, optimizers = [], [], []

    class BudgetExhausted(Exception):
        pass

    def evaluate(x, fresh=False):
        if not fresh and len(trials) + len(failures) >= max_evaluations:
            raise BudgetExhausted
        x = np.asarray(x, dtype=float)
        try:
            if (not np.all(np.isfinite(x)) or any(v < lo or v > hi for v, (lo, hi) in zip(x, bounds))
                    or (simplex and abs(x.sum() - 1) > 1e-7)):
                raise ValueError("Optimizer trial is outside the declared search domain.")
            row = objective(x)
            value = float(row["objective_rt"])
            if not np.isfinite(value):
                raise ValueError("The trial objective is not finite.")
            trials.append({"coordinates": x.tolist(), **row, "fresh_final": fresh})
            return value
        except (ValueError, RuntimeError) as error:
            failures.append({"coordinates": x.tolist(), "reason": str(error), "fresh_final": fresh})
            # This optimizer penalty is never retained as a thermodynamic value.
            return 1e30

    constraints = ({"type": "eq", "fun": lambda x: x.sum() - 1},) if simplex else ()
    exhausted = False
    initial_attempted = 0
    for point in initial_points:
        try:
            evaluate(point)
            initial_attempted += 1
        except BudgetExhausted:
            exhausted = True
            break
    for seed in seeds:
        try:
            result = minimize(evaluate, seed, method="SLSQP", bounds=bounds, constraints=constraints,
                              options={"ftol": 1e-11, "maxiter": max_evaluations})
            optimizers.append({"success": bool(result.success), "message": str(result.message)})
        except BudgetExhausted:
            exhausted = True
            break
    best = min(trials, key=lambda row: row["objective_rt"]) if trials else None
    final = None
    if best is not None:
        before = len(trials)
        evaluate(best["coordinates"], fresh=True)
        if len(trials) > before:
            final = trials[-1]
    negative = final is not None and final["objective_rt"] < -tolerance_rt
    return {"status": "negative_feasible_witness" if negative else "unresolved",
            "minimum_certified": False, "lower_bound_rt": None, "best_fresh_trial": final,
            "tolerance_rt": tolerance_rt, "max_evaluations": max_evaluations,
            "budget_exhausted": exhausted,
            "initial_points_requested": len(initial_points), "initial_points_attempted": initial_attempted,
            "trials": trials, "failed_evaluations": failures,
            "optimizer_attempts": optimizers,
            "interpretation": "A finite negative feasible trial rejects this host in the declared formal model. Nonnegative trials, optimizer success and search exhaustion do not bound the global minimum."}


def _provider_call(properties, evaluator, runtime, python_executable, **kwargs):
    result = evaluator.evaluate_liquid(
        properties["T_K"], properties["P_Pa"], properties["component_moles"],
        runtime=runtime, python_executable=python_executable, common_R=COMMON_R, **kwargs)
    if (result["T_K"] != properties["T_K"] or result["P_Pa"] != properties["P_Pa"]
            or result["model_id"] != PROVIDER_MODEL_ID or result["status"] != "ok_supplied_liquid_properties"
            or result["component_order"] != properties["component_order"]
            or result["phase_policy"]["oxygen_buffer"] != "None" or result["phase_policy"]["equilibrated"]
            or result["basis"]["common_R_J_mol_K"] != COMMON_R
            or not np.allclose(result["component_moles"], properties["component_moles"], rtol=5e-9, atol=0)
            or not np.allclose(result["returned_component_moles"], properties["component_moles"], rtol=5e-9, atol=0)):
        raise ValueError("The property provider changed the requested fixed host or model.")
    return result


def search_competing_solutions(properties, dissolved_h2_moles, *, evaluator, runtime, python_executable,
                               phases=None, max_evaluations=60, tolerance_rt=1e-8):
    """Search native endmember simplexes with the augmented-host H2 correction.

    A native solution may admit compositions beyond its nonnegative endmember
    simplex. This declared subset can supply a counterexample, never absence.
    """
    assess_host_candidates(properties, dissolved_h2_moles, tolerance_rt=tolerance_rt)
    catalog = properties["saturation"]["candidate_order"]
    phases = catalog if phases is None else list(phases)
    if not phases or len(set(phases)) != len(phases) or not set(phases) <= set(catalog):
        raise ValueError("Select unique phases from the complete native catalog.")
    results = []
    basis_provenance = []
    for phase in phases:
        # Each phase gets a fresh worker, so a native endpoint failure cannot
        # invalidate the remaining candidate catalog in the same C session.
        basis_receipt = _provider_call(properties, evaluator, runtime, python_executable,
                                      candidate_compositions=[{"phase": phase}])
        bases = basis_receipt["candidate_evaluations"]
        if len(bases) != 1 or bases[0]["phase"] != phase:
            raise ValueError("The provider changed the requested candidate order.")
        basis_provenance.append({"phase": phase, "provenance": basis_receipt["provenance"]})
        basis = bases[0]
        if "native_endmember_oxide_mass_g_per_mol" not in basis:
            results.append({"phase": phase, "status": "unresolved", "reason": basis.get("reason"),
                            "minimum_certified": False, "lower_bound_rt": None})
            continue
        matrix = np.asarray(basis["native_endmember_oxide_mass_g_per_mol"], dtype=float)
        if (matrix.ndim != 2 or matrix.shape[0] != len(properties["component_moles"])
                or matrix.shape[1] < 1 or not np.all(np.isfinite(matrix))):
            raise ValueError("Invalid native endmember basis.")
        count = matrix.shape[1]
        center = np.full(count, 1. / count)

        def objective(x):
            request = {"phase": phase, "oxide_mass_g": (matrix @ x).tolist()}
            receipt = _provider_call(properties, evaluator, runtime, python_executable,
                                     candidate_compositions=[request])
            candidate = receipt["candidate_evaluations"][0]
            if candidate["phase"] != phase or candidate.get("status") != "ok_candidate_properties":
                raise ValueError(candidate.get("reason", "Unavailable candidate properties."))
            if not np.allclose(candidate["oxide_mass_g"], request["oxide_mass_g"], rtol=5e-9, atol=1e-9):
                raise ValueError("The provider changed the candidate composition.")
            candidate = {"native_affinity_J": None, **candidate}
            trial_properties = {**properties, "saturation": {"candidate_order": [phase],
                                                            "candidates": [candidate], "equilibrated": False}}
            chemical = assess_host_candidates(trial_properties, dissolved_h2_moles,
                                              tolerance_rt=tolerance_rt)["candidates"][0]
            if "insertion_rt_per_mol_atoms" not in chemical:
                raise ValueError(chemical["reason"])
            return {"objective_rt": chemical["insertion_rt_per_mol_atoms"],
                    "chemical_trial": chemical, "provider_candidate": candidate}

        seeds = [center] + [.5 * (center + vertex) for vertex in np.eye(count)]
        # Evaluate independent geometrical probes before a local optimizer can
        # consume the entire budget estimating its first numerical gradient.
        probes = [center] if count == 1 else [center, *np.eye(count)]
        probes += [.5 * (np.eye(count)[i] + np.eye(count)[j])
                   for i in range(count) for j in range(i + 1, count)]
        result = _search(objective, [] if count == 1 else seeds, [(0., 1.)] * count, simplex=True,
                         max_evaluations=max_evaluations, tolerance_rt=tolerance_rt, initial_points=probes)
        complete = count == 1 and result["best_fresh_trial"] is not None and not result["failed_evaluations"]
        if complete and result["status"] == "unresolved":
            result["status"] = "nonnegative_fixed_composition"
        results.append({"phase": phase, "native_basis": basis,
                        "composition_domain_is_single_point": count == 1,
                        "composition_minimum_enumerated": complete,
                        "scope": "Fixed native endmember proportions only; numerical energies and empirical applicability are not interval certified." if count == 1 else
                                 "Nonnegative simplex probes and local optimization; signed-coordinate extensions remain outside this search.",
                        **result})
    return {"assessment_id": "m2_native_solution_composition_search_v1",
            "status": "rejected_by_feasible_trial" if any(r["status"] == "negative_feasible_witness" for r in results) else "unresolved",
            "global_stability_certified": False, "phase_order": phases, "phases": results,
            "domain": "Nonnegative native endmember fractions summing to one; a search subset, not a calibrated composition domain.",
            "provider_provenance": basis_provenance,
            "search_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}


def search_liquid_splitting(properties, dissolved_h2_moles, *, evaluator, runtime, python_executable,
                            max_evaluations=100, tolerance_rt=1e-8):
    """Try two finite liquids whose component amounts sum to the exact host.

    The shared linear dissolved-H2 standard cancels because both daughters
    use the same law and their total H2 is fixed. No source standard is refit.
    """
    assess_host_candidates(properties, dissolved_h2_moles, tolerance_rt=tolerance_rt)
    total = np.r_[np.asarray(properties["component_moles"], dtype=float), dissolved_h2_moles]
    active = total > 0
    rt = COMMON_R * properties["T_K"]

    def dilution(n, h):
        return float(xlogy(n, n / (n + h)) + xlogy(h, h / (n + h)))

    baseline = float(properties["gibbs_J"] / rt + dilution(total[:-1].sum(), total[-1]))
    atoms = np.asarray(properties["basis"]["component_element_matrix"]).T @ total[:-1]
    denominator = float(atoms.sum() + 2 * total[-1])

    def objective(fractions):
        first = np.zeros_like(total)
        first[active] = fractions * total[active]
        second = total - first
        daughters, energies = [], []
        for amounts in (first, second):
            request = {**properties, "component_moles": amounts[:-1].tolist()}
            receipt = _provider_call(request, evaluator, runtime, python_executable)
            energy = float(receipt["gibbs_J"] / rt + dilution(amounts[:-1].sum(), amounts[-1]))
            energies.append(energy)
            daughters.append({"native_component_moles": amounts[:-1].tolist(),
                              "dissolved_h2_moles": float(amounts[-1]), "augmented_gibbs_rt": energy,
                              "native_gibbs_J": receipt["gibbs_J"],
                              "returned_component_moles": receipt["returned_component_moles"]})
        return {"objective_rt": (sum(energies) - baseline) / denominator,
                "delta_gibbs_rt": sum(energies) - baseline, "daughters": daughters}

    center = np.full(int(active.sum()), .5)
    seeds = [center] + [.25 + .5 * vertex for vertex in np.eye(center.size)]
    result = _search(objective, seeds, [(.001, .999)] * center.size,
                     max_evaluations=max_evaluations, tolerance_rt=tolerance_rt)
    return {"assessment_id": "m2_two_liquid_negative_witness_search_v1", **result,
            "global_stability_certified": False, "parent_component_moles": total.tolist(),
            "baseline_augmented_gibbs_rt": baseline, "normalization_mol_atoms": denominator,
            "active_component_indices": np.flatnonzero(active).tolist(), "daughter_fraction_bounds": [.001, .999],
            "domain": "Finite daughter splits of every positive parent component; exact-zero parent components remain zero. This bounded search does not cover the full liquid-splitting domain.",
            "h2_standard": "The identical linear H2 standard cancels between parent and daughters; native H2O remains in MELTS.",
            "search_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}


def assess_liquid_local_curvature(properties, dissolved_h2_moles, *, evaluator, runtime,
                                  python_executable, relative_steps=(1e-3, 5e-4), tolerance_rt=1e-8):
    """Check local splitting curvature and independently evaluate finite splits.

    Chemical-potential differences estimate the Hessian on the positive
    support. Two step sizes and symmetry/homogeneity residuals diagnose
    numerical sensitivity, not rigorous derivative error bounds. Positive
    local curvature cannot exclude a distant, lower-energy second liquid.
    """
    assess_host_candidates(properties, dissolved_h2_moles, tolerance_rt=tolerance_rt)
    steps = np.asarray(relative_steps, dtype=float)
    if (steps.shape != (2,) or not np.all(np.isfinite(steps))
            or not 0 < steps[1] < steps[0] < .1):
        raise ValueError("Supply two decreasing positive relative steps below 0.1.")
    original = np.r_[properties["component_moles"], dissolved_h2_moles].astype(float)
    total = original / original.sum()
    active = total > 0
    root = np.sqrt(total[active])
    projection = null_space(root[None, :])
    rt = COMMON_R * properties["T_K"]
    receipts = []

    def evaluate(amounts):
        request = {**properties, "component_moles": amounts[:-1].tolist()}
        receipt = _provider_call(request, evaluator, runtime, python_executable)
        nhost, h2 = float(amounts[:-1].sum()), float(amounts[-1])
        mu = np.r_[np.asarray(receipt["mu_RT"], dtype=float) + np.log(nhost / (nhost + h2)),
                   np.log(h2 / (nhost + h2)) if h2 > 0 else np.nan]
        dilution = float(xlogy(nhost, nhost / (nhost + h2)) + xlogy(h2, h2 / (nhost + h2)))
        energy = float(receipt["gibbs_J"] / rt + dilution)
        if not np.isfinite(energy) or not np.all(np.isfinite(mu[active])):
            raise ValueError("The provider returned unavailable active-support potentials or energy.")
        receipts.append({"native_component_moles": amounts[:-1].tolist(), "dissolved_h2_moles": h2,
                         "augmented_gibbs_rt": energy, "provider_properties": receipt})
        return energy, mu[active]

    baseline, _ = evaluate(total)
    matrices, diagnostics = [], []
    for step in steps:
        columns = []
        for index in np.flatnonzero(active):
            delta = np.zeros_like(total)
            delta[index] = step * total[index]
            _, plus = evaluate(total + delta)
            _, minus = evaluate(total - delta)
            columns.append((plus - minus) / (2 * delta[index]))
        jacobian = np.asarray(columns).T
        scaled = root[:, None] * jacobian * root[None, :]
        symmetric = .5 * (scaled + scaled.T)
        projected = projection.T @ symmetric @ projection
        matrices.append(projected)
        diagnostics.append({"relative_step": float(step), "scaled_hessian": scaled.tolist(),
                            "projected_hessian": projected.tolist(),
                            "projected_eigenvalues": np.linalg.eigvalsh(projected).tolist(),
                            "symmetry_residual_norm": float(np.linalg.norm(scaled - scaled.T, ord=2)),
                            "homogeneous_direction_residual_norm": float(np.linalg.norm(scaled @ root))})
    variation = float(np.linalg.norm(matrices[0] - matrices[1], ord=2)) if projection.shape[1] else 0.
    margin = 5 * max([variation, 1e-10] + [row[key] for row in diagnostics
                     for key in ("symmetry_residual_norm", "homogeneous_direction_residual_norm")])
    eigvals, eigvecs = np.linalg.eigh(matrices[-1])
    splits = []
    atoms = np.asarray(properties["basis"]["component_element_matrix"]).T @ total[:-1]
    denominator = float(atoms.sum() + 2 * total[-1])
    if eigvals.size:
        direction = np.zeros_like(total)
        direction[active] = root * (projection @ eigvecs[:, 0])
        used = direction != 0
        limit = float(np.min(total[used] / np.abs(direction[used])))
        for fraction in (.001, .01, .1):
            first = .5 * (total + fraction * limit * direction)
            second = total - first
            before = len(receipts)
            energy_a, _ = evaluate(first)
            energy_b, _ = evaluate(second)
            splits.append({"fraction_of_feasible_step": fraction,
                           "objective_rt_per_mol_atoms": (energy_a + energy_b - baseline) / denominator,
                           "component_conservation_max_error_mol": float(np.max(np.abs(first + second - total))),
                           "daughter_evaluation_indices": [before, before + 1]})
    negative = any(row["objective_rt_per_mol_atoms"] < -tolerance_rt for row in splits)
    return {"assessment_id": "m2_liquid_local_curvature_v1",
            "status": "negative_feasible_witness" if negative else
                      "numerically_positive_local_curvature" if eigvals.size and eigvals[0] > margin else "unresolved",
            "global_stability_certified": False, "curvature_bound_certified": False,
            "normalization": "Parent native component plus dissolved-H2 amounts sum to one mol; energy differences per total mol atoms.",
            "parent_component_moles": total.tolist(), "original_component_moles": original.tolist(),
            "active_component_indices": np.flatnonzero(active).tolist(), "composition_subspace_dimension": int(projection.shape[1]),
            "finite_difference_assessments": diagnostics, "step_variation_norm": variation,
            "numerical_sensitivity_margin": margin, "tolerance_rt_per_mol_atoms": tolerance_rt,
            "normalization_mol_atoms": denominator, "baseline_augmented_gibbs_rt": baseline,
            "finite_split_trials": splits, "provider_evaluations": receipts,
            "interpretation": "A finite negative energy difference rejects the host within the formal model. Positive finite-difference curvature is local numerical evidence only; the sensitivity margin is not an interval bound. Exact-zero components are excluded, and a remote liquid minimum is still possible.",
            "search_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
