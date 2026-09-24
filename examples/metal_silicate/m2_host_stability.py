"""Competing native MELTS phases at an unchanged supplied host composition.
========================================================================

This is a one-sided insertion assessment, not a phase-selection solver or
a certificate of a global minimum. Native properties belong to ExoEOS.
"""

from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np

from melts_coupled import COMMON_R, PROVIDER_MODEL_ID


def assess_host_candidates(properties: dict, dissolved_h2_moles: float, *, tolerance_rt: float = 1e-8) -> dict:
    """Evaluate candidate insertion into the formal ideal-H2 augmented host.

    All quantities use the same native amount scale. Candidate amounts have
    the provider's explicit oxide-mass basis. Signed host reaction coefficients
    are allowed, but each consumed component must be present. Potentials of
    absent components are not substituted by zero.
    """
    if properties.get("model_id") != PROVIDER_MODEL_ID or properties.get("status") != "ok_supplied_liquid_properties":
        raise ValueError("Expected the declared supplied-liquid MELTS model.")
    if (not np.isfinite(dissolved_h2_moles) or dissolved_h2_moles < 0
            or not np.isfinite(tolerance_rt) or tolerance_rt <= 0):
        raise ValueError("Supply nonnegative dissolved H2 and a positive finite tolerance.")
    n = np.asarray(properties["component_moles"], dtype=float)
    mu = np.asarray(properties["mu_RT"], dtype=float)
    basis = properties["basis"]
    matrix = np.asarray(basis["component_oxide_matrix"], dtype=float)
    formula = np.asarray(basis["component_element_matrix"], dtype=float)
    masses = np.asarray(properties["oxide_molar_masses_g_mol"], dtype=float)
    saturation = properties["saturation"]
    if (n.ndim != 1 or mu.shape != n.shape or matrix.shape != (n.size, n.size)
            or formula.shape[0] != n.size or masses.shape != n.shape
            or not np.all(np.isfinite(n)) or np.any(n < 0) or n.sum() <= 0
            or not np.all(np.isfinite(matrix)) or not np.all(np.isfinite(formula))
            or not np.all(np.isfinite(masses)) or np.any(masses <= 0)
            or basis["common_R_J_mol_K"] != COMMON_R
            or properties["phase_policy"]["oxygen_buffer"] != "None"
            or properties["phase_policy"]["equilibrated"]
            or saturation["equilibrated"]
            or not np.all(np.isfinite(mu[n > 0]))):
        raise ValueError("Invalid supplied-liquid basis or changed phase policy.")
    rows = saturation["candidates"]
    if [row["phase"] for row in rows] != saturation["candidate_order"] or len({row["phase"] for row in rows}) != len(rows):
        raise ValueError("The complete ordered native candidate catalog is required.")
    log_host_fraction = -float(np.log1p(dissolved_h2_moles / n.sum()))
    rt = COMMON_R * properties["T_K"]
    results = []
    for native in rows:
        name = native["phase"]
        row = {"phase": name, "native_affinity_J": native["native_affinity_J"],
               "role": "alternative_native_alloy" if name.startswith("alloy-") else
                       "alternative_native_water" if name == "water" else "competing_mineral",
               "status": "unresolved", "reason": native["reason"],
               "minimum_certified": False}
        if native["status"] == "ok_candidate_properties":
            oxides = np.asarray(native["oxide_mass_g"], dtype=float)
            coefficients = np.linalg.solve(matrix.T, oxides / masses)
            # Do not round a small coefficient to zero to make an insertion
            # feasible. Even a tiny consumption of an absent component is
            # forbidden, and creation needs its one-sided endpoint potential.
            used = coefficients != 0
            consumed = coefficients > 0
            atom_atol = 1e-12 * max(1., float(np.max(np.abs(coefficients))))
            atoms = formula.T @ coefficients
            row.update(host_component_coefficients_mol=coefficients.tolist(),
                       atomic_roundoff_tolerance_mol=atom_atol,
                       oxide_reconstruction_max_error_mol=float(np.max(np.abs(matrix.T @ coefficients - oxides / masses))),
                       element_amounts_mol=atoms.tolist(),
                       candidate_gibbs_rt=float(native["gibbs_J"] / rt))
            if np.any(atoms < -atom_atol) or not np.isfinite(atoms.sum()) or atoms.sum() <= 0:
                row["reason"] = "Invalid candidate atomic composition."
            elif np.any(consumed & (n == 0)):
                row["reason"] = "Insertion would consume an absent host component."
            elif np.any(used & ~np.isfinite(mu)):
                row["reason"] = "Insertion needs an unavailable host endpoint potential."
            elif not np.any(consumed):
                row["reason"] = "No finite host-consuming insertion direction."
            else:
                # G_new - G_host = G_candidate - c.mu_host to first order.
                # The ideal added H2 shifts every native host potential by
                # ln(N_host/(N_host + n_H2)); native water is already in MELTS.
                native_work = float(coefficients[used] @ mu[used])
                dilution = -float(coefficients[used].sum() * log_host_fraction)
                native_cost = float(native["gibbs_J"] / rt - native_work)
                insertion = (native_cost + dilution) / float(atoms.sum())
                row.update(
                    status="negative_insertion_trial" if insertion < -tolerance_rt else "nonnegative_insertion_trial",
                    reason=None, native_insertion_gibbs_rt=native_cost,
                    h2_dilution_correction_gibbs_rt=dilution,
                    insertion_rt_per_mol_atoms=insertion,
                    maximum_feasible_trial_scale=float(np.min(n[consumed] / coefficients[consumed])),
                )
        results.append(row)
    negative = [row["phase"] for row in results if row["status"] == "negative_insertion_trial"]
    unresolved = [row["phase"] for row in results if row["status"] == "unresolved"]
    return {
        "assessment_id": "m2_native_host_competitor_insertion_v1",
        "status": "rejected_by_feasible_trial" if negative else "unresolved",
        "global_stability_certified": False, "physical_liquid_domain_accepted": False,
        "temperature_K": properties["T_K"], "pressure_Pa": properties["P_Pa"],
        "component_order": properties["component_order"], "element_order": basis["element_order"],
        "native_host_component_amounts_mol": n.tolist(), "native_dissolved_h2_moles": dissolved_h2_moles,
        "log_native_host_fraction": log_host_fraction, "tolerance_rt_per_mol_atoms": tolerance_rt,
        "candidate_order": saturation["candidate_order"], "candidates": results,
        "negative_trial_phases": negative, "unresolved_trial_phases": unresolved,
        "remaining_requirements": [
            "A global minimum over each competing solution-phase composition and liquid splitting.",
            "Empirical applicability of the native mineral, water and alloy models at this composition and T/P.",
            "A common physical material domain for the augmented melt, selected alloy and gas.",
        ],
        "interpretation": "Negative feasible trials reject the declared host within the evaluated formal models. Nonnegative trials cannot certify absence; native incipient compositions are not reoptimized after adding H2 dilution. Native Fe-Ni alloys and water are alternatives to the separately modeled alloy/gas, not replacements for them.",
    }


def evaluate_host_stability(
    record: dict, component_amounts_mol, temperature_k: float, pressure_bar: float,
    *, evaluator, runtime: Path, python_executable: str, amount_scale: float = 1e-24,
    tolerance_rt: float = 1e-8,
) -> dict:
    """Reevaluate an unchanged source host using the selected ExoEOS provider."""
    names = [name for phase in record["phases"].values() for name in phase]
    amounts = np.asarray(component_amounts_mol, dtype=float)
    if (len(set(names)) != len(names) or amounts.shape != (len(names),)
            or not np.all(np.isfinite(amounts)) or np.any(amounts < 0)
            or not np.isfinite(amount_scale) or amount_scale <= 0):
        raise ValueError("Supply a complete finite nonnegative component ledger and positive amount scale.")
    host = np.zeros(len(evaluator.COMPONENTS))
    h2 = 0.
    for name in record["phases"]["silicate"]:
        value = float(amounts[names.index(name)] * amount_scale)
        if name == "H2_dissolved":
            h2 = value
        elif name.endswith("_melts") and name[:-6] in evaluator.COMPONENTS:
            index = evaluator.COMPONENTS.index(name[:-6])
            expected = {element: float(count) for element, count in zip(evaluator.ELEMENTS, evaluator.FORMULA_MATRIX[index]) if count}
            if record["component_formulas"][name] != expected:
                raise ValueError("Source host formula differs from the native MELTS basis.")
            host[index] = value
        else:
            raise ValueError(f"Unsupported supplied host component: {name}")
    if record["component_formulas"].get("H2_dissolved") != {"H": 2}:
        raise ValueError("The molecular dissolved-H2 formula must be explicit.")
    properties = evaluator.evaluate_liquid(
        temperature_k, pressure_bar * 1e5, host, runtime=runtime,
        python_executable=python_executable, common_R=COMMON_R, include_saturation=True,
    )
    if (properties["T_K"] != temperature_k or properties["P_Pa"] != pressure_bar * 1e5
            or tuple(properties["component_order"]) != tuple(evaluator.COMPONENTS)
            or not np.allclose(properties["component_moles"], host, rtol=5e-9, atol=0)):
        raise ValueError("The property provider changed the requested state.")
    assessment = assess_host_candidates(properties, h2, tolerance_rt=tolerance_rt)
    assessment.update(native_amount_scale=amount_scale, provider_properties=properties,
                      assessment_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    return assessment
