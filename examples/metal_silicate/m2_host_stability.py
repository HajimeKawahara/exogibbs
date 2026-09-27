"""Competing native MELTS phases at an unchanged supplied host composition.
========================================================================

This is a one-sided insertion assessment, not a phase-selection solver or
a certificate of a global minimum. Native properties belong to ExoEOS.
"""

from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np

from melts_coupled import COMMON_R, PROVIDER_MODEL_ID, PUBLISHED_MODEL_ID

WATER_MODEL_ID = "dry_melts_thompson2025_water_equivalent_v1"


def assess_host_candidates(properties: dict, dissolved_h2_moles: float, *,
                           candidate_properties=None, tolerance_rt: float = 1e-8,
                           helium_host_mu_rt=None) -> dict:
    """Evaluate candidate insertion into the formal ideal-H2 augmented host.

    All quantities use the same native amount scale. Candidate amounts have
    the provider's explicit oxide-mass basis. Signed host reaction coefficients
    are allowed, but each consumed component must be present. Potentials of
    absent components are not substituted by zero.
    """
    if (properties.get("model_id") not in (PROVIDER_MODEL_ID, PUBLISHED_MODEL_ID, WATER_MODEL_ID)
            or properties.get("status") != "ok_supplied_liquid_properties"):
        raise ValueError("Expected a declared supplied-liquid model.")
    candidates = properties if candidate_properties is None else candidate_properties
    if (candidates.get("model_id") != PROVIDER_MODEL_ID
            or candidates.get("status") != "ok_supplied_liquid_properties"):
        raise ValueError("Require a separate native candidate receipt for a published host.")
    for key in ("T_K", "P_Pa", "component_order"):
        if candidates[key] != properties[key]:
            raise ValueError("Native candidates and selected host differ in state or basis.")
    for key in ("component_moles", "oxide_molar_masses_g_mol"):
        if not np.array_equal(candidates[key], properties[key]):
            raise ValueError("Native candidates and selected host differ in amount or oxide basis.")
    for key in ("component_oxide_matrix", "component_element_matrix", "element_order", "common_R_J_mol_K"):
        if not np.array_equal(candidates["basis"][key], properties["basis"][key]):
            raise ValueError("Native candidates and selected host have different thermochemical bases.")
    if (candidates["phase_policy"]["oxygen_buffer"] != "None"
            or candidates["phase_policy"]["equilibrated"]):
        raise ValueError("The native candidate provider changed the phase policy.")
    if (not np.isfinite(dissolved_h2_moles) or dissolved_h2_moles < 0
            or not np.isfinite(tolerance_rt) or tolerance_rt <= 0):
        raise ValueError("Supply nonnegative dissolved H2 and a positive finite tolerance.")
    n = np.asarray(properties["component_moles"], dtype=float)
    mu = np.asarray(properties["mu_RT"], dtype=float)
    basis = properties["basis"]
    matrix = np.asarray(basis["component_oxide_matrix"], dtype=float)
    formula = np.asarray(basis["component_element_matrix"], dtype=float)
    masses = np.asarray(properties["oxide_molar_masses_g_mol"], dtype=float)
    saturation = candidates["saturation"]
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
    helium_mu = np.zeros_like(n) if helium_host_mu_rt is None else np.asarray(helium_host_mu_rt, dtype=float)
    if helium_mu.shape != n.shape or np.any(~np.isfinite(helium_mu)):
        raise ValueError("Require a finite He correction in the complete native host basis.")
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
                helium_cost = -float(coefficients @ helium_mu)
                insertion = (native_cost + dilution + helium_cost) / float(atoms.sum())
                row.update(
                    status="negative_insertion_trial" if insertion < -tolerance_rt else "nonnegative_insertion_trial",
                    reason=None, native_insertion_gibbs_rt=native_cost,
                    selected_host_insertion_gibbs_rt=native_cost + helium_cost,
                    h2_dilution_correction_gibbs_rt=dilution,
                    helium_dissolution_correction_gibbs_rt=helium_cost,
                    insertion_rt_per_mol_atoms=insertion,
                    maximum_feasible_trial_scale=float(np.min(n[consumed] / coefficients[consumed])),
                )
        results.append(row)
    negative = [row["phase"] for row in results if row["status"] == "negative_insertion_trial"]
    unresolved = [row["phase"] for row in results if row["status"] == "unresolved"]
    return {
        "assessment_id": "m2_native_host_competitor_insertion_v1",
        "host_model_id": properties["model_id"],
        "candidate_model_id": candidates["model_id"],
        "candidate_scope": "Native incipient compositions used as one-sided trials against the selected host potentials; not reoptimized for published mixing or H2 dilution.",
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
    tolerance_rt: float = 1e-8, candidate_evaluator=None,
    helium_dissolution=None, exoeos_checkout=None,
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
    helium = 0.
    has_helium = "He_dissolved" in record["phases"]["silicate"]
    if has_helium != (helium_dissolution is not None):
        raise ValueError("A dissolved-He source requires exactly its saved He recipe.")
    if has_helium and exoeos_checkout is None:
        raise ValueError("A dissolved-He assessment requires its explicit EOS checkout.")
    for name in record["phases"]["silicate"]:
        value = float(amounts[names.index(name)] * amount_scale)
        if name == "H2_dissolved":
            h2 = value
        elif name == "He_dissolved":
            if record["component_formulas"].get(name) != {"He": 1.}:
                raise ValueError("The dissolved-He formula must be explicit atomic He.")
            helium = value
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
    model_id = getattr(evaluator, "MODEL_ID", PROVIDER_MODEL_ID)
    if model_id not in (PROVIDER_MODEL_ID, PUBLISHED_MODEL_ID, WATER_MODEL_ID):
        raise ValueError("Unknown selected host model.")
    if model_id in (PUBLISHED_MODEL_ID, WATER_MODEL_ID) and candidate_evaluator is None:
        raise ValueError("Published host stability requires an explicit native candidate evaluator.")
    options = dict(runtime=runtime, python_executable=python_executable, common_R=COMMON_R)
    properties = evaluator.evaluate_liquid(
        temperature_k, pressure_bar * 1e5, host,
        **options, **({"include_saturation": True} if candidate_evaluator is None else {}))
    if properties["model_id"] != model_id:
        raise ValueError("The provider returned a different selected host model.")
    candidates = properties if candidate_evaluator is None else candidate_evaluator.evaluate_liquid(
        temperature_k, pressure_bar * 1e5, host, **options, include_saturation=True)
    if (properties["T_K"] != temperature_k or properties["P_Pa"] != pressure_bar * 1e5
            or tuple(properties["component_order"]) != tuple(evaluator.COMPONENTS)
            or not np.array_equal(properties["component_moles"], host)):
        raise ValueError("The property provider changed the requested state.")
    helium_mu = None
    helium_result = None
    if has_helium:
        from m2_helium import reconstruct_helium_model

        if (helium_dissolution["temperature_K"] != temperature_k
                or helium_dissolution["pressure_bar"] != pressure_bar
                or helium_dissolution["component_order"] != record["phases"]["silicate"]
                or helium_dissolution["host_component_order"] != record["phases"]["silicate"][:-1]):
            raise ValueError("The He receipt differs from the supplied source state or basis.")
        model = reconstruct_helium_model(exoeos_checkout, helium_dissolution)
        native_amounts = np.asarray([amounts[names.index(name)] * amount_scale
                                     for name in record["phases"]["silicate"]])
        helium_result = model.state(native_amounts)
        helium_mu = np.zeros_like(host)
        for i, name in enumerate(helium_dissolution["host_component_order"]):
            if name.endswith("_melts"):
                helium_mu[evaluator.COMPONENTS.index(name[:-6])] = helium_result["mu_rt"][i]
    assessment = assess_host_candidates(properties, h2, candidate_properties=candidates,
                                        tolerance_rt=tolerance_rt, helium_host_mu_rt=helium_mu)
    if has_helium:
        assessment["helium_dissolution"] = {
            "receipt": helium_dissolution, "native_dissolved_helium_moles": helium,
            "native_additional_gibbs_rt": helium_result["gibbs_rt"],
            "native_host_mu_correction_rt": helium_mu.tolist(),
            "helium_mu_rt": (float(helium_result["mu_rt"][-1])
                             if np.isfinite(helium_result["mu_rt"][-1]) else None),
            "helium_mu_endpoint": ("positive_infinity_at_fixed_zero_dry_mass"
                                   if np.isposinf(helium_result["mu_rt"][-1]) else
                                   "negative_infinity_at_zero_He_positive_dry_mass"
                                   if np.isneginf(helium_result["mu_rt"][-1]) else "finite"),
            "scope": "He-free native competitor against the selected host plus the EOS dry-mass He derivative; He is excluded from the H2 mixing denominator."}
    assessment.update(native_amount_scale=amount_scale, provider_properties=properties,
                      native_candidate_properties=candidates,
                      native_standard_state_receipts=getattr(evaluator, "standard_state_receipts", []),
                      assessment_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    return assessment
