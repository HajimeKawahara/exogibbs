"""M2 common-standard and hydrogen concentration audit
===================================================

Compare independent thermochemical tables without changing their standards.
Numerical audit completion does not establish material calibration.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import jax
import numpy as np

from hydrogen import H2_CALIBRATION, _checkout_provenance
from melts_coupled import COMMON_R, load_melts_evaluator
from m1_chemistry import build_setups, provenance as upper_provenance
from run_bse_common_gibbs import json_value, source_standards_rt
from m2_common_gas import SHARED_SPECIES, UPPER_SPECIES


REACTIONS = {
    "H2 -> 2 H": {"H2": -1, "H": 2},
    "H2 + 0.5 O2 -> H2O": {"H2": -1, "O2": -.5, "H2O": 1},
    "0.5 H2 + 0.5 O2 -> OH": {"H2": -.5, "O2": -.5, "OH": 1},
    "SiO + 2 H2 -> SiH4 + 0.5 O2": {"SiO": -1, "H2": -2, "SiH4": 1, "O2": .5},
}


def audit_shared_standards(formula_matrix, lower_standard_rt, upper_standard_rt,
                           *, species, elements, tolerance=1e-8):
    """Remove only a common elemental gauge from aligned species standards.

    The matrix has element rows and species columns; all potentials use the
    same R and standard pressure. An elemental reference change preserves
    every balanced reaction. A nonzero remaining component cannot be removed
    by changing the energy reference. No supplied standard is modified.
    """
    matrix = np.asarray(formula_matrix, dtype=float)
    lower, upper = np.asarray(lower_standard_rt, dtype=float), np.asarray(upper_standard_rt, dtype=float)
    if (not len(species) or len(set(species)) != len(species) or len(set(elements)) != len(elements)
            or matrix.shape != (len(elements), len(species)) or lower.shape != (len(species),)
            or upper.shape != lower.shape or not np.all(np.isfinite(matrix))
            or not np.all(np.isfinite(lower)) or not np.all(np.isfinite(upper))
            or not np.isfinite(tolerance) or tolerance <= 0):
        raise ValueError("Supply finite aligned element/species arrays and a positive tolerance.")
    difference = lower - upper
    gauge, _, rank, _ = np.linalg.lstsq(matrix.T, difference, rcond=None)
    residual = difference - matrix.T @ gauge
    maximum = float(np.max(np.abs(residual), initial=0.))
    return {"accepted": bool(maximum <= tolerance),
            "species": list(species), "elements": list(elements), "formula_matrix": matrix.tolist(),
            "lower_standard_rt": lower.tolist(), "upper_standard_rt": upper.tolist(),
            "difference_rt": difference.tolist(), "element_gauge_rt": gauge.tolist(),
            "gauge_rank": int(rank), "gauge_unique": bool(rank == len(elements)),
            "nongauge_residual_rt": residual.tolist(),
            "maximum_residual_rt": maximum, "tolerance_rt": tolerance,
            "reasons": ["Independent reaction standards differ beyond the declared tolerance."] if maximum > tolerance else []}


def audit_contact(lower_partial_pressures_bar, upper_partial_pressures_bar,
                  *, species, tolerance=1e-8):
    """Compare aligned partial pressures; this does not certify common standards.

    Both exact-zero values describe a common absent species. A zero on only
    one side fails explicitly, without floors or logarithms of zero.
    """
    lower, upper = np.asarray(lower_partial_pressures_bar), np.asarray(upper_partial_pressures_bar)
    if (not len(species) or len(set(species)) != len(species)
            or lower.shape != (len(species),) or upper.shape != lower.shape
            or not np.all(np.isfinite(lower)) or not np.all(np.isfinite(upper))
            or np.any(lower < 0) or np.any(upper < 0)
            or not np.isfinite(tolerance) or tolerance <= 0):
        raise ValueError("Supply finite nonnegative aligned pressures and a positive tolerance.")
    supported = (lower > 0) & (upper > 0)
    mismatch = (lower > 0) != (upper > 0)
    residual = np.zeros(len(species))
    residual[supported] = np.log(lower[supported]) - np.log(upper[supported])
    maximum = None if np.any(mismatch) else float(np.max(np.abs(residual), initial=0.))
    return {"accepted": bool(maximum is not None and maximum <= tolerance),
            "species": list(species), "lower_partial_pressures_bar": lower.tolist(),
            "upper_partial_pressures_bar": upper.tolist(),
            "log_pressure_residual": [None if bad else float(value) for value, bad in zip(residual, mismatch)],
            "support_mismatch_species": [name for name, bad in zip(species, mismatch) if bad],
            "maximum_residual": maximum, "tolerance": tolerance,
            "scope": "Shared partial pressures only; standards, additional species and condensates need separate audits."}


def h2_mass_fraction_to_amount(mass_fraction, host_mass_kg, host_amount_mol, h2_mass_kg_mol):
    """Convert H2 mass / total liquid mass to mol and endmember mole fraction.

    This is an amount conversion, not a solubility calibration or a new
    scalar-energy law. The host excludes the added H2 but includes native
    dissolved water, if present. Its amount follows the stated endmembers.
    """
    values = np.asarray([mass_fraction, host_mass_kg, host_amount_mol, h2_mass_kg_mol])
    if (not np.all(np.isfinite(values)) or not 0 <= mass_fraction < 1
            or np.any(values[1:] <= 0)):
        raise ValueError("Require 0 <= mass_fraction < 1 and positive finite host amounts and H2 molar mass.")
    hydrogen = mass_fraction / (1 - mass_fraction) * host_mass_kg / h2_mass_kg_mol
    return float(hydrogen), float(hydrogen / (host_amount_mol + hydrogen))


def gas_standard_audit(temperature_k):
    """Compare BSE's actual gas standard construction with the existing M1 table."""
    if not np.isfinite(temperature_k) or temperature_k <= 0:
        raise ValueError("Temperature must be positive and finite.")
    lower, _ = source_standards_rt(temperature_k)
    _, upper = build_setups()
    columns = [upper.gas_species.index(name) for name in UPPER_SPECIES]
    matrix = np.asarray(upper.gas_setup.formula_matrix)[:, columns]
    lower_h = np.array([lower[name + "_gas"] for name in SHARED_SPECIES])
    upper_h = np.asarray(upper.gas_setup.hvector_func(temperature_k))[columns]
    result = audit_shared_standards(matrix, lower_h, upper_h, species=SHARED_SPECIES,
                                    elements=upper.gas_setup.elements)
    cycles = []
    for name, coefficients in REACTIONS.items():
        row = np.array([coefficients.get(species, 0.) for species in SHARED_SPECIES])
        imbalance = matrix @ row
        if np.any(imbalance != 0):
            raise ValueError("An independent gas reaction does not conserve atoms.")
        cycles.append({"reaction": name, "stoichiometry": row.tolist(),
                       "element_imbalance": imbalance.tolist(),
                       "lower_delta_g_rt": float(row @ lower_h), "upper_delta_g_rt": float(row @ upper_h),
                       "difference_rt": float(row @ (lower_h - upper_h))})
    result.update(temperature_K=temperature_k, pressure_standard_bar=1., independent_reactions=cycles,
                  independent_reaction_rank=int(np.linalg.matrix_rank([item["stoichiometry"] for item in cycles])))
    return result


def native_standard_audit(inventory_path, checkout, runtime, python_executable, temperature_k):
    """Evaluate dry BSE once; compare named native standards with source data.

    Virtual sums of pure-oxide standards are explicitly distinguished from
    actual MELTS endmembers. Their differences are not assumed to be errors.
    """
    inventory = json.loads(Path(inventory_path).read_text())
    evaluator = load_melts_evaluator(checkout)
    oxide = {name.lower(): amount for name, amount in zip(inventory["oxide_order"], inventory["oxide_amounts_mol"])}
    host = np.linalg.solve(evaluator.NU.T, [oxide.get(name, 0.) for name in evaluator.OXIDES])
    scale = .1 / inventory["dry_rock_mass_kg"]
    state = evaluator.evaluate_liquid(temperature_k, 1e5, host * scale, runtime=runtime,
                                      common_R=COMMON_R, python_executable=python_executable)
    source, formulas = source_standards_rt(temperature_k)
    substitutions = {"sio2": {"SiO2_silicate": 1},
                     "fe2sio4": {"FeO_silicate": 2, "SiO2_silicate": 1},
                     "mg2sio4": {"MgO_silicate": 2, "SiO2_silicate": 1},
                     "na2sio3": {"Na2SiO3_silicate": 1}}
    products = {"sio2": {"SiO_gas": 1, "O2_gas": .5},
                "fe2sio4": {"Fe_gas": 2, "SiO_gas": 1, "O2_gas": 1.5},
                "mg2sio4": {"Mg_gas": 2, "SiO_gas": 1, "O2_gas": 1.5},
                "na2sio3": {"Na_gas": 2, "SiO_gas": 1, "O2_gas": 1}}
    rows, reactions = [], []
    for component, coefficients in substitutions.items():
        reference = sum(source[name] * value for name, value in coefficients.items())
        native = state["mu0_RT"][evaluator.COMPONENTS.index(component)]
        rows.append({"melts_component": component, "source_combination": coefficients,
                     "source_combination_mu0_rt": reference, "native_mu0_rt": native,
                     "difference_rt": native - reference,
                     "interpretation": "Different model data; formal oxide sum is not a measured standard of this MELTS endmember."})
        gas = products[component]
        atoms = np.array([sum(value * formulas[name].get(element, 0) for name, value in gas.items())
                          for element in evaluator.ELEMENTS])
        imbalance = atoms - evaluator.FORMULA_MATRIX[evaluator.COMPONENTS.index(component)]
        if np.any(imbalance != 0):
            raise ValueError("Native endmember vaporization reaction does not conserve atoms.")
        gas_g = sum(value * source[name] for name, value in gas.items())
        reactions.append({"reactant_melts_component": component, "gas_products": gas,
                          "element_order": list(evaluator.ELEMENTS), "element_imbalance": imbalance.tolist(),
                          "source_virtual_reactant_delta_g_rt": gas_g - reference,
                          "native_reactant_delta_g_rt": gas_g - native,
                          "difference_rt": reference - native})
    h2_mass = 2 * inventory["atomic_masses_kg_mol"][inventory["elements"].index("H")]
    conversions = []
    for ppm in (10., 100., 1000.):
        mass_fraction = ppm * 1e-6
        amount, fraction = h2_mass_fraction_to_amount(mass_fraction, .1, float(host.sum() * scale), h2_mass)
        restored = amount * h2_mass / (.1 + amount * h2_mass)
        conversions.append({"diagnostic_H2_mass_ppm": ppm, "h2_amount_mol": amount,
                            "h2_endmember_mole_fraction": fraction,
                            "reconstructed_mass_fraction": restored,
                            "roundtrip_error": abs(restored - mass_fraction)})
    import exoeos
    from exoeos import MaFeSiOHLiquid

    if Path(exoeos.__file__).resolve().parent != Path(checkout).resolve() / "src/exoeos":
        raise ValueError("Set PYTHONPATH to the selected ExoEOS checkout.")

    model = MaFeSiOHLiquid()
    shift = np.asarray(model.standard_state_shift_RT(temperature_k))
    alloy_source = np.array([source[name + "_metal"] for name in model.components])
    return {"native_evaluations": 1, "native_state": state, "standard_comparisons": rows,
            "independent_vaporization_reactions": reactions,
            "alloy_standards": {"component_order": list(model.components),
                                "source_standard_rt": alloy_source.tolist(),
                                "native_convention_shift_rt": shift.tolist(),
                                "applied_standard_rt": (alloy_source + shift).tolist(),
                                "independent_calibration_accepted": False,
                                "reason": "Native convention shifts preserve declared source anchors; they do not independently calibrate MELTS exchange or alloy H."},
            "cross_phase_standards_accepted": False,
            "reason": "No independent common reaction calibration or phase-specific reference conversion establishes alignment.",
            "inventory_sha256": hashlib.sha256(Path(inventory_path).read_bytes()).hexdigest(),
            "hydrogen_basis": {"dry_host_mass_kg": .1, "dry_host_amount_mol": float(host.sum() * scale),
                               "host_mean_component_mass_kg_mol": .1 / float(host.sum() * scale),
                               "h2_mass_kg_mol": h2_mass, "diagnostic_conversions": conversions,
                               "scope": "Exact amount conversion only; diagnostic ppm values are not experimental observations."}}


def run_audit(*, inventory_path=None, exoeos_checkout=None, runtime=None, python_executable=None):
    """Return reproducible gas comparisons and optional fresh native evidence."""
    if not jax.config.jax_enable_x64:
        raise RuntimeError("Set JAX_ENABLE_X64=1 for the declared audit tolerance.")
    gases = [gas_standard_audit(temperature) for temperature in (2173.15, 2350.)]
    native = None
    native_inputs = (inventory_path, exoeos_checkout, runtime, python_executable)
    if any(value is not None for value in native_inputs):
        if any(value is None for value in native_inputs):
            raise ValueError("Native audit requires the selected ExoEOS checkout, runtime and worker Python.")
        native = native_standard_audit(inventory_path, exoeos_checkout, runtime, python_executable, 2173.15)
    directory = Path(__file__).resolve().parent
    files = ("m2_standards_audit.py", "run_bse_common_gibbs.py", "source.py", "hydrogen.py",
             "reference.json", "reduced_gas_reference.json", "m1_chemistry.py")
    return {"model_id": "m2_independent_standards_hydrogen_audit_v1",
            "numerical_audit_completed": all(item["independent_reaction_rank"] == 4 for item in gases),
            "scientific_acceptance": {"M2_A": "pending", "M2_B": "pending"},
            "gas_standard_comparisons": gases, "native_standard_comparison": native,
            "h2_calibration": H2_CALIBRATION,
            "temperature_policy": {"source_branch_K": 2350., "BSE_evaluation_K": 2173.15,
                                   "source_MgO_liquid_fit_K": [3105., 5000.],
                                   "common_calibrated_domain": None,
                                   "reason": "Fixed source branch and BSE point are not a validated joint material domain."},
            "provenance": {"exogibbs": _checkout_provenance(directory.parents[1]),
                           "upper_thermochemistry": upper_provenance(),
                           "file_sha256": {name: hashlib.sha256((directory / name).read_bytes()).hexdigest() for name in files}}}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inventory", type=Path)
    parser.add_argument("--exoeos-checkout", type=Path)
    parser.add_argument("--runtime", type=Path)
    parser.add_argument("--python")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error("Preserve previous audit records; output already exists.")
    result = run_audit(inventory_path=args.inventory, exoeos_checkout=args.exoeos_checkout,
                       runtime=args.runtime, python_executable=args.python)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x") as stream:
        stream.write(json.dumps(json_value(result), indent=2, allow_nan=False) + "\n")
    if not result["numerical_audit_completed"]:
        raise SystemExit("The requested numerical audit is incomplete.")


if __name__ == "__main__":
    main()
