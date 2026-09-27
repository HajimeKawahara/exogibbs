"""Saved-alloy insertion bound
===========================

Bound a saved source alloy against its fixed numerical elemental plane.

This supplementary proof leaves the saved phase selection and material flags
unchanged. It invokes no native MELTS worker and performs no new equilibrium.
"""

import argparse
import hashlib
import json
from pathlib import Path
import subprocess

import jax
import numpy as np

from hydrogen import _checkout_provenance
from m2_scenarios import normalize_scenario
from run_bse_common_gibbs import source_standards_rt


COMPONENTS = ("Fe_metal", "Si_metal", "O_metal", "H_metal")
ELEMENTS = ("Fe", "Si", "O", "H")
HERE = Path(__file__).resolve().parent


def sha256(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def saved_inputs(closure_path: Path, physical_path: Path) -> tuple:
    """Require the exact accepted root already named by a physical audit."""
    report = json.loads(Path(closure_path).read_text())
    audit = json.loads(Path(physical_path).read_text())
    provenance = audit["source"]
    if (provenance["kind"] != "global_closure_root"
            or provenance["sha256"] != sha256(closure_path)
            or not provenance["numerical_source_accepted"]
            or not provenance["source_contact_accepted"]):
        raise ValueError("Require a physical audit of this exact accepted pressure root.")
    if (provenance["executed_checkouts"] != report["checkouts"]
            or any(item["status"] for item in report["checkouts"].values())):
        raise ValueError("Require the same clean executed checkouts in both records.")
    selector = provenance["selection"]
    run = report["runs"][selector["run_index"]]
    state = run["roots"][selector["root_index"]]
    if (not report["numerically_accepted"] or not state["accepted"]
            or not state["global_closure_numerically_accepted"]
            or not state["pressure_closure_performed"]
            or not state["contact_accepted"] or not state["column_numerically_accepted"]):
        raise ValueError("A trial or unaccepted closure cannot supply a final-root proof.")
    source = state["source"]
    for owner in ("exogibbs", "exoeos"):
        origin = source["source_metadata"]["provenance"][owner]
        if origin["commit"] != report["checkouts"][owner]["head"] or origin["changed_tracked_file_sha256"]:
            raise ValueError("The source recipe must belong to the recorded clean provider checkout.")
    temperature, pressure = source["temperature_K"], source["pressure_bar"]
    if (selector["layers"] != len(state["layers"])
            or selector["temperature_k"] != temperature
            or selector["pressure_bar"] != pressure
            or state["temperature_base_k"] != temperature
            or state["pressure_base_pa"] != pressure * 1e5
            or audit["host_stability"]["temperature_K"] != temperature
            or audit["host_stability"]["pressure_Pa"] != state["pressure_base_pa"]):
        raise ValueError("The audit and selected root must have identical T/P and layers.")
    record = source["source_internal_record"]
    if tuple(record["phases"]["metal"]) != COMPONENTS:
        raise ValueError("Only the declared four-component source alloy is supported.")
    # Unit atomic columns select lambda exactly; a rounded matrix product
    # must not be mistaken for the exact external supporting plane.
    for name, element in zip(COMPONENTS, ELEMENTS):
        if record["component_formulas"][name] != {element: 1}:
            raise ValueError("The source alloy requires unit atomic formula columns.")
    basis = record["elements"]
    result = source["source_internal_result"]
    if not result["accepted"] or len(basis) != len(set(basis)):
        raise ValueError("Require an accepted source and a unique elemental basis.")
    potential = np.asarray(result["elemental_potentials_rt"], dtype=float)
    if potential.shape != (len(basis),) or not np.all(np.isfinite(potential)):
        raise ValueError("The saved elemental plane must be finite and complete.")
    plane = potential[[basis.index(element) for element in ELEMENTS]]
    selection = source["metal_selection"]
    domain = selection["metal_composition_domain"]
    composition = np.asarray(selection["metal_composition"], dtype=float)
    if (composition.shape != (4,) or np.any(composition <= 0)
            or not np.all(np.isfinite(composition)) or selection["metal_amount_mol"] <= 0
            or domain["component_order"] != list(COMPONENTS)
            or not np.array_equal(composition, domain["selected_metal"]["composition"])):
        raise ValueError("Require the saved positive selected alloy and its exact domain record.")
    names = [name for phase in record["phases"].values() for name in phase]
    amounts = np.asarray(result["component_amounts_mol"], dtype=float)
    if len(names) != len(set(names)) or amounts.shape != (len(names),):
        raise ValueError("Require a unique complete source component ledger.")
    metal = amounts[[names.index(name) for name in COMPONENTS]]
    if (not np.all(np.isfinite(metal)) or np.any(metal <= 0)
            or not np.array_equal(metal / metal.sum(), composition)):
        raise ValueError("The saved alloy composition must match the source primitive amounts.")
    scenario_record = report["provider_scenario"]
    scenario = normalize_scenario(None if scenario_record is None else scenario_record["values"])
    expected_scenario_sha = None if scenario_record is None else scenario_record["sha256"]
    if (normalize_scenario(source["source_metadata"].get("provider_scenario")) != scenario
            or source.get("provider_scenario_sha256") != expected_scenario_sha):
        raise ValueError("The closure and executed source must name the same scenario and bytes.")
    if (scenario["metal_bounds"]["lower"] != domain["lower_atomic_fractions"]
            or scenario["metal_bounds"]["upper"] != domain["upper_atomic_fractions"]
            or domain["effective_upper_atomic_fractions"] != domain["upper_atomic_fractions"]):
        raise ValueError("The source scenario and supported alloy domain must agree exactly.")
    return report, audit, state, scenario, plane, composition


def verified_recipe(source: dict, eos_checkout: Path) -> dict:
    """Bind the reconstructed binary64 standards to the executed recipe."""
    provenance = source["source_metadata"]["provenance"]
    hashes = provenance["file_sha256"]
    required = {"run_bse_common_gibbs.py", "source.py", "reference.json", "m2_scenarios.py"}
    if not required <= set(hashes):
        raise ValueError("The source does not record the required standard-state recipe.")
    verified = {}
    for name, expected in hashes.items():
        path = HERE / name
        if not path.is_file() or sha256(path) != expected:
            raise ValueError("The source recipe has changed: " + name)
        verified[str(path)] = expected
    if provenance["jax_version"] != jax.__version__ or provenance["numpy_version"] != np.__version__:
        raise ValueError("Reconstruct standards with the recorded JAX and NumPy versions.")
    revision = provenance["exoeos"]["commit"]
    for relative in ("src/exoeos/ma_fe_si_o.py", "src/exoeos/ma_fe_si_o_h.py", "src/exoeos/_arrays.py"):
        try:
            original = subprocess.check_output(
                ["git", "-C", str(eos_checkout), "show", revision + ":" + relative], stderr=subprocess.PIPE)
        except subprocess.CalledProcessError as error:
            raise ValueError("Fetch the recorded source ExoEOS commit before proving this recipe: " + revision) from error
        path = eos_checkout / relative
        if path.read_bytes() != original:
            raise ValueError("The executed alloy model has changed: " + relative)
        verified[str(path)] = sha256(path)
    return verified


def assess(closure_path: Path, physical_path: Path, eos_checkout: Path) -> dict:
    import exoeos
    from exoeos.ma_fe_si_o_h import MaFeSiOHLiquid
    from exoeos.ma_interval import ma_alloy_curvature_lower_bound, ma_alloy_insertion_lower_bound

    eos_checkout = Path(eos_checkout).resolve()
    if Path(exoeos.__file__).resolve().parent != eos_checkout / "src" / "exoeos":
        raise ValueError("Import the explicitly selected ExoEOS checkout.")
    if not jax.config.read("jax_enable_x64"):
        raise ValueError("Reconstruction and proof require JAX_ENABLE_X64=1.")
    report, audit, state, scenario, plane, composition = saved_inputs(closure_path, physical_path)
    source = state["source"]
    files = verified_recipe(source, eos_checkout)
    temperature, pressure = source["temperature_K"], source["pressure_bar"]
    model = MaFeSiOHLiquid()
    reference, _ = source_standards_rt(temperature, pressure)
    shift = np.asarray(model.standard_state_shift_RT(temperature))
    # This is exactly the source builder's binary64 base-standard recipe.
    # Scenario terms remain separate because the source adds them afterward.
    standards = np.array([reference[name] for name in COMPONENTS]) + shift
    offsets = np.array([scenario["standard_offsets_rt"].get(name, 0.) for name in COMPONENTS])
    extracted = audit["formal_reaction_standards"]["extraction"]
    if (extracted["alloy_component_order"] != list(ELEMENTS)
            or not np.array_equal(shift, extracted["alloy_standard_shift_rt"])
            or standards[0] != extracted["standards_rt"]["Fe_metal"]
            or standards[1] != extracted["standards_rt"]["Si_metal"]):
        raise ValueError("Reconstructed standards disagree with the recorded independent extraction.")
    lower, upper = (np.asarray(scenario["metal_bounds"][name]) for name in ("lower", "upper"))
    curvature = ma_alloy_curvature_lower_bound(model, temperature, lower, upper)
    bound = ma_alloy_insertion_lower_bound(model, temperature, lower, upper, composition[1:],
                                          standards, plane, standard_offsets_rt=offsets)
    files[str(eos_checkout / "src/exoeos/ma_interval.py")] = sha256(eos_checkout / "src/exoeos/ma_interval.py")
    files[str(Path(__file__).resolve())] = sha256(Path(__file__))
    return {
        "model_id": "saved_m2_alloy_fixed_plane_interval_bound_v1",
        "completed": True,
        "source": {"path": str(Path(closure_path).resolve()), "sha256": sha256(closure_path),
                   "selection": audit["source"]["selection"]},
        "physical_audit": {"path": str(Path(physical_path).resolve()), "sha256": sha256(physical_path)},
        "temperature_K": temperature, "pressure_bar": pressure, "component_order": list(ELEMENTS),
        "base_standard_potentials_rt": standards.tolist(), "linear_standard_offsets_rt": offsets.tolist(),
        "supporting_potentials_rt": plane.tolist(), "saved_selected_composition": composition.tolist(),
        "evaluation_point": {"solute_fractions": composition[1:].tolist(), "Fe": "exact 1-Si-O-H"},
        "lower_atomic_fractions": lower.tolist(), "upper_atomic_fractions": upper.tolist(),
        "interaction_K": np.asarray(model.dry_model.interaction_K).tolist(),
        "reduced_curvature_lower_bound_rt": curvature,
        "insertion_lower_bound_rt_per_mol_atoms": bound,
        "strict_nonnegative_bound": bool(bound >= 0),
        "source_metal_selection_status": source["metal_selection"]["status"],
        "empirical_material_certified": False,
        "scope": "Whole declared alloy domain against the fixed saved elemental plane. Reconstructed binary64 base standards and separately added scenario constants define the exact scalar; original Shomate arithmetic and empirical uncertainty are not interval-certified. Negative bounds and original source status are preserved. Other phases and coupled feasibility are not certified here.",
        "source_checkouts": report["checkouts"],
        "proof_checkouts": {"exogibbs": _checkout_provenance(HERE.parents[1]),
                            "exoeos": _checkout_provenance(eos_checkout)},
        "file_sha256": files,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--saved-closure", type=Path, required=True)
    parser.add_argument("--saved-physical-audit", type=Path, required=True)
    parser.add_argument("--exoeos-checkout", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    result = assess(args.saved_closure, args.saved_physical_audit, args.exoeos_checkout)
    with args.output.open("x") as stream:
        json.dump(result, stream, indent=2, allow_nan=False)
        stream.write("\n")
    print(json.dumps({"completed": result["completed"],
                      "insertion_lower_bound_rt_per_mol_atoms": result["insertion_lower_bound_rt_per_mol_atoms"],
                      "strict_nonnegative_bound": result["strict_nonnegative_bound"]}))


if __name__ == "__main__":
    main()
