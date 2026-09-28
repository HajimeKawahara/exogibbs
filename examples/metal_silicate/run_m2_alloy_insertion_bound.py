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


def selected_source_state(report: dict, audit: dict, source_sha256: str, *,
                          allow_fixed_pressure=False) -> dict:
    """Select a bound source without promoting a fixed pressure to a root."""
    provenance = audit["source"]
    kind = provenance["kind"]
    if (kind not in ("global_closure_root", "fixed_pressure_internal_source")
            or provenance["sha256"] != source_sha256
            or provenance["numerical_source_accepted"] is not True
            or provenance["source_contact_accepted"] is not True
            or provenance["executed_checkouts"] != report["checkouts"]
            or any(item["status"] for item in report["checkouts"].values())):
        raise ValueError("Require the exact accepted source and clean executed provider records.")
    selector = provenance["selection"]
    if kind == "global_closure_root":
        state = report["runs"][selector["run_index"]]["roots"][selector["root_index"]]
        if (report["numerically_accepted"] is not True or state["accepted"] is not True
                or state["global_closure_numerically_accepted"] is not True
                or state["pressure_closure_performed"] is not True):
            raise ValueError("An unaccepted pressure trial cannot supply a final-root proof.")
    else:
        if not allow_fixed_pressure:
            raise ValueError("A fixed-pressure source requires explicit opt-in; it is not a pressure root.")
        state = report["source_state"]
        if (report.get("model_id") != "m2_fixed_pressure_internal_source_v1"
                or report.get("fixed_pressure_numerically_accepted") is not True
                or report.get("pressure_closure_performed") is not False
                or report.get("failure") or state.get("failure")
                or any(key in item and item[key] is not False for item in (report, state)
                       for key in ("accepted", "numerically_accepted", "global_closure_numerically_accepted",
                                   "pressure_closure_performed"))
                or "runs" in report or "roots" in report
                or "run_index" in selector or "root_index" in selector
                or type(report["arguments"]["nlayer"]) is not int or report["arguments"]["nlayer"] < 2
                or type(selector["layers"]) is not int
                or selector["layers"] != report["arguments"]["nlayer"]):
            raise ValueError("A fixed-pressure envelope must preserve its explicit non-root scope.")
        source = state["source"]
        record, result = source["source_internal_record"], source["source_internal_result"]
        elements = record["elements"]
        names = [name for group in record["phases"].values() for name in group]
        amounts = np.asarray(result["component_amounts_mol"], dtype=float)
        budget = np.asarray(report["inventory"]["total_element_amounts_mol"], dtype=float)
        matrix = np.asarray([[record["component_formulas"][name].get(element, 0.)
                              for name in names] for element in elements], dtype=float)
        if (result["accepted"] is not True or len(names) != len(set(names))
                or len(elements) != len(set(elements))
                or elements != report["inventory"]["elements"]
                or any(set(record["component_formulas"][name])-set(elements) for name in names)
                or amounts.shape != (len(names),) or budget.shape != (len(elements),)
                or np.any(~np.isfinite(amounts)) or np.any(amounts < 0)
                or np.any(~np.isfinite(budget)) or np.any(budget < 0)
                or not np.any(budget > 0) or np.any(~np.isfinite(matrix)) or np.any(matrix < 0)
                or np.any(np.abs(matrix @ amounts-budget) > 1e-9*budget)):
            raise ValueError("The fixed-pressure source primitives must close the exact inventory atoms.")
    if state["contact_accepted"] is not True or state["column_numerically_accepted"] is not True:
        raise ValueError("Require accepted source contact and column diagnostics at the declared pressure.")
    return state


def saved_inputs(closure_path: Path, physical_path: Path, *, allow_extended=False,
                 allow_fixed_pressure=False) -> tuple:
    """Require the exact accepted source already named by a physical audit."""
    report = json.loads(Path(closure_path).read_text())
    audit = json.loads(Path(physical_path).read_text())
    provenance = audit["source"]
    state = selected_source_state(report, audit, sha256(closure_path),
                                  allow_fixed_pressure=allow_fixed_pressure)
    selector = provenance["selection"]
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
    metadata = source["source_metadata"]
    kind = metadata.get("metal_model", "ma")
    components = COMPONENTS
    formulas = [{element: 1} for element in ELEMENTS]
    extended = None
    if allow_extended and kind in ("phosphorus", "associated", "associated_k", "associated_k_na"):
        extended = metadata["phosphorus_metal" if kind == "phosphorus" else "associated_metal"]
        components = tuple(name+"_metal" for name in extended["component_order"])
        formulas = ([{name: 1} for name in extended["component_order"]] if kind == "phosphorus"
                    else extended["component_formulas"])
    if tuple(record["phases"]["metal"]) != components:
        raise ValueError("Only an explicitly selected declared source alloy is supported.")
    if len(formulas) != len(components):
        raise ValueError("Every declared alloy component requires its complete atom column.")
    for name, formula in zip(components, formulas):
        if record["component_formulas"][name] != formula:
            raise ValueError("The source alloy atom columns differ from its declaration.")
    basis = record["elements"]
    result = source["source_internal_result"]
    if not result["accepted"] or len(basis) != len(set(basis)):
        raise ValueError("Require an accepted source and a unique elemental basis.")
    potential = np.asarray(result["elemental_potentials_rt"], dtype=float)
    if potential.shape != (len(basis),) or not np.all(np.isfinite(potential)):
        raise ValueError("The saved elemental plane must be finite and complete.")
    plane = potential.copy() if extended is not None else potential[[basis.index(element) for element in ELEMENTS]]
    selection = source["metal_selection"]
    domain = selection["metal_composition_domain"]
    composition = np.asarray(selection["metal_composition"], dtype=float)
    if (composition.shape != (len(components),) or np.any(composition <= 0)
            or not np.all(np.isfinite(composition)) or selection["metal_amount_mol"] <= 0
            or domain["component_order"] != list(components)
            or not np.array_equal(composition, domain["selected_metal"]["composition"])):
        raise ValueError("Require the saved positive selected alloy and its exact domain record.")
    names = [name for phase in record["phases"].values() for name in phase]
    amounts = np.asarray(result["component_amounts_mol"], dtype=float)
    if len(names) != len(set(names)) or amounts.shape != (len(names),):
        raise ValueError("Require a unique complete source component ledger.")
    metal = amounts[[names.index(name) for name in components]]
    if (not np.all(np.isfinite(metal)) or np.any(metal <= 0)
            or not np.array_equal(metal / metal.sum(), composition)):
        raise ValueError("The saved alloy composition must match the source primitive amounts.")
    scenario_record = report["provider_scenario"]
    scenario = normalize_scenario(None if scenario_record is None else scenario_record["values"])
    expected_scenario_sha = None if scenario_record is None else scenario_record["sha256"]
    if (normalize_scenario(source["source_metadata"].get("provider_scenario")) != scenario
            or source.get("provider_scenario_sha256") != expected_scenario_sha):
        raise ValueError("The closure and executed source must name the same scenario and bytes.")
    expected_lower = scenario["metal_bounds"]["lower"] if extended is None else extended[
        "lower_atomic_fractions" if kind == "phosphorus" else "lower_species_fractions"]
    expected_upper = scenario["metal_bounds"]["upper"] if extended is None else extended[
        "upper_atomic_fractions" if kind == "phosphorus" else "upper_species_fractions"]
    if (expected_lower != domain["lower_atomic_fractions"]
            or expected_upper != domain["upper_atomic_fractions"]
            or domain["effective_upper_atomic_fractions"] != domain["upper_atomic_fractions"]):
        raise ValueError("The source scenario and supported alloy domain must agree exactly.")
    return report, audit, state, scenario, plane, composition


def verified_recipe(source: dict, eos_checkout: Path, *, allow_historical_builders=False, receipt=None) -> dict:
    """Bind the reconstructed binary64 standards to the executed recipe."""
    provenance = source["source_metadata"]["provenance"]
    hashes = provenance["file_sha256"]
    required = {"run_bse_common_gibbs.py", "source.py", "reference.json", "m2_scenarios.py"}
    if not required <= set(hashes):
        raise ValueError("The source does not record the required standard-state recipe.")
    verified = {}
    historical = {}
    current = {}
    builders = {'melts_coupled.py','m2_expanded_source.py','m2_scenarios.py','phase_selection.py'}
    gibbs_revision = provenance['exogibbs']['commit']
    for name, expected in hashes.items():
        path = HERE / name
        if allow_historical_builders:
            try:
                original = subprocess.check_output(['git','-C',str(HERE.parents[1]),'show',
                    gibbs_revision+':examples/metal_silicate/'+name],stderr=subprocess.PIPE)
            except subprocess.CalledProcessError as error:
                raise ValueError('Fetch the executed source recipe commit before verification: '+gibbs_revision) from error
            if hashlib.sha256(original).hexdigest() != expected:
                raise ValueError('The saved recipe hash differs from its executed git blob: '+name)
            historical[name] = expected
        digest = sha256(path) if path.is_file() else None
        if digest is None or (digest != expected and not (allow_historical_builders and name in builders)):
            raise ValueError("The source recipe has changed: " + name)
        verified[str(path)] = digest
        current[name] = digest
    if receipt is not None:
        receipt.update(executed_gibbs_commit=gibbs_revision,
            historical_builder_policy=bool(allow_historical_builders),
            executed_git_blob_sha256=historical,proof_checkout_file_sha256=current,
            changed_nonreplayed_builder_files=[name for name in hashes if current[name] != hashes[name]],
            scope='All original recipe blobs must match their executed commit. Only explicitly listed construction/selection files may differ; source-building control flow is not replayed. Explicit saved scenario values and domains are rebound independently, and replayed thermochemical standards/data remain byte-identical. Saved constitutive expressions, phase gradients and primal energies have separate binding checks.')
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


def assess(closure_path: Path, physical_path: Path, eos_checkout: Path, *, allow_historical_builders=False) -> dict:
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
    recipe_receipt = {}
    files = verified_recipe(source, eos_checkout,allow_historical_builders=allow_historical_builders,receipt=recipe_receipt)
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
    if any(sha256(Path(path)) != digest for path,digest in files.items()):
        raise ValueError('A replayed proof recipe changed during verification.')
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
        "source_recipe_verification":recipe_receipt,
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
    parser.add_argument("--allow-historical-builders", action="store_true",
                        help="Verify original builder git blobs while retaining byte-identical thermochemical replay files.")
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    result = assess(args.saved_closure, args.saved_physical_audit, args.exoeos_checkout,
                    allow_historical_builders=args.allow_historical_builders)
    with args.output.open("x") as stream:
        json.dump(result, stream, indent=2, allow_nan=False)
        stream.write("\n")
    print(json.dumps({"completed": result["completed"],
                      "insertion_lower_bound_rt_per_mol_atoms": result["insertion_lower_bound_rt_per_mol_atoms"],
                      "strict_nonnegative_bound": result["strict_nonnegative_bound"]}))


if __name__ == "__main__":
    main()
