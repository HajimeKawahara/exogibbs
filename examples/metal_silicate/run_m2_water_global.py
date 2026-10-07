"""Bound reconstructed water and arbitrary H2 against a saved common plane."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import time

import numpy as np

from m2_common_plane import _I
from m2_host_standards import source_hydrogen_standard_receipt
from m2_scenarios import normalize_scenario
from m2_water_global import certify_water_common_plane, parameters_from_saved_water
from run_m2_alloy_insertion_bound import selected_source_state


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def saved_scenario_values(scenario, source):
    """Bind original scenario bytes and compare explicitly normalized values."""
    values = normalize_scenario(None if scenario is None else scenario["values"])
    expected_sha = None if scenario is None else scenario["sha256"]
    if (values != normalize_scenario(source["source_metadata"].get("provider_scenario"))
            or expected_sha != source.get("provider_scenario_sha256")):
        raise ValueError("The root and executed source name different scenario values or bytes.")
    return values


def saved_water_problem(report: dict, audit: dict, closure_sha256: str, *, exoeos_checkout=None,
                        allow_fixed_pressure=False) -> tuple:
    """Bind an accepted source; fixed pressure requires explicit separate scope."""
    state = selected_source_state(report, audit, closure_sha256,
                                  allow_fixed_pressure=allow_fixed_pressure)
    provenance = audit["source"]
    selector = provenance["selection"]
    source = state["source"]
    metadata = source["source_metadata"]
    scenario = saved_scenario_values(report["provider_scenario"], source)
    for owner in ("exogibbs", "exoeos"):
        origin = metadata["provenance"][owner]
        if origin["commit"] != report["checkouts"][owner]["head"] or origin["changed_tracked_file_sha256"]:
            raise ValueError("The executed source and root provider pins differ.")
    host = audit["host_stability"]
    properties = host["provider_properties"]
    temperature, pressure = source["temperature_K"], source["pressure_bar"]
    if (selector["layers"] != len(state["layers"]) or selector["temperature_k"] != temperature
            or selector["pressure_bar"] != pressure or state["temperature_base_k"] != temperature
            or state["pressure_base_pa"] != pressure*1e5 or host["temperature_K"] != temperature
            or host["pressure_Pa"] != pressure*1e5 or properties["T_K"] != temperature
            or properties["P_Pa"] != pressure*1e5 or metadata["liquid_model"] != "published_water"
            or properties["model_id"] != metadata["host_ledger"]["model_id"]):
        raise ValueError("The saved root, selected host model and audited T/P must agree exactly.")
    record, result = source["source_internal_record"], source["source_internal_result"]
    elements = record["elements"]
    names = [name for phase in record["phases"].values() for name in phase]
    amounts = np.asarray(result["component_amounts_mol"], dtype=float)
    budget = metadata["input"]["element_amounts_mol"]
    potentials = result["elemental_potentials_rt"]
    if (not result["accepted"] or len(elements) != len(set(elements)) or len(names) != len(set(names))
            or amounts.shape != (len(names),) or np.any(~np.isfinite(amounts)) or np.any(amounts < 0)
            or len(budget) != len(elements) or len(potentials) != len(elements)
            or np.any(~np.isfinite(budget)) or np.any(np.asarray(budget) < 0) or np.any(~np.isfinite(potentials))
            or report["inventory"]["elements"] != elements or report["inventory"]["total_element_amounts_mol"] != budget):
        raise ValueError("A complete accepted finite source and exact global inventory are required.")
    scale = metadata["input"]["native_amount_scale"]
    if not np.isfinite(scale) or scale <= 0 or host["native_amount_scale"] != scale:
        raise ValueError("The source and audit amount scales differ.")
    component_names = properties["component_order"]
    actual = [0.]*len(component_names)
    dissolved = 0.
    has_helium = 'He_dissolved' in record['phases']['silicate']
    if has_helium != ('helium_dissolution' in metadata) or has_helium != ('helium_dissolution' in host):
        raise ValueError('Source and audit must declare the same dissolved-He scalar.')
    for name in record["phases"]["silicate"]:
        value = float(amounts[names.index(name)]*scale)
        if name == "H2_dissolved":
            if record["component_formulas"][name] != {"H": 2}:
                raise ValueError("Require the molecular dissolved-H2 atom column.")
            dissolved = value
        elif name == 'He_dissolved':
            if record['component_formulas'][name] != {'He':1.}:
                raise ValueError('The dissolved-He component must preserve atomic He.')
        else:
            if not name.endswith("_melts") or name[:-6] not in component_names:
                raise ValueError("Unknown primitive host component.")
            i = component_names.index(name[:-6])
            expected = {e: count for e, count in zip(properties["basis"]["element_order"],
                         properties["basis"]["component_element_matrix"][i]) if count}
            if record["component_formulas"][name] != expected:
                raise ValueError("The source and provider host atom columns differ.")
            actual[i] = value
    if (actual != properties["component_moles"] or actual != host["native_host_component_amounts_mol"]
            or dissolved != host["native_dissolved_h2_moles"]):
        raise ValueError("The audited host amounts differ from the exact saved source.")
    water = properties["water_reconstruction"]
    matches = [row for row in metadata["host_ledger"]["water_standard_receipts"]
               if row["T_K"] == temperature and row["P_Pa"] == pressure*1e5]
    if (not matches or any(row["H2O_gas_standard_RT"] != water["gas_H2O_standard_RT"]
            or row["common_R_J_mol_K"] != properties["basis"]["common_R_J_mol_K"]
            or row["gas_standard_pressure_Pa"] != 1e5 for row in matches)):
        raise ValueError("The audited water gas anchor differs from the executed source receipt.")
    hydrogen = source_hydrogen_standard_receipt(source)
    if audit["dissolved_hydrogen_standard_receipt"] != hydrogen:
        raise ValueError("The saved and replayed hydrogen standard receipts differ.")
    offsets = scenario["standard_offsets_rt"]
    plane, inventory = dict(zip(elements, potentials)), dict(zip(elements, budget))
    for element in properties["basis"]["element_order"]:
        # These elements are absent from the declared finite source inventory.
        # Their arbitrary plane values do not affect any admissible phase.
        plane.setdefault(element, 0.)
        inventory.setdefault(element, 0.)
    parameters, reference, binding = parameters_from_saved_water(
        properties, plane, inventory,
        _I(hydrogen["dissolved_h2_base_standard_rt"])+_I(hydrogen["H2_dissolved_standard_offset_rt"]),
        water_standard_offset_rt=offsets.get("h2o_melts", 0.))
    if has_helium:
        from m2_helium_global import add_saved_helium
        parameters, binding = add_saved_helium(parameters,binding,source,host,exoeos_checkout,inventory=report['inventory'])
    return parameters, reference, {**binding, "dissolved_hydrogen_standard_receipt": hydrogen,
                                  "source_selection": selector, "source_provider_checkouts": report["checkouts"],
                                  **({"source_kind": provenance["kind"], "pressure_closure_performed": False}
                                     if provenance["kind"] == "fixed_pressure_internal_source" else {})}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--saved-closure", type=Path, required=True,
                        help="Saved closure, or an explicitly selected fixed-pressure internal-source envelope.")
    parser.add_argument("--allow-fixed-pressure-source", action="store_true",
                        help="Certify only the declared fixed-T/P source; never a pressure closure.")
    parser.add_argument("--saved-physical-audit", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--max-nodes", type=int, default=20000)
    parser.add_argument("--tolerance-rt", type=float, default=1e-10)
    parser.add_argument("--exoeos-checkout", type=Path, help="Required for the declared dissolved-He scalar.")
    args = parser.parse_args()
    files = {str(path.resolve()): sha256(path) for path in (args.saved_closure, args.saved_physical_audit)}
    for name in ("run_m2_water_global.py", "m2_water_global.py", "m2_common_plane.py", "m2_liquid_global.py", "m2_host_standards.py", "m2_helium_global.py", "m2_helium.py", "m2_scenarios.py", "run_m2_alloy_insertion_bound.py"):
        path = Path(__file__).with_name(name)
        files[str(path.resolve())] = sha256(path)
    report, audit = [json.loads(path.read_text()) for path in (args.saved_closure, args.saved_physical_audit)]
    parameters, reference, binding = saved_water_problem(report, audit, files[str(args.saved_closure.resolve())],exoeos_checkout=args.exoeos_checkout,
        allow_fixed_pressure=args.allow_fixed_pressure_source)
    if 'helium_elimination' in binding:
        for name,digest in binding['helium_elimination']['receipt']['provider_recipe_file_sha256'].items():
            files[str((args.exoeos_checkout/name).resolve())]=digest
    started = time.monotonic()
    result = certify_water_common_plane(parameters, reference, tolerance_rt=args.tolerance_rt, max_nodes=args.max_nodes,
        progress_callback=lambda nodes, best: print(json.dumps({"nodes_evaluated": nodes, "best_trial": best}), flush=True))
    if any(sha256(Path(path)) != digest for path, digest in files.items()):
        raise ValueError("An input or proof source changed during verification.")
    result.update(binding=binding, command=sys.argv, elapsed_seconds=time.monotonic()-started,
                  input_and_code_sha256=files, proof_commit=subprocess.check_output(
                      ["git", "rev-parse", "HEAD"], cwd=Path(__file__).resolve().parents[2], text=True).strip())
    if audit["source"]["kind"] == "fixed_pressure_internal_source":
        result.update(source_kind="fixed_pressure_internal_source", pressure_closure_performed=False)
    with args.output.open("x") as stream:
        json.dump(result, stream, indent=2, allow_nan=False)
        stream.write("\n")
    print(result["lower_bound_rt_per_dry_component"], result["bound_within_requested_tolerance"], flush=True)


if __name__ == "__main__":
    main()
