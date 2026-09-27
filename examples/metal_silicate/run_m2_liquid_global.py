"""Reevaluate one saved host and bound its complete formal liquid tangent plane."""

import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import subprocess
import sys
import time

import numpy as np

from m2_liquid_global import assess_liquid_global_tangent_plane
from melts_coupled import COMMON_R, load_melts_evaluator


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--saved-physical-audit", type=Path, required=True)
    parser.add_argument("--exoeos-checkout", type=Path, required=True)
    parser.add_argument("--runtime", type=Path, required=True)
    parser.add_argument("--python", required=True)
    parser.add_argument("--output-directory", type=Path, required=True)
    parser.add_argument("--max-nodes", type=int, default=20000)
    args = parser.parse_args()
    args.output_directory.mkdir(parents=True, exist_ok=False)
    raw = args.saved_physical_audit.read_bytes()
    saved = json.loads(raw)
    host = saved["host_stability"]
    request = host["provider_properties"]
    models_by_id = {"melts_v102_published_mixing_native_standard_states_v1": "published",
                    "alphamelts_2_3_2_rhyolite_melts_1_0_2_supplied_liquid_v1": "native"}
    if request["model_id"] not in models_by_id:
        raise ValueError("The saved host model is not supported.")
    liquid_model = models_by_id[request["model_id"]]
    if (not saved["source"].get("numerical_source_accepted")
            or not saved["source"].get("source_contact_accepted")
            or host["temperature_K"] != request["T_K"]
            or host["pressure_Pa"] != request["P_Pa"]
            or not np.array_equal(host["native_host_component_amounts_mol"], request["component_moles"])):
        raise ValueError("The saved accepted source and host T/P/amount ledger must agree.")
    native = load_melts_evaluator(args.exoeos_checkout)
    evaluator = load_melts_evaluator(args.exoeos_checkout, liquid_model=liquid_model,
                                     runtime=args.runtime, python_executable=args.python)
    mixing_path = args.exoeos_checkout / "examples/melts_liquid_mixing.py"
    spec = importlib.util.spec_from_file_location("melts_liquid_mixing", mixing_path)
    mixing_model = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mixing_model)
    root = Path(__file__).resolve().parents[2]
    started = time.time()

    def evaluate(n):
        return native.evaluate_liquid(request["T_K"], request["P_Pa"], n, runtime=args.runtime,
                                      python_executable=args.python, common_R=COMMON_R)

    def write(name, value):
        with (args.output_directory / name).open("x") as stream:
            json.dump(value, stream, indent=2, allow_nan=False)
            stream.write("\n")

    protocol = {"command": sys.argv, "source_path": str(args.saved_physical_audit.resolve()),
                "source_sha256": hashlib.sha256(raw).hexdigest(),
                "gibbs_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=root, text=True).strip(),
                "exoeos_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=args.exoeos_checkout, text=True).strip(),
                "source_hashes": {str(path.relative_to(root)): hashlib.sha256(path.read_bytes()).hexdigest()
                                  for path in (Path(__file__), Path(__file__).with_name("m2_liquid_global.py"))},
                "mixing_provider_sha256": hashlib.sha256(mixing_path.read_bytes()).hexdigest(),
                "max_nodes": args.max_nodes, "new_pressure_root": False, "fresh_native_properties": True,
                "selected_liquid_model": liquid_model, "native_comparisons_are_separate_controls": True}
    (args.output_directory / "source_physical_audit.json").write_bytes(raw)
    write("protocol.json", protocol)
    native_parent = evaluate(request["component_moles"])
    properties = native_parent if liquid_model == "native" else evaluator.evaluate_liquid(
        request["T_K"], request["P_Pa"], request["component_moles"], runtime=args.runtime,
        python_executable=args.python, common_R=COMMON_R)
    write("host_properties.json", properties)
    comparison = mixing_model.compare_native_mixing(native_parent)
    checks = [{"role": "independent_native_parent", "comparison": comparison, "provider_properties": native_parent}]
    parent = np.asarray(properties["component_moles"])
    parent /= parent.sum()
    # Independent native probes validate the published expression away from
    # the host. Their agreement is not used as an interval error bound.
    for index in np.flatnonzero(parent):
        for weight in (.98, .5):
            point = (1-weight)*parent
            point[index] += weight
            row = {"role": "toward_vertex", "component_index": int(index),
                   "vertex_weight": weight, "requested_component_moles": point.tolist()}
            try:
                receipt = evaluate(point.tolist())
                checked = mixing_model.compare_native_mixing(receipt)
                row.update(comparison=checked, provider_properties=receipt)
            except (ValueError, RuntimeError) as error:
                # Oxide inversion becomes ill-conditioned close to some
                # vertices. Preserve the failed request; never clip its
                # composition or invent native properties at that endpoint.
                row.update(comparison={"status": "native_evaluation_unavailable"}, reason=str(error))
            checks.append(row)
            write(f"native_probe_{len(checks)-1:02d}.json", row)
            if row["comparison"]["status"] != "native_evaluation_unavailable":
                break
    write("native_expression_checks.json", checks)
    if any(row["comparison"]["status"] == "mismatch" for row in checks):
        write("assessment.json", {"status": "native_expression_mismatch", "formal_two_liquid_bound_accepted": False})
        raise RuntimeError("Native expression compatibility failed; no stability certificate was attempted.")
    result = assess_liquid_global_tangent_plane(properties, host["native_dissolved_h2_moles"],
                                               mixing_model=mixing_model, max_nodes=args.max_nodes)
    write("assessment.json", result)
    write("summary.json", {**protocol, "completed": True, "elapsed_seconds": time.time()-started,
                           "status": result["status"], "lower_bound_rt": result.get("lower_bound_rt"),
                           "formal_two_liquid_bound_accepted": result.get("formal_two_liquid_bound_accepted", False),
                           "nodes_evaluated": result.get("nodes_evaluated"),
                           "native_compatibility_checks": len(checks),
                           "native_evaluation_failures": sum(row["comparison"]["status"] == "native_evaluation_unavailable" for row in checks),
                           "maximum_native_potential_difference_rt": max(row["comparison"]["maximum_potential_difference_rt"] for row in checks if "maximum_potential_difference_rt" in row["comparison"]),
                           "native_binary_error_bound_certified": False,
                           "global_empirical_stability_certified": False})
    print(result["status"], result.get("lower_bound_rt"), flush=True)


if __name__ == "__main__":
    main()
