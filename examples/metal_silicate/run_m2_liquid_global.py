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
    evaluator = load_melts_evaluator(args.exoeos_checkout)
    mixing_path = args.exoeos_checkout / "examples/melts_liquid_mixing.py"
    spec = importlib.util.spec_from_file_location("melts_liquid_mixing", mixing_path)
    mixing_model = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mixing_model)
    root = Path(__file__).resolve().parents[2]
    started = time.time()

    def evaluate(n):
        return evaluator.evaluate_liquid(request["T_K"], request["P_Pa"], n, runtime=args.runtime,
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
                "max_nodes": args.max_nodes, "new_pressure_root": False, "fresh_native_properties": True}
    (args.output_directory / "source_physical_audit.json").write_bytes(raw)
    write("protocol.json", protocol)
    properties = evaluate(request["component_moles"])
    write("host_properties.json", properties)
    comparison = mixing_model.compare_native_mixing(properties)
    checks = [{"role": "parent", "comparison": comparison, "provider_properties": properties}]
    parent = np.asarray(properties["component_moles"])
    parent /= parent.sum()
    # Independent native probes validate the published expression away from
    # the host. Their agreement is not used as an interval error bound.
    for index in np.flatnonzero(parent):
        point = .02*parent
        point[index] += .98
        receipt = evaluate(point.tolist())
        checked = mixing_model.compare_native_mixing(receipt)
        checks.append({"role": "near_vertex", "component_index": int(index),
                       "comparison": checked, "provider_properties": receipt})
    write("native_expression_checks.json", checks)
    if any(row["comparison"]["status"] != "compatible_at_supplied_state" for row in checks):
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
                           "maximum_native_potential_difference_rt": max(row["comparison"]["maximum_potential_difference_rt"] for row in checks),
                           "native_binary_error_bound_certified": False,
                           "global_empirical_stability_certified": False})
    print(result["status"], result.get("lower_bound_rt"), flush=True)


if __name__ == "__main__":
    main()
