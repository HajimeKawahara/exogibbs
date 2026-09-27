"""Bound every declared mineral model against one freshly evaluated host."""

import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import subprocess
import sys
import time

import numpy as np

from m2_liquid_global import require_saved_liquid_expression
from m2_solid_global import certify_solid_insertion
from melts_coupled import COMMON_R, load_melts_evaluator


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--saved-physical-audit", type=Path, required=True)
    parser.add_argument("--exoeos-checkout", type=Path, required=True)
    parser.add_argument("--runtime", type=Path, required=True)
    parser.add_argument("--python", required=True)
    parser.add_argument("--output-directory", type=Path, required=True)
    parser.add_argument("--max-nodes", type=int, default=100000)
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
    mixing_path = args.exoeos_checkout / "examples/melts_solid_mixing.py"
    spec = importlib.util.spec_from_file_location("melts_solid_mixing", mixing_path)
    provider = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(provider)
    models = json.loads(provider.PARAMETER_PATH.read_text())["models"]
    root = Path(__file__).resolve().parents[2]
    started = time.time()

    def write(name, value):
        with (args.output_directory / name).open("x") as stream:
            json.dump(value, stream, indent=2, allow_nan=False)
            stream.write("\n")

    protocol = {"command": sys.argv, "source_path": str(args.saved_physical_audit.resolve()),
                "source_sha256": hashlib.sha256(raw).hexdigest(), "liquid_model": liquid_model,
                "gibbs_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=root, text=True).strip(),
                "exoeos_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=args.exoeos_checkout, text=True).strip(),
                "source_hashes": {str(path.relative_to(root)): hashlib.sha256(path.read_bytes()).hexdigest()
                                  for path in (Path(__file__), Path(__file__).with_name("m2_solid_global.py"),
                                               Path(__file__).with_name("m2_liquid_global.py"))},
                "provider_hashes": {str(path.relative_to(args.exoeos_checkout)): hashlib.sha256(path.read_bytes()).hexdigest()
                                    for path in (mixing_path, provider.PARAMETER_PATH,
                                                 args.exoeos_checkout / "examples/melts_liquid_evaluator.py")},
                "max_nodes": args.max_nodes, "new_pressure_root": False,
                "fresh_native_standard_states": True}
    (args.output_directory / "source_physical_audit.json").write_bytes(raw)
    write("protocol.json", protocol)
    kwargs = dict(runtime=args.runtime, python_executable=args.python, common_R=COMMON_R)
    native_properties = native.evaluate_liquid(request["T_K"], request["P_Pa"], request["component_moles"],
                                               candidate_standard_states=list(models), **kwargs)
    write("native_standard_state_receipt.json", native_properties)
    properties = native_properties if liquid_model == "native" else evaluator.evaluate_liquid(
        request["T_K"], request["P_Pa"], request["component_moles"], **kwargs)
    require_saved_liquid_expression(request, properties)
    write("host_properties.json", properties)
    rows = []
    for standards in native_properties["candidate_standard_states"]:
        phase = standards["phase"]
        parameters = provider.solid_mixing_parameters(phase, properties["T_K"], properties["P_Pa"])
        result = certify_solid_insertion(parameters, standards, properties, host["native_dissolved_h2_moles"],
                                         max_nodes=args.max_nodes)
        write(phase + ".json", result)
        row = {key: result[key] for key in ("phase", "formal_global_insertion_bound_accepted",
                                           "lower_bound_rt_per_formula_unit", "node_count", "unresolved_box_count",
                                           "bound_within_requested_tolerance")}
        rows.append(row)
        print(json.dumps(row), flush=True)
    write("summary.json", {**protocol, "completed": True, "elapsed_seconds": time.time()-started,
                           "phases": rows, "declared_models": len(models),
                           "accepted_models": sum(row["formal_global_insertion_bound_accepted"] for row in rows),
                           "native_binary_error_bound_certified": False,
                           "global_empirical_stability_certified": False})


if __name__ == "__main__":
    main()
