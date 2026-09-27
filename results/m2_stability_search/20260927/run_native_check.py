"""Reevaluate a pinned host and record independent whole-catalog searches."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[3]
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "examples/metal_silicate")]
from m2_host_stability import assess_host_candidates
from m2_stability_search import assess_liquid_local_curvature, search_competing_solutions
from melts_coupled import COMMON_R, load_melts_evaluator


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--saved-physical-audit", type=Path, required=True)
    parser.add_argument("--exoeos-checkout", type=Path, required=True)
    parser.add_argument("--runtime", type=Path, required=True)
    parser.add_argument("--python", required=True)
    parser.add_argument("--output-directory", type=Path, required=True)
    parser.add_argument("--evaluations-per-phase", type=int, default=60)
    args = parser.parse_args()
    output = args.output_directory
    output.mkdir(parents=True, exist_ok=False)
    raw = args.saved_physical_audit.read_bytes()
    saved = json.loads(raw)
    previous = saved["host_stability"]
    h2 = previous["native_dissolved_h2_moles"]
    request = previous["provider_properties"]
    evaluator = load_melts_evaluator(args.exoeos_checkout)
    parameters = dict(evaluator=evaluator, runtime=args.runtime, python_executable=args.python)
    started = time.time()
    properties = evaluator.evaluate_liquid(request["T_K"], request["P_Pa"], request["component_moles"],
        runtime=args.runtime, python_executable=args.python, common_R=COMMON_R, include_saturation=True)
    provenance = {
        "gibbs_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
        "command": sys.argv, "source_path": str(args.saved_physical_audit.resolve()),
        "source_sha256": hashlib.sha256(raw).hexdigest(), "source_closure": saved["source"],
        "search_sha256": hashlib.sha256((ROOT / "examples/metal_silicate/m2_stability_search.py").read_bytes()).hexdigest(),
        "runner_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "scope": "Fresh fixed-host properties and local/composition diagnostics; no pressure or equilibrium root is recomputed.",
    }
    def write(name, value):
        with (output / name).open("x") as stream:
            json.dump(value, stream, indent=2, allow_nan=False)
            stream.write("\n")
    write("source_physical_audit.json", saved)
    write("protocol.json", {**provenance, "max_evaluations_per_phase": args.evaluations_per_phase,
                            "local_curvature_steps": [.001, .0005], "tolerance_rt": 1e-8})
    write("fresh_host_assessment.json", {**provenance, "assessment": assess_host_candidates(properties, h2),
                                         "provider_properties": properties})
    rows = []
    for index, phase in enumerate(properties["saturation"]["candidate_order"]):
        result = search_competing_solutions(properties, h2, phases=[phase],
                    max_evaluations=args.evaluations_per_phase, **parameters)
        write(f"phase_{index:02d}_{phase}.json", result)
        row = result["phases"][0]
        best = row.get("best_fresh_trial")
        rows.append({"phase": phase, "status": row["status"],
                     "minimum_fresh_rt_per_mol_atoms": None if best is None else best["objective_rt"],
                     "initial_points_requested": row.get("initial_points_requested"),
                     "initial_points_attempted": row.get("initial_points_attempted"),
                     "successful_evaluations": len(row.get("trials", [])),
                     "failed_evaluations": len(row.get("failed_evaluations", [])),
                     "composition_minimum_enumerated": row.get("composition_minimum_enumerated", False),
                     "reason": row.get("reason")})
        print(phase, row["status"], rows[-1]["minimum_fresh_rt_per_mol_atoms"], flush=True)
    curvature = assess_liquid_local_curvature(properties, h2, **parameters)
    write("local_curvature.json", curvature)
    print("local_curvature", curvature["status"], flush=True)
    write("summary.json", {**provenance, "completed": True, "elapsed_seconds": time.time() - started,
                           "temperature_K": properties["T_K"], "pressure_bar": properties["P_Pa"] / 1e5,
                           "native_dissolved_h2_moles": h2, "phase_count": len(rows), "phases": rows,
                           "local_curvature_status": curvature["status"],
                           "global_stability_certified": False})


if __name__ == "__main__":
    main()
