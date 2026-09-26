"""Assess native competing phases of an archived, unchanged M2 source host.
========================================================================

This command never resolves the finite source or edits the input archive.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from m2_host_stability import evaluate_host_stability
from melts_coupled import load_melts_evaluator


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-json", required=True, type=Path)
    parser.add_argument("--exoeos-checkout", required=True, type=Path)
    parser.add_argument("--runtime", required=True, type=Path)
    parser.add_argument("--python", required=True, dest="python_executable")
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--search-solutions", nargs="+", metavar="PHASE",
                        help="Search these native endmember simplexes; nonnegative samples do not certify stability.")
    parser.add_argument("--search-liquid-splitting", action="store_true")
    parser.add_argument("--search-evaluations", type=int, default=60,
                        help="Objective evaluations per search, plus one fresh final evaluation.")
    args = parser.parse_args()
    raw = args.source_json.read_bytes()
    source = json.loads(raw)
    record = source.get("record", source.get("source_record"))
    parameters = source.get("arguments", {})
    temperature = source.get("temperature_K", parameters.get("temperature_k"))
    pressure = source.get("pressure_bar", parameters.get("bottom_pressure_bar"))
    evaluator = load_melts_evaluator(args.exoeos_checkout)
    assessment = evaluate_host_stability(
        record, source["source_result"]["component_amounts_mol"], temperature, pressure,
        evaluator=evaluator, runtime=args.runtime,
        python_executable=args.python_executable,
    )
    if args.search_solutions or args.search_liquid_splitting:
        from m2_stability_search import search_competing_solutions, search_liquid_splitting
        parameters = dict(evaluator=evaluator, runtime=args.runtime, python_executable=args.python_executable,
                          max_evaluations=args.search_evaluations)
        properties, h2 = assessment["provider_properties"], assessment["native_dissolved_h2_moles"]
        if args.search_solutions:
            assessment["solution_search"] = search_competing_solutions(properties, h2,
                phases=args.search_solutions, **parameters)
        if args.search_liquid_splitting:
            assessment["liquid_splitting_search"] = search_liquid_splitting(properties, h2, **parameters)
        if any(assessment.get(key, {}).get("status") in {"rejected_by_feasible_trial", "negative_feasible_witness"}
               for key in ("solution_search", "liquid_splitting_search")):
            assessment["status"] = "rejected_by_feasible_trial"
    result = {
        "scope": "New native-property postprocessing of the recorded source composition; no source re-equilibration.",
        "input": {"path": str(args.source_json.resolve()), "sha256": hashlib.sha256(raw).hexdigest(),
                  "source_accepted": source["source_result"].get("accepted"),
                  "source_scope": source.get("scope", source.get("model_id")),
                  "source_provenance": source.get("source_metadata", {}).get("provenance")},
        "assessment": assessment,
        "runner_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
    }
    with args.output.open("x") as stream:
        stream.write(json.dumps(result, indent=2, allow_nan=False) + "\n")


if __name__ == "__main__":
    main()
