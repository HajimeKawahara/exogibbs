"""Screen a preserved source's omitted gases without a native chemistry solve."""

import argparse
import hashlib
import json
from pathlib import Path

from m2_omitted_gas import screen_source


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--layers", type=int)
    parser.add_argument("--root-index", type=int, default=0)
    parser.add_argument("--mole-fraction-target", type=float, required=True)
    parser.add_argument("--atomic-references", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error("Choose a new output; historical sources and screens are preserved.")
    raw = args.source.read_bytes()
    document = json.loads(raw)
    if "runs" in document:
        runs = [run for run in document["runs"] if args.layers is None or run["nlayer"] == args.layers]
        if len(runs) != 1 or not 0 <= args.root_index < len(runs[0]["roots"]):
            parser.error("Select exactly one saved layer count and an existing root index.")
        state = runs[0]["roots"][args.root_index]
        source = state["source"]
        if (source["temperature_K"] != state["temperature_base_k"]
                or abs(source["pressure_bar"] * 1e5 / state["pressure_base_pa"] - 1.) > 1e-12):
            parser.error("Saved source and root conditions differ.")
    else:
        source = document
    references = None if args.atomic_references is None else json.loads(args.atomic_references.read_bytes())
    report = screen_source(source, mole_fraction_target=args.mole_fraction_target,
                           additional_atomic_references=references)
    report["saved_source"] = {"path": str(args.source.resolve()), "sha256": hashlib.sha256(raw).hexdigest(),
                              "layers": args.layers, "root_index": args.root_index}
    if args.atomic_references is not None:
        report["atomic_reference_input_sha256"] = hashlib.sha256(args.atomic_references.read_bytes()).hexdigest()
    with args.output.open("x") as stream:
        json.dump(report, stream, indent=2, allow_nan=False)
        stream.write("\n")


if __name__ == "__main__":
    main()
