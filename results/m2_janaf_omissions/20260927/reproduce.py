"""Reassess three immutable saved closures; perform no native or planetary solve."""

import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys

REPOSITORY = "https://github.com/HajimeKawahara/exoinventory"
REPLAY_IMPLEMENTATION_COMMIT = "09453dd87e0aab9d7c0b76f91a7d9d024db7c848"
CASES = {"h1e22": "native_h_recovery/case_000", "h1e24": "native_h_pilot/case_001", "h2e24": "native_h_pilot/case_002"}
SOURCE_ROOT = Path("examples/subneptune_taxonomy/finite_melt/validation/20260925_m2_hydrogen_pilot")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inventory-checkout", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error("Choose a new output directory; preserve prior assessments.")
    repository = Path(__file__).resolve().parents[3]
    sys.path.insert(0, str(repository / "examples/metal_silicate"))
    from m2_janaf import screen_source_janaf
    actual_commit = subprocess.check_output(["git", "-C", str(repository), "rev-parse", "HEAD"], text=True).strip()
    for relative in ("examples/metal_silicate/m2_janaf.py", "examples/metal_silicate/m2_omitted_gas.py",
                     "examples/metal_silicate/data/janaf_atomic.json"):
        pinned = subprocess.check_output(["git", "-C", str(repository), "show", REPLAY_IMPLEMENTATION_COMMIT + ":" + relative])
        if (repository / relative).read_bytes() != pinned:
            raise ValueError("Reproduction code or data changed from the archived implementation.")
    source_commit = subprocess.check_output(["git", "-C", str(args.inventory_checkout), "rev-parse", "HEAD"], text=True).strip()
    args.output.mkdir(parents=True)
    reports = []
    for label, location in CASES.items():
        path = SOURCE_ROOT / location / "closure/closure.json"
        raw = (args.inventory_checkout / path).read_bytes()
        git_raw = subprocess.check_output(["git", "-C", str(args.inventory_checkout), "show", source_commit + ":" + str(path)])
        if raw != git_raw:
            raise ValueError("The saved closure differs from its recorded source commit.")
        document = json.loads(raw)
        run, = [run for run in document["runs"] if run["nlayer"] == 16]
        root, = run["roots"]
        source = root["source"]
        if source["temperature_K"] != root["temperature_base_k"] or abs(source["pressure_bar"] * 1e5 / root["pressure_base_pa"] - 1) > 1e-12:
            raise ValueError("Source and saved root conditions disagree.")
        result = screen_source_janaf(source, mole_fraction_target=1e-8)
        result["saved_source"] = {"repository": REPOSITORY, "commit": source_commit, "path": str(path),
            "url": REPOSITORY + "/blob/" + source_commit + "/" + str(path), "sha256": hashlib.sha256(raw).hexdigest(),
            "layers": 16, "root_index": 0}
        result["execution"] = {"gibbs_commit": actual_commit, "code_equivalent_to_commit": REPLAY_IMPLEMENTATION_COMMIT, "new_native_solve": False,
                               "new_pressure_closure": False, "input_bytes_unchanged": True}
        output = args.output / (label + ".json")
        output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
        demand = result["fixed_reservoir_demand"]
        p, = [row for row in demand["element_demands"] if row["element"] == "P"]
        reports.append({"case": label, "pressure_bar": source["pressure_bar"],
                        "sum_omitted_mole_fractions": demand["sum_mole_fractions"],
                        "P_demand_over_local_source_P": p["demand_over_source_budget"],
                        "trace_screen_passed": demand["trace_screen_passed"],
                        "path": output.name, "sha256": hashlib.sha256(output.read_bytes()).hexdigest()})
    (args.output / "summary.json").write_text(json.dumps(reports, indent=2, allow_nan=False) + "\n")
    print(json.dumps(reports, indent=2))


if __name__ == "__main__":
    main()
