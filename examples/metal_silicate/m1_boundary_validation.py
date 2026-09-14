"""Reevaluate the vendored low-O boundary inputs without overwriting records."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import importlib.util
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
SPEC = importlib.util.spec_from_file_location("m1_boundary_chemistry", Path(__file__).with_name("m1_chemistry.py"))
M1 = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(M1)


def main() -> None:
    """Save fresh source/upper solves and boundary audits on absolute atom moles."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error("Output exists; use a new path.")
    fixture = ROOT / "tests/unittests/examples/data/m1_boundary_cases.json"
    saved = json.loads(fixture.read_text())
    _, upper = M1.build_setups()
    record = {
        "scope": "Fresh local source and upper solves; approximate contract only, no M1-A acceptance.",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "input_fixture": {"path": str(fixture), "sha256": hashlib.sha256(fixture.read_bytes()).hexdigest()},
        "evaluation_script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "element_mass_tolerance": M1.ELEMENT_TOLERANCE,
        "chemical_tolerance": M1.CHEMICAL_TOLERANCE,
        "provenance": M1.provenance(), "cases": [],
    }
    for case_input in saved["cases"]:
        network, case, _ = M1.source_inputs(case_input["oxygen_factor"])
        source = M1.SOURCE.solve_reduced_source(
            network, case, element_amounts_mol=case_input["source_element_amounts_mol"],
            pressure_bar=case_input["P_bar"],
        )
        budget = M1.source_gas_inventory(network, source)
        report = M1.solve_parcel(upper, case_input["T_K"], case_input["P_bar"], budget)
        boundary = M1.audit_boundary(network, case, source, upper, report)
        record["cases"].append({"input": case_input, "source": source, "upper": report, "boundary": boundary})
        print(json.dumps({
            "id": case_input["id"], "contract_met": boundary["contract_met"],
            "cloud_element_fraction": boundary["cloud_element_fraction"],
            "max_abs_delta_source_reaction_residual_rt": boundary["max_abs_delta_source_reaction_residual_rt"],
        }), flush=True)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x", encoding="utf-8") as stream:
        json.dump(record, stream, indent=2, allow_nan=False)
        stream.write("\n")
    if not all(item["boundary"]["contract_met"] for item in record["cases"]):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
