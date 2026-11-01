"""Replay the scoped SSO source investigation from archived inputs, offline."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import math
from pathlib import Path
import platform
import re
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent))
from gas_source_comparison import ROOT, check_equal, file_hash, janaf_g, load_inputs, nasa_g, verify_hashes

SNAPSHOT = ROOT / "tests/data/validation/s2o_source_investigation/20261102"
BASELINE = ROOT / "tests/data/validation/gas_source_comparison/baseline-001"


def read_janaf(path: Path) -> dict:
    """Read exact tabulated nodes, including the suspect formation columns."""
    rows = {}
    for line in path.read_text().splitlines()[2:]:
        fields = line.split()
        if len(fields) != 8:
            continue
        values = [float(value.replace("INFINITE", "inf")) for value in fields]
        t, cp, entropy, _, increment, hf, gf, _ = values
        rows[t] = dict(temperature_K=t, cp_J_mol_K=cp, entropy_J_mol_K=entropy,
                       enthalpy_increment_kJ_mol=increment,
                       formation_enthalpy_kJ_mol=hf, formation_gibbs_kJ_mol=gf)
    return dict(hf298_kJ_mol=rows[298.15]["formation_enthalpy_kJ_mol"],
                samples=list(rows.values()))


def investigate() -> dict:
    """Separate the formation-column error from source and energy differences."""
    inputs = load_inputs()
    sources = json.loads((SNAPSHOT / "sources.json").read_text())
    if file_hash(BASELINE / "inputs.json") != sources["baseline_sha256"]:
        raise ValueError("Baseline input hash mismatch")
    for source in sources["sources"]:
        if source["archive"] and file_hash(SNAPSHOT / source["archive"]) != source["sha256"]:
            raise ValueError(f"Source hash mismatch: {source['id']}")
    nasa = inputs["nasa9_records"]
    # Printed NASA/TP Appendix D values, independently retrieved this session.
    printed = [float(value.replace("D", "E")) for value in re.findall(
        r"[-+]?\d+\.\d+D[-+]\d+", (SNAPSHOT / "nasa-s2o.txt").read_text())]
    if printed != [c for interval in nasa["S2O"]["intervals"] for c in interval["coefficients"]]:
        raise ValueError("Printed NASA S2O coefficient mismatch")
    fresh = {"O1S2": read_janaf(SNAPSHOT / "O-011.txt"),
             "O1": read_janaf(SNAPSHOT / "O-001.txt"),
             "S1": read_janaf(SNAPSHOT / "S-006.txt")}
    # Establish the narrow fresh-acquisition match without authenticating other species.
    for name, record in fresh.items():
        old = inputs["janaf_records"][name]
        if record["hf298_kJ_mol"] != old["hf298_kJ_mol"]:
            raise ValueError(f"JANAF anchor differs: {name}")
        nodes = {s["temperature_K"]: s for s in record["samples"]}
        for sample in old["samples"]:
            if any(nodes[sample["temperature_K"]][key] != value for key, value in sample.items()):
                raise ValueError(f"JANAF sample differs: {name}")
    r = inputs["gas_constant_J_mol_K"]
    lodders = sources["lodders2004"]
    baseline = json.loads((BASELINE / "comparison.json").read_text())
    samples = []
    for t in sources["temperatures_K"]:
        atoms = 2 * janaf_g(fresh["S1"], t, r) + janaf_g(fresh["O1"], t, r)
        janaf = atoms - janaf_g(fresh["O1S2"], t, r)
        raw = (2 * janaf_g(fresh["S1"], t, r, True)
               + janaf_g(fresh["O1"], t, r, True) - janaf_g(fresh["O1S2"], t, r, True))
        nasa_ln_k = 2 * nasa_g(nasa["S"], t) + nasa_g(nasa["O"], t) - nasa_g(nasa["S2O"], t)
        old = next(s for s in baseline["samples"] if s["species_id"] == "O1S2" and s["temperature_K"] == t)
        j = next(s for s in fresh["O1S2"]["samples"] if s["temperature_K"] == t)
        l = next(s for s in lodders["samples"] if s["temperature_K"] == t)
        # A molecular replacement with fixed JANAF atoms is a diagnostic, not a new dataset.
        anchor = lodders["hf298_kJ_mol"] - fresh["O1S2"]["hf298_kJ_mol"]
        increment = l["enthalpy_increment_kJ_mol"] - j["enthalpy_increment_kJ_mol"]
        entropy = l["entropy_J_mol_K"] - j["entropy_J_mol_K"]
        delta_g = anchor + increment - t * entropy / 1000
        lodders_ln_k = atoms - janaf_g(lodders, t, r)
        samples.append(dict(temperature_K=t,ln_Kp_janaf_hs=janaf,ln_Kp_janaf_raw=raw,
                            ln_Kp_nasa=nasa_ln_k,ln_Kp_curated=old["ln_Kp"]["curated"],
                            janaf_internal_difference_dex=(raw-janaf)/math.log(10),
                            janaf_over_nasa=math.exp(janaf-nasa_ln_k),
                            curated_over_nasa=None if old["ln_Kp"]["curated"] is None else math.exp(old["ln_Kp"]["curated"]-nasa_ln_k),
                            lodders_with_janaf_atoms_over_janaf=math.exp(lodders_ln_k-janaf),
                            molecular_delta_g_kJ_mol=dict(anchor=anchor,enthalpy_increment=increment,
                                                         minus_T_delta_S=-t*entropy/1000,total=delta_g),
                            cp_lodders_minus_janaf_J_mol_K=l["cp_J_mol_K"]-j["cp_J_mol_K"]))
    hf0 = sources["atct_1_172"]["hf0_kJ_mol"]
    d0 = 2*hf0["S"] + hf0["O"] - hf0["SSO"]
    return dict(snapshot_id=sources["snapshot_id"],reaction=sources["reaction"],
                standard_pressure_Pa=100000,official_janaf_matching_records=3,
                official_nasa_matching_coefficients=len(printed),samples=samples,
                atct_atomization_0K_kJ_mol_from_rounded_values=d0,
                atct_minus_lodders_atomization_kJ_mol=d0-lodders["atomization_energy_0K_kJ_mol"],
                assessment=dict(source_diagnosis="supported",production_decision="retain_pending_energy_and_application_review",
                                full_physical_accuracy="unknown",equilibrium_abundance_impact="not_run"))


def selected_hashes() -> dict:
    paths = [Path(__file__), ROOT / "mceg/reviews/gas_source_comparison.py",
             ROOT / "examples/comparisons/audit_atomic_ions.py",
             BASELINE / "inputs.json", BASELINE / "comparison.json"]
    paths += [SNAPSHOT / name for name in ("sources.json", "O-001.txt", "O-011.txt", "S-006.txt", "nasa-s2o.txt")]
    return {str(p.relative_to(ROOT)): file_hash(p) for p in paths}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    result = investigate()
    if args.check:
        receipt = json.loads((SNAPSHOT / "receipt.json").read_text())
        verify_hashes(receipt["input_sha256"])
        verify_hashes(receipt["output_sha256"])
        check_equal(result, json.loads((SNAPSHOT / "comparison.json").read_text()))
        print("S2O: 3 fresh JANAF records, 18 NASA coefficients and 4 samples verified")
    elif args.output_dir:
        args.output_dir.mkdir(parents=True, exist_ok=False)
        output = args.output_dir / "comparison.json"
        output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
        receipt = dict(created_utc=datetime.now(timezone.utc).isoformat(),
                       command="python mceg/reviews/s2o_source_investigation.py --output-dir " + str(args.output_dir),
                       python=platform.python_version(),input_sha256=selected_hashes(),
                       output_sha256={str((SNAPSHOT / "comparison.json").relative_to(ROOT)): file_hash(output)})
        (args.output_dir / "receipt.json").write_text(json.dumps(receipt, indent=2) + "\n")
        print(output)
    else:
        parser.error("Use --check or --output-dir")


if __name__ == "__main__":
    main()
