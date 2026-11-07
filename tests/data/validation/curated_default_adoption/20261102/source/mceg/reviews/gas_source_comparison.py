"""Replay the frozen gas-source screening offline; MCEG is not required."""

from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
from hashlib import sha256
import io
import json
import math
import os
from pathlib import Path
import platform
import subprocess
import sys


ROOT = Path(__file__).resolve().parents[2]
BASELINE = ROOT / "tests/data/validation/gas_source_comparison/baseline-001"
DEFAULT_INPUT = BASELINE / "inputs.json"
SCRIPT = Path(__file__).resolve()
LN10 = math.log(10)
REPLAY_ABS_TOL = 2e-11
REPLAY_REL_TOL = 2e-12
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "examples/comparisons"))

from audit_atomic_ions import nasa9_g_rt


class MissingSample(LookupError):
    """A declared source has no value at the requested temperature."""


def file_hash(path: Path) -> str:
    return sha256(path.read_bytes()).hexdigest()


def verify_hashes(hashes: dict) -> None:
    for name, expected in hashes.items():
        path = (ROOT / name).resolve()
        if not path.is_relative_to(ROOT) or file_hash(path) != expected:
            raise ValueError(f"SHA256 mismatch: {name}")


def load_inputs(path: Path = DEFAULT_INPUT) -> dict:
    """Load a frozen selection, rejecting drift in its repository dependencies."""
    inputs = json.loads(Path(path).read_text())
    verify_hashes(inputs["selected_input_sha256"])
    if inputs["schema_version"] != "1" or inputs["standard_pressure_Pa"] != 100000:
        raise ValueError("Unsupported schema or standard pressure")
    return inputs


def curated_basis(provenance: dict) -> str:
    """Describe the adopted combination or the inherited fit's source label."""
    if provenance["status"] != "inherited_unreviewed":
        labels = {
            ("nasa9", "atct"): "NASA CEA + ATcT 1.172",
            ("janaf", "atct"): "JANAF + ATcT 1.172",
            ("janaf", "janaf"): "JANAF (thermal + D0)",
            ("janaf", "barklem_collet"): "JANAF + Barklem-Collet 2016 D0",
            ("janaf", "spectroscopy"): "JANAF + Merriles 2020 D0",
            ("fastchem", "janaf"): "JANAF (FastChem fit)",
            ("fastchem", "barklem_collet"): "Barklem-Collet 2016 (FastChem fit)",
        }
        return labels[provenance["thermal_source"], provenance["energy_source"]]
    source = provenance["record"]["record_line"].split("#", 1)[1].strip()
    if "JANAF" in source:
        label = "JANAF 1998"
    elif source.startswith("Barklem,"):
        label = "Barklem-Collet 2016"
    elif source.startswith("Dorofeeva,") and "2003" in source:
        label = "Dorofeeva 2003"
    elif source.startswith("Lodders,") and "1999" in source:
        label = "Lodders 1999"
    elif source.startswith("Lodders,") and "2004" in source:
        label = "Lodders 2004"
    elif source.startswith("Goos et al.,") and "2016" in source:
        label = "Goos/Burcat 2016"
    elif source.startswith("Goos et al.,") and "2023" in source:
        label = "Goos/Burcat 2023"
    else:
        raise ValueError(f"Unrecognized inherited source: {source}")
    return label + " (FastChem fit) [U]"


def nasa_g(record: dict, temperature: float) -> float:
    # Preserve the historical low-interval choice at a shared boundary.
    for interval in record["intervals"]:
        if interval["bounds_K"][0] <= temperature <= interval["bounds_K"][1]:
            return nasa9_g_rt(temperature, interval)
    raise MissingSample("No NASA interval at this temperature")


def janaf_g(record: dict, temperature: float, gas_constant: float,
            formation_column: bool = False) -> float:
    for sample in record["samples"]:
        if sample["temperature_K"] == temperature:
            if formation_column:
                return 1000 * sample["formation_gibbs_kJ_mol"] / (gas_constant * temperature)
            return (1000 * (record["hf298_kJ_mol"] + sample["enthalpy_increment_kJ_mol"])
                    - temperature * sample["entropy_J_mol_K"]) / (gas_constant * temperature)
    raise MissingSample("No JANAF knot at this temperature")


def compare(inputs: dict) -> dict:
    """Evaluate explicitly mapped species and preserve unavailable samples."""
    os.environ.setdefault("JAX_PLATFORMS", "cpu")
    import jax
    import numpy as np
    from exogibbs.presets import curated

    jax.config.update("jax_enable_x64", True)
    setup = curated.chemsetup(ions=False)
    matrix = np.asarray(setup.formula_matrix)
    species_ids = [row["id"] for row in inputs["species"]]
    if len(set(species_ids)) != len(species_ids):
        raise ValueError("Duplicate species identity")
    for name, record in inputs["nasa9_records"].items():
        if record["source_name"] != name:
            raise ValueError(f"NASA identity mismatch: {name}")
    for element, mapping in inputs["atomic_references"].items():
        if inputs["nasa9_records"][mapping["nasa_id"]]["composition"] != {element: 1}:
            raise ValueError(f"Atomic reference composition mismatch: {element}")
    species = []
    for row in inputs["species"]:
        index = setup.species.index(row["id"])
        actual = {e: float(matrix[i, index]) for i, e in enumerate(setup.elements) if matrix[i, index]}
        if actual != row["composition"] or actual != inputs["nasa9_records"][row["nasa_id"]]["composition"]:
            raise ValueError(f"Species composition mismatch: {row['id']}")
        provenance = setup.metadata["species_provenance"][row["id"]]
        species.append({**row, "curated_basis": curated_basis(provenance),
                        "curated_provenance": provenance})
    lower, upper = setup.metadata["temperature_range_K"]
    samples = []
    for temperature in inputs["temperatures_K"]:
        h = np.asarray(setup.hvector_func(float(temperature))) if lower <= temperature <= upper else None
        for row in species:
            values, missing = {}, {}
            for source in ("janaf", "nasa"):
                records = inputs["janaf_records" if source == "janaf" else "nasa9_records"]
                key = source + "_id"
                def g(identifier, raw=False):
                    if source == "janaf":
                        return janaf_g(records[identifier], temperature, inputs["gas_constant_J_mol_K"], raw)
                    return nasa_g(records[identifier], temperature)
                try:
                    values[source] = sum(count * g(inputs["atomic_references"][e][key])
                                         for e, count in row["composition"].items()) - g(row[key])
                    if source == "janaf":
                        raw = sum(count * g(inputs["atomic_references"][e][key], True)
                                  for e, count in row["composition"].items()) - g(row[key], True)
                        internal = (raw - values[source]) / LN10
                except MissingSample as error:
                    values[source] = None
                    missing[source] = str(error)
                    if source == "janaf":
                        internal = None
            values["curated"] = None if h is None else -float(h[setup.species.index(row["id"])])
            if h is None:
                missing["curated"] = f"Outside curated range [{lower}, {upper}] K"
            if any(value is not None and not math.isfinite(value) for value in values.values()):
                raise ValueError(f"Non-finite result: {row['id']} at {temperature} K")
            samples.append({"species_id": row["id"], "temperature_K": temperature,
                            "ln_Kp": values, "missing": missing,
                            "janaf_internal_difference_dex": internal})
    return {"schema_version": "1", "snapshot_id": inputs["snapshot_id"],
            "reaction": inputs["reaction"], "janaf_method": inputs["janaf_method"],
            "standard_pressure_Pa": inputs["standard_pressure_Pa"],
            "temperatures_K": inputs["temperatures_K"], "display_policy": inputs["display_policy"],
            "curated_temperature_range_K": [lower, upper], "species": species,
            "excluded": inputs["excluded"], "samples": samples}


def grade(ratio: float, yellow_factor: float = 2, red_factor: float = 10) -> str:
    """Classify the symmetric multiplicative difference before rounding."""
    factor = max(ratio, 1 / ratio)
    return "red" if factor >= red_factor else "yellow" if factor >= yellow_factor else "green"


def major_marker(ratio: float, is_major: bool, lower: float = 0.9, upper: float = 1.1) -> bool:
    return bool(is_major and (ratio <= lower or ratio >= upper))


def summary_rows(report: dict, inputs: dict, temperature: float = 1500) -> list:
    """Project the canonical numerical record into the historical table."""
    samples = {r["species_id"]: r for r in report["samples"] if r["temperature_K"] == temperature}
    policy = inputs["display_policy"]
    rows = []
    for species in report["species"]:
        sample = samples[species["id"]]
        row = {"species_id": species["id"], "molecule": species["label"],
               "janaf_id": species["janaf_id"], "nasa_id": species["nasa_id"],
               "major_molecule": species["major_molecule"],
               "temperature_K": temperature, "identity_status": species["identity_status"],
               "janaf_internal_table_flag": species["janaf_internal_table_flag"],
               "curated_basis": species["curated_basis"]}
        for source in ("janaf", "curated"):
            value, reference = sample["ln_Kp"][source], sample["ln_Kp"]["nasa"]
            dex = None if value is None or reference is None else (value - reference) / LN10
            ratio = None if dex is None else 10 ** dex
            row[source + "_over_nasa_dex"] = dex
            row[source + "_over_nasa"] = ratio
            row[source + "_grade"] = None if ratio is None else grade(ratio, policy["yellow_factor"], policy["red_factor"])
            row[source + "_major_10pct"] = False if ratio is None else major_marker(
                ratio, species["major_molecule"], policy["major_ratio_lower"], policy["major_ratio_upper"])
        rows.append(row)
    rank = {"red": 0, "yellow": 1, "green": 2, None: 3}
    return sorted(rows, key=lambda r: (rank[r["curated_grade"]], rank[r["janaf_grade"]],
                  -abs(r["curated_over_nasa_dex"] or 0), r["molecule"]))


HEADERS = ["Molecule", "JANAF / NASA", "Grade", "Current curated / NASA", "Grade", "Current curated basis"]


def display_rows(rows: list) -> list:
    result = []
    symbols = {"green": "🟢", "yellow": "🟡", "red": "🔴", None: "Missing"}
    for row in rows:
        name = row["molecule"] + (" *" if row["identity_status"] == "provisional" else "")
        name += " †" if row["janaf_internal_table_flag"] else ""
        cells = [name]
        for source in ("janaf", "curated"):
            ratio = row[source + "_over_nasa"]
            cells += ["Missing" if ratio is None else f"{ratio:.4g}",
                      symbols[row[source + "_grade"]] + (" 🔷" if row[source + "_major_10pct"] else "")]
        result.append(cells + [row["curated_basis"]])
    return result


def render_markdown(rows: list) -> str:
    """Render the six-column Issue projection from canonical numeric rows."""
    lines = ["| " + " | ".join(HEADERS) + " |", "|---|---:|:---:|---:|:---:|---|"]
    lines.extend("| " + " | ".join(cells) + " |" for cells in display_rows(rows))
    return "\n".join(lines) + "\n"


def render_tables(rows: list) -> dict:
    stream = io.StringIO(newline="")
    writer = csv.writer(stream, lineterminator="\n")
    writer.writerows([HEADERS] + display_rows(rows))
    csv_text = stream.getvalue()
    rst = ".. csv-table:: Frozen source comparison at 1500 K\n   :header-rows: 1\n\n"
    rst += "".join("   " + line + "\n" for line in csv_text.splitlines())
    return {"table.csv": csv_text, "table.rst": rst, "table.md": render_markdown(rows)}


def check_historical(rows: list, inputs: dict) -> dict:
    actual = {row["species_id"]: row for row in rows}
    expected = inputs["historical_table"]
    if set(actual) != {row["curated_species"] for row in expected}:
        raise ValueError("Historical table coverage changed")
    maximum = 0.0
    for old in expected:
        row = actual[old["curated_species"]]
        metadata = {
            "janaf_id": old["janaf_species"], "nasa_id": old["nasa_species"],
            "identity_status": "provisional" if old["identity_provisional"] else "reviewed",
            "major_molecule": old["major_molecule"],
            "janaf_internal_table_flag": old["janaf_internal_table_flag"],
        }
        if any(row[key] != value for key, value in metadata.items()):
            raise ValueError(f"Historical mapping or flag changed: {row['species_id']}")
        for source, historical in (("janaf", "JANAF"), ("curated", "curated")):
            delta = abs(row[source + "_over_nasa_dex"] - old[historical + "_over_NASA_dex"])
            maximum = max(maximum, delta)
            if delta > 1e-10 or row[source + "_grade"] != old[historical + "_grade"]:
                raise ValueError(f"Historical value or grade changed: {row['species_id']}")
            if row[source + "_major_10pct"] != old[historical + "_major_10pct_flag"]:
                raise ValueError(f"Historical major marker changed: {row['species_id']}")
        if row["curated_basis"] != old["curated_basis"]:
            raise ValueError(f"Historical curated basis changed: {row['species_id']}")
    return {"result": "passed", "rows": len(rows), "max_difference_dex": maximum,
            "absolute_tolerance_dex": 1e-10}


def check_equal(actual, expected, location="comparison") -> None:
    """Require exact metadata and bounded floating-point replay differences."""
    if isinstance(expected, dict) and isinstance(actual, dict) and actual.keys() == expected.keys():
        for key in expected:
            check_equal(actual[key], expected[key], location + "." + key)
    elif isinstance(expected, list) and isinstance(actual, list) and len(actual) == len(expected):
        for index, (value, saved) in enumerate(zip(actual, expected)):
            check_equal(value, saved, f"{location}[{index}]")
    elif isinstance(expected, float) and isinstance(actual, (float, int)):
        if not math.isclose(actual, expected, abs_tol=REPLAY_ABS_TOL, rel_tol=REPLAY_REL_TOL):
            raise ValueError(f"Numerical replay mismatch: {location}")
    elif actual != expected:
        raise ValueError(f"Replay mismatch: {location}")


def json_text(value: dict) -> str:
    return json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--check", action="store_true", help="Verify an existing receipt and replay (default: archived baseline)")
    args = parser.parse_args()
    output = args.output_dir or (BASELINE if args.check else ROOT / "results/gas_source_comparison/replay-001")
    if not args.check and output.exists():
        parser.error(f"Output directory already exists: {output}")
    inputs = load_inputs(args.input)
    hashes = {**inputs["selected_input_sha256"], str(SCRIPT.relative_to(ROOT)): file_hash(SCRIPT),
              str(args.input.resolve().relative_to(ROOT)): file_hash(args.input)}
    if args.check:
        receipt = json.loads((output / "receipt.json").read_text())
        if hashes != receipt["selected_input_sha256"]:
            raise ValueError("Receipt input/code SHA256 mismatch")
        verify_hashes(receipt["selected_input_sha256"])
        for name, expected in receipt["output_sha256"].items():
            if file_hash(output / name) != expected:
                raise ValueError(f"Output SHA256 mismatch: {name}")
    report = compare(inputs)
    rows = summary_rows(report, inputs, inputs["display_temperature_K"])
    historical = check_historical(rows, inputs)
    tables = render_tables(rows)
    if args.check:
        check_equal(report, json.loads((output / "comparison.json").read_text()))
        for name, content in tables.items():
            if (output / name).read_text() != content:
                raise ValueError(f"Table replay mismatch: {name}")
        print(json.dumps({"result": "passed", "samples": len(report["samples"]), "historical_table": historical}))
        return
    output.mkdir(parents=True, exist_ok=False)
    for name, content in {"comparison.json": json_text(report), **tables}.items():
        (output / name).write_text(content)
    import jax
    import jaxlib
    import numpy as np

    receipt = {"schema_version": "1", "snapshot_id": inputs["snapshot_id"],
               "execution_kind": "fresh_portable_execution", "executed_at_utc": datetime.now(timezone.utc).isoformat(),
               "command": [sys.executable, *sys.argv],
               "repository_head": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
               "implementation_identity": "Selected file hashes include uncommitted files; repository HEAD alone does not identify this execution.",
               "environment": {"python": platform.python_version(), "platform": platform.platform(),
                               "numpy": np.__version__, "jax": jax.__version__, "jaxlib": jaxlib.__version__,
                               "backend": jax.default_backend(), "jax_enable_x64": jax.config.jax_enable_x64},
               "selected_input_sha256": hashes,
               "output_sha256": {name: file_hash(output / name) for name in ["comparison.json", *tables]},
               "historical_table_replay": historical,
               "replay_tolerances": {"absolute": REPLAY_ABS_TOL, "relative": REPLAY_REL_TOL,
                                     "metadata_and_display_tables": "exact"}}
    (output / "receipt.json").write_text(json_text(receipt))
    print(json.dumps({"result": "written", "output_dir": str(output), "historical_table": historical}))


if __name__ == "__main__":
    main()
