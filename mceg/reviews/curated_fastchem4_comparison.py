"""Replay neutral formation-constant ratios against updated FastChem 4.0.4."""

from __future__ import annotations

import argparse
from collections import Counter
import csv
from datetime import datetime, timezone
from hashlib import sha256
import json
import math
import os
from pathlib import Path
import platform
import sys
import zipfile

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "tests/data/validation/curated_fastchem4_comparison/20261106"
BASELINE = ROOT / "tests/data/validation/gas_source_comparison/baseline-001/inputs.json"
TABLE = DATA / "source/fastchem-v4.0.4-logK.dat"
TEMPERATURES = (1500., 2000., 2500., 3000., 4000.)
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "examples/comparisons"))


def digest(path: Path) -> str:
    return sha256(path.read_bytes()).hexdigest()


def load_inputs() -> dict:
    """Reject source drift before evaluating the frozen comparison."""
    inputs = json.loads((DATA / "inputs.json").read_text())
    for name, expected in inputs["selected_file_sha256"].items():
        if digest(ROOT / name) != expected:
            raise ValueError(f"SHA256 mismatch: {name}")
    if inputs["curated_dataset"] != "curated-gas-v2" or inputs["standard_pressure_Pa"] != 100000:
        raise ValueError("Unexpected model or standard pressure")
    return inputs


def scalar_logk(temperature: float, coefficients: list) -> float:
    """Evaluate the published natural-log formation fit in scalar arithmetic."""
    a, b, c, d, e = coefficients
    return a / temperature + b * math.log(temperature) + c + d * temperature + e * temperature**2


def grade(ratio: float) -> str:
    factor = max(ratio, 1 / ratio)
    return "red" if factor >= 10 else "yellow" if factor >= 2 else "green"


def compute() -> dict:
    """Compare 108 explicit molecular records without solving abundances."""
    inputs = load_inputs()
    os.environ.setdefault("JAX_PLATFORMS", "cpu")
    import jax
    import numpy as np
    from exogibbs.presets import curated_v2, fastchem4
    from exogibbs.presets.fastchem import _parse_fastchem_coeffs_with_metadata
    from audit_atomic_ions import nasa9_g_rt

    jax.config.update("jax_enable_x64", True)
    curated = curated_v2.chemsetup(ions=False)
    if curated.metadata["dataset"] != inputs["curated_dataset"]:
        raise ValueError("Curated dataset changed")
    targets = json.loads((DATA / "targets.json").read_text())
    ids = [row["species_id"] for row in targets]
    if len(ids) != 108 or len(set(ids)) != len(ids):
        raise ValueError("Expected 108 unique tracked molecular records")
    coefficients, compositions, records = _parse_fastchem_coeffs_with_metadata(TABLE.read_text())
    # The provider accepts package-relative paths. Resolve the immutable fixture
    # explicitly rather than silently using its older default wo_ions table.
    relative = os.path.relpath(TABLE, ROOT / "src/exogibbs/data")
    fastchem = fastchem4.chemsetup(path=relative, silent=True)
    baseline = json.loads(BASELINE.read_text())
    screened = {row["id"]: row for row in baseline["species"]}
    matrix = np.asarray(curated.formula_matrix)
    for target in targets:
        name = target["species_id"]
        index = curated.species.index(name)
        actual = {element: int(matrix[i, index]) for i, element in enumerate(curated.elements)
                  if matrix[i, index]}
        if name not in coefficients or actual != compositions[name] or "e-" in actual:
            raise ValueError(f"Neutral record/composition mismatch: {name}")
        if name in screened and actual != screened[name]["composition"]:
            raise ValueError(f"Historical NASA mapping changed: {name}")
        target["composition"] = actual
        target["fastchem_record"] = records[name]["record_line"]
        target["fastchem_coefficients"] = coefficients[name]

    hcn_zip = ROOT / "tests/data/validation/new_fastchem_hcn/logK_hcn.zip"
    with zipfile.ZipFile(hcn_zip) as archive:
        if archive.read("logK.dat") != TABLE.read_bytes():
            raise ValueError("Official full table differs from the archived HCN update")

    def nasa_logk(name, temperature):
        row = screened[name]
        def g(identifier):
            record = baseline["nasa9_records"][identifier]
            for interval in record["intervals"]:
                if interval["bounds_K"][0] <= temperature <= interval["bounds_K"][1]:
                    return nasa9_g_rt(temperature, interval)
            raise ValueError(f"NASA temperature outside source domain: {identifier}")
        return sum(count * g(baseline["atomic_references"][element]["nasa_id"])
                   for element, count in row["composition"].items()) - g(row["nasa_id"])

    samples, maximum_provider_error = [], 0.
    for temperature in TEMPERATURES:
        current = -np.asarray(curated.hvector_func(temperature))
        native = -np.asarray(fastchem.hvector_func(temperature))
        for element in curated.elements:
            atom = element + "1"
            if current[curated.species.index(atom)] != 0 or native[fastchem.species.index(atom)] != 0:
                raise ValueError(f"Neutral-atom reference gauge changed: {element}")
        for target in targets:
            name = target["species_id"]
            current_k = float(current[curated.species.index(name)])
            fastchem_k = scalar_logk(temperature, coefficients[name])
            error = abs(fastchem_k - float(native[fastchem.species.index(name)]))
            maximum_provider_error = max(maximum_provider_error, error)
            difference = current_k - fastchem_k
            ratio = math.exp(difference)
            nasa_k = nasa_logk(name, temperature) if name in screened else None
            nasa_ratio = math.exp(current_k - nasa_k) if nasa_k is not None else None
            if not all(math.isfinite(value) for value in (current_k, fastchem_k, ratio)):
                raise ValueError(f"Non-finite comparison: {name}")
            samples.append({"species_id": name, "molecule": target["molecule"],
                            "scope": target["scope"], "temperature_K": temperature,
                            "ln_K_curated": current_k, "ln_K_fastchem4": fastchem_k,
                            "log10_curated_over_fastchem4": difference / math.log(10),
                            "curated_over_fastchem4": ratio, "fastchem4_grade": grade(ratio),
                            "fastchem4_major_10pct": target["major_molecule"] and (ratio <= .9 or ratio >= 1.1),
                            "ln_K_frozen_NASA": nasa_k, "curated_over_frozen_NASA": nasa_ratio,
                            "nasa_grade": grade(nasa_ratio) if nasa_ratio is not None else None,
                            "nasa_major_10pct": bool(target["major_molecule"] and nasa_ratio is not None
                                                     and (nasa_ratio <= .9 or nasa_ratio >= 1.1))})
    if maximum_provider_error > 2e-11:
        raise ValueError("Scalar and native FastChem providers disagree")
    table = [row for row in samples if row["temperature_K"] == 1500]
    main = [row for row in table if row["scope"] == "main93"]
    return {"curated_dataset": inputs["curated_dataset"], "fastchem_version": "4.0.4",
            "fastchem_commit": inputs["fastchem_commit"], "standard_pressure_Pa": 100000,
            "ratio_definition": "Kp(curated-gas-v2) / Kp(FastChem 4.0.4 full logK.dat)",
            "reaction": "neutral gas atoms -> one neutral gas molecule",
            "temperatures_K": list(TEMPERATURES), "molecule_count": len(targets),
            "sample_count": len(samples), "equilibrium_abundance_solves": 0,
            "native_provider_max_abs_ln_K_error": maximum_provider_error,
            "updated_HCN_archive_full_table_identical": True,
            "grades_main93_1500K": dict(Counter(row["fastchem4_grade"] for row in main)),
            "grades_all108_1500K": dict(Counter(row["fastchem4_grade"] for row in table)),
            "largest_differences_1500K": sorted(table, key=lambda row: abs(row["log10_curated_over_fastchem4"]), reverse=True)[:12],
            "HCN": [row for row in samples if row["species_id"] == "C1H1N1_hcn"],
            "targets": targets, "samples": samples}


def write_report(report: dict, output: Path) -> None:
    output.mkdir(parents=True, exist_ok=True)
    (output / "comparison.json").write_text(json.dumps(report, indent=2) + "\n")
    with (output / "comparison.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(report["samples"][0]))
        writer.writeheader()
        writer.writerows(report["samples"])
    with (output / "table1500.csv").open("w", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(["Molecule", "Curated / FastChem 4.0.4", "Grade", "Curated / frozen NASA", "NASA grade"])
        for row in report["samples"]:
            if row["temperature_K"] == 1500:
                nasa = row["curated_over_frozen_NASA"]
                writer.writerow([row["molecule"], f"{row['curated_over_fastchem4']:.8g}",
                                 row["fastchem4_grade"], "" if nasa is None else f"{nasa:.8g}", row["nasa_grade"] or ""])
    import jax
    import numpy
    receipt = {"created_utc": datetime.now(timezone.utc).isoformat(), "result": "passed",
               "command": ["python", "mceg/reviews/curated_fastchem4_comparison.py", "--output", str(output.relative_to(ROOT))],
               "python": platform.python_version(), "numpy": numpy.__version__, "jax": jax.__version__,
               "backend": jax.default_backend(), "dtype": "float64", "equilibrium_abundance_solves": 0,
               "inputs_sha256": digest(DATA / "inputs.json"),
               "selected_file_sha256": load_inputs()["selected_file_sha256"],
               "output_sha256": {name: digest(output / name) for name in ("comparison.json", "comparison.csv", "table1500.csv")}}
    (output / "receipt.json").write_text(json.dumps(receipt, indent=2) + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT / "results/curated_fastchem4_comparison")
    args = parser.parse_args()
    result = compute()
    write_report(result, args.output.resolve())
    print(json.dumps({"molecules": result["molecule_count"], "samples": result["sample_count"],
                      "HCN_1500K": result["HCN"][0]["curated_over_fastchem4"],
                      "grades": result["grades_main93_1500K"]}))
