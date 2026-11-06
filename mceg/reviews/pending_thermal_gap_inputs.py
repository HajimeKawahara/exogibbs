"""Bounded energy and NASA controls for species with unresolved hot thermals."""

import argparse
from datetime import datetime, timezone
import json
import math
from pathlib import Path
import platform
import sys

import numpy as np

from c2h_source_review import read_table
from gas_source_comparison import ROOT, file_hash
from pending_nitrogen_inputs import (
    R, NASA_R, FASTCHEM, independent_properties, nasa_spec, retained_spec,
    source_record as nitrogen_source_record,
)
import pending_candidate as model

SLUGS = ("c2n2", "c3o2", "o3")


def source_record(slug):
    """Balance physical source zeros while preserving original source packets."""
    if slug == "o3":
        return nitrogen_source_record(slug)
    folder = ROOT / f"tests/data/validation/{slug}_source_review/20261106"
    inputs = json.loads((folder / "inputs.json").read_text())
    comparison = json.loads((folder / "comparison.json").read_text())
    name, composition = inputs["name"], inputs["composition"]
    balance = lambda values: math.fsum(n * values[s] for s, n in composition.items()) - values[name]
    tables = {s: read_table(folder / f) for s, f in inputs["janaf_files"].items()}
    jh0 = {s: round(1000 * (table[298.15][4] + table[0.][3])) for s, table in tables.items()}
    atct = inputs["atct"]["species"]
    assert inputs["atct"]["version"] == "1.228"
    ah0 = {s: round(1000 * row["hf0_kJ_mol"]) for s, row in atct.items()}
    nasa = inputs["nasa_records"]
    nh0 = {s: row["hf298_J_mol"] - row["h298_minus_h0_J_mol"] for s, row in nasa.items()}
    jd0, ad0, nd0 = map(balance, (jh0, ah0, nh0))
    assert abs(jd0 - comparison["janaf_D0_J_mol"]) < 1e-8
    assert abs(ad0 - comparison["evaluated_energy"]["D0_J_mol"]) < 1e-8
    assert abs(nd0 - comparison["nasa_D0_J_mol"]) < 1e-8
    lines = FASTCHEM.read_text().splitlines()
    i = next(i for i, line in enumerate(lines) if line.startswith(inputs["species_id"] + " "))
    coefficients = list(map(float, lines[i + 1].split()))
    files = [FASTCHEM, folder / "inputs.json", folder / "comparison.json", folder / "sources.json", folder / "nasa-selected.txt"]
    files.extend(folder / f for f in inputs["janaf_files"].values())
    return {"slug": slug, "molecule": name, "species": inputs["species_id"], "composition": composition,
            "janaf_fixed_reference_h0_J_mol": jh0, "same_version_hf0_J_mol": ah0,
            "janaf_D0_J_mol": jd0, "atct_D0_J_mol": ad0, "delta_D0_J_mol": ad0-jd0,
            "original_coefficients": coefficients, "formal_fit_intercept_Ra_J_mol": R*coefficients[0],
            "nasa_same_identity": True, "nasa_h0_J_mol": nh0, "nasa_D0_J_mol": nd0, "nasa_records": nasa,
            "quoted_ATcT_marginal_uncertainties": {s: row["uncertainty"] for s, row in atct.items()},
            "source_sha256": {str(p.relative_to(ROOT)): file_hash(p) for p in files}}


def build(slug):
    """Define alternatives without assuming NASA resolves the thermal source gap."""
    if slug not in SLUGS:
        raise ValueError("Unknown thermal-gap control")
    record = source_record(slug)
    pins = dict(record["source_sha256"])
    for name in ("pending_thermal_gap_inputs.py", "pending_nitrogen_inputs.py"):
        path = ROOT / "mceg/reviews" / name
        pins[str(path.relative_to(ROOT))] = file_hash(path)
    return {"id": f"{slug}_bounded_candidates_20261106", "slug": slug,
            "baseline_dataset": "curated-ch2-v1", "temperature_range_K": [1500, 4000],
            "standard_pressure_Pa": 100000, "gas_constant_J_mol_K": R, "native_NASA_R_J_mol_K": NASA_R,
            "target_species": record["species"], "tracked_elements": list(record["composition"]),
            "primary_variant": "retained", "variants": [
                {"name": "retained", "replacements": {record["species"]: retained_spec(record)}},
                {"name": "nasa", "replacements": {record["species"]: nasa_spec(record)}}],
            "selected_variants": [], "source_records": {slug: record}, "source_sha256": dict(sorted(pins.items())),
            "scope": "The retained candidate changes only balanced physical ATcT-minus-JANAF D0. NASA is a separate thermal/gauge/fit-residual control at the same ATcT energy; native NASA energy is removed with its original R before SI-R insertion. No second ZPE. Solver agreement cannot establish either thermal model's physical superiority.",
            "limitations": "No invented hot state sums, low-temperature partition extrapolation, independent-quadrature ATcT covariance or full physical uncertainty. The source gap remains documented after the finite substitution calculation.",
            "production_change": "none"}


def validate(definition):
    """Check source arithmetic, caloric derivatives and the named production base."""
    import jax
    from exogibbs.presets import curated_ch2
    assert definition == build(definition["slug"])
    jax.config.update("jax_enable_x64", True)
    setup = curated_ch2.chemsetup(ions=True)
    record = definition["source_records"][definition["slug"]]
    ts = np.unique(np.r_[np.arange(1500., 4001.), np.linspace(1500.137, 3999.813, 251)])
    a, b, c, d, e = record["original_coefficients"]
    literal = -(a/ts + b*np.log(ts) + c + d*ts + e*ts**2)
    runtime_error = float(abs(literal-np.asarray(setup.hvector_func(ts))[:, setup.species.index(record["species"])]).max())
    assert runtime_error < 1e-12
    checks = []
    for variant in definition["variants"]:
        spec = variant["replacements"][record["species"]]
        actual = np.array([list(model.reaction_properties(t, spec).values()) for t in ts])
        expected = np.array([list(independent_properties(record, t, spec["kind"]).values()) for t in ts])
        errors = dict(zip(model.PROPERTY_KEYS, map(float, abs(actual-expected).max(axis=0))))
        assert max(errors["G_J_mol"], errors["H_J_mol"]) < 2e-8  # NASA O3 cancellation: <1.3e-8 J/mol.
        assert max(errors["S_J_mol_K"], errors["Cp_J_mol_K"]) < 2e-10
        checks.append({"variant": variant["name"], "independent_source_property_errors": errors})
    return {"temperature_count": len(ts), "maximum_current_h_error": runtime_error,
            "independent_source_checks": checks, "jax_derivative_checks": model.validate_definition(definition),
            "production_change": "none", "physical_accuracy": "unknown"}


def write_packet(slug):
    """Record the actual validation without modifying earlier execution evidence."""
    import jax
    directory = ROOT / f"tests/data/validation/{slug}_candidate/20261106"
    directory.mkdir(parents=True, exist_ok=True)
    if (directory / "receipt.json").exists():
        raise FileExistsError("Refuse to overwrite executed candidate packet")
    definition = build(slug)
    result = validate(definition)
    for name, data in (("candidate.json", definition), ("validation.json", result)):
        (directory / name).write_text(json.dumps(data, indent=2, allow_nan=False) + "\n")
    paths = {ROOT / p for p in definition["source_sha256"]}
    paths.update(Path(m.__file__).resolve() for m in tuple(sys.modules.values())
                 if getattr(m, "__file__", None) and Path(m.__file__).resolve().is_relative_to(ROOT)
                 and Path(m.__file__).suffix == ".py" and Path(m.__file__).name not in ("curated.py", "exogibbs_version.py"))
    paths.update((ROOT / "src/exogibbs/data/curated").glob("*.json"))
    receipt = {"created_utc": datetime.now(timezone.utc).isoformat(), "command": [sys.executable, *sys.argv],
               "result": "passed", "python": platform.python_version(), "jax": jax.__version__, "numpy": np.__version__,
               "backend": jax.default_backend(), "dtype": "float64",
               "selected_file_sha256": {str(p.relative_to(ROOT)): file_hash(p) for p in sorted(paths)},
               "output_sha256": {name: file_hash(directory / name) for name in ("candidate.json", "validation.json")}}
    (directory / "receipt.json").write_text(json.dumps(receipt, indent=2) + "\n")
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("slug", choices=SLUGS)
    args = parser.parse_args()
    print(json.dumps(write_packet(args.slug), indent=2))
