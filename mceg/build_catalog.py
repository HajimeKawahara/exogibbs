"""Copy provider-owned evidence into the wheel without importing MCEG.

Run from any directory with ``python mceg/build_catalog.py``. Generated files
are committed so ordinary builds and installations need no evidence tooling.
"""

import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "src/exogibbs/data/mceg"
RECORD_VERSION = "2026-10-28"


def build_catalog(output: Path = OUTPUT) -> None:
    """Generate a deterministic, small catalog from the archived source records."""
    def copy(source: str, destination: str) -> dict:
        content = (ROOT / source).read_bytes()
        target = output / destination
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(content)
        return {"path": destination, "sha256": hashlib.sha256(content).hexdigest()}

    curated = copy("mceg/contracts/curated_gas.yaml", "contracts/curated_gas.yaml")
    hcn = copy("mceg/contracts/fastchem_hcn_followup.yaml", "contracts/fastchem_hcn_followup.yaml")
    summary_pin = copy("tests/data/validation/new_fastchem_hcn/summary.json", "artifacts/hcn_summary.json")
    audit_pin = copy("mceg/reviews/fastchem_hcn_sources_20261028.json", "artifacts/hcn_sources.json")
    summary = json.loads((output / summary_pin["path"]).read_text())
    extrema = [r for r in summary["HCN_extrema"]
               if r["numerator"] == "updated" and r["denominator"] == "curated"]
    differences = "; ".join(
        f"{r['profile']}: {100 * (r['ratio'] - 1):+.2f}% at "
        f"{r['temperature_K']:g} K and {r['pressure_bar']:.6g} bar"
        for r in extrema)
    data_bytes = (ROOT / "src/exogibbs/data/curated/gas_v1.json").read_bytes()
    data_digest = hashlib.sha256(data_bytes).hexdigest()
    archived_digest = summary["selected_file_sha256"]["src/exogibbs/data/curated/gas_v1.json"]
    if data_digest != archived_digest:
        raise ValueError("Curated data changed; review new evidence before rebinding the HCN study")
    data = json.loads(data_bytes)
    species = [{"id": name, "aliases": [record["name"]]
                if record.get("name", name) != name else []}
               for name, record in sorted(data["decisions"].items())]
    catalog = {
        "schema_version": "1",
        "provider": "exogibbs",
        "record_version": RECORD_VERSION,
        "models": [{
            "id": "exogibbs.curated_gas.v1",
            "label": "Curated gas thermochemistry (ion-inclusive comparison)",
            "dataset": data["id"],
            "identity": {"data_sha256": data_digest,
                         "settings": {"ions": True}},
            "species": species,
            "records": [
                {"id": "curated_gas", **curated, "species": ["*"], "assessments": [],
                 "summary": data["interpretation"],
                 "links": ["https://github.com/HajimeKawahara/exogibbs/blob/"
                           "4258624c1b015eeaae4967e8e84e8af25f15c116/documents/presets/curated.rst"]},
                {"id": "hcn_followup", **hcn, "species": ["C1H1N1_hcn"],
                 "assessments": ["hcn_refit_scope_review", "column_closure_and_hcn_factorization_review",
                                 "whole_mixture_accuracy_review"],
                 "artifacts": [summary_pin, audit_pin],
                 "summary": (
                     "Revised preliminary FastChem / curated HCN mole-fraction differences at the "
                     "largest absolute log-ratio points on each sampled profile: " + differences + ". "
                     + summary["scope"] + " Normalized Asplund 2021 composition; the same ExoGibbs "
                     "solver for all tables. Extrema include only points with mole fraction >=1e-12 "
                     "in either table. Differences include coupled atomic abundances. These are "
                     "recorded model sensitivities, not accuracy rankings, error bounds or a "
                     "calibrated uncertainty distribution. Your conditions have not been evaluated."),
                 "links": [summary["source_url"],
                           "https://github.com/HajimeKawahara/exogibbs/issues/297#issuecomment-6457152016",
                           "https://github.com/HajimeKawahara/exogibbs/blob/"
                           "d09e8e16275db5151158b884a53a3ff2c3cd31b6/documents/mceg/fastchem_hcn_followup.rst"]},
            ],
        }],
    }
    (output / "index.json").write_text(json.dumps(catalog, indent=2) + "\n")


if __name__ == "__main__":
    build_catalog()
