"""Copy provider-owned evidence into the wheel without importing MCEG.

Run from any directory with ``python mceg/build_catalog.py``. Generated files
are committed so ordinary builds and installations need no evidence tooling.
"""

import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "src/exogibbs/data/mceg"
RECORD_VERSION = "2026-11-02"


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
    c2h_bytes = (ROOT / "src/exogibbs/data/curated/c2h_v1.json").read_bytes()
    c2h_digest = hashlib.sha256(c2h_bytes).hexdigest()
    c2h_data = json.loads(c2h_bytes)
    impact = "tests/data/validation/c2h_equilibrium_impact/20261102"
    receipt = json.loads((ROOT / impact / "receipt.json").read_text())
    if (c2h_digest != receipt["selected_file_sha256"]["src/exogibbs/data/curated/c2h_v1.json"]
            or data_digest != c2h_data["base_data_sha256"]):
        raise ValueError("C2H data changed; review new evidence before rebinding the substitution study")
    c2h_contract = copy("mceg/contracts/curated_c2h.yaml", "contracts/curated_c2h.yaml")
    c2h_summary = copy(f"{impact}/summary.json", "artifacts/c2h_summary.json")
    if c2h_summary["sha256"] != receipt["output_sha256"]["summary.json"]:
        raise ValueError("C2H summary differs from the archived execution")
    c2h_species = {s["id"]: s for s in species}
    c2h_species["C2H1"] = {"id": "C2H1", "aliases": ["C2H", "CCH"]}
    catalog["models"].append({
        "id": "exogibbs.curated_c2h.v1",
        "label": "Curated gas with conditional C2H correction (ion-inclusive study)",
        "dataset": c2h_data["id"],
        "identity": {"data_sha256": c2h_digest, "settings": {"ions": True}},
        "species": [c2h_species[k] for k in sorted(c2h_species)],
        "records": [{
            "id": "c2h_correction", **c2h_contract, "species": ["*", "C2H1"],
            "assessments": ["conditional_selection_review", "single_species_impact_review",
                            "whole_mixture_accuracy_review"],
            "artifacts": [c2h_summary],
            "summary": (
                "Opt-in C2H-only correction of curated-gas-v1: NASA thermal functions and balanced "
                "ATcT 1.172 atomization energy. The archived 75 paired gas-only conditions span "
                "1500-4000 K, 1e-6-100 bar and C/O=0.5888436553555908, 1, 1.5. At C/O=1.5, "
                "2000 K and 1e-6 bar, HCN increases 47.16% and C2H2 increases by a factor of 2.5585. "
                "All species re-equilibrate; this is source sensitivity, not an accuracy ranking. "
                "Conditional source adoption leaves thermal and whole-mixture physical uncertainty "
                "unquantified. The old HCN study does not validate this changed dataset. "
                "Your conditions have not been evaluated."),
            "links": ["https://github.com/HajimeKawahara/exogibbs/pull/310"],
        }],
    })
    (output / "index.json").write_text(json.dumps(catalog, indent=2) + "\n")


if __name__ == "__main__":
    build_catalog()
