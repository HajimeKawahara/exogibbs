"""Copy provider-owned evidence into the wheel without importing MCEG.

Run from any directory with ``python mceg/build_catalog.py``. Generated files
are committed so ordinary builds and installations need no evidence tooling.
"""

import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "src/exogibbs/data/mceg"
RECORD_VERSION = "2026-11-05"


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
    adoption = "tests/data/validation/curated_default_adoption/20261102"
    default_receipt = copy(f"{adoption}/receipt.json", "artifacts/c2h_default_receipt.json")
    default_summary = copy(f"{adoption}/summary.json", "artifacts/c2h_default_summary.json")
    adopted = json.loads((ROOT / adoption / "receipt.json").read_text())
    if (default_summary["sha256"] != adopted["output_sha256"]["summary.json"]
            or c2h_digest != adopted["selected_file_sha256"]["src/exogibbs/data/curated/c2h_v1.json"]):
        raise ValueError("Default adoption differs from the archived execution")
    c2h_species = {s["id"]: s for s in species}
    c2h_species["C2H1"] = {"id": "C2H1", "aliases": ["C2H", "CCH"]}
    catalog["models"].append({
        "id": "exogibbs.curated_c2h.v1",
        "label": "Versioned curated C2H correction (ion-inclusive lookup)",
        "dataset": c2h_data["id"],
        "identity": {"data_sha256": c2h_digest, "settings": {"ions": True}},
        "species": [c2h_species[k] for k in sorted(c2h_species)],
        "records": [{
            "id": "c2h_correction", **c2h_contract, "species": ["*", "C2H1"],
            "assessments": ["conditional_selection_review", "single_species_impact_review",
                            "whole_mixture_accuracy_review", "default_adoption_review"],
            "artifacts": [c2h_summary, default_receipt, default_summary],
            "summary": (
                "Former curated default, retained as curated_c2h: NASA C2H thermal functions and balanced ATcT 1.172 "
                "atomization energy supersede the 1967 JANAF estimate. Energy alignment alone "
                "does not remove the old thermal discrepancy. curated_v1 preserves the original. "
                "New cold solves pass in both networks (75 conditions each); ion-inclusive results "
                "reproduce the archived corrected candidate. The archived 75 paired gas-only conditions span "
                "1500-4000 K, 1e-6-100 bar and C/O=0.5888436553555908, 1, 1.5. At C/O=1.5, "
                "2000 K and 1e-6 bar, HCN increases 47.16% and C2H2 increases by a factor of 2.5585. "
                "All species re-equilibrate; this is source sensitivity, not an accuracy ranking. "
                "Conditional source adoption leaves thermal and whole-mixture physical uncertainty "
                "unquantified. The old HCN study does not validate this changed dataset. "
                "Your conditions have not been evaluated."),
            "links": ["https://github.com/HajimeKawahara/exogibbs/pull/310"],
        }],
    })
    tio_path = "src/exogibbs/data/curated/tio_v1.json"
    tio_bytes = (ROOT / tio_path).read_bytes()
    tio_data = json.loads(tio_bytes)
    tio_digest = hashlib.sha256(tio_bytes).hexdigest()
    tio_adoption = "tests/data/validation/curated_tio_adoption/20261103"
    tio_execution = json.loads((ROOT / tio_adoption / "receipt.json").read_text())
    if (tio_digest != tio_execution["selected_file_sha256"][tio_path]
            or c2h_digest != tio_data["base_data_sha256"]):
        raise ValueError("TiO data changed; review new evidence before rebinding the adoption")
    tio_contract = copy("mceg/contracts/curated_tio.yaml", "contracts/curated_tio.yaml")
    tio_receipt = copy(f"{tio_adoption}/receipt.json", "artifacts/tio_adoption_receipt.json")
    tio_summary = copy(f"{tio_adoption}/summary.json", "artifacts/tio_adoption_summary.json")
    if tio_summary["sha256"] != tio_execution["output_sha256"]["summary.json"]:
        raise ValueError("TiO summary differs from the archived execution")
    catalog["models"].append({
        "id": "exogibbs.curated_tio.v1",
        "label": "Versioned curated TOTO/Jain TiO (ion-inclusive lookup)",
        "dataset": tio_data["id"],
        "identity": {"data_sha256": tio_digest, "settings": {"ions": True}},
        "species": [c2h_species[k] for k in sorted(c2h_species)],
        "records": [{
            "id": "tio_adoption", **tio_contract, "species": ["*", "O1Ti1"],
            "assessments": ["scientific_selection_review", "production_adoption_review",
                            "physical_accuracy_review"],
            "artifacts": [tio_receipt, tio_summary],
            "summary": (
                "Former default, retained as curated_tio: curated-tio-v1 replaces only TiO in curated-c2h-v1 with TOTO "
                "state thermodynamics and Jain 2023 D0=6.824 +/- 0.010 eV, on 1500-4000 K. "
                "The representative 48Ti16O internal partition uses NASA elemental mass "
                "and atomic reference functions. C2H and all other functions are retained. "
                "The obsolete JANAF low-singlet assumption motivates the thermal update; "
                "the D0 difference is within the former quoted uncertainty. Both factory "
                "modes pass 75 new cold solves, and ion-inclusive results reproduce the "
                "reviewed candidate. Physical atmospheric accuracy and complete "
                "thermochemical uncertainty remain unquantified. curated_c2h and curated_v1 "
                "preserve earlier datasets. Your conditions have not been evaluated."),
            "links": ["https://github.com/HajimeKawahara/exogibbs/pull/325"],
        }],
    })
    co2_path = "src/exogibbs/data/curated/co2_v1.json"
    co2_bytes = (ROOT / co2_path).read_bytes()
    co2_data = json.loads(co2_bytes)
    co2_digest = hashlib.sha256(co2_bytes).hexdigest()
    co2_adoption = "tests/data/validation/curated_co2_adoption/20261104"
    co2_execution = json.loads((ROOT / co2_adoption / "receipt.json").read_text())
    if (co2_digest != co2_execution["selected_file_sha256"][co2_path]
            or tio_digest != co2_data["base_data_sha256"]):
        raise ValueError("CO2 data changed; review new evidence before rebinding the adoption")
    co2_contract = copy("mceg/contracts/curated_co2.yaml", "contracts/curated_co2.yaml")
    co2_receipt = copy(f"{co2_adoption}/receipt.json", "artifacts/co2_adoption_receipt.json")
    co2_summary = copy(f"{co2_adoption}/summary.json", "artifacts/co2_adoption_summary.json")
    if co2_summary["sha256"] != co2_execution["output_sha256"]["summary.json"]:
        raise ValueError("CO2 summary differs from the archived execution")
    catalog["models"].append({
        "id": "exogibbs.curated_co2.v1",
        "label": "Versioned curated NIST CO2 (ion-inclusive lookup)",
        "dataset": co2_data["id"],
        "identity": {"data_sha256": co2_digest, "settings": {"ions": True}},
        "species": [c2h_species[k] for k in sorted(c2h_species)],
        "records": [{
            "id": "co2_adoption", **co2_contract, "species": ["*", "C1O2"],
            "assessments": ["scientific_selection_review", "production_adoption_review",
                            "physical_accuracy_review"],
            "artifacts": [co2_receipt, co2_summary],
            "summary": (
                "Former default, retained as curated_co2: curated-co2-v1 adopts the exact reviewed NIST 626 thermal model "
                "on curated-tio-v1 at 1500-4000 K. NASA molecular mass 44.0095 u, NASA "
                "atomic functions and ATcT 1.172 total atomization 1598.180 kJ/mol are retained. "
                "Only CO2 thermochemistry changes; TiO, C2H, all other species and budgets are preserved. "
                "Both factory modes pass 75 new cold solves, and ion-inclusive results reproduce "
                "the reviewed candidate. Scoped scientific selection and numerical adoption are "
                "supported; complete isotope, continuum, high-temperature cutoff and atmospheric "
                "physical accuracy remain unknown. The fixed-isotope-average diagnostic is not "
                "adopted as a natural-isotope correction. curated_tio, curated_c2h and curated_v1 "
                "preserve earlier datasets. Your conditions have not been evaluated."),
            "links": ["https://github.com/HajimeKawahara/exogibbs/pull/329"],
        }],
    })
    h2o_path = "src/exogibbs/data/curated/h2o_v1.json"
    h2o_bytes = (ROOT / h2o_path).read_bytes()
    h2o_data = json.loads(h2o_bytes)
    h2o_digest = hashlib.sha256(h2o_bytes).hexdigest()
    h2o_adoption = "tests/data/validation/curated_h2o_adoption/20261104"
    h2o_execution = json.loads((ROOT / h2o_adoption / "receipt.json").read_text())
    if (h2o_digest != h2o_execution["selected_file_sha256"][h2o_path]
            or co2_digest != h2o_data["base_data_sha256"]):
        raise ValueError("H2O data changed; review new evidence before rebinding the adoption")
    h2o_contract = copy("mceg/contracts/curated_h2o.yaml", "contracts/curated_h2o.yaml")
    h2o_receipt = copy(f"{h2o_adoption}/receipt.json", "artifacts/h2o_adoption_receipt.json")
    h2o_summary = copy(f"{h2o_adoption}/summary.json", "artifacts/h2o_adoption_summary.json")
    if h2o_summary["sha256"] != h2o_execution["output_sha256"]["summary.json"]:
        raise ValueError("H2O summary differs from the archived execution")
    catalog["models"].append({
        "id": "exogibbs.curated_h2o.v1",
        "label": "Versioned curated evaluated H2O and NIST CO2 (ion-inclusive lookup)",
        "dataset": h2o_data["id"],
        "identity": {"data_sha256": h2o_digest, "settings": {"ions": True}},
        "species": [c2h_species[k] for k in sorted(c2h_species)],
        "records": [{
            "id": "h2o_adoption", **h2o_contract, "species": ["*", "H2O1"],
            "assessments": ["scientific_selection_review", "production_adoption_review",
                            "physical_accuracy_review"],
            "artifacts": [h2o_receipt, h2o_summary],
            "summary": (
                "Former default, retained as curated_h2o: curated-h2o-v1 adopts the exact reviewed evaluated-2025 water "
                "Cp polynomial and analytic H/S integrals on curated-co2-v1 at 1500-4000 K. "
                "The representative H2(16O) internal model uses NASA aggregate mass 18.01528 u, "
                "NASA atoms and retained ATcT 1.172 total atomization 917.819 kJ/mol. "
                "CO2, TiO, C2H and all other formation functions are preserved from the base. "
                "Joint CO2/H2O production passes 150 new cold solves across both networks; "
                "six tighter independently assembled candidate solves and two FastChem "
                "comparisons check numerical response. Earlier single-species responses "
                "are not relabeled as joint results. Scoped scientific selection and numerical "
                "adoption are supported; scattering assignment, complete isotope treatment "
                "and physical atmospheric accuracy remain unknown. Earlier versioned presets "
                "and their evidence remain available. Your conditions have not been evaluated."),
            "links": ["https://github.com/HajimeKawahara/exogibbs/pull/330"],
        }],
    })
    oh_path = "src/exogibbs/data/curated/oh_v1.json"
    oh_bytes = (ROOT / oh_path).read_bytes()
    oh_data = json.loads(oh_bytes)
    oh_digest = hashlib.sha256(oh_bytes).hexdigest()
    oh_adoption = "tests/data/validation/curated_oh_adoption/20261105"
    oh_execution = json.loads((ROOT / oh_adoption / "receipt.json").read_text())
    if (oh_digest != oh_execution["selected_file_sha256"][oh_path]
            or h2o_digest != oh_data["base_data_sha256"]):
        raise ValueError("OH data changed; review new evidence before rebinding the adoption")
    oh_contract = copy("mceg/contracts/curated_oh.yaml", "contracts/curated_oh.yaml")
    oh_receipt = copy(f"{oh_adoption}/receipt.json", "artifacts/oh_adoption_receipt.json")
    oh_summary = copy(f"{oh_adoption}/summary.json", "artifacts/oh_adoption_summary.json")
    if oh_summary["sha256"] != oh_execution["output_sha256"]["summary.json"]:
        raise ValueError("OH summary differs from the archived execution")
    oh_species = {**c2h_species, "H1O1": {"id": "H1O1", "aliases": ["OH"]}}
    catalog["models"].append({
        "id": "exogibbs.curated_oh.v1",
        "label": "Curated default with NASA/ATcT OH (ion-inclusive lookup)",
        "dataset": oh_data["id"],
        "identity": {"data_sha256": oh_digest, "settings": {"ions": True}},
        "species": [oh_species[k] for k in sorted(oh_species)],
        "records": [{
            "id": "oh_adoption", **oh_contract, "species": ["*", "H1O1"],
            "assessments": ["scientific_selection_review", "production_adoption_review",
                            "physical_accuracy_review"],
            "artifacts": [oh_receipt, oh_summary],
            "summary": (
                "Former default, retained as curated_oh: curated-oh-v1 replaces the inherited JANAF OH electronic/caloric defect "
                "with the exact NASA9 reaction and balanced ATcT 1.172 D0=425.604 kJ/mol, "
                "on curated-h2o-v1 at 1500-4000 K. Only OH changes; H2O, CO2, TiO, C2H, "
                "other formation functions, species and budgets are retained. MYTHOS is a "
                "finite-state thermal control, differing by less than 0.098 percent in K; "
                "it is not adopted. Both network modes pass 75 fresh cold solves, six tighter "
                "candidate controls and two independent FastChem comparisons. Scoped source "
                "selection and numerical adoption are supported; complete isotope, continuum, "
                "energy covariance and atmospheric physical accuracy remain unknown. "
                "Earlier versioned presets and immutable evidence remain available. "
                "Your conditions have not been evaluated."),
            "links": ["https://github.com/HajimeKawahara/exogibbs/pull/334"],
        }],
    })
    kh_path = "src/exogibbs/data/curated/kh_v1.json"
    kh_bytes = (ROOT / kh_path).read_bytes()
    kh_data = json.loads(kh_bytes)
    kh_digest = hashlib.sha256(kh_bytes).hexdigest()
    kh_adoption = "tests/data/validation/curated_kh_adoption/20261105"
    kh_execution = json.loads((ROOT / kh_adoption / "receipt.json").read_text())
    if (kh_digest != kh_execution["selected_file_sha256"][kh_path]
            or oh_digest != kh_data["base_data_sha256"]):
        raise ValueError("KH data changed; review new evidence before rebinding the adoption")
    kh_contract = copy("mceg/contracts/curated_kh.yaml", "contracts/curated_kh.yaml")
    kh_receipt = copy(f"{kh_adoption}/receipt.json", "artifacts/kh_adoption_receipt.json")
    kh_summary = copy(f"{kh_adoption}/summary.json", "artifacts/kh_adoption_summary.json")
    if kh_summary["sha256"] != kh_execution["output_sha256"]["summary.json"]:
        raise ValueError("KH summary differs from the archived execution")
    kh_species = {**oh_species, "H1K1": {"id": "H1K1", "aliases": ["KH"]}}
    catalog["models"].append({
        "id": "exogibbs.curated_kh.v1",
        "label": "Curated default with spectroscopic KH energy (ion-inclusive lookup)",
        "dataset": kh_data["id"],
        "identity": {"data_sha256": kh_digest, "settings": {"ions": True}},
        "species": [kh_species[k] for k in sorted(kh_species)],
        "records": [{
            "id": "kh_adoption", **kh_contract, "species": ["*", "H1K1"],
            "assessments": ["scientific_selection_review", "production_adoption_review",
                            "physical_accuracy_review"],
            "artifacts": [kh_receipt, kh_summary],
            "summary": (
                "Default curated-kh-v1 adopts spectroscopic KH D0=170857.838639 J/mol "
                "on curated-oh-v1 at 1500-4000 K, reducing the inherited balanced energy "
                "by 9532.161361 J/mol while retaining thermal functions, S/Cp and atomic "
                "conventions. Only KH changes; OH, H2O, CO2, TiO, C2H, species and "
                "budgets remain fixed. Both networks pass 75 fresh cold solves, six tighter "
                "candidate controls and two FastChem comparisons. The 1988 abstract's "
                "De uncertainty is not a D0 uncertainty. NASA thermal-reference controls "
                "change molecular and atomic thermal functions and are not adopted. "
                "Scoped source selection and numerical adoption are supported; complete "
                "D0/thermal/isotope/continuum and atmospheric physical accuracy remain unknown. "
                "Earlier versioned presets and immutable evidence remain available. "
                "Your conditions have not been evaluated."),
            "links": ["https://github.com/HajimeKawahara/exogibbs/pull/339"],
        }],
    })
    (output / "index.json").write_text(json.dumps(catalog, indent=2) + "\n")


if __name__ == "__main__":
    build_catalog()
