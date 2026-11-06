"""Explicit source-energy ledgers for the remaining carbon/nitrogen candidates.

ATcT atomization, a directly measured diatomic D0, a NASA-derived corrected
anchor, and a finite H298 formation shift remain distinct constructions.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np

from c2h_source_review import nasa_properties, read_table
from gas_source_comparison import ROOT, file_hash

R, NASA_R = 8.31446261815324, 8.314510
SLUGS = ("nh2", "c2", "hco", "c2o", "c4n2", "cnn", "ncn", "cnc", "hnco", "nco", "cis_h2n2", "vn", "p4s3")
NASA_ALTERNATIVES = {"nh2", "hco", "c2o", "c4n2", "cnn", "ncn", "cnc"}
FASTCHEM = ROOT/"src/exogibbs/data/FastChem4/logK/logK.dat"


def source_record(slug: str, directory: Path | None = None) -> dict:
    """Reconstruct source energies from printed anchors and exact identity."""
    directory = directory or ROOT/f"tests/data/validation/{slug}_source_review/20261106"
    inputs = json.loads((directory/"inputs.json").read_text())
    name = inputs.get("molecule", inputs.get("name"))
    species, composition = inputs["species_id"], inputs["composition"]
    table_files = inputs.get("tables", inputs.get("janaf_files"))
    tables = {n: read_table(directory/f) for n, f in table_files.items()}
    balance = lambda values: math.fsum(count*values[n] for n, count in composition.items())-values[name]
    h0 = {n: (round(1000*(tb[298.15][4]+tb[0.][3]))
              if 0. in tb and math.isfinite(tb[0.][3]) else None) for n, tb in tables.items()}
    old_d0 = balance(h0) if all(v is not None for v in h0.values()) else None
    nasa = inputs.get("nasa_records", inputs.get("nasa9"))
    if nasa is None:
        nasa = json.loads((directory/"nasa_records.json").read_text())
    same_identity = inputs.get("nasa_same_identity", True)
    nh0 = {n: rec["hf298_J_mol"]-rec["h298_minus_h0_J_mol"] for n, rec in nasa.items()}
    nasa_d0 = balance(nh0) if nasa and same_identity else None
    lines = FASTCHEM.read_text().splitlines()
    i = next(i for i, line in enumerate(lines) if line.startswith(species+" "))
    coefficients = list(map(float, lines[i+1].split()))
    if "fastchem_record" in inputs:
        assert coefficients == inputs["fastchem_record"]["coefficients"]
    source_details = {}
    selected_d0, anchor_type = None, None
    if "atct" in inputs:
        atct = inputs["atct"]
        version = atct.get("version", inputs.get("atct_version"))
        assert version == "1.228"
        rows = atct.get("species", atct)
        formation = {n: round(1000*row["hf0_kJ_mol"]) for n, row in rows.items()}
        selected_d0 = balance(formation)
        delta_gh = old_d0-selected_d0
        anchor_type = "ATcT1.228_balanced_D0"
        source_details = {"version": version, "same_version_hf0_J_mol": formation,
                          "original_ATcT_rows": rows,
                          "uncertainty_scope": "Marginal source uncertainties are not independent reaction errors; covariance is unavailable."}
    elif slug == "c4n2":
        selected_d0 = nasa_d0
        delta_gh = old_d0-selected_d0
        anchor_type = "corrected_energy_NASA_physical_D0_control"
        source_details = json.loads((directory/"correction_evidence.json").read_text())
        source_details["scope"] = "Choose the available NASA balanced physical D0 as a corrected-energy control. NIST's corrected529.3 and NASA529.2 kJ/mol Hf298 remain distinct; this is not a new ATcT result."
    elif slug == "vn":
        literature = json.loads((directory/"literature.json").read_text())
        selected_d0 = literature["neutral_2016"]["D0_eV"]*literature["eV_to_J_mol"]
        delta_gh = old_d0-selected_d0
        anchor_type = "neutral_VN_measured_D0"
        source_details = {"measurement": literature["neutral_2016"],
                          "eV_to_J_mol": literature["eV_to_J_mol"],
                          "reported_D0_width_J_mol": literature["neutral_2016"]["reported_uncertainty_eV"]*literature["eV_to_J_mol"],
                          "independent_theory": literature["theory_2023"]}
    elif slug == "p4s3":
        literature = json.loads((directory/"literature.json").read_text())
        assert old_d0 is None and not nasa
        old_h298 = round(1000*tables[name][298.15][4])
        preferred = literature["kroeger_karton2019"]["preferred_Hf298_J_mol"]
        delta_gh = preferred-old_h298
        anchor_type = "preferred_reaction_cycle_Hf298"
        for element in ("P", "S"):
            tb = tables[element]
            heat = -1000*tb[0.][3]-1000*(tb[298.15][4]-tb[0.][4])
            assert abs(heat-literature["elemental_reference"][element]["H298_minus_H0_J_mol"]) < 1e-8
        source_details = {"original_Hf298_J_mol": old_h298,
                          "selected_Hf298_J_mol": preferred,
                          "preferred_cycle": literature["kroeger_karton2019"],
                          "elemental_reference": literature["elemental_reference"],
                          "D0_available": False,
                          "uncertainty_scope": "Method/cycle differences are sensitivities; atomization-route TAE0 uncertainty does not belong to the preferred cycle."}
    else:
        raise ValueError("No explicit selected energy source")
    paths = [FASTCHEM, *[p for p in directory.iterdir() if p.suffix in (".json", ".txt", ".tsv") and p.name != "receipt.json"]]
    return {"slug": slug, "molecule": name, "species": species, "composition": composition,
            "anchor_type": anchor_type, "janaf_fixed_reference_h0_J_mol": h0,
            "janaf_D0_J_mol": old_d0, "selected_D0_J_mol": selected_d0,
            "delta_D0_J_mol": selected_d0-old_d0 if old_d0 is not None else None,
            "delta_GH_J_mol": delta_gh, "original_coefficients": coefficients,
            "formal_fit_intercept_Ra_J_mol": R*coefficients[0],
            "nasa_same_identity": same_identity, "nasa_h0_J_mol": nh0,
            "nasa_D0_J_mol": nasa_d0, "nasa_records": nasa, "source_details": source_details,
            "source_sha256": {str(p.relative_to(ROOT)): file_hash(p) for p in sorted(paths)}}


def retained_spec(record: dict, additional_GH_J_mol: float = 0.) -> dict:
    """Retain all thermal/fit terms and add the explicit physical energy shift."""
    return {"kind": "retained_energy", "coefficients": record["original_coefficients"],
            "delta_GH_J_mol": record["delta_GH_J_mol"]+additional_GH_J_mol}


def nasa_spec(record: dict) -> dict:
    """Build the exact identity-matched NASA reaction with one energy alignment."""
    if not record["nasa_same_identity"] or record["selected_D0_J_mol"] is None:
        raise ValueError("NASA identity and physical zero-K anchors must both be available")
    coefficients = {n: np.asarray(next(row["coefficients"] for row in rec["intervals"]
                    if row["bounds_K"] == [1000., 6000.])) for n, rec in record["nasa_records"].items()}
    reaction = coefficients[record["molecule"]].copy()
    for element, count in record["composition"].items():
        reaction -= count*coefficients[element]
    correction = record["selected_D0_J_mol"]/R-record["nasa_D0_J_mol"]/NASA_R
    reaction[7] -= correction
    return {"kind": "nasa9", "coefficients": reaction.tolist(),
            "native_energy_removal_D0_over_R_K": record["nasa_D0_J_mol"]/NASA_R,
            "selected_energy_insertion_D0_over_R_K": record["selected_D0_J_mol"]/R}


def independent_properties(record: dict, temperature_K: float, kind: str, additional_GH_J_mol: float = 0.) -> dict:
    """Expand each source separately, rather than call the candidate evaluator."""
    t = float(temperature_K)
    if kind == "retained_energy":
        a, b, c, d, e = record["original_coefficients"]
        shift = record["delta_GH_J_mol"]+additional_GH_J_mol
        g = -R*(a+b*t*math.log(t)+c*t+d*t*t+e*t**3)+shift
        heat = R*(-a+b*t+d*t*t+2*e*t**3)+shift
        entropy = R*(b*math.log(t)+b+c+2*d*t+3*e*t*t)
        cp = R*(b+2*d*t+6*e*t*t)
    elif kind == "nasa9":
        props = {n: nasa_properties(rec, t, NASA_R) for n, rec in record["nasa_records"].items()}
        reaction = [math.fsum([props[record["molecule"]][i], *(-count*props[e][i] for e, count in record["composition"].items())]) for i in range(3)]
        correction = record["selected_D0_J_mol"]/R-record["nasa_D0_J_mol"]/NASA_R
        heat, entropy, cp = R*(reaction[0]/NASA_R-correction), R*reaction[1]/NASA_R, R*reaction[2]/NASA_R
        g = heat-t*entropy
    else:
        raise ValueError("Unknown independent source representation")
    return dict(zip(("G_J_mol", "H_J_mol", "S_J_mol_K", "Cp_J_mol_K"), (g, heat, entropy, cp)))


def build(slug: str) -> dict:
    """Define independent complete grids and narrowly scoped selected controls."""
    if slug not in SLUGS:
        raise ValueError("Unknown molecule candidate")
    record = source_record(slug)
    sources = {slug: record}
    variants = [{"name": "retained", "replacements": {record["species"]: retained_spec(record)}}]
    if slug in NASA_ALTERNATIVES:
        variants.append({"name": "nasa", "replacements": {record["species"]: nasa_spec(record)}})
    selected, special = [], {}
    if slug == "nco":
        other = source_record("hnco", ROOT/"tests/data/validation/nco_candidate/20261106/source/hnco_source_review")
        sources["hnco"] = other
        variants.append({"name": "joint_hnco_nco", "replacements": {
            record["species"]: retained_spec(record), other["species"]: retained_spec(other)}})
        special = {"dissociation_reaction": "HNCO -> H + NCO",
                   "dissociation_delta_GH_J_mol": record["delta_GH_J_mol"]-other["delta_GH_J_mol"],
                   "scope": "HNCO-only belongs to its separate candidate; this independently reconstructs both source records for a matched joint control, without importing another new candidate packet."}
    if slug == "vn":
        width = record["source_details"]["reported_D0_width_J_mol"]
        selected = [{"name": name, "replacements": {record["species"]: retained_spec(record, -sign*width)}}
                    for sign, name in ((-1, "lower_measured_D0"), (1, "upper_measured_D0"))]
        special = {"reported_width_J_mol": width, "scope": "Two selected conditions per network; source-reported D0 width, not a full thermal uncertainty or confidence envelope."}
    if slug == "p4s3":
        source = record["source_details"]
        atomization = source["preferred_cycle"]["atomization_route_Hf298_J_mol"]
        selected = [{"name": "atomization_route_H298", "replacements": {record["species"]: retained_spec(record, atomization-source["selected_Hf298_J_mol"])}}]
        special = {"alternate_Hf298_J_mol": atomization,
                   "scope": "Less-converged atomization-route central value at selected conditions only; its TAE0 uncertainty is not attached to the preferred reaction cycle. No zero-K anchor is invented."}
    pins = {str(Path(__file__).resolve().relative_to(ROOT)): file_hash(Path(__file__).resolve())}
    for source in sources.values():
        pins.update(source["source_sha256"])
    return {"id": slug+"_bounded_candidates_20261106", "slug": slug, "baseline_dataset": "curated-ch2-v1",
            "temperature_range_K": [1500, 4000], "standard_pressure_Pa": 100000,
            "gas_constant_J_mol_K": R, "native_NASA_R_J_mol_K": NASA_R,
            "target_species": record["species"], "tracked_elements": [e for e in record["composition"] if e != "H"],
            "primary_variant": "retained", "variants": variants, "selected_variants": selected,
            "source_records": sources, "special_controls": special, "source_sha256": dict(sorted(pins.items())),
            "retained_scope": "The explicit physical energy difference changes G/H equally and leaves S/Cp and the entire literal thermal/fit residual intact. No physical D0 is inferred from R*a.",
            "NASA_scope": "When present, exact NASA molecular-minus-neutral-atoms thermal shape and atomic gauge replace the inherited fit residual. Native energy is removed with NASA R and chosen energy inserted with SI R exactly once. This is a separate thermal/reference sensitivity, not proof of greater accuracy.",
            "physical_accuracy": "unknown", "production_change": "none"}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("slug", choices=SLUGS)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    text = json.dumps(build(args.slug), indent=2, allow_nan=False)+"\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text)
    else:
        print(text, end="")
