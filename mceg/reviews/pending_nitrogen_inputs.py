"""Build bounded nitrogen candidates from immutable, molecule-specific sources.

This review-only adapter makes the physical source-energy arithmetic explicit.
The shared evaluator accepts the resulting potentials without reinterpreting
their atomic reference, thermal model or fitted intercept.
"""

from __future__ import annotations

import argparse
from copy import deepcopy
import json
import math
from pathlib import Path

import numpy as np

from c2h_source_review import nasa_properties, read_table
from gas_source_comparison import ROOT, file_hash

R, NASA_R = 8.31446261815324, 8.314510
SLUGS = ("no", "n2o", "no2", "hno", "n2h4", "no3", "n2o4", "n2o5", "cis_hno2")
NASA_ALTERNATIVES = {"n2o", "no2", "hno", "n2h4", "n2o5"}
FASTCHEM = ROOT / "src/exogibbs/data/FastChem4/logK/logK.dat"


def source_record(slug: str) -> dict:
    """Reconstruct physical anchors from source tables, not frozen differences."""
    directory = ROOT / f"tests/data/validation/{slug}_source_review/20261106"
    inputs = json.loads((directory / "inputs.json").read_text())
    comparison = json.loads((directory / "comparison.json").read_text())
    composition, name = inputs["composition"], inputs["molecule"]
    tables = {n: read_table(directory / p) for n, p in inputs["tables"].items()}
    balance = lambda values: math.fsum(count * values[n] for n, count in composition.items()) - values[name]
    # These JANAF fixed-reference offsets have integer-J precision in the source
    # tables. N2O5's blank zero-K cells remain unavailable, never extrapolated.
    jh0 = {n: (round(1000 * (tb[298.15][4] + tb[0.][3]))
               if 0. in tb and math.isfinite(tb[0.][3]) else None)
           for n, tb in tables.items()}
    janaf_d0 = balance(jh0) if all(v is not None for v in jh0.values()) else None
    atct_hf0 = {n: round(1000 * row["hf0_kJ_mol"]) for n, row in inputs["atct"].items()}
    atct_d0 = balance(atct_hf0)
    expected = comparison["balanced_D0_J_mol"]
    assert expected["janaf"] is None if janaf_d0 is None else abs(janaf_d0 - expected["janaf"]) < 1e-8
    assert abs(atct_d0 - expected["atct_1.228"]) < 1e-8
    assert inputs["atct_version"] == "1.228"
    lines = FASTCHEM.read_text().splitlines()
    index = next(i for i, line in enumerate(lines) if line.startswith(inputs["species_id"] + " "))
    coefficients = list(map(float, lines[index + 1].split()))
    assert coefficients == inputs["fastchem_record"]["coefficients"]
    nasa = json.loads((directory / "nasa_records.json").read_text())
    same_identity = inputs.get("nasa_same_identity", True)
    nasa_h0 = {n: rec["hf298_J_mol"] - rec["h298_minus_h0_J_mol"] for n, rec in nasa.items()}
    nasa_d0 = balance(nasa_h0) if same_identity else None
    if nasa_d0 is not None:
        assert abs(nasa_d0 - expected["nasa"]) < 1e-8
    paths = [FASTCHEM, directory / "inputs.json", directory / "comparison.json",
             directory / "nasa_records.json", directory / "sources.json"]
    paths.extend(directory / p for p in inputs["tables"].values())
    for optional in ("literature.json", "partition_inputs.json", "conformer_gap_inputs.json", "identity.json"):
        if (directory / optional).is_file():
            paths.append(directory / optional)
    return {"slug": slug, "molecule": name, "species": inputs["species_id"],
            "composition": composition, "atct_version": "1.228", "same_version_hf0_J_mol": atct_hf0,
            "janaf_fixed_reference_h0_J_mol": jh0, "janaf_D0_J_mol": janaf_d0,
            "atct_D0_J_mol": atct_d0, "delta_D0_J_mol": atct_d0 - janaf_d0 if janaf_d0 is not None else None,
            "original_coefficients": coefficients, "formal_fit_intercept_Ra_J_mol": R * coefficients[0],
            "nasa_same_identity": same_identity, "nasa_h0_J_mol": nasa_h0,
            "nasa_D0_J_mol": nasa_d0, "nasa_records": nasa,
            "quoted_ATcT_marginal_uncertainties_kJ_mol": {n: row["uncertainty_95pct_kJ_mol"] for n, row in inputs["atct"].items()},
            "source_sha256": {str(p.relative_to(ROOT)): file_hash(p) for p in sorted(set(paths))}}


def retained_spec(record: dict, d0_J_mol: float | None = None) -> dict:
    """Apply one physical energy difference to unchanged literal thermals."""
    if record["janaf_D0_J_mol"] is None:
        raise ValueError("A missing JANAF zero-K anchor cannot be replaced by the fit intercept")
    chosen = record["atct_D0_J_mol"] if d0_J_mol is None else d0_J_mol
    return {"kind": "retained_energy", "coefficients": record["original_coefficients"],
            "delta_GH_J_mol": record["janaf_D0_J_mol"] - chosen}


def nasa_spec(record: dict) -> dict:
    """Remove native NASA H0 once, then insert balanced ATcT D0 with SI R."""
    if not record["nasa_same_identity"]:
        raise ValueError("An isomer-mixture NASA record cannot replace one conformer")
    coefficients = {n: np.asarray(next(row["coefficients"] for row in rec["intervals"]
                    if row["bounds_K"] == [1000., 6000.])) for n, rec in record["nasa_records"].items()}
    reaction = coefficients[record["molecule"]].copy()
    for element, count in record["composition"].items():
        reaction -= count * coefficients[element]
    correction = record["atct_D0_J_mol"] / R - record["nasa_D0_J_mol"] / NASA_R
    reaction[7] -= correction
    return {"kind": "nasa9", "coefficients": reaction.tolist(),
            "native_energy_removal_D0_over_R_K": record["nasa_D0_J_mol"] / NASA_R,
            "selected_energy_insertion_D0_over_R_K": record["atct_D0_J_mol"] / R}


def independent_properties(record: dict, temperature_K: float, kind: str) -> dict:
    """Use independent source expressions for replay, including all calorics."""
    t = float(temperature_K)
    if not 1500 <= t <= 4000:
        raise ValueError("Temperature outside bounded candidate domain")
    if kind == "retained_energy":
        a, b, c, d, e = record["original_coefficients"]
        delta = -record["delta_D0_J_mol"]
        g = -R * (a + b*t*math.log(t) + c*t + d*t*t + e*t**3) + delta
        h = R * (-a + b*t + d*t*t + 2*e*t**3) + delta
        entropy = R * (b*math.log(t) + b + c + 2*d*t + 3*e*t*t)
        cp = R * (b + 2*d*t + 6*e*t*t)
    elif kind == "nasa9":
        props = {n: nasa_properties(rec, t, NASA_R) for n, rec in record["nasa_records"].items()}
        reaction = [math.fsum([props[record["molecule"]][i],
                    *(-count*props[element][i] for element, count in record["composition"].items())]) for i in range(3)]
        correction = record["atct_D0_J_mol"] / R - record["nasa_D0_J_mol"] / NASA_R
        h = R * (reaction[0] / NASA_R - correction)
        entropy, cp = R * reaction[1] / NASA_R, R * reaction[2] / NASA_R
        g = h - t * entropy
    else:
        raise ValueError("Unknown independent source model")
    return dict(zip(("G_J_mol", "H_J_mol", "S_J_mol_K", "Cp_J_mol_K"), (g, h, entropy, cp)))


def build(slug: str) -> dict:
    """Build independent definitions, including explicit coupled NOx controls."""
    if slug not in SLUGS:
        raise ValueError("Unknown nitrogen candidate")
    target = source_record(slug)
    sources = {slug: target}
    variants = []
    if target["janaf_D0_J_mol"] is not None:
        variants.append({"name": "retained", "replacements": {target["species"]: retained_spec(target)}})
    if slug in NASA_ALTERNATIVES:
        variants.append({"name": "nasa", "replacements": {target["species"]: nasa_spec(target)}})
    special = {}
    selected_variants = []
    if slug == "n2o4":
        other = source_record("no2")
        sources["no2"] = other
        variants.append({"name": "joint_no2_n2o4_atct", "replacements": {
            target["species"]: retained_spec(target), other["species"]: retained_spec(other)}})
        special["joint_scope"] = "Compare dimer-only and matched monomer/dimer energy changes on the same hot base. No low-temperature extrapolation or reactive-fluid EOS is used."
        special["association_delta_GH_J_mol"] = -target["delta_D0_J_mol"] + 2*other["delta_D0_J_mol"]
    if slug == "cis_hno2":
        other = source_record("trans_hno2")
        sources["trans_hno2"] = other
        gap_source = ROOT / "tests/data/validation/cis_hno2_source_review/20261106/comparison.json"
        gap = json.loads(gap_source.read_text())["conformer_gap"]
        pair = {target["species"]: retained_spec(target), other["species"]: retained_spec(other)}
        variants.append({"name": "joint_cis_trans_atct", "replacements": pair})
        measured_d0 = other["atct_D0_J_mol"] - gap["measured_2023_gap_J_mol"]
        measured = deepcopy(pair)
        measured[target["species"]] = retained_spec(target, measured_d0)
        variants.append({"name": "joint_measured_gap", "replacements": measured})
        for sign, label in ((-1, "lower_gap"), (1, "upper_gap")):
            replacements = deepcopy(pair)
            replacements[target["species"]] = retained_spec(target, measured_d0-sign*gap["quoted_gap_uncertainty_J_mol"])
            selected_variants.append({"name": label, "replacements": replacements})
        special.update({"conformer_gap": gap, "fixed_trans_reference": "ATcT 1.228",
                        "measured_gap_cis_D0_J_mol": measured_d0,
                        "joint_scope": "Single-isomer thermal functions remain separate; an equilibrium-mixture Q is never copied into either species. The measured gap and its quoted width are distinct sensitivity scenarios, not an ATcT covariance update."})
    pins = {str(Path(__file__).resolve().relative_to(ROOT)): file_hash(Path(__file__).resolve())}
    for record in sources.values():
        pins.update(record["source_sha256"])
    return {"id": f"{slug}_bounded_candidates_20261106", "slug": slug,
            "baseline_dataset": "curated-ch2-v1", "temperature_range_K": [1500, 4000],
            "standard_pressure_Pa": 100000, "gas_constant_J_mol_K": R, "native_NASA_R_J_mol_K": NASA_R,
            "target_species": target["species"], "tracked_elements": list(target["composition"]),
            "primary_variant": 0, "variants": variants, "selected_variants": selected_variants,
            "source_records": sources, "special_controls": special,
            "source_sha256": dict(sorted(pins.items())),
            "retained_scope": "Apply only the physical ATcT-minus-JANAF energy difference to the literal fit; retain all thermal terms and the source-fit residual. Formal R*a is not physical D0. No additional ZPE.",
            "NASA_scope": "NASA9 is molecular-minus-stoichiometric-neutral-atoms. Remove the native source D0 using NASA R, insert selected ATcT D0 with SI R, retain dimensionless NASA molecular/atomic thermals. This also replaces the inherited fit residual; it is a separate model sensitivity.",
            "uncertainty_scope": "Rounded same-version central ATcT values; marginal uncertainties are recorded but not combined as independent reaction errors. No complete thermal uncertainty or atmospheric accuracy is claimed.",
            "production_change": "none"}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("slug", choices=SLUGS)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    text = json.dumps(build(args.slug), indent=2, allow_nan=False) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text)
    else:
        print(text, end="")
