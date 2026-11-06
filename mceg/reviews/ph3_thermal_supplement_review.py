"""Audit retrieved PH3 supplement files; never construct a production model."""

import argparse
import csv
import json
import math
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "tests/data/validation/ph3_thermal_supplement/20261105"
R = 8.31446261815324
CENTERS = (300, 500, 1000, 1500, 2000, 3000, 4000)


def rounding_half_width(value: float) -> float:
    """Conservative half-unit bound for five significant source digits."""
    return 0.5 * 10 ** (math.floor(math.log10(abs(value))) - 4) if value else 0.0


def read_sources() -> tuple:
    """Read exact archived nodes without interpolating or relabelling J."""
    with (DATA / "moments.csv").open() as stream:
        moments = {(int(r["row_label"]), int(r["temperature_K"])):
                   tuple(float(r[k]) for k in ("Q", "Q1", "Q2"))
                   for r in csv.DictReader(stream)}
    with (DATA / "calorics.csv").open() as stream:
        calorics = {int(r["temperature_K"]):
                    {k: float(v) for k, v in r.items() if k != "temperature_K" and v}
                    for r in csv.DictReader(stream)}
    return moments, calorics


def derivative(values: dict, temperature: int, step: int) -> tuple:
    """Five-point first derivative and source-rounding bound, not fit error."""
    weights = (1, -8, 8, -1)
    nodes = (temperature - 2 * step, temperature - step,
             temperature + step, temperature + 2 * step)
    value = math.fsum(w * values[t] for w, t in zip(weights, nodes)) / (12 * step)
    bound = math.fsum(abs(w) * rounding_half_width(values[t])
                     for w, t in zip(weights, nodes)) / (12 * step)
    return value, bound


def review() -> dict:
    """Separate exact positivity failures from derivative sensitivity checks."""
    moments, calorics = read_sources()
    samples = []
    for temperature in CENTERS:
        q, q1, q2 = moments[100, temperature]
        dq, dq1, dq2 = map(rounding_half_width, (q, q1, q2))
        variance = q2 / q - (q1 / q) ** 2
        variance_upper = (q2 + dq2) / (q - dq) - ((q1 - dq1) / (q + dq)) ** 2
        before, after = (moments[label, temperature][2] for label in (41, 42))
        row = {
            "temperature_K": temperature,
            "final_row_Q": q,
            "final_row_variance": variance,
            "variance_upper_after_source_rounding": variance_upper,
            "moment_Cp_with_translation_J_mol_K": R * (2.5 + variance),
            "published_Cp_J_mol_K": calorics[temperature]["Cp_J_mol_K"],
            "Q2_row_41": before,
            "Q2_row_42": after,
            "Q2_drop_lower_after_source_rounding": before - after
            - rounding_half_width(before) - rounding_half_width(after),
            "closure": [],
        }
        for step in (30, 60):
            h = {t: r["H_increment_kJ_mol"] for t, r in calorics.items()}
            s = {t: r["S_J_mol_K"] for t, r in calorics.items()}
            dh, dh_bound = derivative(h, temperature, step)
            ds, ds_bound = derivative(s, temperature, step)
            row["closure"].append({
                "step_K": step,
                "dH_dT_J_mol_K": dh * 1000,
                "dH_dT_rounding_bound_J_mol_K": dh_bound * 1000,
                "T_dS_dT_J_mol_K": temperature * ds,
                "T_dS_dT_rounding_bound_J_mol_K": temperature * ds_bound,
                "dH_dT_minus_published_Cp_J_mol_K": dh * 1000 - row["published_Cp_J_mol_K"],
            })
        samples.append(row)
    return {
        "species": "PH3",
        "doi": "10.1016/j.jqsrt.2014.03.012",
        "interpretation": "Appendix A cumulative energy moments; published row labels retained",
        "nuclear_spin_divisor_for_thermochemical_Q": 16,
        "spin_normalization_cancels_from_moment_ratios": True,
        "final_row_Q296": moments[100, 296][0],
        "paper_direct_Q296": 3249.46,
        "Q296_rounding_half_width": rounding_half_width(moments[100, 296][0]),
        "published_Cp_at_1_K_J_mol_K": calorics[1]["Cp_J_mol_K"],
        "translation_Cp_J_mol_K": 2.5 * R,
        "samples": samples,
        "decision": "retain_current; resolve distributed-file moment and caloric inconsistencies",
        "production_change": "none",
        "candidate_created": False,
        "equilibrium_substitution": "not_run",
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(review(), indent=2) + "\n")
