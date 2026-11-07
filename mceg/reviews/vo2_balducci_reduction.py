"""Replay published VO2 equilibria and explicitly limited thermal re-reductions."""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path

import numpy as np
from scipy.interpolate import CubicHermiteSpline
from scipy.stats import linregress

from c2h_source_review import read_table
from gas_source_comparison import ROOT, file_hash
from vo2_source_review import electronic_partition

DATA = ROOT / "tests/data/validation/vo2_balducci_reduction/20261102"
SOURCE = ROOT / "tests/data/validation/vo2_source_review/20261102"
CURATED = ROOT / "src/exogibbs/data/curated/gas_v1.json"


def phi_curve(rows: np.ndarray) -> CubicHermiteSpline:
    """Interpolate Phi=(H0-G)/T using dPhi/dT=(H-H0)/T**2; do not extrapolate."""
    t, phi, h = np.asarray(rows, dtype=float).T
    return CubicHermiteSpline(t, phi, h*1000/t**2, extrapolate=False)


def janaf_phi(path: Path) -> CubicHermiteSpline:
    """Convert a JANAF H298 reference to H0 before interpolating its thermal term."""
    table = read_table(path)
    return phi_curve(np.array([[t, row[2]+table[0][3]*1000/t,
                                row[3]-table[0][3]]
                               for t, row in sorted(table.items()) if t >= 298.15]))


def thermal_change(temperature: float, inputs: dict, mode: str) -> dict:
    """Return one consistent, zero-point-excluded VO2 G/H/S/Cp perturbation."""
    r, c2 = inputs["gas_constant_J_mol_K"], inputs["second_radiation_constant_cm_K"]
    t = temperature
    if not math.isfinite(t) or t <= 0:
        raise ValueError("Temperature must be finite and positive")
    if mode not in ("vibrations", "electronic", "combined"):
        raise ValueError("Unknown thermal perturbation")
    dg, dh, dcp = 0., 0., 0.
    if mode in ("vibrations", "combined"):
        for sign, frequencies in ((-1, inputs["old_frequencies_cm-1"]),
                                  (1, inputs["partial_new_frequencies_cm-1"])):
            for frequency in frequencies:
                theta = frequency*c2
                x = theta/t
                denominator = -math.expm1(-x)
                occupation = math.exp(-x)/denominator
                dg += sign*r*t*math.log(denominator)
                dh += sign*r*theta*occupation
                dcp += sign*r*x*x*occupation*(1+occupation)
    if mode in ("electronic", "combined"):
        levels = [e*inputs["electronvolt_J_mol"]/r for e in inputs["levels_eV"]]
        terms = electronic_partition(levels, inputs["weights"], t, r)
        dg -= r*t*math.log(terms["Q_el"]/inputs["ground_weight"])
        dh += terms["H_el_J_mol"]
        dcp += terms["Cp_el_J_mol_K"]
    return {"temperature_K": t, "delta_G_J_mol": dg, "delta_H_J_mol": dh,
            "delta_S_J_mol_K": (dh-dg)/t, "delta_Cp_J_mol_K": dcp,
            "delta_Phi_J_mol_K": -dg/t,
            "delta_third_law_reaction2_or3_H0_kJ_mol": dg/1000}


def review() -> dict:
    """Separate raw observations, source reproduction and incomplete model changes."""
    sources = json.loads((DATA / "sources.json").read_text())
    for source in sources["fixed_inputs"]:
        if file_hash(ROOT/source["path"]) != source["sha256"]:
            raise ValueError("Source hash mismatch: " + source["path"])
    inputs = json.loads((DATA / "inputs.json").read_text())
    source = json.loads((SOURCE / "literature_inputs.json").read_text())
    with (DATA / "table_I.csv").open() as stream:
        raw = list(csv.DictReader(stream))
    r = inputs["gas_constant_J_mol_K"]
    second_law, observations = [], {}
    for i, column, sign in ((1, "minus_lnK_1", -1), (2, "lnK_2", 1),
                            (3, "minus_lnK_3", -1)):
        rows = [row for row in raw if row[column]]
        t = np.array([float(row["temperature_K"]) for row in rows])
        lnk = np.array([sign*float(row[column]) for row in rows])
        h0 = np.array([float(row[f"H0_{i}_kJ_mol"]) for row in rows])
        observations[i] = (t, lnk, h0)
        fit = linregress(1/t, lnk)
        printed = inputs["printed_summary"][str(i)]
        second_law.append({"reaction": i, "count": len(t),
                           "arithmetic_mean_temperature_K": float(t.mean()),
                           "harmonic_mean_temperature_K": float(1/np.mean(1/t)),
                           "printed_temperature_K": printed["temperature_K"],
                           "OLS_H_T_kJ_mol": float(-r*fit.slope/1000),
                           "OLS_slope_standard_error_kJ_mol": float(r*fit.stderr/1000),
                           "printed_H_T_kJ_mol": printed["H_T_kJ_mol"],
                           "printed_H_T_uncertainty_kJ_mol": printed["uncertainty_kJ_mol"],
                           "printed_row_H0_mean_kJ_mol": float(h0.mean()),
                           "printed_row_H0_sample_std_kJ_mol": float(h0.std(ddof=1)),
                           "scope": "Unweighted OLS of printed lnK, not a new combined physical error budget."})

    vo, v = janaf_phi(DATA / "O-026.txt"), janaf_phi(SOURCE / "V-005.txt")
    vo2 = phi_curve(np.array(source["balducci1983"]["table_IV_VO2"]))
    tmean = second_law[1]["harmonic_mean_temperature_K"]
    delta_h_thermal = tmean**2*(2*vo.derivative()(tmean)
                               - v.derivative()(tmean)-vo2.derivative()(tmean))/1000
    second_law[1]["H0_reduced_at_harmonic_temperature_kJ_mol"] = float(
        second_law[1]["OLS_H_T_kJ_mol"] - delta_h_thermal)
    second_law[1]["printed_second_law_H0_kJ_mol"] = inputs["printed_summary"]["2"]["H0_kJ_mol"]
    t, lnk, printed_h0 = observations[2]
    # Every reaction has Delta nu = 0. Pressure shifts cancel only when all
    # species use one convention; the historical VO2 table does not specify it.
    unconverted = (-r*t*lnk + t*(2*vo(t)-v(t)-vo2(t)))/1000
    conversion = -r*t*math.log(101325/100000)/1000
    converted = unconverted + conversion
    modern_vo = json.loads(CURATED.read_text())["sources"]["spectroscopy"]["VO"]
    anchor = modern_vo["dissociation_eV"]*inputs["electronvolt_J_mol"]/1000
    nodes = [{"temperature_K": float(ti), "lnK2": float(li),
              "printed_H0_kJ_mol": float(pi),
              "H0_assume_tableIV_1bar_kJ_mol": float(hi),
              "H0_assume_tableIV_1atm_kJ_mol": float(ci),
              "residual_assume_1atm_kJ_mol": float(ci-pi)}
             for ti, li, pi, hi, ci in zip(t, lnk, printed_h0, unconverted, converted)]
    shifted = {}
    for mode in ("vibrations", "electronic", "combined"):
        rows = [thermal_change(float(ti), inputs, mode) for ti in t]
        delta = np.array([row["delta_third_law_reaction2_or3_H0_kJ_mol"] for row in rows])
        mean_thermal = thermal_change(tmean, inputs, mode)
        # With VO2 on the reactant side, second-law reduction shifts by +delta H,
        # whereas third-law reduction shifts by +delta G. Check both methods.
        second_h0 = (second_law[1]["H0_reduced_at_harmonic_temperature_kJ_mol"]
                     + mean_thermal["delta_H_J_mol"]/1000)
        shifted[mode] = {"mean_delta_reaction2_H0_kJ_mol": float(delta.mean()),
                         "second_law_delta_H0_kJ_mol": mean_thermal["delta_H_J_mol"]/1000,
                         "second_law_H0_kJ_mol": second_h0,
                         "third_law_mean_H0_assume_1atm_kJ_mol": float(np.mean(converted+delta)),
                         "third_law_mean_H0_assume_1bar_kJ_mol": float(np.mean(unconverted+delta)),
                         "second_minus_third_law_H0_assume_1atm_kJ_mol": float(second_h0-np.mean(converted+delta)),
                         "second_minus_third_law_H0_assume_1bar_kJ_mol": float(second_h0-np.mean(unconverted+delta)),
                         "modern_VO_reanchored_second_law_D0_kJ_mol": 2*anchor+second_h0,
                         "modern_VO_reanchored_D0_assume_1atm_kJ_mol": float(2*anchor+np.mean(converted+delta)),
                         "modern_VO_reanchored_D0_assume_1bar_kJ_mol": float(2*anchor+np.mean(unconverted+delta)),
                         "observation_nodes": rows}
    cycle = [{"temperature_K": float(row["temperature_K"]),
              "lnK3_minus_lnK1_minus_lnK2": -float(row["minus_lnK_3"])
              + float(row["minus_lnK_1"])-float(row["lnK_2"])}
             for row in raw if all(row[key] for key in
                                  ("minus_lnK_1", "lnK_2", "minus_lnK_3"))]
    return {"source": "Balducci et al. 1983 Table I; printed observations retained without correction",
            "second_law": second_law,
            "reaction_cycle": {
                "identity": "Reaction 3 = reaction 1 + reaction 2",
                "overlapping_rows": cycle,
                "max_abs_lnK_closure": max(abs(row["lnK3_minus_lnK1_minus_lnK2"])
                                          for row in cycle),
                "historical_D0_route2": "2 D0(EuO) + 2 H0(1) + H0(2)",
                "historical_D0_route3": "2 D0(EuO) + H0(1) + H0(3)",
                "covariance": "Both routes share EuO twice, VO, and overlapping equilibrium observations; no independent pooling or new combined uncertainty."},
            "third_law_reaction2": {
                "nodes": nodes,
                "mean_H0_assume_1atm_kJ_mol": float(converted.mean()),
                "mean_H0_assume_1bar_kJ_mol": float(unconverted.mean()),
                "modern_VO_D0_kJ_mol": anchor,
                "modern_VO_reanchor_keep_selected_minus74_kJ_mol": 2*anchor-74,
                "modern_VO_reanchor_replayed_assume_1atm_kJ_mol": float(2*anchor+converted.mean()),
                "modern_VO_reanchor_replayed_assume_1bar_kJ_mol": float(2*anchor+unconverted.mean()),
                "thermal_variants": shifted},
            "thermal_samples": {mode: [thermal_change(ti, inputs, mode)
                                        for ti in (1500., 2000., 3000., 4000.)]
                                for mode in ("vibrations", "electronic", "combined")},
            "vibration_components_2000K": {
                name: thermal_change(2000., {**inputs, "old_frequencies_cm-1": [old],
                                            "partial_new_frequencies_cm-1": [new]},
                                     "vibrations")
                for name, old, new in (("bend", 538, 291), ("symmetric_stretch", 983, 960))},
            "limitations": inputs["limitations"],
            "adoption": "none; partial thermochemical re-reduction, not a calibrated recommended D0"}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    text = json.dumps(review(), indent=2, allow_nan=False) + "\n"
    if args.output:
        args.output.write_text(text)
    else:
        print(text, end="")
