"""Screen declared gas equilibria for insertion of source-bounded pure phases.

Positive log saturation proves a descent direction from a gas-only solution
for the selected thermochemical records. This is not a multiphase solve or
proof that the candidate phase catalog is physically complete.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
SOURCE_SHA256 = "1ec5fe226af698ef6c900537d30c2a640b31f83483d3f6f51c47c813f8d009ee"
ELEMENTS = set("H He C N O P S".split())


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read_condensed(path: Path) -> tuple:
    """Retain original NASA9 phase identities and intervals, without extrapolation."""
    if digest(path) != SOURCE_SHA256:
        raise ValueError("Pinned CEA source hash mismatch")
    lines = path.read_text().splitlines()
    records, excluded = {}, []
    for i, header in enumerate(lines[:-1]):
        metadata = lines[i + 1]
        if (not header or not header[0].isalpha() or len(metadata) < 80
                or not metadata[:2].strip().isdigit() or metadata[51:52] == "0"):
            continue
        composition = {metadata[k:k+2].strip().capitalize(): float(metadata[k+2:k+8])
                       for k in range(10, 50, 8) if metadata[k:k+2].strip()}
        if not set(composition) <= ELEMENTS:
            continue
        name = header.split()[0]
        count = int(metadata[:2])
        if not count or name == "Jet-A(L)" or any(not n.is_integer() for n in composition.values()):
            excluded.append({"name": name, "reason": "No thermal polynomial or non-molecular fuel surrogate"})
            continue
        intervals = []
        for j in range(count):
            bounds, first, second = lines[i+2+3*j:i+5+3*j]
            coefficients = [float(first[k:k+16]) for k in range(0, 80, 16)]
            coefficients += [float(second[k:k+16]) for k in (0, 16, 48, 64)]
            intervals.append({"bounds_K": [float(bounds[:11]), float(bounds[11:22])],
                              "coefficients": coefficients})
        if name in records:
            raise ValueError(f"Ambiguous condensed identity: {name}")
        records[name] = {"composition": composition, "intervals": intervals,
                         "source_line": i+1, "reference": header.strip(),
                         "phase_code": int(metadata[51:52]),
                         "raw_record_sha256": hashlib.sha256(
                             ("\n".join(lines[i:i+2+3*count])+"\n").encode()).hexdigest()}
    return records, excluded


def standard_gibbs_rt(record: dict, temperatures) -> np.ndarray:
    """Evaluate absolute NASA9 Gibbs energy only inside the record's intervals."""
    t = np.asarray(temperatures, dtype=float)
    result = np.full(t.shape, np.nan)
    for interval in record["intervals"]:
        lo, hi = interval["bounds_K"]
        selected = np.isfinite(t) & (t >= lo) & (t <= hi) & (t > 0)
        x = t[selected]
        a, b, c, d, e, f, g, h, s = interval["coefficients"]
        enthalpy = -a/x**2 + b*np.log(x)/x + c + d*x/2 + e*x**2/3 + f*x**3/4 + g*x**4/5 + h/x
        entropy = -a/(2*x**2) - b/x + c*np.log(x) + d*x + e*x**2/2 + f*x**3/3 + g*x**4/4 + s
        result[selected] = enthalpy - entropy
    return result


def log_saturation(mu_elements_rt, composition: dict, elements: list,
                   condensed_g_rt) -> np.ndarray:
    """Return ln S = A_cond dot mu_elements/RT - g_cond/RT; positive favors insertion."""
    if not set(composition) <= set(elements):
        raise ValueError("Condensate contains an element absent from the gas model")
    stoichiometry = np.array([composition.get(e, 0.) for e in elements])
    return np.asarray(mu_elements_rt) @ stoichiometry - np.asarray(condensed_g_rt)


def json_values(values) -> list:
    return [float(x) if np.isfinite(x) else None for x in values]


def minimum_cp_r(record: dict) -> float:
    """Check endpoints and all real stationary points on used source intervals."""
    minima = []
    for interval in record["intervals"]:
        lo, hi = interval["bounds_K"]
        lo, hi = max(lo, 200.), min(hi, 1500.)
        if lo >= hi:
            continue
        coefficients = np.asarray(interval["coefficients"][:7]) * 1000.**np.arange(-2, 5)
        derivative = np.array([-2*coefficients[0], -coefficients[1], 0., coefficients[3],
                               2*coefficients[4], 3*coefficients[5], 4*coefficients[6]])
        roots = np.polynomial.polynomial.polyroots(derivative)
        candidates = [lo/1000., hi/1000.]
        candidates.extend(r.real for r in roots if abs(r.imag) < 1e-9 and lo/1000. < r.real < hi/1000.)
        minima.append(min(float(np.sum(coefficients*x**np.arange(-2, 5))) for x in candidates))
    return min(minima)


def screen_case(case: dict, phases: dict, gas_data: dict) -> dict:
    """Use independently reconstructed absolute atomic potentials and checked gas inputs."""
    temperatures = np.asarray(case["temperature_K"])
    pressure = np.broadcast_to(np.asarray(case["pressure_bar"]), temperatures.shape)
    elements, species = list(case["elements"]), case["species"]
    logx = np.asarray(case["log_mole_fractions"])
    atoms = [species.index(e) for e in elements]
    atom_g = np.stack([standard_gibbs_rt(gas_data["records"][e], temperatures)
                       for e in elements], axis=-1)
    mu = atom_g + logx[:, atoms] + np.log(pressure[:, None])
    supplied = np.asarray(case["element_chemical_potential_RT"])
    difference = float(np.max(abs(mu-supplied)))
    if not all(case["accepted"]):
        raise ValueError("Cannot screen a gas state that failed its numerical gates")
    if not np.isfinite(mu).all() or difference > 1e-10:
        raise ValueError(f"Atomic chemical-potential reconstruction failed: {difference}")
    results = {}
    p_favorable, any_favorable = [[] for _ in temperatures], [[] for _ in temperatures]
    for name, record in phases.items():
        value = log_saturation(mu, record["composition"], elements,
                               standard_gibbs_rt(record, temperatures))
        evaluated = np.isfinite(value)
        for index in np.flatnonzero(evaluated & (value > 1e-8)):
            any_favorable[index].append(name)
            if record["composition"].get("P", 0):
                p_favorable[index].append(name)
        results[name] = {"ln_saturation": json_values(value), "evaluated": evaluated.tolist()}
    return {"temperature_K": temperatures.tolist(), "pressure_bar": pressure.tolist(),
            "metallicity": case["metallicity"], "elements": elements,
            "kind": case.get("kind"), "intervention": case.get("intervention"),
            "max_atomic_mu_reconstruction_abs_RT": difference,
            "gas_states_accepted": len(case["accepted"]),
            "phases": results, "supersaturated_P_phases": p_favorable,
            "supersaturated_phases": any_favorable,
            "positive_tolerance_lnS": 1e-8}


def plot_results(report: dict, output: Path) -> None:
    """Plot bounded phase affinity separately from the native condensation scenario."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    native_path = output.parent / "native7_phase_receipt.json"
    native = json.loads(native_path.read_text()) if native_path.exists() else None
    fig, axes = plt.subplots(1, 2 if native else 1, figsize=(11 if native else 6, 4.4),
                             layout="constrained", squeeze=False)
    ax = axes[0, 0]
    for suffix, color, label in (("Z1", "C0", "Solar"), ("Z100", "C1", "100x solar")):
        for prefix, style, extra in (("full", "-", ""), ("H3PO4_alternative_anchor", "--", " / gas energy alternative")):
            case = report["cases"][f"{prefix}_{suffix}"]
            t, p = np.asarray(case["temperature_K"]), np.asarray(case["pressure_bar"])
            selected = p == .001
            solid = np.array([np.nan if x is None else x for x in case["phases"]["H3PO4(cr)"]["ln_saturation"]])
            liquid = np.array([np.nan if x is None else x for x in case["phases"]["H3PO4(L)"]["ln_saturation"]])
            affinity = np.where(np.isfinite(solid), solid, liquid) / np.log(10.)
            ax.plot(t[selected], affinity[selected], style, color=color,
                    marker=".", label=label+extra)
    ax.axhline(0., color="black", lw=.8)
    ax.axvline(315.5, color="gray", lw=.8, ls=":")
    ax.set(xscale="log", xlim=(200, 1500), xlabel="Temperature [K]",
           ylabel="H3PO4 phase log10 saturation", title="Bounded NASA9 phase insertion\n78-gas source and energy scenarios")
    ax.legend(fontsize=7)
    ax.grid(alpha=.2)
    ax.text(.03, .04, "Above zero: gas-only state unstable\nSolid below 315.5 K; liquid above\nNo acid-phase extrapolation beyond 1000 K",
            transform=ax.transAxes, ha="left", va="bottom", fontsize=7)
    if native:
        ax = axes[0, 1]
        t = np.asarray(native["temperature_K"])
        selected = t <= 1500
        ax.plot(t[selected], np.asarray(native["matched_gas_only"]["gas_P_fraction"])[selected],
                color="gray", ls=":", label="Gas-only: total gaseous P")
        ax.plot(t[selected], np.asarray(native["gas_P_fraction"])[selected],
                color="C0", marker=".", label="Condensation allowed: gaseous P")
        total = sum(np.asarray(v) for v in native["condensed_P_fractions"].values())
        ax.plot(t[selected], np.maximum(total[selected], 1e-300),
                color="C3", marker=".", label="Condensed P (H3PO4)")
        ax.set(xscale="log", yscale="log", xlim=(200, 1500), ylim=(1e-24, 2),
               xlabel="Temperature [K]", ylabel="Fraction of total P atoms",
               title="Native GGChem, same gas inputs\n1156 gases + 22 candidate condensates")
        ax.legend(fontsize=7)
        ax.grid(alpha=.2)
    fig.suptitle("Cold phosphorus phase checks | 1 mbar | distinct source scenarios")
    fig.savefig(output, dpi=170)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path,
                        default=ROOT.parent / "VULCAN/thermo/NASA9/nasa9_2002_E.txt")
    parser.add_argument("--gas-input", type=Path,
                        default=ROOT / "results/phosphorus_cold_validation/gas/primitive_cases.json")
    parser.add_argument("--output", type=Path,
                        default=ROOT / "results/phosphorus_cold_validation/phase_summary.json")
    args = parser.parse_args()
    phases, excluded = read_condensed(args.source)
    gas_fixture = ROOT / "tests/data/validation/phosphorus_cold/thermo.json"
    gas_data = json.loads(gas_fixture.read_text())
    inputs = json.loads(args.gas_input.read_text())
    cases = {name: screen_case(case, phases, gas_data) for name, case in inputs.items()}
    metadata = {name: {k: v for k, v in record.items() if k != "intervals"}
                | {"temperature_intervals_K": [i["bounds_K"] for i in record["intervals"]],
                   "minimum_cp_R_used_intervals": minimum_cp_r(record)}
                for name, record in phases.items()}
    if min(r["minimum_cp_R_used_intervals"] for r in metadata.values()) <= 0:
        raise ValueError("A selected condensed polynomial has nonpositive heat capacity")
    transitions = []
    for solid, liquid, t in (("H3PO4(cr)", "H3PO4(L)", 315.5),
                              ("P4O10(cr)", "P4O10(L)", 699.),
                              ("P(cr)", "P(L)", 317.3),
                              ("H2O(cr)", "H2O(L)", 273.15)):
        delta = standard_gibbs_rt(phases[liquid], [t])[0] - standard_gibbs_rt(phases[solid], [t])[0]
        transitions.append({"solid": solid, "liquid": liquid, "temperature_K": t,
                            "liquid_minus_solid_g_RT": float(delta)})
    report = {
        "scope": "Pure-phase insertion into the declared 78/107-species gas networks and source/energy interventions. Positive lnS disproves gas-only stability for these records; no multiphase abundances or complete physical validation are claimed.",
        "criterion": "lnS = sum_e nu_e * mu_e/(RT) - g_cond/(RT); lnS > 0 favors condensation; unit pure-phase activity and negligible condensed-volume pressure correction.",
        "standard_pressure_bar": 1.0,
        "cases": cases, "phase_records": metadata, "excluded_records": excluded,
        "phase_transition_checks": transitions,
        "phase_count": len(phases),
        "P_phase_count": sum(bool(r["composition"].get("P")) for r in phases.values()),
        "minimum_condensed_cp_R_used_intervals": min(r["minimum_cp_R_used_intervals"] for r in metadata.values()),
        "maximum_abs_transition_delta_g_RT": max(abs(t["liquid_minus_solid_g_RT"]) for t in transitions),
        "baseline_case_ids": ["full_Z1", "full_Z100"],
        "scenario_note": "Nonbaseline gas interventions retain unchanged condensed thermochemistry; their phase affinities are conditional sensitivity scenarios, not uncertainty bounds.",
        "missing_competitors": ["NH4H2PO4(s)", "Metal phosphates and phosphides outside the seven-element gas model", "Solid solutions, aqueous solutions and rainout"],
        "source_url": "https://github.com/shami-EEG/VULCAN/blob/aa447277372f964b9c9c4608dd1073fd1bcc5236/thermo/NASA9/nasa9_2002_E.txt",
        "input_sha256": {str(path.resolve()): digest(path) for path in
                         (args.source, args.gas_input, gas_fixture, Path(__file__))},
        "limitations": ["Source polynomial intervals are respected but do not certify physical accuracy.",
                        "Out-of-range values are null with an explicit evaluation mask.",
                        "Gas-state instability does not identify the final stable phase assemblage or its amounts."]}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, allow_nan=False)+"\n")
    plot_results(report, args.output.with_name("phase_stability.png"))
    for name in ("full_Z1", "full_Z100"):
        case = cases[name]
        print(name, "P-phase unstable states", sum(bool(p) for p in case["supersaturated_P_phases"]),
              "/", len(case["temperature_K"]))


if __name__ == "__main__":
    main()
