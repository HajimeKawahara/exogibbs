"""Audit NH3 energy and partition sources without changing production data.

The default command replays a frozen offline packet. --extract-hapi extracts
the original NH3 arrays from the pinned official source without importing HAPI.
"""

import argparse
import ast
import json
import math
from pathlib import Path

import numpy as np

from c2h_source_review import nasa_properties
from gas_source_comparison import ROOT, file_hash
import select_major_molecules as major


DATA = ROOT / "tests/data/validation/nh3_source_review/20261104"
GRID = ROOT / "tests/data/validation/curated_h2o_adoption/20261104/ions.npz"
CURATED = ROOT / "src/exogibbs/data/curated/gas_v1.json"
TEMPERATURES = (500, 1000, 1500, 2000, 2500, 3000, 3500, 4000)
TABLES = {"TIPS2021": "hapi_tips2021_nh3.txt",
          "TIPS2025": "hapi_tips2025_nh3.txt", "CoYuTe": "CoYuTe.pf"}
KB = 1.380649e-23
PLANCK = 6.62607015e-34
C2 = PLANCK * 29979245800 / KB


def extract_hapi(path: Path, output: Path) -> None:
    """Extract only literal TIPS temperature and NH3 Q arrays with AST parsing."""
    environment = {}
    for node in ast.parse(path.read_text()).body:
        if not isinstance(node, ast.Assign) or len(node.targets) != 1:
            continue
        target = node.targets[0]
        if isinstance(target, ast.Name) and target.id in ("M", "I"):
            environment[target.id] = ast.literal_eval(node.value)
        if not isinstance(target, ast.Subscript) or not isinstance(target.value, ast.Name):
            continue
        name = target.value.id
        if name not in {f"TIPS_{year}_{suffix}" for year in (2021, 2025)
                        for suffix in ("ISOT", "ISOT_HASH", "ISOQ_HASH")}:
            continue
        if name.endswith("ISOT"):
            environment.setdefault(name, {})[ast.literal_eval(target.slice)] = (
                ast.literal_eval(node.value.args[0]))
        elif (environment.get("M"), environment.get("I")) == (11, 1):
            if name.endswith("ISOT_HASH"):
                value = environment[node.value.value.id][ast.literal_eval(node.value.slice)]
            else:
                value = ast.literal_eval(node.value.args[0])
            environment[name] = value
    output.mkdir(parents=True, exist_ok=True)
    for year in (2021, 2025):
        values = np.column_stack((environment[f"TIPS_{year}_ISOT_HASH"],
                                  environment[f"TIPS_{year}_ISOQ_HASH"]))
        np.savetxt(output / f"hapi_tips{year}_nh3.txt", values,
                   fmt=["%.1f", "%.8e"])


def log_translation(temperature: float, mass_u: float) -> float:
    """Return the one-bar molecular translation factor in logarithmic form."""
    kt = KB * temperature
    return (1.5 * math.log(2 * math.pi * mass_u * major.ATOMIC_MASS_KG * kt / PLANCK**2)
            + math.log(kt / 100000))


def printed_fit(temperature: float, coefficients: list) -> float:
    """Replay the printed 2014 fit diagnostically, not as accepted source data."""
    if not 1 <= temperature < 5000:
        raise ValueError("Temperature outside the printed-fit diagnostic range")
    return 10**np.polynomial.polynomial.polyval(math.log10(temperature), coefficients)


def curated_ln_k(temperature: float) -> float:
    """Evaluate the NH3 column retained by every current curated preset."""
    data = json.loads(CURATED.read_text())
    lower, upper = data["temperature_range_K"]
    if not lower <= temperature <= upper:
        raise ValueError("Temperature outside curated range")
    return -major.audit.nasa9_g_rt(temperature, {
        "coefficients": data["nasa9_reactions"]["H3N1"], "bounds_K": [lower, upper]})


def verify_inputs() -> None:
    """Reject changed source tables and implementation inputs before replay."""
    sources = json.loads((DATA / "sources.json").read_text())
    for source in sources["downloads"]:
        if source.get("file") and file_hash(DATA / source["file"]) != source["archive_sha256"]:
            raise ValueError("Source hash mismatch: " + source["file"])
    for path, digest in sources["existing_inputs"].items():
        if file_hash(ROOT / path) != digest:
            raise ValueError("Input hash mismatch: " + path)


def footprint() -> dict:
    """Describe the existing 75-case NH3 allocation; perform no new solve."""
    with np.load(GRID) as grid:
        column = grid["species"].tolist().index("H3N1")
        n = np.exp(grid["ln_n"])
        points = {}
        for element, count in (("N", 1), ("H", 3)):
            e = grid["elements"].tolist().index(element)
            share = count*n[:, column]/grid["budget"][:, e]
            i = int(np.argmax(share))
            points[element] = {
                "case_index": i, "element_fraction": float(share[i]),
                "mole_fraction": float(np.exp(grid["ln_x"][i, column])),
                "temperature_K": float(grid["temperature_K"][i]),
                "pressure_bar": float(grid["pressure_bar"][i]),
                "C_over_O": float(grid["C_over_O"][i]),
                "maximum_relative_conservation_residual": float(np.max(
                    np.abs(n @ grid["formula_matrix"][e]/grid["budget"][:, e]-1))),
            }
        return {"model": "curated-h2o-v1", "cases": len(n), "species": n.shape[1],
                "all_converged": bool(np.all(grid["converged"])),
                "maximum_abs_h_error": max(abs(h+curated_ln_k(t)) for t, h in
                    zip(grid["temperature_K"], grid["h"][:, column])),
                "maximum_element_shares": points}


def review() -> dict:
    """Separate balanced energy, thermal shape, state conventions and scope."""
    verify_inputs()
    reference = json.loads(major.REFERENCE_PATH.read_text())
    literature = json.loads((DATA / "literature.json").read_text())
    energy = {s: major.atomization_kelvin("NH3", s, reference)
              for s in ("atct", "nasa9", "janaf")}
    record = reference["nasa9_records"]["NH3"]
    mass_factor = (literature["CoYuTe"]["mass_u"]/record["mass_u"])**1.5
    tables = {name: dict(np.loadtxt(DATA / file)) for name, file in TABLES.items()}
    fit = literature["SousaSilva2014"]
    fit_q296 = printed_fit(296, fit["printed_coefficients"])
    rows = []
    for t in TEMPERATURES:
        ng = major.thermal_g_rt("NH3", t, "nasa9", reference)
        qn = math.exp(-ng-log_translation(t, record["mass_u"]))
        h, s, cp = nasa_properties(record, t, major.NASA_R)
        nh = (h-major.assigned_h0("NH3", "nasa9", reference))/major.NASA_R
        nasa_k = major.formation_log_kp("NH3", t, "nasa9", energy["nasa9"], reference)
        hybrid = major.formation_log_kp("NH3", t, "nasa9", energy["atct"], reference)
        modern = {}
        for name, table in tables.items():
            q = table.get(t)
            if q is not None:
                ratio = q/24/qn
                modern[name] = {
                    "Q_including_nuclear_spin": float(q), "Q_over_24_over_nasa": ratio,
                    "same_D0_atoms_natural_mass_K_proxy": ratio,
                    "isotope_mass_K_proxy": ratio*mass_factor,
                    "reference_zero_shift_K_proxy": ratio*math.exp(.793374*C2/t),
                    "delta_thermal_G_J_mol": -major.R*t*math.log(ratio),
                    "method": ("product_approximation" if name == "TIPS2025" and t >= 692
                               else "direct_sum_distribution"),
                    "within_published_CoYuTe_opacity_range": t <= 1500 if name == "CoYuTe" else None,
                }
        janaf = None
        if str(float(t)) in reference["janaf"]["NH3"]["rows"]:
            j = reference["janaf"]["NH3"]["rows"][str(float(t))]
            jh = 1000*(j[3]-reference["janaf"]["NH3"]["rows"]["0.0"][3])
            janaf = {
                "same_D0_atoms_K_ratio": math.exp(ng-major.thermal_g_rt("NH3", t, "janaf", reference)),
                "native_ln_K": major.formation_log_kp("NH3", t, "janaf", energy["janaf"], reference, "janaf"),
                "atomic_thermal_shift_ln_K": sum(n*(major.thermal_g_rt(a, t, "janaf", reference)
                    - major.thermal_g_rt(a, t, "nasa9", reference)) for a, n in (("N", 1), ("H", 3))),
                "delta_H_increment_J_mol": jh-major.R*nh,
                "delta_S_J_mol_K": j[1]-major.R*s/major.NASA_R,
                "delta_Cp_J_mol_K": j[0]-major.R*cp/major.NASA_R,
            }
        rows.append({"temperature_K": t, "nasa_ln_K": nasa_k,
                     "curated_ln_K": curated_ln_k(t) if t >= 1500 else None,
                     "reconstructed_hybrid_ln_K": hybrid,
                     "curated_over_nasa": math.exp(hybrid-nasa_k),
                     "Q_nasa_natural_mass": qn, "modern": modern, "janaf": janaf,
                     "printed_2014_fit_diagnostic_over_24_over_nasa":
                         printed_fit(t, fit["printed_coefficients"])/24/qn})
    return {
        "reaction": "N(g) + 3 H(g) -> NH3(g)", "standard_pressure_Pa": 100000,
        "atct_version": reference["atct"]["version"],
        "balanced_D0_J_mol": {s: e*(major.NASA_R if s == "nasa9" else major.R) for s, e in energy.items()},
        "balanced_D0_over_k_K": energy,
        "adopted_minus_nasa_D0_over_k_K": energy["atct"]-energy["nasa9"],
        "isotope_to_natural_translation_factor": mass_factor,
        "source_samples": rows,
        "printed_fit_audit": {"temperature_K": 296, "printed_fit_Q": fit_q296,
            "reported_direct_Q": fit["reported_Q296"],
            "relative_mismatch": fit_q296/fit["reported_Q296"]-1,
            "declared_relative_fit_accuracy": .004,
            "coefficient_last_digit_rounding_relative_bound": math.expm1(math.log(10)*sum(
                e*math.log10(296)**i for i, e in enumerate(fit["coefficient_half_last_digits"]))),
            "accepted_as_direct_sum_data": False},
        "lowest_allowed_state_check": {
            "published_energy_cm-1": .793374, "published_gtot": 12,
            "single_state_Q_at_1K": 12*math.exp(-.793374*C2),
            "distributed_Q_at_1K": float(tables["CoYuTe"][1]),
            "reference_shift_energy_J_mol": .793374*C2*major.R},
        "footprint": footprint(),
        "decision": "retain production provisionally; reconcile direct-sum supplement before a candidate",
        "production_change": "none", "equilibrium_substitution": "not_run",
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--extract-hapi", type=Path)
    parser.add_argument("--packet", type=Path, default=DATA)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.extract_hapi:
        extract_hapi(args.extract_hapi, args.packet)
    else:
        serialized = json.dumps(review(), indent=2, allow_nan=False) + "\n"
        if args.output:
            args.output.write_text(serialized)
        else:
            print(serialized, end="")
