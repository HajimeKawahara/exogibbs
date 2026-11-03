"""A bounded NIST CO2 thermal candidate in the existing elemental reference.

The main model represents elemental CO2 by 12C16O2 internal thermodynamics
and NASA natural-mass translation. The isotope-average alternative is a
fixed-composition thermal diagnostic, not isotope-resolved chemistry.
"""

import argparse
from functools import lru_cache
from hashlib import sha256
import json
import math
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "tests/data/validation/co2_candidate/20261104"
SOURCE = ROOT / "tests/data/validation/co2_source_review/20261103"
sys.path.insert(0, str(ROOT / "examples/comparisons"))
import select_major_molecules as major
from tio_candidate import _nasa, _polynomial, hermite_coefficients

R = major.R
KB, PLANCK = 1.380649e-23, 6.62607015e-34
MASS_U = 44.0095
KNOTS = np.arange(1500., 4001., 250.)
CODES = ("626", "627", "628", "636", "637", "638", "727", "728", "737", "738", "828", "838")
OXYGEN = {"6": .99757, "7": .00038, "8": .00205}
CARBON = {"2": .9893, "3": .0107}
ISOTOPE_WEIGHTS = {s: OXYGEN[s[0]]*CARBON[s[1]]*OXYGEN[s[2]]*(1 if s[0] == s[2] else 2) for s in CODES}


def translation(temperature: float) -> float:
    """Log single-molecule translational factor at NASA mass and 1 bar."""
    kt = KB*temperature
    return 1.5*math.log(2*math.pi*MASS_U*major.ATOMIC_MASS_KG*kt/PLANCK**2)+math.log(kt/1e5)


def source_properties(row: np.ndarray, code: str = "626") -> dict:
    """Reconstruct calorics from Q, H and Cp with spin-free common translation."""
    t, q = row[:2]
    spin = (2 if code[1] == "3" else 1)*(6 if code[0] == "7" else 1)*(6 if code[2] == "7" else 1)
    g = -R*t*(math.log(q/spin)+translation(t))
    h = R*t*row[7]
    return {"temperature_K": float(t), "G_relative_J_mol": g, "H_relative_J_mol": h,
            "S_J_mol_K": (h-g)/t, "Cp_J_mol_K": R*row[3],
            "relative_standard_u_Q": row[2]/q, "standard_u_Cp_J_mol_K": R*row[4]}


@lru_cache(maxsize=1)
def load_candidate() -> dict:
    """Read frozen coefficients without external NIST files."""
    return json.loads((DATA/"candidate.json").read_text())


def molecular_properties(temperature_K: float, model: str = "nist626") -> dict:
    """Return derivative-consistent molecular G-H0, H-H0, S and Cp."""
    if not math.isfinite(temperature_K) or not 1500 <= temperature_K <= 4000:
        raise ValueError("Temperature outside CO2 candidate domain")
    record = load_candidate()
    f, slope, curvature = _polynomial(float(temperature_K), record["models"][model], np)
    return {"G_relative_J_mol": float(R*f), "H_relative_J_mol": float(R*(f-temperature_K*slope)),
            "S_J_mol_K": float(-R*slope), "Cp_J_mol_K": float(-R*temperature_K*curvature)}


def formation_h(temperature_K: float, model: str = "nist626") -> float:
    """Return Gformation/(RT) with unchanged ATcT energy and NASA atoms."""
    props = molecular_properties(temperature_K, model)
    record = load_candidate()
    background = _nasa(temperature_K, record["atomic_background_nasa9"], np)[0]
    return props["G_relative_J_mol"]/(R*temperature_K)+(background-record["D0_over_k_K"])/temperature_K


def formation_h_jax(temperature_K, model: str = "nist626"):
    """Evaluate a traced temperature in x64; NaN outside the bounded domain."""
    import jax.numpy as jnp
    record = load_candidate()
    t = jnp.asarray(temperature_K)
    f = _polynomial(t, record["models"][model], jnp)[0]
    background = _nasa(t, record["atomic_background_nasa9"], jnp)[0]
    value = (f+background-record["D0_over_k_K"])/t
    return jnp.where(jnp.isfinite(t) & (t >= 1500) & (t <= 4000), value, jnp.nan)


def extract(directory: Path, output: Path) -> None:
    """Build from complete pinned official tables, preserving their constants."""
    pins = json.loads((SOURCE/"source_nodes.json").read_text())["source_sha256"]
    tables, hashes = {}, {}
    for code in CODES:
        name = "tcfun_"+code+".txt"
        hashes[name] = sha256((directory/name).read_bytes()).hexdigest()
        if hashes[name] != pins[name]:
            raise ValueError("Unreviewed NIST table: "+name)
        tables[code] = np.loadtxt(directory/name, skiprows=9)
    def node(t, model):
        values = {c: source_properties(tables[c][tables[c][:, 0] == t][0], c) for c in CODES}
        if model == "nist626":
            return values["626"]
        # Mean thermodynamic functions, not the logarithm of a mean partition.
        return {k: float(sum(ISOTOPE_WEIGHTS[c]*values[c][k] for c in CODES))
                for k in ("G_relative_J_mol", "H_relative_J_mol", "S_J_mol_K", "Cp_J_mol_K")} | {"temperature_K": float(t)}
    nodes = {m: [node(t, m) for t in KNOTS] for m in ("nist626", "isotope_average")}
    background = np.zeros(9)
    reference = json.loads(major.REFERENCE_PATH.read_text())
    for atom, count in (("C", 1), ("O", 2)):
        background -= count*np.array(major.nasa_interval(atom, 2500, reference)["coefficients"])
        background[7] += count*major.assigned_h0(atom, "nasa9", reference)/major.NASA_R
    record = {"id": "co2_nist626_elemental_candidate_v1", "temperature_range_K": [1500, 4000],
              "standard_pressure_Pa": 100000, "molecular_mass_u": MASS_U,
              "D0_J_mol": 1598180., "D0_over_k_K": major.atomization_kelvin("CO2", "atct", reference),
              "atomic_background_nasa9": background.tolist(),
              "models": {m: {"knots_K": KNOTS.tolist(), "coefficients": hermite_coefficients(rows)} for m, rows in nodes.items()},
              "representation": "C2 quintic Hermite of (G-H0)/R; H,S,Cp are analytic derivatives of one function",
              "source": {"paper_doi": "10.1063/5.0276615", "dataset_doi": "10.18434/mds2-2364", "data_year": 2021, "sha256": hashes},
              "conventions": {"main": "626 lowest rovibrational zero, nuclear spin1, allowed-level exchange symmetry already counted, NASA molecular mass and atomic thermal functions, ATcT1.172 total atomization energy unchanged",
                              "isotope_average": "Fixed terrestrial fractions; remove each isotope nuclear-spin product, replace each translation by44.0095u, average G/H/S/Cp at each isotope ground zero. No extra mixing entropy or isotope D0/ZPE correction; diagnostic only, not a natural-composition formation equilibrium.",
                              "source_constants": "Retain NIST2014 c2=1.43877736Kcm in internal moments; use currentSI translation and common modernR for calorics. Do not silently refit Hamiltonian constants.",
                              "high_temperature": "Ground-electronic bound-species functions with nominal626 cutoff44000cm-1; frozen pureCO2 Cp warning does not model reacting mixture Cp. Parameter covariance excludes complete continuum/model error."}}
    checks = sorted({int(t+offset) for t in KNOTS[:-1] for offset in (13,31,62,94,125,156,188,219,237)})
    off = {m: [node(t, m) for t in checks] for m in nodes}
    source_rows = {c: [tables[c][tables[c][:, 0] == t][0].tolist() for t in sorted(set(KNOTS.tolist()+checks))]
                   for c in CODES}
    output.mkdir(parents=True, exist_ok=True)
    for filename, data in (("candidate.json", record), ("nodes.json", nodes),
                           ("off_knot_reference.json", off), ("nist_samples.json", source_rows)):
        (output/filename).write_text(json.dumps(data, indent=2, allow_nan=False)+"\n")


def validate() -> dict:
    """Compare analytic fit calorics to withheld independently printed rows."""
    result = {}
    for model, rows in json.loads((DATA/"off_knot_reference.json").read_text()).items():
        maxima = {k: 0. for k in ("ln_K", "H_J_mol", "S_J_mol_K", "Cp_J_mol_K")}
        for row in rows:
            t = row["temperature_K"]
            props = molecular_properties(t, model)
            errors = {"ln_K": abs(props["G_relative_J_mol"]-row["G_relative_J_mol"])/(R*t),
                      "H_J_mol": abs(props["H_relative_J_mol"]-row["H_relative_J_mol"]),
                      "S_J_mol_K": abs(props["S_J_mol_K"]-row["S_J_mol_K"]),
                      "Cp_J_mol_K": abs(props["Cp_J_mol_K"]-row["Cp_J_mol_K"])}
            maxima = {k: max(maxima[k], v) for k, v in errors.items()}
        minima = []
        record = load_candidate()["models"][model]
        for t, width, values in zip(KNOTS[:-1], np.diff(KNOTS), record["coefficients"]):
            curvature = np.arange(2, 6)*np.arange(1, 5)*np.asarray(values)[2:]/width**2
            cp = -R*np.polynomial.polynomial.polymul([t, width], curvature)
            roots = np.polynomial.polynomial.polyroots(np.arange(1, len(cp))*cp[1:])
            points = [0., 1.]+[float(v.real) for v in roots if abs(v.imag)<1e-10 and 0<v.real<1]
            minima.append(float(min(np.polynomial.polynomial.polyval(x, cp) for x in points)))
        result[model] = {"off_knot_checks": len(rows), "maximum_errors": maxima, "minimum_Cp_J_mol_K": min(minima)}
    rows = []
    from co2_source_review import curated_ln_k
    for t in KNOTS:
        main, average = molecular_properties(t), molecular_properties(t, "isotope_average")
        rows.append({"temperature_K": t, "candidate_over_current_K": math.exp(-formation_h(t)-curated_ln_k(t)),
                     "isotope_average_over_main_K_thermal_diagnostic": math.exp(-(average["G_relative_J_mol"]-main["G_relative_J_mol"])/(R*t)),
                     "isotope_average_minus_main_Cp_J_mol_K": average["Cp_J_mol_K"]-main["Cp_J_mol_K"],
                     "source_constant_delta_ln_Q_first_order": (main["H_relative_J_mol"]/(R*t)-2.5)*(1.43877736/(PLANCK*29979245800/KB)-1)})
    return {"models": result, "thermal_diagnostics": rows, "curated_change": "none",
            "source_precision": "NIST Q/H/Cp columns retain printed precision; caloric fit error is separate from parameter covariance and physical error."}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--extract", type=Path)
    parser.add_argument("--output-dir", type=Path, default=DATA)
    args = parser.parse_args()
    if args.extract:
        extract(args.extract, args.output_dir)
    else:
        print(json.dumps(validate(), indent=2, allow_nan=False))
