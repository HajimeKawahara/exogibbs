"""Evaluate and rebuild a scoped TiO candidate without changing production data.

The elemental TiO approximation uses 48Ti16O internal levels, molecular
translation at 63.8664 u, and the existing NASA neutral-atom thermal gauge.
The central 6.824 eV bound-level cutoff is fixed when varying only D0.
"""

import argparse
import bz2
from functools import lru_cache
from hashlib import sha256
import json
import math
from pathlib import Path
import sys

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "tests/data/validation/tio_candidate/20261103"
REFERENCE = ROOT / "examples/comparisons/data/major_molecule_reference.json"
SOURCE_REVIEW = ROOT / "tests/data/validation/tio_source_review/20261103"
sys.path.insert(0, str(ROOT / "examples/comparisons"))
import select_major_molecules as major


R = major.R
KB = 1.380649e-23
PLANCK = 6.62607015e-34
C2 = PLANCK*299792458*100/KB
EV_K = 1.602176634e-19/KB
D0_EV = 6.824
MASS_U = 63.8664
KNOTS = np.arange(1500., 4001., 500.)
STATE_FILE = "48Ti-16O__Toto.states.bz2"
EXPECTED_STATES_SHA256 = "7d54bf8d3ba2fb1a1f7a5d5f25b9335f0ee381625a873813e6a4ca1979734067"


@lru_cache(maxsize=1)
def load_candidate() -> dict:
    """Load the compact coefficients; no external line list is needed to evaluate."""
    return json.loads((DATA/"candidate.json").read_text())


def _polynomial(temperature, record: dict, xp):
    knots = xp.asarray(record["knots_K"])
    i = xp.clip(xp.searchsorted(knots, temperature, side="right")-1, 0, len(knots)-2)
    width = knots[i+1]-knots[i]
    u = (temperature-knots[i])/width
    c = xp.asarray(record["coefficients"])[i]
    value = sum(c[..., p]*u**p for p in range(6))
    slope = sum(p*c[..., p]*u**(p-1) for p in range(1, 6))/width
    curvature = sum(p*(p-1)*c[..., p]*u**(p-2) for p in range(2, 6))/width**2
    return value, slope, curvature


def _nasa(temperature, coefficients, xp):
    a, b, c, d, e, f, g, h, s = coefficients
    t = temperature
    entropy = -a/(2*t*t)-b/t+c*xp.log(t)+d*t+e*t*t/2+f*t**3/3+g*t**4/4+s
    enthalpy = -a/t+b*xp.log(t)+c*t+d*t*t/2+e*t**3/3+f*t**4/4+g*t**5/5+h
    cp = a/t**2+b/t+c+d*t+e*t*t+f*t**3+g*t**4
    return enthalpy-t*entropy, -entropy, -cp/t


def _thermal(temperature, thermal: str, xp):
    record = load_candidate()
    if thermal == "states":
        return _polynomial(temperature, record, xp)
    if thermal == "nasa":
        return _nasa(temperature, record["nasa_molecular_thermal_coefficients"], xp)
    raise ValueError("thermal must be 'states' or 'nasa'")


def molecular_properties(temperature_K: float, thermal: str = "states") -> dict:
    """Return G-H0, H-H0 in J/mol and S, Cp in J/mol/K on 1500-4000 K."""
    if not math.isfinite(temperature_K) or not 1500 <= temperature_K <= 4000:
        raise ValueError("Temperature outside TiO candidate domain")
    t = float(temperature_K)
    f, slope, curvature = _thermal(t, thermal, np)
    return {"G_relative_J_mol": float(R*f), "H_relative_J_mol": float(R*(f-t*slope)),
            "S_J_mol_K": float(-R*slope), "Cp_J_mol_K": float(-R*t*curvature)}


def formation_h(temperature_K: float, thermal: str = "states", d0_eV: float = D0_EV) -> float:
    """Return Gformation/(RT), or minus ln K, in the existing atomic gauge.

    Varying d0_eV changes only the reaction energy; the central-D0 level cutoff
    and thermal representation stay fixed. This is an elemental approximation,
    not an isotope-resolved reaction or a natural-isotope mixture calculation.
    """
    properties = molecular_properties(temperature_K, thermal)
    background = _nasa(float(temperature_K), load_candidate()["atomic_background_nasa9"], np)[0]
    return properties["G_relative_J_mol"]/(R*temperature_K)+(background-d0_eV*EV_K)/temperature_K


def formation_h_jax(temperature_K, thermal: str = "states", d0_eV: float = D0_EV):
    """JIT/grad/vmap compatible formation h; return NaN outside the declared domain.

    Use 64-bit JAX for the recorded numerical precision. The thermal selector
    is a static Python option; temperature and energy may be traced values.
    """
    import jax.numpy as jnp

    t = jnp.asarray(temperature_K, dtype=jnp.asarray(load_candidate()["coefficients"]).dtype)
    molecular = _thermal(t, thermal, jnp)[0]
    background = _nasa(t, load_candidate()["atomic_background_nasa9"], jnp)[0]
    value = (molecular+background-d0_eV*EV_K)/t
    valid = jnp.isfinite(t) & (t >= 1500) & (t <= 4000)
    return jnp.where(valid, value, jnp.nan)


def direct_properties(temperature: float, energy_K: np.ndarray, weights: np.ndarray) -> dict:
    """Compute level moments and ideal-gas molecular properties independently."""
    t = temperature
    population = weights*np.exp(-energy_K/t)
    q = float(population.sum())
    mean = float(np.sum(population*energy_K)/q)
    variance = float(np.sum(population*(energy_K-mean)**2)/q)
    kt = KB*t
    ln_translation = (1.5*math.log(2*math.pi*MASS_U*major.ATOMIC_MASS_KG*kt/PLANCK**2)
                      + math.log(kt/100000))
    g = -R*t*(math.log(q)+ln_translation)
    h = R*(2.5*t+mean)
    return {"temperature_K": t, "Q": q, "mean_energy_K": mean,
            "variance_energy_K2": variance, "G_relative_J_mol": g,
            "H_relative_J_mol": h, "S_J_mol_K": (h-g)/t,
            "Cp_J_mol_K": R*(2.5+variance/t**2)}


def hermite_coefficients(nodes: list) -> list:
    """Use the same endpoint G, -S, -Cp/T construction as curated JANAF fits."""
    coefficients = []
    for left, right in zip(nodes[:-1], nodes[1:]):
        t0, t1 = left["temperature_K"], right["temperature_K"]
        width = t1-t0
        c0 = left["G_relative_J_mol"]/R
        c1 = -width*left["S_J_mol_K"]/R
        c2 = -width**2*left["Cp_J_mol_K"]/(2*R*t0)
        r0 = right["G_relative_J_mol"]/R-c0-c1-c2
        r1 = -width*right["S_J_mol_K"]/R-c1-2*c2
        r2 = -width**2*right["Cp_J_mol_K"]/(R*t1)-2*c2
        coefficients.append([c0, c1, c2, 10*r0-4*r1+r2/2,
                             -15*r0+7*r1-r2, 6*r0-3*r1+r2/2])
    return coefficients


def extract(source_directory: Path, output_directory: Path) -> None:
    """Freeze a compact model and direct off-knot checks from pinned source files."""
    source = source_directory/STATE_FILE
    digest = sha256(source.read_bytes()).hexdigest()
    if digest != EXPECTED_STATES_SHA256:
        raise ValueError("Unreviewed TiO states file")
    reviewed = json.loads((SOURCE_REVIEW/"exomol_nodes.json").read_text())
    source_hashes = {STATE_FILE: digest}
    for isotope, item in reviewed["isotopologues"].items():
        for suffix in ("pf", "def"):
            filename = isotope+"__Toto."+suffix
            source_hashes[filename] = sha256((source_directory/filename).read_bytes()).hexdigest()
            if source_hashes[filename] != item[suffix+"_sha256"]:
                raise ValueError("Unreviewed TiO partition or definition file")
    with bz2.open(source, "rt") as stream:
        energy, weights, j = np.loadtxt(stream, usecols=(1, 2, 3), unpack=True)
    with bz2.open(source, "rt") as stream:
        labels = np.loadtxt(stream, usecols=9, dtype=str)
    if len(energy) != 301245 or np.any(weights != 2*j+1) or energy.min() != 0:
        raise ValueError("Unexpected state identity or degeneracies")
    theta = energy*C2
    bound = theta <= D0_EV*EV_K
    nodes = [direct_properties(float(t), theta[bound], weights[bound]) for t in KNOTS]
    reference = json.loads(REFERENCE.read_text())
    background = np.zeros(9)
    for atom in ("Ti", "O"):
        background -= np.asarray(major.nasa_interval(atom, 2500, reference)["coefficients"])
        background[7] += major.assigned_h0(atom, "nasa9", reference)/major.NASA_R
    nasa = np.array(major.nasa_interval("TiO", 2500, reference)["coefficients"])
    nasa[7] -= major.assigned_h0("TiO", "nasa9", reference)/major.NASA_R
    record = {"id": "tio_48_internal_elemental_candidate_v1", "temperature_range_K": [1500, 4000],
              "standard_pressure_Pa": 100000, "D0_eV": D0_EV, "molecular_mass_u": MASS_U,
              "D0_source": {"doi": "10.1063/5.0146365", "quoted_uncertainty_eV": .010},
              "nuclear_spin_factor": 1, "knots_K": KNOTS.tolist(),
              "coefficients": hermite_coefficients(nodes), "atomic_background_nasa9": background.tolist(),
              "nasa_molecular_thermal_coefficients": nasa.tolist(),
              "representation": "C2 quintic Hermite of molecular (G-H0)/R; no production update",
              "isotope_convention": "48Ti16O internal levels represent elemental TiO; NASA natural-mass translation and atoms; not a complete natural-isotope or isotope-resolved reaction",
              "energy_zero": "Lowest 48Ti16O rovibronic level; Ti(a3F2)+O(3P2) atomization; no additional ZPE or spin-orbit correction",
              "cutoff": {"D0_eV": D0_EV, "energy_cm-1": D0_EV*EV_K/C2,
                         "original_states": len(energy), "retained_states": int(bound.sum()),
                         "excluded_states": int((~bound).sum()), "maximum_original_energy_cm-1": float(energy.max()),
                         "J_max": int(j.max()), "energy_uncertainty_policy": "Thermal levels and central cutoff stay fixed when varying only D0"},
              "completeness": "All current-state electronic manifolds included; approximate D,F,g,h,i already present and are not added again",
              "source_sha256": {str(path.relative_to(ROOT)): sha256(path.read_bytes()).hexdigest()
                                for path in (REFERENCE, SOURCE_REVIEW/"exomol_nodes.json",
                                             SOURCE_REVIEW/"literature.json")},
              "original_states": {"url": "https://www.exomol.com/db/TiO/48Ti-16O/Toto/"+STATE_FILE,
                                  "sha256": digest, "version": "20240509", "archived": False}}
    off_knots = [float(t+fraction*500) for t in KNOTS[:-1]
                 for fraction in (.05, .125, .25, .375, .5, .625, .75, .875, .95)]
    references = [direct_properties(t, theta[bound], weights[bound]) for t in off_knots]
    cutoff = []
    scope_rows = []
    published = dict(np.loadtxt(source_directory/"48Ti-16O__Toto.pf"))
    fractions = {46: .0825, 47: .0744, 48: .7372, 49: .0541, 50: .0518}
    spin = {46: 1, 47: 6, 48: 1, 49: 8, 50: 1}
    partitions = {a: dict(np.loadtxt(source_directory/f"{a}Ti-16O__Toto.pf")) for a in fractions}
    approximate = np.isin(labels, ["D(3SIGMA-)", "F(3SIGMA+)", "g(1GAMMA)", "h(1SIGMA+)", "i(1PI)"])
    for node in nodes:
        t = node["temperature_K"]
        full = direct_properties(t, theta, weights)
        excluded_fraction = float(np.sum(weights[~bound]*np.exp(-theta[~bound]/t))/full["Q"])
        cutoff.append({"temperature_K": t, "excluded_Q_fraction": excluded_fraction,
                       "full_minus_bound_G_J_mol": full["G_relative_J_mol"]-node["G_relative_J_mol"],
                       "full_minus_bound_H_J_mol": full["H_relative_J_mol"]-node["H_relative_J_mol"],
                       "full_minus_bound_S_J_mol_K": full["S_J_mol_K"]-node["S_J_mol_K"],
                       "full_minus_bound_Cp_J_mol_K": full["Cp_J_mol_K"]-node["Cp_J_mol_K"],
                       "published_Q": float(published[t]), "full_states_Q_over_published": full["Q"]/published[t]})
        without_high = direct_properties(t, theta[~approximate], weights[~approximate])
        without_shell = direct_properties(t, theta[j <= 190], weights[j <= 190])
        q_isotopes = {a: partitions[a][t]/spin[a] for a in fractions}
        scope_rows.append({"temperature_K": t,
                           "titanium_fraction_weighted_internal_Q_over_48":
                           sum(fractions[a]*q_isotopes[a] for a in fractions)/q_isotopes[48],
                           "omit_five_approximate_states_Q_over_full": without_high["Q"]/full["Q"],
                           "omit_five_approximate_states_delta_Cp_J_mol_K": without_high["Cp_J_mol_K"]-full["Cp_J_mol_K"],
                           "omit_J_above_190_Q_over_full": without_shell["Q"]/full["Q"],
                           "omit_J_above_190_delta_Cp_J_mol_K": without_shell["Cp_J_mol_K"]-full["Cp_J_mol_K"]})
    scope = {"source_file_sha256": source_hashes,
             "source_url_pattern": "https://www.exomol.com/db/TiO/{isotope}/Toto/{filename}",
             "ciaaw_titanium": {"url": "https://www.ciaaw.org/titanium.htm", "fractions": fractions},
             "ciaaw_oxygen": {"url": "https://www.ciaaw.org/oxygen.htm", "O16_fraction_range": [.99738, .99776]},
             "electronic_states": len(set(labels)), "five_approximate_state_count": int(approximate.sum()),
             "rows": scope_rows,
             "limitations": ["Isotope weighted Q is a terrestrial-ratio sensitivity only; isotope-dependent D0, 17O/18O and network isotope fractionation are omitted.",
                             "Omitting represented high states or last rotational shells is a sensitivity, not an upper bound on unrepresented states or physical error.",
                             "All 16 electronic states, including the five approximate states, are included before the central bound-level cut; no supplement is added twice."]}
    output_directory.mkdir(parents=True, exist_ok=True)
    for name, data in (("candidate.json", record), ("nodes.json", nodes),
                       ("off_knot_reference.json", references), ("cutoff_sensitivity.json", cutoff),
                       ("scientific_scope.json", scope)):
        (output_directory/name).write_text(json.dumps(data, indent=2, allow_nan=False)+"\n")


def validate() -> dict:
    """Compare the frozen fit with direct states at withheld temperatures."""
    references = json.loads((DATA/"off_knot_reference.json").read_text())
    maxima = dict.fromkeys(("ln_K", "H_J_mol", "S_J_mol_K", "Cp_J_mol_K"), 0.)
    for row in references:
        t = row["temperature_K"]
        result = molecular_properties(t)
        errors = {"ln_K": abs(result["G_relative_J_mol"]-row["G_relative_J_mol"])/(R*t),
                  "H_J_mol": abs(result["H_relative_J_mol"]-row["H_relative_J_mol"]),
                  "S_J_mol_K": abs(result["S_J_mol_K"]-row["S_J_mol_K"]),
                  "Cp_J_mol_K": abs(result["Cp_J_mol_K"]-row["Cp_J_mol_K"])}
        maxima = {key: max(maxima[key], value) for key, value in errors.items()}
    minima = []
    for t, width, values in zip(KNOTS[:-1], np.diff(KNOTS), load_candidate()["coefficients"]):
        curvature = np.arange(2, 6)*np.arange(1, 5)*np.asarray(values)[2:]/width**2
        cp = -R*np.polynomial.polynomial.polymul([t, width], curvature)
        roots = np.polynomial.polynomial.polyroots(np.arange(1, len(cp))*cp[1:])
        points = [0., 1.]+[x.real for x in roots if abs(x.imag) < 1e-10 and 0 < x.real < 1]
        minima.append(float(min(np.polynomial.polynomial.polyval(u, cp) for u in points)))
    return {"off_knot_points": len(references), "maximum_absolute_fit_errors": maxima,
            "minimum_molecular_Cp_J_mol_K": min(minima),
            "fit_targets": {"ln_K": 1e-5, "Cp_J_mol_K": .01},
            "scope": "Numerical interpolation error against the named finite level model; not physical uncertainty or equilibrium validation"}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--extract", type=Path, help="Directory containing the pinned original states and PF")
    parser.add_argument("--output-directory", type=Path, default=DATA)
    parser.add_argument("--temperature", type=float, default=2500.)
    arguments = parser.parse_args()
    if arguments.extract:
        extract(arguments.extract, arguments.output_directory)
        if arguments.output_directory.resolve() == DATA.resolve():
            load_candidate.cache_clear()
            (DATA/"validation.json").write_text(json.dumps(validate(), indent=2)+"\n")
            inputs = [Path(__file__).resolve(), REFERENCE, SOURCE_REVIEW/"exomol_nodes.json",
                      SOURCE_REVIEW/"literature.json",
                      ROOT/"examples/comparisons/select_major_molecules.py",
                      ROOT/"examples/comparisons/audit_atomic_ions.py"]
            outputs = [DATA/name for name in ("candidate.json", "nodes.json", "off_knot_reference.json",
                                              "cutoff_sensitivity.json", "scientific_scope.json", "validation.json")]
            hashes = lambda paths: {str(p.relative_to(ROOT)): sha256(p.read_bytes()).hexdigest() for p in paths}
            receipt = {"command": "PYTHONPATH=src python mceg/reviews/tio_candidate.py --extract ORIGINAL_SOURCE_DIRECTORY",
                       "selected_input_sha256": hashes(inputs), "output_sha256": hashes(outputs),
                       "external_sources": "Original full files are not vendored; URLs and hashes in candidate.json and scientific_scope.json",
                       "scope": "Six direct-state knots and 45 independent off-knot references; bound-state model fixed at central D0; no equilibrium solve or production change"}
            (DATA/"receipt.json").write_text(json.dumps(receipt, indent=2)+"\n")
    else:
        print(json.dumps({"temperature_K": arguments.temperature,
                          "formation_h": formation_h(arguments.temperature),
                          "nasa_proxy_h": formation_h(arguments.temperature, "nasa"),
                          "molecular": molecular_properties(arguments.temperature)}, indent=2))
