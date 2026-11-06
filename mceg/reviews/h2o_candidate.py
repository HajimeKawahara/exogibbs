"""Build two scoped H2O thermal candidates without changing production.

The evaluated model integrates a smooth fit to published 2025 Cp; the discrete
alternative interpolates direct PW2024 level moments. Both retain the current
ATcT total atomization energy and NASA atomic reference and aggregate mass.
"""

import argparse
import csv
from functools import lru_cache
from hashlib import sha256
import json
import math
from pathlib import Path

import numpy as np
from numpy.polynomial import Chebyshev, Polynomial

import h2o_source_review as source
from tio_candidate import _nasa, _polynomial, hermite_coefficients


ROOT = source.ROOT
DATA = ROOT / "tests/data/validation/h2o_candidate/20261104"
R = source.R
KNOTS = np.arange(1500., 4001., 500.)
MASS_U = source.NATURAL_MASS_U
THERMAL = ("evaluated", "discrete")
PROPERTY_KEYS = ("G_relative_J_mol", "H_relative_J_mol", "S_J_mol_K", "Cp_J_mol_K")


@lru_cache(maxsize=1)
def load_candidate() -> dict:
    """Read the frozen compact model; original downloads are unnecessary."""
    return json.loads((DATA / "candidate.json").read_text())


def _powers(value, coefficients):
    result = value * 0
    for coefficient in reversed(coefficients):
        result = result * value + coefficient
    return result


def _evaluated(temperature, record: dict, xp):
    """Return molecular G/R and its derivatives from one analytic Cp/R fit."""
    t = temperature
    u = (t - record["center_K"]) / record["scale_K"]
    u0 = (record["anchor_temperature_K"] - record["center_K"]) / record["scale_K"]
    cp = _powers(u, record["Cp_over_R_polynomial_u"])
    h = record["anchor_H_over_R_K"] + record["scale_K"] * (
        _powers(u, record["enthalpy_integral_u"]) - _powers(u0, record["enthalpy_integral_u"]))
    s = (record["anchor_S_over_R"] + _powers(u, record["entropy_integral_u"])
         - _powers(u0, record["entropy_integral_u"])
         + record["entropy_log_coefficient"] * xp.log(t / record["anchor_temperature_K"]))
    return h - t * s, -s, -cp / t


def _thermal(temperature, thermal: str, xp):
    model = load_candidate()
    if thermal == "evaluated":
        return _evaluated(temperature, model[thermal], xp)
    if thermal == "discrete":
        return _polynomial(temperature, model[thermal], xp)
    raise ValueError("thermal must be 'evaluated' or 'discrete'")


def molecular_properties(temperature_K: float, thermal: str = "evaluated") -> dict:
    """Return ground-relative G/H, absolute S and Cp on 1500-4000 K."""
    t = float(temperature_K)
    if not math.isfinite(t) or not 1500 <= t <= 4000:
        raise ValueError("Temperature outside H2O candidate domain")
    g, slope, curvature = _thermal(t, thermal, np)
    return dict(zip(PROPERTY_KEYS, map(float, (R * g, R * (g - t * slope),
                                              -R * slope, -R * t * curvature))))


def formation_h(temperature_K: float, thermal: str = "evaluated") -> float:
    """Return Gformation/(RT) for 2H+O -> H2O with the retained total D0."""
    molecular = molecular_properties(temperature_K, thermal)
    model = load_candidate()
    background = _nasa(temperature_K, model["atomic_background_nasa9"], np)[0]
    return (molecular["G_relative_J_mol"] / R + background - model["D0_over_k_K"]) / temperature_K


def formation_h_jax(temperature_K, thermal: str = "evaluated"):
    """JIT/batch/grad-compatible h; NaN outside the domain, using 64-bit JAX."""
    import jax.numpy as jnp

    model = load_candidate()
    t = jnp.asarray(temperature_K, dtype=jnp.asarray(model["atomic_background_nasa9"]).dtype)
    molecular = _thermal(t, thermal, jnp)[0]
    background = _nasa(t, model["atomic_background_nasa9"], jnp)[0]
    h = (molecular + background - model["D0_over_k_K"]) / t
    return jnp.where(jnp.isfinite(t) & (t >= 1500) & (t <= 4000), h, jnp.nan)


def evaluated_fit(tables: dict) -> dict:
    """Fit 251 Cp values; hold H/S/Q and 2250 intermediate Cp rows for checks.

    Inverse published U(Cp) weights balance numerical approximation across the
    interval. They imply neither independent errors nor a new fit uncertainty.
    """
    temperatures = tables["temperature_K"]
    selected = np.flatnonzero((temperatures - 1500) % 10 == 0)
    u = (temperatures[selected] - 2750) / 1250
    polynomial = Chebyshev.fit(
        u, tables["Cp_J_mol_K"][selected] / R, deg=8,
        w=1 / tables["expanded_U_Cp_J_mol_K"][selected], domain=[-1, 1]).convert(kind=Polynomial)
    quotient, remainder = divmod(polynomial, Polynomial([2750 / 1250, 1]))
    return {"representation": "Degree-8 Cp/R in scaled temperature, analytically integrated to H/R and S/R",
            "center_K": 2750., "scale_K": 1250., "anchor_temperature_K": 1500.,
            "anchor_H_over_R_K": float(tables["H_relative_J_mol"][0] / R),
            "anchor_S_over_R": float(tables["S_J_mol_K"][0] / R),
            "Cp_over_R_polynomial_u": polynomial.coef.tolist(),
            "enthalpy_integral_u": polynomial.integ().coef.tolist(),
            "entropy_integral_u": quotient.integ().coef.tolist(),
            "entropy_log_coefficient": float(remainder.coef[0]),
            "fit_temperature_K": temperatures[selected].tolist(),
            "fit_weights": "Inverse published expanded Cp uncertainty; numerical weighting, not a covariance model"}


def direct_properties(t: float, energy_K: np.ndarray, weights: np.ndarray) -> dict:
    """Sum all supplied PW2024 states with shared para ground zero and Q/4."""
    population = weights * np.exp(-energy_K / t)
    q = float(population.sum())
    mean = float(np.sum(population * energy_K) / q)
    variance = float(np.sum(population * (energy_K - mean)**2) / q)
    moment = {"Q": q, "mean_energy_K": mean, "variance_energy_K2": variance}
    properties = source.molecular_properties(t, moment, MASS_U)
    values = [properties[k] for k in ("Grel_J_mol", "Hrel_J_mol", "S_J_mol_K", "Cp_J_mol_K")]
    return {"temperature_K": t, **moment, **dict(zip(PROPERTY_KEYS, values))}


def extract(directory: Path, output: Path) -> None:
    """Build coefficients and independent references from three pinned originals."""
    names = ("H216O_ThermoFunctions.csv", "H216O_PartitionFunctions.csv", "PW2024.txt")
    expected = json.loads((source.DATA / "sources.json").read_text())["downloads"]
    hashes = {}
    for name in names:
        hashes[name] = sha256((directory / name).read_bytes()).hexdigest()
        if hashes[name] != next(r["sha256"] for r in expected if r["file"] == name):
            raise ValueError("Changed original source: " + name)
    rows = []
    for name in names[:2]:
        with (directory / name).open(newline="") as stream:
            rows.append([{k: float(v) for k, v in row.items()} for row in csv.DictReader(stream)
                         if 1500 <= float(row["T/K"]) <= 4000])
    thermo, partition = rows
    shift = 1.5 * R * math.log(MASS_U / source.ISOTOPE_MASS_U)
    tables = {"temperature_K": np.array([r["T/K"] for r in thermo]),
              "H_relative_J_mol": np.array([r["H/kJ/mol"] for r in thermo]),
              "S_J_mol_K": np.array([r["S/J/K mol"] + shift for r in thermo]),
              "Cp_J_mol_K": np.array([r["C_p/J/K mol"] for r in thermo]),
              "expanded_U_H_J_mol": np.array([r["U_ULB(H)/kJ/mol"] for r in thermo]),
              "expanded_U_S_J_mol_K": np.array([r["U_ULB(S)/J/K mol"] for r in thermo]),
              "expanded_U_Cp_J_mol_K": np.array([r["U_ULB(C_p)/J/K mol"] for r in thermo]),
              "Q_raw": np.array([r["Q"] for r in partition]),
              "expanded_U_Q": np.array([r["U_ULB(Q)"] for r in partition]),
              "Q_first_moment": np.array([r["Q'"] for r in partition]),
              "Q_second_moment": np.array([r['Q"'] for r in partition])}
    if not np.array_equal(tables["temperature_K"], [r["T/K"] for r in partition]):
        raise ValueError("Inconsistent table temperature columns")
    levels = np.genfromtxt(directory / names[2], skip_header=1,
                          dtype=[("energy", float), ("j", int), ("spin", "U1"), ("uncertainty", float)])
    if len(levels) != 810270 or set(levels["spin"]) != {"O", "P"} or levels["energy"].min() != 0:
        raise ValueError("Unexpected PW2024 state identity")
    energy = levels["energy"] * source.C2
    weights = np.where(levels["spin"] == "O", 3, 1) * (2 * levels["j"] + 1)
    nodes = [direct_properties(t, energy, weights) for t in KNOTS]
    withheld = [direct_properties(float(t + f * 500), energy, weights)
                for t in KNOTS[:-1] for f in (.05, .125, .25, .375, .5, .625, .75, .875, .95)]
    reference = json.loads(source.major.REFERENCE_PATH.read_text())
    background = np.zeros(9)
    for atom, count in (("H", 2), ("O", 1)):
        background -= count * np.asarray(source.major.nasa_interval(atom, 2500, reference)["coefficients"])
        background[7] += count * source.major.assigned_h0(atom, "nasa9", reference) / source.major.NASA_R
    record = {"id": "h2o_2025_evaluated_and_PW2024_discrete_candidates_v1",
              "temperature_range_K": [1500, 4000], "standard_pressure_Pa": 100000,
              "D0_J_mol": 917819., "D0_over_k_K": source.major.atomization_kelvin("H2O", "atct", reference),
              "atomic_background_nasa9": background.tolist(), "molecular_mass_u": MASS_U,
              "isotope_internal_states": "H2(16O), dominant isotope representing element-aggregate H2O; not a complete natural-isotope reaction",
              "spin_and_zero": "Raw discrete Q/4, state-dependent 3/1 weights retained; published S already spin-free; shared para ground zero; no extra symmetry division or ZPE",
              "evaluated_scope": "2025 evaluated Cp integrated with its 1500 K H/S anchors; numeric CSV H is J/mol despite header; additional evaluated contributions retained without claiming a unique scattering attribution",
              "discrete_scope": "All 810270 PW2024 supplied levels; no separate OH+H scattering calculation or additional continuum term",
              "scattering_limit": "Scattering phase-shift terms need not equal a duplicate free OH+H population, but their interaction/cluster assignment in this ideal reacting mixture is unverified",
              "source_sha256": hashes,
              "source_doi": ["10.1063/5.0273661", "10.60893/figshare.jpr.29087330.v1",
                             "10.60893/figshare.jpr.29087339.v1", "10.60893/figshare.jpr.29087342.v1"],
              "evaluated": evaluated_fit(tables),
              "discrete": {"representation": "C2 quintic Hermite of (G-H0)/R from full discrete-level moments",
                           "knots_K": KNOTS.tolist(), "coefficients": hermite_coefficients(nodes)}}
    output.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output / "published_tables.npz", **tables)
    for name, value in (("candidate.json", record), ("discrete_nodes.json", nodes),
                        ("discrete_off_knot_reference.json", withheld)):
        (output / name).write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def validate() -> dict:
    """Measure held-out caloric, Q, smoothness and alternative-scope residuals."""
    with np.load(DATA / "published_tables.npz") as tables:
        temperatures = tables["temperature_K"]
        prediction = np.array([[molecular_properties(t)[k] for k in PROPERTY_KEYS] for t in temperatures])
        actual = np.array([tables["H_relative_J_mol"] - temperatures * tables["S_J_mol_K"],
                           tables["H_relative_J_mol"], tables["S_J_mol_K"], tables["Cp_J_mol_K"]]).T
        errors = prediction - actual
        published = {k: float(abs(errors[:, i]).max()) for i, k in enumerate(PROPERTY_KEYS)}
        ratios = {k: float(np.max(abs(errors[:, i]) / tables[name])) for i, k, name in
                  ((1, "H", "expanded_U_H_J_mol"), (2, "S", "expanded_U_S_J_mol_K"),
                   (3, "Cp", "expanded_U_Cp_J_mol_K"))}
        qg = np.array([-R * t * (math.log(q / 4) + source.translation_log_q(t, MASS_U))
                       for t, q in zip(temperatures, tables["Q_raw"])])
        independent_q = float(np.max(abs((prediction[:, 0] - qg) / (R * temperatures))))
        withheld = (temperatures - 1500) % 10 != 0
        withheld_cp = float(np.max(abs(errors[withheld, 3])))
    references = json.loads((DATA / "discrete_off_knot_reference.json").read_text())
    discrete = {k: max(abs(molecular_properties(r["temperature_K"], "discrete")[k] - r[k])
                       for r in references) for k in PROPERTY_KEYS}
    discrete["ln_K"] = max(abs(molecular_properties(r["temperature_K"], "discrete")[PROPERTY_KEYS[0]]
                               - r[PROPERTY_KEYS[0]]) / (R * r["temperature_K"]) for r in references)
    rows = []
    for t in KNOTS:
        evaluated, direct = (molecular_properties(t, thermal) for thermal in THERMAL)
        current = -source.curated_ln_k(t)
        rows.append({"temperature_K": t, "evaluated_over_current_K": math.exp(current - formation_h(t)),
                     "discrete_over_current_K": math.exp(current - formation_h(t, "discrete")),
                     "evaluated_over_discrete_K": math.exp(formation_h(t, "discrete") - formation_h(t)),
                     "evaluated_minus_discrete_properties": {k: evaluated[k] - direct[k] for k in PROPERTY_KEYS}})
    polynomial = Polynomial(load_candidate()["evaluated"]["Cp_over_R_polynomial_u"])
    critical = [-1, 1] + [r.real for r in polynomial.deriv().roots() if abs(r.imag) < 1e-10 and -1 < r.real < 1]
    return {"published_rows": len(temperatures), "fit_Cp_rows": 251, "withheld_Cp_rows": int(withheld.sum()),
            "evaluated_max_absolute_table_errors": published, "maximum_errors_over_published_expanded_U": ratios,
            "evaluated_withheld_Cp_max_error_J_mol_K": withheld_cp,
            "max_ln_K_difference_from_independent_rounded_Q": independent_q,
            "discrete_off_knot_points": len(references), "discrete_maximum_fit_errors": discrete,
            "evaluated_minimum_Cp_J_mol_K": float(min(polynomial(x) * R for x in critical)),
            "thermal_scope_comparison": rows,
            "uncertainty_scope": "Approximation errors versus source values; neither source-independent errors nor whole-model physical uncertainties"}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--extract", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.extract:
        extract(args.extract, args.output or DATA)
    else:
        result = json.dumps(validate(), indent=2, allow_nan=False) + "\n"
        if args.output:
            args.output.write_text(result)
        else:
            print(result, end="")
