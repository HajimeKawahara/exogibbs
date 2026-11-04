"""Build and evaluate the bounded OH NASA thermal + ATcT candidate offline.

The NASA9 reaction is an exact linear combination, with only its energy
constant replaced. MYTHOS is a direct finite-state thermal control, not a fit.
Neither path changes the production model or the NASA atomic thermal gauge.
"""

import argparse
from functools import lru_cache
import json
import math
from pathlib import Path

import numpy as np

import oh_source_review as source
from tio_candidate import _nasa


ROOT = source.ROOT
DATA = ROOT / "tests/data/validation/oh_candidate/20261105"
R = source.major.R
THERMAL = ("nasa", "mythos")
PROPERTY_KEYS = ("G_relative_J_mol", "H_relative_J_mol", "S_J_mol_K", "Cp_J_mol_K")
INPUTS = (source.BASELINE / "inputs.json", source.major.REFERENCE_PATH,
          source.DATA / "literature.json", source.DATA / "states.npz",
          source.DATA / "state_inventory.json")


@lru_cache(maxsize=1)
def load_candidate() -> dict:
    """Load the exact reaction and ground-relative thermal coefficients."""
    return json.loads((DATA / "candidate.json").read_text())


@lru_cache(maxsize=1)
def _states():
    with np.load(source.DATA / "states.npz") as archive:
        states = archive["MYTHOS"]
    return states[:, 0] * source.C2, states[:, 1] / 2


def _thermal(temperature, thermal, xp):
    model = load_candidate()
    if thermal == "nasa":
        return _nasa(temperature, model["nasa_molecular_thermal_coefficients"], xp)
    if thermal != "mythos":
        raise ValueError("thermal must be 'nasa' or 'mythos'")
    t = xp.asarray(temperature)
    energy, weights = (xp.asarray(values) for values in _states())
    population = weights * xp.exp(-energy / t[..., None])
    q = xp.sum(population, axis=-1)
    mean = xp.sum(population * energy, axis=-1) / q
    variance = xp.sum(population * (energy - mean[..., None])**2, axis=-1) / q
    kt = source.KB * t
    translation = (1.5 * xp.log(2 * math.pi * model["molecular_mass_u"]
                   * source.major.ATOMIC_MASS_KG * kt / source.PLANCK**2)
                   + xp.log(kt / model["standard_pressure_Pa"]))
    log_q = xp.log(q) + translation
    return -t * log_q, -(log_q + 2.5 + mean / t), -(2.5 + variance / t**2) / t


def molecular_properties(temperature_K: float, thermal: str = "nasa") -> dict:
    """Return molecular G-H0 and H-H0 (J/mol), S and Cp (J/mol/K)."""
    if not math.isfinite(temperature_K) or not 1500 <= temperature_K <= 4000:
        raise ValueError("Temperature outside OH candidate domain")
    t = float(temperature_K)
    g, slope, curvature = _thermal(t, thermal, np)
    return dict(zip(PROPERTY_KEYS, map(float, (
        R * g, R * (g - t * slope), -R * slope, -R * t * curvature))))


def _formation(temperature, thermal, xp):
    model = load_candidate()
    if thermal == "nasa":
        return _nasa(temperature, model["formation_nasa9_coefficients"], xp)[0] / temperature
    molecular = _thermal(temperature, thermal, xp)[0]
    background = _nasa(temperature, model["atomic_background_nasa9"], xp)[0]
    return (molecular + background - model["D0_over_k_K"]) / temperature


def formation_h(temperature_K: float, thermal: str = "nasa") -> float:
    """Return Gformation/(RT) = -ln K for H + O -> OH at one bar."""
    if not math.isfinite(temperature_K) or not 1500 <= temperature_K <= 4000:
        raise ValueError("Temperature outside OH candidate domain")
    return float(_formation(float(temperature_K), thermal, np))


def formation_h_jax(temperature_K, thermal: str = "nasa"):
    """Evaluate traced scalar/array temperatures; return NaN outside 1500-4000 K.

    The thermal selector is static. Use JAX float64 for archived precision.
    MYTHOS sums all archived states directly, including supplied quasi-bound
    levels; it adds no continuum or scattering contribution.
    """
    import jax.numpy as jnp

    t = jnp.asarray(temperature_K,
                    dtype=jnp.asarray(load_candidate()["formation_nasa9_coefficients"]).dtype)
    value = _formation(t, thermal, jnp)
    return jnp.where(jnp.isfinite(t) & (t >= 1500) & (t <= 4000), value, jnp.nan)


def build() -> dict:
    """Construct the exact NASA9 reaction from archived, hash-recorded inputs."""
    records = json.loads(INPUTS[0].read_text())["nasa9_records"]
    reference = json.loads(INPUTS[1].read_text())
    literature = json.loads(INPUTS[2].read_text())
    coefficients = {s: np.asarray(next(row["coefficients"] for row in records[s]["intervals"]
                    if row["bounds_K"] == [1000., 6000.])) for s in ("OH", "H", "O")}
    headers = literature["nasa_headers"]
    h0 = {s: h["hf298_J_mol"] - h["h298_minus_h0_J_mol"] for s, h in headers.items()}
    d0_nasa = h0["H"] + h0["O"] - h0["OH"]
    d0 = 1000 * (sum(reference["atct"]["records"][s]["hf0_kJ_mol"] for s in ("H", "O"))
                 - literature["atct"]["OH_hf0_kJ_mol"])
    shift = d0 / R - d0_nasa / source.major.NASA_R
    reaction = coefficients["OH"] - coefficients["H"] - coefficients["O"]
    reaction[7] -= shift
    molecule = coefficients["OH"].copy()
    molecule[7] -= h0["OH"] / source.major.NASA_R
    background = -coefficients["H"] - coefficients["O"]
    background[7] += (h0["H"] + h0["O"]) / source.major.NASA_R
    return {"id": "oh_nasa_thermal_atct_1_172_candidate_v1", "species": "H1O1",
            "reaction": "H(g)+O(g)->OH(g)", "temperature_range_K": [1500, 4000],
            "standard_pressure_Pa": 100000, "source_interval_K": [1000, 6000],
            "representation": "Exact OH-H-O NASA9 linear combination; only b1 receives the ATcT energy correction; no fit",
            "D0_J_mol": d0, "D0_over_k_K": d0 / R, "nasa_D0_J_mol": d0_nasa,
            "ATcT_minus_nasa_D0_over_k_K": shift,
            "gas_constant_J_mol_K": R, "nasa_native_gas_constant_J_mol_K": source.major.NASA_R,
            "formation_nasa9_coefficients": reaction.tolist(),
            "nasa_molecular_thermal_coefficients": molecule.tolist(),
            "atomic_background_nasa9": background.tolist(),
            "molecular_mass_u": source.MASS,
            "thermal_convention": "NASA dimensionless thermal functions retained; assigned H0 removed with native R, calorics expressed with modern SI R; balanced ATcT D0 inserted once",
            "mythos_control": "Direct sum of all 10606 archived 16O1H MYTHOS states; Q/2 removes nuclear spin once; parity and electronic/fine structure retained; natural-mass translation; no extra symmetry factor, ZPE, cutoff, continuum or scattering term",
            "uncertainty_scope": "Numerical reproduction and a finite-state thermal comparison; no new physical uncertainty or ATcT covariance estimate",
            "source_sha256": {str(p.relative_to(ROOT)): source.file_hash(p) for p in INPUTS}}


def validate() -> dict:
    """Check 2501 temperatures against separate NASA expressions and MYTHOS."""
    import jax

    model = load_candidate()
    for name, expected in model["source_sha256"].items():
        if source.file_hash(ROOT / name) != expected:
            raise ValueError("Changed candidate input: " + name)
    records = json.loads(INPUTS[0].read_text())["nasa9_records"]
    headers = json.loads(INPUTS[2].read_text())["nasa_headers"]
    h0_oh = headers["OH"]["hf298_J_mol"] - headers["OH"]["h298_minus_h0_J_mol"]
    t_grid = np.arange(1500., 4001.)
    independent, reaction_h, reaction_cp, properties, control = [], [], [], [], []
    for t in t_grid:
        raw = {s: source.nasa_properties(records[s], t, source.major.NASA_R) for s in headers}
        independent.append(sum(sign * source.nasa_g(records[s], t)
                              for s, sign in (("OH", 1), ("H", -1), ("O", -1))))
        reaction_h.append(R * ((raw["OH"][0] - raw["H"][0] - raw["O"][0])
                               / source.major.NASA_R - model["ATcT_minus_nasa_D0_over_k_K"]))
        reaction_cp.append(R * (raw["OH"][2] - raw["H"][2] - raw["O"][2]) / source.major.NASA_R)
        h, s, cp = raw["OH"]
        h, s, cp = (h - h0_oh) * R / source.major.NASA_R, s * R / source.major.NASA_R, cp * R / source.major.NASA_R
        properties.append([h - t * s, h, s, cp])
        control.append(formation_h(t, "mythos"))
    independent = np.asarray(independent) - model["ATcT_minus_nasa_D0_over_k_K"] / t_grid
    actual = np.array([formation_h(t) for t in t_grid])
    molecular = np.array([[molecular_properties(t)[k] for k in PROPERTY_KEYS] for t in t_grid])
    ratio = np.exp(actual - control)
    previous = jax.config.x64_enabled
    jax.config.update("jax_enable_x64", True)
    try:
        evaluated = np.asarray(jax.jit(formation_h_jax)(t_grid))
        first = np.asarray(jax.jit(jax.vmap(jax.grad(formation_h_jax)))(t_grid))
        second = np.asarray(jax.jit(jax.vmap(jax.grad(jax.grad(formation_h_jax))))(t_grid))
    finally:
        jax.config.update("jax_enable_x64", previous)
    samples = [{"temperature_K": float(t), "nasa_ATcT_over_current_K": math.exp(-source.curated_ln_k(t) - formation_h(t)),
                "mythos_ATcT_over_current_K": math.exp(-source.curated_ln_k(t) - formation_h(t, "mythos")),
                "mythos_over_nasa_K": float(ratio[int(t - 1500)]),
                "mythos_minus_nasa_molecular_properties": {k: molecular_properties(t, "mythos")[k]
                 - molecular_properties(t)[k] for k in PROPERTY_KEYS}} for t in np.arange(1500., 4001., 500.)]
    return {"temperature_count": len(t_grid), "temperature_range_K": [1500, 4000], "step_K": 1,
            "maximum_absolute_formation_h_reconstruction_error": float(np.max(abs(actual - independent))),
            "maximum_absolute_molecular_property_reconstruction_errors": dict(zip(PROPERTY_KEYS,
                np.max(abs(molecular - properties), axis=0).tolist())),
            "jax_float64_maximum_absolute_errors": {
                "formation_h": float(np.max(abs(evaluated - independent))),
                "reaction_H_J_mol": float(np.max(abs(-R * t_grid**2 * first - reaction_h))),
                "reaction_Cp_J_mol_K": float(np.max(abs(-R * (2 * t_grid * first + t_grid**2 * second) - reaction_cp)))},
            "mythos_over_nasa_K": {"minimum": float(ratio.min()), "minimum_temperature_K": float(t_grid[ratio.argmin()]),
                "maximum": float(ratio.max()), "maximum_temperature_K": float(t_grid[ratio.argmax()]),
                "maximum_absolute_relative_difference": float(np.max(abs(ratio - 1)))},
            "thermal_comparison": samples,
            "uncertainty_scope": model["uncertainty_scope"], "production_change": "none"}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--build", action="store_true", help="Rebuild candidate.json from frozen inputs")
    parser.add_argument("--output", type=Path, help="Write numerical validation JSON")
    args = parser.parse_args()
    if args.build:
        DATA.mkdir(parents=True, exist_ok=True)
        (DATA / "candidate.json").write_text(json.dumps(build(), indent=2, allow_nan=False) + "\n")
    result = json.dumps(validate(), indent=2, allow_nan=False) + "\n"
    if args.output:
        args.output.write_text(result)
    else:
        print(result, end="")
