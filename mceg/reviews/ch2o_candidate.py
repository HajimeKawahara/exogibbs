"""Construct bounded formaldehyde energy and thermal candidates offline."""

import argparse
from datetime import datetime, timezone
from functools import lru_cache
import json
import math
from pathlib import Path
import shlex
import sys

import numpy as np

import ch2o_source_review as source
from tio_candidate import _nasa

ROOT, R, NASA_R = source.ROOT, source.R, source.NASA_R
DATA = ROOT / "tests/data/validation/ch2o_candidate/20261105"
FASTCHEM = ROOT / "src/exogibbs/data/fastchem/logK/logK.dat"
THERMAL = ("retained", "nasa")
PROPERTY_KEYS = ("G_J_mol", "H_J_mol", "S_J_mol_K", "Cp_J_mol_K")


@lru_cache(maxsize=1)
def load_candidate() -> dict:
    """Read the frozen five-coefficient fit and common-energy NASA control."""
    return json.loads((DATA / "candidate.json").read_text())


def build() -> dict:
    """Apply one balanced source-anchor shift without extrapolating the fit."""
    inputs = json.loads((source.DATA / "inputs.json").read_text())
    paths = [source.DATA / "inputs.json", source.DATA / "sources.json", FASTCHEM,
             source.DATA / "C-061.txt"]
    tables = {name: source.read_table(ROOT / path)
              for name, path in inputs["atomic_janaf_tables"].items()}
    paths.extend(ROOT / path for path in inputs["atomic_janaf_tables"].values())
    tables["CH2O"] = source.read_table(source.DATA / "C-061.txt")
    h0 = {name: 1000*(table[298.15][4]+table[0.0][3]) for name, table in tables.items()}
    balance = lambda values: sum(n*values[e] for e, n in source.COMPOSITION.items())-values["CH2O"]
    original_d0 = balance(h0)
    atct = inputs["atct_1.228"]
    # The published millikJ values are integral J/mol. Balance those exact
    # values before division, independent of Python's float-sum algorithm.
    d0 = balance({name: 1000*row["hf0_kJ_mol"] for name, row in atct.items()})
    lines = FASTCHEM.read_text().splitlines()
    index = next(i for i, line in enumerate(lines) if line.startswith(source.SPECIES+" "))
    coefficients = np.array(list(map(float, lines[index+1].split())))
    updated = coefficients.copy()
    updated[0] += (d0-original_d0)/R
    records = inputs["nasa_records"]
    nh0 = {name: row["hf298_J_mol"]-row["h298_minus_h0_J_mol"] for name, row in records.items()}
    nasa_d0 = balance(nh0)
    nasa = {name: np.array(next(row["coefficients"] for row in record["intervals"]
            if row["bounds_K"] == [1000., 6000.])) for name, record in records.items()}
    reaction = nasa["CH2O"]-sum(n*nasa[e] for e, n in source.COMPOSITION.items())
    reaction[7] -= d0/R-nasa_d0/NASA_R
    return {"id": "ch2o_retained_thermal_atct_1_228_candidate_v1", "species": source.SPECIES,
            "reaction": "C(g)+2H(g)+O(g)->CH2O(g)", "baseline_dataset": "curated-cos-v1",
            "temperature_range_K": [1500, 4000], "standard_pressure_Pa": 100000,
            "gas_constant_J_mol_K": R, "nasa_native_R_J_mol_K": NASA_R,
            "atct_version": "1.228", "same_version_hf0_kJ_mol": {
                name: row["hf0_kJ_mol"] for name, row in atct.items()},
            "D0_J_mol": d0, "retained_source_D0_J_mol": original_d0,
            "delta_D0_J_mol": d0-original_d0, "delta_D0_over_R_K": (d0-original_d0)/R,
            "original_logK_coefficients": coefficients.tolist(),
            "candidate_logK_coefficients": updated.tolist(),
            "nasa_D0_J_mol": nasa_d0, "nasa_control_coefficients": reaction.tolist(),
            "representation": "Exact inherited formation fit with one 1/T energy shift; no refit",
            "source_anchor_scope": "JANAF Hf298 plus H0-H298 for molecule and atoms gives 1502104 J/mol; separately rounded Hf0 entries give 1502102. The fit is not extrapolated to zero K or claimed to reproduce every JANAF caloric node.",
            "nasa_control_scope": "Same balanced ATcT energy; NASA molecular and atomic dimensionless thermals, native-R removal of H0 and SI-R calorics. A thermal-reference alternative, not an uncertainty interval.",
            "uncertainty_scope": "Same-version ATcT central energies; molecular uncertainty is not reaction covariance. Historical thermal, natural-isotope and electronic-completeness limitations remain.",
            "source_sha256": {str(path.relative_to(ROOT)): source.file_hash(path) for path in paths}}


def _formation(t, thermal, xp):
    model = load_candidate()
    if thermal == "retained":
        a, b, c, d, e = model["candidate_logK_coefficients"]
        return -(a/t+b*xp.log(t)+c+d*t+e*t*t)
    if thermal == "nasa":
        return _nasa(t, model["nasa_control_coefficients"], xp)[0]/t
    raise ValueError("thermal must be 'retained' or 'nasa'")


def formation_h(temperature_K: float, thermal: str = "retained") -> float:
    """Return Gformation/(RT) in the bounded production-temperature domain."""
    if not math.isfinite(temperature_K) or not 1500 <= temperature_K <= 4000:
        raise ValueError("Temperature outside CH2O candidate domain")
    return float(_formation(float(temperature_K), thermal, np))


def formation_h_jax(temperature_K, thermal: str = "retained"):
    """Support differentiation and arrays; invalid traced inputs yield NaN."""
    import jax.numpy as jnp

    t = jnp.asarray(temperature_K, dtype=jnp.asarray(load_candidate()["candidate_logK_coefficients"]).dtype)
    valid = jnp.isfinite(t) & (t >= 1500) & (t <= 4000)
    safe = jnp.where(valid, t, 1500.)
    return jnp.where(valid, _formation(safe, thermal, jnp), jnp.nan)


def reaction_properties(temperature_K: float, thermal: str = "retained") -> dict:
    """Derive reaction H/S/Cp from the same continuous Gibbs representation."""
    t = float(temperature_K)
    g = R*t*formation_h(t, thermal)
    if thermal == "retained":
        a, b, c, d, e = load_candidate()["candidate_logK_coefficients"]
        h = R*(-a+b*t+d*t*t+2*e*t**3)
        entropy = R*(b*math.log(t)+b+c+2*d*t+3*e*t*t)
        cp = R*(b+2*d*t+6*e*t*t)
    else:
        value, slope, curvature = _nasa(t, load_candidate()["nasa_control_coefficients"], np)
        h, entropy, cp = R*(value-t*slope), -R*slope, -R*t*curvature
    return dict(zip(PROPERTY_KEYS, map(float, (g, h, entropy, cp))))


def validate() -> dict:
    """Check source reconstruction, runtime replay, calorics and derivatives."""
    import jax
    from exogibbs.presets import curated_cos

    model = load_candidate()
    assert model == build()
    records = json.loads((source.DATA / "inputs.json").read_text())["nasa_records"]
    temperatures = np.arange(1500., 4001.)
    old = [source.curated_properties(t) for t in temperatures]
    current = np.array([-row["ln_K"] for row in old])
    retained = np.array([formation_h(t) for t in temperatures])
    expected = current-model["delta_D0_over_R_K"]/temperatures
    nasa = []
    for t in temperatures:
        raw = {s: np.array(source.nasa_properties(record, t, NASA_R)) for s, record in records.items()}
        reaction = raw["CH2O"]-sum(n*raw[s] for s, n in source.COMPOSITION.items())
        nasa.append((reaction[0]/t-reaction[1])/NASA_R
                    -(model["D0_J_mol"]/R-model["nasa_D0_J_mol"]/NASA_R)/t)
    errors = {"retained_h": float(abs(retained-expected).max()),
              "nasa_h": float(abs(np.array([formation_h(t, "nasa") for t in temperatures])-nasa).max())}
    previous = jax.config.x64_enabled
    jax.config.update("jax_enable_x64", True)
    try:
        setup = curated_cos.chemsetup(ions=True)
        assert setup.metadata["dataset"] == model["baseline_dataset"]
        errors["current_runtime_h"] = float(abs(np.asarray(setup.hvector_func(temperatures))[
            :, setup.species.index(source.SPECIES)]-current).max())
        derivative = {}
        for thermal in THERMAL:
            f = jax.jit(lambda t: formation_h_jax(t, thermal))
            g = lambda t: R*t*f(t)
            first = np.asarray(jax.jit(jax.vmap(jax.grad(g)))(temperatures))
            second = np.asarray(jax.jit(jax.vmap(jax.grad(jax.grad(g))))(temperatures))
            jg = R*temperatures*np.asarray(f(temperatures))
            actual = np.stack([jg, jg-temperatures*first, -first, -temperatures*second], axis=-1)
            exact = np.array([[reaction_properties(t, thermal)[key] for key in PROPERTY_KEYS] for t in temperatures])
            derivative[thermal] = dict(zip(PROPERTY_KEYS, map(float, abs(actual-exact).max(axis=0))))
    finally:
        jax.config.update("jax_enable_x64", previous)
    change = np.array([[reaction_properties(t)[key]-row[key] for key in PROPERTY_KEYS[1:]]
                       for t, row in zip(temperatures, old)])
    change[:, 0] += model["delta_D0_J_mol"]
    return {"temperature_count": len(temperatures), "temperature_range_K": [1500, 4000], "step_K": 1,
            "maximum_absolute_reconstruction_error": errors, "jax_float64_derivative_errors": derivative,
            "energy_only_property_errors": dict(zip(PROPERTY_KEYS[1:], map(float, abs(change).max(axis=0)))),
            "thermal_comparison": [{"temperature_K": t,
                "candidate_over_current_K": math.exp(-formation_h(t)-source.curated_properties(t)["ln_K"]),
                "nasa_control_over_current_K": math.exp(-formation_h(t, "nasa")-source.curated_properties(t)["ln_K"]),
                "nasa_control_over_primary_K": math.exp(formation_h(t)-formation_h(t, "nasa")),
                "retained_reaction_properties": reaction_properties(t),
                "nasa_control_reaction_properties": reaction_properties(t, "nasa")}
                for t in (1500., 2000., 2500., 3000., 3500., 4000.)],
            "production_change": "none", "physical_accuracy": "unknown"}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--build", action="store_true")
    args = parser.parse_args()
    if args.build:
        DATA.mkdir(parents=True, exist_ok=True)
        (DATA / "candidate.json").write_text(json.dumps(build(), indent=2)+"\n")
        load_candidate.cache_clear()
    result = validate()
    if args.build:
        (DATA / "validation.json").write_text(json.dumps(result, indent=2)+"\n")
        paths = [Path(__file__), Path(source.__file__), ROOT / "mceg/reviews/tio_candidate.py",
                 ROOT / "mceg/reviews/c2h_source_review.py", ROOT / "mceg/reviews/gas_source_comparison.py"]
        paths.extend(ROOT / path for path in load_candidate()["source_sha256"])
        prior = json.loads((source.DATA / "receipt.json").read_text())
        paths.extend(ROOT / path for path in prior["selected_input_sha256"])
        paths.extend(Path(module.__file__).resolve() for module in tuple(sys.modules.values())
                     if getattr(module, "__file__", None)
                     and str(Path(module.__file__).resolve()).startswith(str(ROOT)+"/")
                     and Path(module.__file__).suffix == ".py"
                     and Path(module.__file__).name != "exogibbs_version.py")
        paths.extend((ROOT / "src/exogibbs").rglob("*.py"))
        paths.extend((ROOT / "src/exogibbs/data/curated").glob("*.json"))
        receipt = {"created_utc": datetime.now(timezone.utc).isoformat(), "result": "passed", "dtype": "float64",
                   "command": shlex.join([sys.executable, *sys.argv]),
                   "selected_file_sha256": {str(path.relative_to(ROOT)): source.file_hash(path) for path in paths},
                   "output_sha256": {name: source.file_hash(DATA / name) for name in ("candidate.json", "validation.json")}}
        (DATA / "receipt.json").write_text(json.dumps(receipt, indent=2)+"\n")
    print(json.dumps(result, indent=2))
