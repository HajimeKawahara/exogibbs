"""Compare retained acetylene thermals and exact NASA9 at one ATcT energy."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
from functools import lru_cache
import json
import math
from pathlib import Path
import platform
import sys

import numpy as np

import c2h2_source_review as source
from gas_source_comparison import file_hash
from tio_candidate import _nasa

ROOT, R, NASA_R = source.ROOT, source.R, source.NASA_R
DATA = ROOT / "tests/data/validation/c2h2_candidate/20261106"
FASTCHEM = ROOT / "src/exogibbs/data/FastChem4/logK/logK.dat"
SPECIES = "C2H2"
COMPOSITION = {"C": 2, "H": 2}
THERMAL = ("retained", "nasa")
PROPERTY_KEYS = ("G_J_mol", "H_J_mol", "S_J_mol_K", "Cp_J_mol_K")


@lru_cache(maxsize=1)
def load_candidate() -> dict:
    """Load literal coefficients and two independently balanced energy shifts."""
    return json.loads((DATA / "candidate.json").read_text())


def build() -> dict:
    """Balance physical source anchors without treating a fit intercept as D0."""
    inputs = json.loads((source.DATA / "inputs.json").read_text())
    paths = [source.DATA / "inputs.json", source.DATA / "sources.json", FASTCHEM,
             source.DATA / "nasa-selected.txt", source.DATA / "comparison.json"]
    tables = {n: source.read_table(source.DATA / f) for n, f in inputs["janaf_files"].items()}
    paths.extend(source.DATA / f for f in inputs["janaf_files"].values())
    h0 = {n: round(1000 * (table[298.15][4] + table[0.][3])) for n, table in tables.items()}
    balance = lambda v: math.fsum(c * v[n] for n, c in COMPOSITION.items()) - v[SPECIES]
    old_d0 = balance(h0)
    evaluated = inputs["atct"]["species"]
    d0 = balance({n: round(1000 * row["hf0_kJ_mol"]) for n, row in evaluated.items()})
    lines = FASTCHEM.read_text().splitlines()
    index = next(i for i, line in enumerate(lines) if line.startswith(SPECIES + " "))
    coefficients = list(map(float, lines[index + 1].split()))
    records = inputs["nasa_records"]
    nh0 = {n: row["hf298_J_mol"] - row["h298_minus_h0_J_mol"] for n, row in records.items()}
    nasa_d0 = balance(nh0)
    nasa = {n: np.asarray(next(row["coefficients"] for row in rec["intervals"]
            if row["bounds_K"] == [1000., 6000.])) for n, rec in records.items()}
    reaction = nasa[SPECIES] - 2 * nasa["C"] - 2 * nasa["H"]
    energy_shift = d0 / R - nasa_d0 / NASA_R
    reaction[7] -= energy_shift
    return {"id": "c2h2_retained_and_nasa_atct1228_candidates_v1", "species": SPECIES,
            "reaction": inputs["reaction"], "composition": COMPOSITION,
            "baseline_dataset": "curated-tio2-v1", "temperature_range_K": [1500, 4000],
            "standard_pressure_Pa": 100000, "gas_constant_J_mol_K": R,
            "nasa_native_R_J_mol_K": NASA_R, "atct_version": "1.228",
            "same_version_hf0_kJ_mol": {n: row["hf0_kJ_mol"] for n, row in evaluated.items()},
            "janaf_h0_J_mol": h0, "retained_source_D0_J_mol": old_d0, "D0_J_mol": d0,
            "delta_D0_J_mol": d0 - old_d0, "delta_D0_over_R_K": (d0 - old_d0) / R,
            "original_record": lines[index], "original_logK_coefficients": coefficients,
            "formal_fit_intercept_Ra_J_mol": R * coefficients[0],
            "nasa_h0_J_mol": nh0, "nasa_D0_J_mol": nasa_d0,
            "nasa_energy_shift_over_R_K": energy_shift, "nasa_control_coefficients": reaction.tolist(),
            "representation": "Retained: add physical delta(D0)/(RT) to the unchanged literal formation fit. NASA: exact acetylene-2C-2H NASA9, replace b1 energy once. Neither candidate is a new thermal fit.",
            "energy_scope": "D0 is a balanced physical source anchor, not R times the fitted 1/T coefficient. Retain the JANAF/fit residual. No extra ZPE, isotope shift or extrapolation to zero K.",
            "nasa_scope": "Remove NASA H0 with native R=8.314510, insert ATcT D0 with SI R. Retain NASA dimensionless molecular and atomic thermals and express all candidate calorics using SI R. This changes the thermal-reference model and fit residual as well as the molecular thermal shape.",
            "aceTY_scope": "The source packet's four finite PF diagnostics remain limited controls; no aCeTY extrapolation or complete hot state-sum claim is used.",
            "uncertainty_scope": "Same-version ATcT central values; marginal molecular uncertainties are not a balanced reaction covariance or a thermal uncertainty interval.",
            "source_sha256": {str(p.relative_to(ROOT)): file_hash(p) for p in paths}}


def _formation(t, thermal, xp):
    model = load_candidate()
    if thermal == "retained":
        a, b, c, d, e = model["original_logK_coefficients"]
        g = -a - b*t*xp.log(t) - c*t - d*t*t - e*t**3 - model["delta_D0_over_R_K"]
        slope = -b*(xp.log(t)+1) - c - 2*d*t - 3*e*t*t
        curvature = -b/t - 2*d - 6*e*t
        return g, slope, curvature
    if thermal == "nasa":
        return _nasa(t, model["nasa_control_coefficients"], xp)
    raise ValueError("thermal must be 'retained' or 'nasa'")


def formation_h(temperature_K: float, thermal: str = "retained") -> float:
    """Return one-bar Gformation/(RT) on 1500-4000 K only."""
    if not math.isfinite(temperature_K) or not 1500 <= temperature_K <= 4000:
        raise ValueError("Temperature outside C2H2 candidate domain")
    return float(_formation(float(temperature_K), thermal, np)[0] / temperature_K)


def formation_h_jax(temperature_K, thermal: str = "retained"):
    """Support traced scalar/array/JIT/vmap; invalid inputs yield NaN."""
    import jax.numpy as jnp
    t = jnp.asarray(temperature_K, dtype=jnp.asarray(load_candidate()["original_logK_coefficients"]).dtype)
    valid = jnp.isfinite(t) & (t >= 1500) & (t <= 4000)
    safe = jnp.where(valid, t, 1500.)
    return jnp.where(valid, _formation(safe, thermal, jnp)[0] / safe, jnp.nan)


def reaction_properties(temperature_K: float, thermal: str = "retained") -> dict:
    """Differentiate the same continuous Gibbs function for G/H/S/Cp."""
    formation_h(temperature_K, thermal)
    t = float(temperature_K)
    g, slope, curvature = _formation(t, thermal, np)
    return dict(zip(PROPERTY_KEYS, map(float, (R*g, R*(g-t*slope), -R*slope, -R*t*curvature))))


def validate() -> dict:
    """Use independent source expressions, current runtime and JAX derivatives."""
    import jax
    from exogibbs.presets import curated_tio2
    jax.config.update("jax_enable_x64", True)
    model = load_candidate()
    assert model == build()
    records = json.loads((source.DATA / "inputs.json").read_text())["nasa_records"]
    ts = np.unique(np.r_[np.arange(1500., 4001.), np.linspace(1500.137, 3999.813, 251)])
    setup = curated_tio2.chemsetup(ions=True)
    assert setup.metadata["dataset"] == model["baseline_dataset"]
    current = np.asarray(setup.hvector_func(ts))[:, setup.species.index(SPECIES)]
    a, b, c, d, e = model["original_logK_coefficients"]
    independent_fit = -(a/ts + b*np.log(ts) + c + d*ts + e*ts*ts)
    old_props = np.stack([R*ts*independent_fit, R*(-a+b*ts+d*ts*ts+2*e*ts**3),
        R*(b*np.log(ts)+b+c+2*d*ts+3*e*ts*ts), R*(b+2*d*ts+6*e*ts*ts)], axis=-1)
    expected_retained = old_props.copy(); expected_retained[:, :2] -= model["delta_D0_J_mol"]
    nasa = []
    for t in ts:
        raw = {n: source.nasa_properties(rec, t, NASA_R) for n, rec in records.items()}
        dh, ds, cp = [math.fsum([raw[SPECIES][i], -2*raw["C"][i], -2*raw["H"][i]]) for i in range(3)]
        h = R * (dh / NASA_R - model["nasa_energy_shift_over_R_K"])
        entropy, heat = R * ds / NASA_R, R * cp / NASA_R
        nasa.append([h-t*entropy, h, entropy, heat])
    reconstruction, derivatives = {}, {}
    for thermal, expected in zip(THERMAL, (expected_retained, np.asarray(nasa))):
        actual = np.array([list(reaction_properties(t, thermal).values()) for t in ts])
        reconstruction[thermal] = dict(zip(PROPERTY_KEYS, map(float, abs(actual-expected).max(axis=0))))
        f = jax.jit(lambda t: formation_h_jax(t, thermal))
        g = lambda t: R*t*f(t)
        first = np.asarray(jax.jit(jax.vmap(jax.grad(g)))(ts))
        second = np.asarray(jax.jit(jax.vmap(jax.grad(jax.grad(g))))(ts))
        jg = R*ts*np.asarray(f(ts))
        jprops = np.stack([jg, jg-ts*first, -first, -ts*second], axis=-1)
        derivatives[thermal] = dict(zip(PROPERTY_KEYS, map(float, abs(jprops-actual).max(axis=0))))
    assert max(v for group in (reconstruction, derivatives) for props in group.values() for v in props.values()) < 1e-8
    residuals = []
    for row in json.loads((source.DATA / "comparison.json").read_text())["samples"]:
        t = row["temperature_K"]
        before = row["reaction"]["curated"]["ln_K"] - row["reaction"]["janaf"]["ln_K"]
        after = -formation_h(t) - (row["reaction"]["janaf"]["ln_K"] + model["delta_D0_over_R_K"]/t)
        residuals.append({"temperature_K":t,"original_fit_residual_ln_K":before,"preserved_fit_residual_ln_K":after,"error_ln_K":after-before})
    samples = []
    for t in (1500., 2000., 2500., 3000., 3500., 4000.):
        old_h = float(setup.hvector_func(t)[setup.species.index(SPECIES)])
        samples.append({"temperature_K":t,"retained_over_current_K":math.exp(model["delta_D0_over_R_K"]/t),
            "nasa_over_current_K":math.exp(old_h-formation_h(t,"nasa")),
            "nasa_over_retained_K":math.exp(formation_h(t)-formation_h(t,"nasa")),
            "retained_reaction":reaction_properties(t),"nasa_reaction":reaction_properties(t,"nasa")})
    return {"temperature_count":len(ts),"integer_grid_count":2501,"offgrid_count":251,"range_K":[1500,4000],
            "maximum_abs_current_h_error":float(abs(current-independent_fit).max()),
            "independent_property_errors":reconstruction,"jax_property_errors":derivatives,
            "source_fit_residuals":residuals,"samples":samples,"production_change":"none",
            "physical_accuracy":"unknown"}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--build", action="store_true")
    args = parser.parse_args()
    if args.build:
        DATA.mkdir(parents=True, exist_ok=True)
        (DATA / "candidate.json").write_text(json.dumps(build(),indent=2)+"\n")
        load_candidate.cache_clear()
    result = validate()
    if args.build:
        (DATA / "validation.json").write_text(json.dumps(result,indent=2)+"\n")
        paths = {ROOT/n for n in load_candidate()["source_sha256"]}
        paths.update(Path(m.__file__).resolve() for m in tuple(sys.modules.values())
                     if getattr(m,"__file__",None) and Path(m.__file__).resolve().is_relative_to(ROOT)
                     and Path(m.__file__).suffix==".py" and Path(m.__file__).name not in ("exogibbs_version.py","curated.py"))
        paths.add(Path(__file__).resolve())
        paths.update((ROOT/"src/exogibbs/data/curated").glob("*.json"))
        receipt={"created_utc":datetime.now(timezone.utc).isoformat(),"result":"passed","python":platform.python_version(),
                 "command":[sys.executable,*sys.argv],"dtype":"float64","selected_file_sha256":{str(p.relative_to(ROOT)):file_hash(p) for p in sorted(paths)},
                 "output_sha256":{n:file_hash(DATA/n) for n in ("candidate.json","validation.json")}}
        (DATA/"receipt.json").write_text(json.dumps(receipt,indent=2)+"\n")
    print(json.dumps(result,indent=2))
