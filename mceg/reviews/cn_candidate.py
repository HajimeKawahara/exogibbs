"""Evaluate a CN ATcT energy correction while retaining the literal current fit."""

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

from carbon_radical_source_review import R, curated_properties
from c2h_source_review import read_table
from gas_source_comparison import ROOT, file_hash

NAME, SPECIES = "CN", "C1N1"
COMPOSITION = {"C": 1, "N": 1}
DATA = ROOT / "tests/data/validation/cn_candidate/20261106"
SOURCE = ROOT / "tests/data/validation/cn_source_review/20261106"
FIT = ROOT / "src/exogibbs/data/fastchem/logK/logK.dat"


@lru_cache(maxsize=1)
def load_candidate() -> dict:
    """Load the frozen energy correction and unmodified thermal coefficients."""
    return json.loads((DATA / "candidate.json").read_text())


def build() -> dict:
    """Balance physical JANAF/ATcT energies independently of the empirical fit."""
    facts = json.loads((SOURCE / "inputs.json").read_text())
    paths = {n: ROOT / p for n, p in facts["atomic_tables"].items()}
    paths[NAME] = SOURCE / facts["molecular_janaf_table"]
    tables = {n: read_table(p) for n, p in paths.items()}
    # These are fixed-reference offsets. Only the balanced difference is D0.
    offsets = {n: round(1000 * (tb[298.15][4] + tb[0.0][3])) for n, tb in tables.items()}
    old = sum(c * offsets[n] for n, c in COMPOSITION.items()) - offsets[NAME]
    modern = {n: row["hf0_kJ_mol"] for n, row in facts["atct"].items()}
    new = sum(c * round(1000 * modern[n]) for n, c in COMPOSITION.items()) - round(1000 * modern[NAME])
    lines = FIT.read_text().splitlines()
    i = next(i for i, line in enumerate(lines) if line.startswith(SPECIES + " "))
    coefficients = list(map(float, lines[i + 1].split()))
    inputs = [FIT, SOURCE / "inputs.json", SOURCE / "sources.json", DATA / "sources.json", *paths.values()]
    return {"id": "cn_retained_fastchem_thermal_atct1228_candidate_v1", "species": SPECIES,
            "reaction": "C(g) + N(g) -> CN(g)", "composition": COMPOSITION,
            "temperature_range_K": [1500, 4000], "standard_pressure_Pa": 100000,
            "gas_constant_J_mol_K": R, "base_dataset": "curated-cos-v1",
            "baseline_energy_source": "JANAF 1998", "candidate_ATcT_version": "1.228",
            "janaf_fixed_reference_offsets_J_mol": offsets, "candidate_hf0_kJ_mol": modern,
            "base_D0_J_mol": old, "D0_J_mol": new, "delta_D0_J_mol": new - old,
            "delta_D0_over_R_K": (new - old) / R,
            "original_record": lines[i], "original_coefficients": coefficients,
            "formal_fit_intercept_Ra_J_mol": coefficients[0] * R,
            "representation": "ln K = a/T + b ln(T) + c + d T + e T^2 + delta(D0)/(R T). Retain every literal coefficient and fit/source residual; no thermal refit, isotope change or second energy alignment.",
            "anchor_scope": "The physical source D0 values set only the energy difference. R*a is a formal extrapolated fit intercept, not an independently measured D0; the fit is never evaluated at zero temperature. Ground-level atomization already includes ZPE.",
            "uncertainty_scope": "Same-version correlated ATcT atomic and molecular central values, not independent uncertainties. Reaction covariance, historical fit weights and complete thermal uncertainty are unavailable.",
            "source_sha256": {str(p.relative_to(ROOT)): file_hash(p) for p in inputs}}


def _formation(t, updated, xp):
    a, b, c, d, e = load_candidate()["original_coefficients"]
    g = -a - b * t * xp.log(t) - c * t - d * t**2 - e * t**3
    slope = -b * (xp.log(t) + 1) - c - 2 * d * t - 3 * e * t**2
    curvature = -b / t - 2 * d - 6 * e * t
    return g - (load_candidate()["delta_D0_over_R_K"] if updated else 0), slope, curvature


def formation_h(temperature_K: float, updated: bool = True) -> float:
    """Return one-bar Gformation/(RT) in the bounded review domain."""
    if not math.isfinite(temperature_K) or not 1500 <= temperature_K <= 4000:
        raise ValueError("Temperature outside CN candidate domain")
    return float(_formation(float(temperature_K), updated, np)[0] / temperature_K)


def formation_h_jax(temperature_K, updated: bool = True):
    """Support scalar/array/JIT/vmap calls and return NaN outside the domain."""
    import jax.numpy as jnp
    t = jnp.asarray(temperature_K, dtype=jnp.asarray(load_candidate()["original_coefficients"]).dtype)
    valid = jnp.isfinite(t) & (t >= 1500) & (t <= 4000)
    safe = jnp.where(valid, t, 1500.)
    return jnp.where(valid, _formation(safe, updated, jnp)[0] / safe, jnp.nan)


def reaction_properties(temperature_K: float, updated: bool = True) -> dict:
    """Differentiate the same continuous potential for consistent G/H/S/Cp."""
    formation_h(temperature_K, updated)
    t = float(temperature_K)
    g, slope, curvature = _formation(t, updated, np)
    return dict(zip(("G_J_mol", "H_J_mol", "S_J_mol_K", "Cp_J_mol_K"),
                    map(float, (R * g, R * (g - t * slope), -R * slope, -R * t * curvature))))


def validate() -> dict:
    """Check the current factory, independent fit formula and energy-only change."""
    import jax
    from exogibbs.presets import curated_cos
    model = load_candidate()
    assert model == build()
    previous = jax.config.x64_enabled
    jax.config.update("jax_enable_x64", True)
    try:
        setup = curated_cos.chemsetup(ions=True)
        assert setup.metadata["dataset"] == model["base_dataset"]
        ts = np.arange(1500., 4001.)
        current = np.asarray(setup.hvector_func(ts))[:, setup.species.index(SPECIES)]
        before = np.array([formation_h(t, False) for t in ts])
        after = np.array([formation_h(t) for t in ts])
        independent = -np.array([curated_properties(SPECIES, t)["ln_K"] for t in ts]) - model["delta_D0_over_R_K"] / ts
        oldprops = np.array([list(reaction_properties(t, False).values()) for t in ts])
        props = np.array([list(reaction_properties(t).values()) for t in ts])
        expected = np.zeros_like(props)
        expected[:, :2] = -model["delta_D0_J_mol"]
        f = jax.jit(formation_h_jax)
        g = lambda t: R * t * f(t)
        slope = np.asarray(jax.jit(jax.vmap(jax.grad(g)))(ts))
        curvature = np.asarray(jax.jit(jax.vmap(jax.grad(jax.grad(g))))(ts))
        jg = R * ts * np.asarray(f(ts))
        jprops = np.stack([jg, jg - ts * slope, -slope, -ts * curvature], axis=-1)
        keys = list(reaction_properties(1500))
        old_report = json.loads((SOURCE / "comparison.json").read_text())
        residuals = []
        for row in old_report["samples"]:
            t = row["temperature_K"]
            if 1500 <= t <= 4000:
                source = row["source"]["ln_K"]
                before_residual = -formation_h(t, False) - source
                after_residual = -formation_h(t) - (source + model["delta_D0_over_R_K"] / t)
                residuals.append({"temperature_K": t, "current_minus_JANAF_ln_K": before_residual,
                                  "candidate_minus_shifted_JANAF_ln_K": after_residual,
                                  "residual_preservation_error_ln_K": after_residual - before_residual})
        result = {"temperature_count": len(ts), "range_K": [1500, 4000], "step_K": 1,
                  "maximum_abs_current_h_replay_error": float(abs(before - current).max()),
                  "maximum_abs_independent_fit_h_error": float(abs(after - independent).max()),
                  "maximum_abs_energy_shift_h_error": float(abs(after - current + model["delta_D0_over_R_K"] / ts).max()),
                  "maximum_abs_energy_only_property_errors": dict(zip(keys, map(float, abs(props - oldprops - expected).max(axis=0)))),
                  "maximum_abs_jax_property_errors": dict(zip(keys, map(float, abs(jprops - props).max(axis=0)))),
                  "source_fit_residuals": residuals,
                  "samples": [{"temperature_K": t, "candidate_over_current_K": math.exp(model["delta_D0_over_R_K"] / t),
                               "reaction_properties": reaction_properties(t)} for t in (1500, 2000, 2500, 3000, 4000)],
                  "production_change": "none", "new_fit": "none"}
        assert max(result[k] for k in result if k.endswith("_h_error") or k.endswith("_replay_error")) < 1e-12
        return result
    finally:
        jax.config.update("jax_enable_x64", previous)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--build", action="store_true")
    args = parser.parse_args()
    if args.build:
        DATA.mkdir(parents=True, exist_ok=True)
        (DATA / "candidate.json").write_text(json.dumps(build(), indent=2) + "\n")
        load_candidate.cache_clear()
    result = validate()
    if args.build:
        import jax
        (DATA / "validation.json").write_text(json.dumps(result, indent=2) + "\n")
        paths = {Path(module.__file__).resolve() for module in list(sys.modules.values())
                 if getattr(module, "__file__", None) and Path(module.__file__).resolve().is_relative_to(ROOT)
                 and Path(module.__file__).suffix == ".py" and Path(module.__file__).name != "exogibbs_version.py"}
        paths.update(ROOT / name for name in load_candidate()["source_sha256"])
        paths.update([Path(__file__).resolve(), SOURCE / "comparison.json"])
        paths.update((ROOT / "src/exogibbs/data/curated").glob("*.json"))
        paths.update(ROOT / "src/exogibbs/data" / n for n in ("FastChem4/logK/logK.dat", "FastChem4/element_abundances/asplund_2021.dat"))
        receipt = {"created_utc": datetime.now(timezone.utc).isoformat(), "command": [sys.executable, *sys.argv],
                   "python": platform.python_version(), "jax": jax.__version__, "numpy": np.__version__,
                   "dtype": "float64", "backend": "cpu", "result": "passed",
                   "selected_file_sha256": {str(p.relative_to(ROOT)): file_hash(p) for p in sorted(paths)},
                   "output_sha256": {name: file_hash(DATA / name) for name in ("candidate.json", "validation.json")}}
        (DATA / "receipt.json").write_text(json.dumps(receipt, indent=2) + "\n")
    print(json.dumps(result, indent=2))
