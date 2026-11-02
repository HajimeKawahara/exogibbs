"""Isolate VO2 energy and electronic-state sensitivity on a fixed gas grid."""

from __future__ import annotations

import argparse
from dataclasses import replace
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import platform
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT), str(ROOT / "src"), str(Path(__file__).resolve().parent)]
os.environ.setdefault("JAX_PLATFORMS", "cpu")

import jax
import jax.numpy as jnp
import numpy as np
from scipy.special import logsumexp

from c2h_equilibrium_impact import carbon_budget, independent_check, solve_cases
from gas_source_comparison import file_hash
from exogibbs.presets import curated


DATA = ROOT / "tests/data/validation/vo2_equilibrium_impact/20261102"
INPUT = DATA / "inputs.json"
PRIOR = ROOT / "tests/data/validation/curated_default_adoption/20261102/ions.npz"
TEMPERATURES = np.array([1500., 2000., 2500., 3000., 4000.])
PRESSURES = np.array([1e-6, 1e-4, .01, 1., 100.])
KINDS = ("baseline", "energy", "electronic", "combined")
PRIMARY_TOLERANCE, STRICT_TOLERANCE = 1e-13, 5e-14
TRACKED = ("O2V1", "O1V1", "V1", "C1H1N1_hcn", "C1O1", "C1H4", "H2O1",
           "O2", "C2H2", "H3N1", "N2", "e1-")


def correction(temperature, kind: str, inputs: dict) -> dict:
    """Derive G/H/S/Cp changes from one constant energy and electronic sum."""
    if kind not in KINDS:
        raise ValueError("Unknown diagnostic: " + kind)
    t = jnp.asarray(temperature, dtype=jnp.result_type(temperature, float))
    valid = jnp.isfinite(t) & (t >= 1500) & (t <= 4000)
    t = jnp.where(valid, t, 1500.)
    r = inputs["gas_constant_J_mol_K"]
    energy = inputs["energy_shift_J_mol"] if kind in ("energy", "combined") else 0.
    levels = jnp.asarray(inputs["electronic_levels_eV"]) * inputs["electronvolt_J_mol"]
    weights = jnp.asarray(inputs["electronic_weights"])
    terms = weights * jnp.exp(-levels / (r * t[..., None]))
    total = jnp.sum(terms, axis=-1)
    mean = jnp.sum(terms * levels, axis=-1) / total
    variance = jnp.sum(terms * (levels - mean[..., None])**2, axis=-1) / total
    active = kind in ("electronic", "combined")
    logq = jnp.log(total / inputs["reference_ground_weight"]) if active else jnp.zeros_like(t)
    mean = mean if active else jnp.zeros_like(t)
    variance = variance if active else jnp.zeros_like(t)
    values = {"G_J_mol": energy-r*t*logq, "H_J_mol": energy+mean,
              "S_J_mol_K": r*logq+mean/t, "Cp_J_mol_K": variance/(r*t*t),
              "formation_K_ratio": jnp.exp(-energy/(r*t)+logq)}
    return {key: jnp.where(valid, value, jnp.nan) for key, value in values.items()}


def models() -> tuple:
    """Change only VO2; all diagnostics retain the curated network and budget."""
    inputs = json.loads(INPUT.read_text())
    base = curated.chemsetup()
    index = base.species.index(inputs["species"])
    result = [base]
    for kind in KINDS[1:]:
        def hvector(temperature, kind=kind):
            t = jnp.asarray(temperature, dtype=base.formula_matrix.dtype)
            shift = correction(t, kind, inputs)["G_J_mol"] / (inputs["gas_constant_J_mol_K"]*t)
            return base.hvector_func(t).at[..., index].add(shift)
        result.append(replace(base, hvector_func=jax.jit(hvector), metadata={
            **base.metadata, "dataset": "curated-c2h-v1+vo2-"+kind+"-diagnostic",
            "interpretation": "VO2-only finite diagnostic; not adoption or physical accuracy validation."}))
    return tuple(result), inputs


def run(output: Path, executable: Path | None = None) -> None:
    """Save four endpoints and independent tighter cold starts at all 75 cases."""
    jax.config.update("jax_enable_x64", True)
    output.mkdir(parents=True, exist_ok=False)
    setups, inputs = models()
    base = setups[0]
    c, o, v = [base.elements.index(e) for e in ("C", "O", "V")]
    ratios = [float(base.element_vector_reference[c]/base.element_vector_reference[o]), 1., 1.5]
    t, p = np.repeat(TEMPERATURES, 5), np.tile(PRESSURES, 5)
    cases, budgets, diagnostics = [], [], []
    amounts, strict = [[[] for _ in setups] for _ in range(2)]
    for ratio in ratios:
        budget = carbon_budget(base, ratio)
        cases.extend({"C_over_O": ratio, "temperature_K": float(ti), "pressure_bar": float(pi)}
                     for ti, pi in zip(t, p))
        budgets.extend([budget]*len(t))
        for mi, model in enumerate(setups):
            for tolerance, destination in ((PRIMARY_TOLERANCE, amounts), (STRICT_TOLERANCE, strict)):
                values, rows = solve_cases(model, budget, t, p, tolerance)
                destination[mi].append(values)
                diagnostics.extend({"dataset": model.metadata["dataset"], "C_over_O": ratio,
                                    "epsilon_crit": tolerance, **row} for row in rows)
        print(f"C/O={ratio:.12g}: 200 endpoint/strict cold solves passed", flush=True)
    ln_n, strict_ln_n = [np.array([np.concatenate(row) for row in rows]) for rows in (amounts, strict)]
    ln_x = ln_n-logsumexp(ln_n, axis=-1, keepdims=True)
    strict_ln_x = strict_ln_n-logsumexp(strict_ln_n, axis=-1, keepdims=True)
    n, x, budgets = np.exp(ln_n), np.exp(ln_x), np.array(budgets)
    stability = {"max_abs_delta_ln_x": float(np.max(abs(strict_ln_x-ln_x))),
                 "max_abs_delta_x": float(np.max(abs(np.exp(strict_ln_x)-x)))}
    if stability["max_abs_delta_ln_x"] > 1e-4 or stability["max_abs_delta_x"] > 1e-9:
        raise ValueError(f"Tolerance stability failed: {stability}")
    a = np.asarray(base.formula_matrix)
    shares = {e: n*a[base.elements.index(e)]/budgets[None, :, base.elements.index(e), None]
              for e in ("V", "C", "O", "N")}
    delta_log10_x = (ln_x[1:]-ln_x[0])/np.log(10.)

    def point(i, j, mi):
        return {**cases[i], "species": base.species[j], "x_before": float(x[0, i, j]),
                "x_after": float(x[mi, i, j]), "delta_x": float(x[mi, i, j]-x[0, i, j]),
                "ratio": float(np.exp(ln_x[mi, i, j]-ln_x[0, i, j])),
                "delta_log10_x": float(delta_log10_x[mi-1, i, j]),
                "element_fractions": {e: values[[0, mi], i, j].tolist() for e, values in shares.items()}}

    vo2 = base.species.index(inputs["species"])
    index = int(np.argmax(shares["V"][0, :, vo2]))
    allocations, extrema = [], []
    for mi, kind in enumerate(KINDS[1:], 1):
        allocations.append({"diagnostic": kind, "case": cases[index],
                            "vanadium": [point(index, int(j), mi) for j in np.flatnonzero(a[v])],
                            "tracked": [point(index, base.species.index(name), mi) for name in TRACKED]})
        for name in TRACKED:
            j = base.species.index(name)
            significant = np.flatnonzero(np.max(x[[0, mi], :, j], axis=0) >= 1e-12)
            extrema.append({"diagnostic": kind, "species": name,
                "largest_absolute_change": point(int(np.argmax(abs(x[mi, :, j]-x[0, :, j]))), j, mi),
                "largest_log_change": point(int(np.argmax(abs(delta_log10_x[mi-1, :, j]))), j, mi),
                "largest_log_change_above_display_threshold": None if not significant.size else
                point(int(significant[np.argmax(abs(delta_log10_x[mi-1, significant, j]))]), j, mi)})
    case = cases[index]
    independent = (independent_check(setups, budgets[index], case["temperature_K"], case["pressure_bar"],
                                    strict_ln_n[:, index], executable.resolve()) if executable else "not_run")
    arrays = dict(ln_n=ln_n, n=n, ln_x=ln_x, x=x, strict_ln_n=strict_ln_n,
                  delta_x=x[1:]-x[0], delta_log10_x=delta_log10_x, budget=budgets,
                  species=base.species, elements=base.elements, formula_matrix=a,
                  temperature_K=[r["temperature_K"] for r in cases], pressure_bar=[r["pressure_bar"] for r in cases],
                  C_over_O=[r["C_over_O"] for r in cases], diagnostics=KINDS,
                  h=np.array([m.hvector_func(np.array([r["temperature_K"] for r in cases])) for m in setups]),
                  **{e+"_fraction": values for e, values in shares.items()})
    np.savez_compressed(output / "equilibria.npz", **arrays)
    caloric = [{"diagnostic": kind, "temperature_K": float(ti),
                **{key: float(value) for key, value in correction(ti, kind, inputs).items()}}
               for kind in KINDS[1:] for ti in TEMPERATURES]
    summary = {"datasets": [m.metadata["dataset"] for m in setups], "diagnostics": KINDS,
               "network_species": len(base.species), "paired_cases": len(cases), "endpoint_solves": 300,
               "tighter_cold_solves": 300, "temperature_K": TEMPERATURES.tolist(),
               "pressure_bar": PRESSURES.tolist(), "carbon_to_oxygen": ratios,
               "input_sha256": file_hash(INPUT), "energy_shift_J_mol": inputs["energy_shift_J_mol"],
               "tolerance_stability": stability, "caloric_corrections": caloric,
               "highlighted_case": case, "highlighted_allocations": allocations, "extrema": extrema,
               "independent_solver_case": case, "independent_solver": independent,
               "scope": "Fixed-network gas-only finite sensitivity, not adoption, physical validation or an uncertainty interval. "
                        "The energy shift is a historical source-gap diagnostic, not a precise assigned D0. "
                        "Electronic-only assumes identical rovibrational factors; no complete thermal replacement. "
                        "No condensation, nonideality, quenching or spectra; no concentration floors."}
    for filename, value in (("summary.json", summary), ("diagnostics.json", diagnostics)):
        (output / filename).write_text(json.dumps(value, indent=2, allow_nan=False)+"\n")
    paths = [Path(__file__), INPUT, PRIOR]
    paths += [ROOT / name for name in ("mceg/reviews/c2h_equilibrium_impact.py", "mceg/reviews/gas_source_comparison.py",
              "examples/comparisons/compare_gas_thermochemistry.py", "src/exogibbs/presets/curated.py",
              "src/exogibbs/presets/curated_c2h.py", "src/exogibbs/presets/curated_v1.py",
              "src/exogibbs/presets/fastchem4.py", "src/exogibbs/presets/fastchem.py", "src/exogibbs/thermo/standard.py",
              "src/exogibbs/data/curated/gas_v1.json", "src/exogibbs/data/curated/c2h_v1.json",
              "src/exogibbs/data/FastChem4/logK/logK.dat", "src/exogibbs/data/FastChem4/element_abundances/asplund_2021.dat")]
    paths += sorted((ROOT / "src/exogibbs/equilibrium/gas").rglob("*.py"))
    receipt = {"created_utc": datetime.now(timezone.utc).isoformat(), "result": "passed",
               "datasets": summary["datasets"], "scope": summary["scope"],
               "python": platform.python_version(), "numpy": np.__version__, "jax": jax.__version__,
               "backend": jax.default_backend(), "dtype": "float64", "command": " ".join(sys.argv),
               "primary_tolerance": PRIMARY_TOLERANCE, "strict_tolerance": STRICT_TOLERANCE,
               "selected_file_sha256": {str(path.relative_to(ROOT)): file_hash(path) for path in paths},
               "output_sha256": {path.name: file_hash(path) for path in sorted(output.iterdir())},
               "fastchem_binary_sha256": file_hash(executable) if executable else None}
    (output / "receipt.json").write_text(json.dumps(receipt, indent=2)+"\n")
    print(json.dumps({"highlighted_case": case, "tolerance_stability": stability,
                      "independent_solver": independent}), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--fastchem-executable", type=Path)
    args = parser.parse_args()
    run(args.output_dir, args.fastchem_executable)
