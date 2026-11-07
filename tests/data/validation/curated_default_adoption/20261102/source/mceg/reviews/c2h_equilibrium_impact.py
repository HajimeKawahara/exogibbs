"""Measure a single C2H correction on fixed gas-only networks and budgets."""

from __future__ import annotations

import argparse
from dataclasses import replace
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import platform
import sys
import tempfile

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT), str(ROOT / "src"), str(ROOT / "examples/comparisons"),
                str(Path(__file__).resolve().parent)]
os.environ.setdefault("JAX_PLATFORMS", "cpu")

import jax
import jax.numpy as jnp
import numpy as np
from scipy.special import logsumexp

from compare_gas_thermochemistry import equilibrium_metrics, fastchem_node_text, validate_diagnostics
from gas_source_comparison import file_hash
from exogibbs.api.gas import EquilibriumOptions, solve_profile
from exogibbs.presets import curated, curated_c2h


TEMPERATURES = np.array([1500., 2000., 2500., 3000., 4000.])
PRESSURES = np.array([1e-6, 1e-4, .01, 1., 100.])
SNAPSHOT = ROOT / "tests/data/validation/c2h_equilibrium_impact/20261102"
PRIMARY_TOLERANCE = 1e-13
STRICT_TOLERANCE = 5e-14


def carbon_budget(setup, ratio: float) -> np.ndarray:
    """Change C/H while preserving every other element/H ratio, then normalize."""
    budget = np.array(setup.element_vector_reference)
    budget[setup.elements.index("C")] = ratio * budget[setup.elements.index("O")]
    return budget / budget.sum()


def solve_cases(setup, budget: np.ndarray, temperatures: np.ndarray,
                pressures: np.ndarray, tolerance: float) -> tuple:
    """Use cold starts and verify all element, charge and formation constraints."""
    result, info = solve_profile(
        setup, T=jnp.asarray(temperatures), P=jnp.asarray(pressures), b=jnp.asarray(budget),
        options=EquilibriumOptions(epsilon_crit=tolerance, method="vmap_cold"),
        return_diagnostics=True)
    ln_n = np.asarray(result.ln_n)
    if not np.all(np.isfinite(ln_n)):
        raise ValueError("Non-finite log amount")
    metrics = equilibrium_metrics(setup, budget, ln_n, np.asarray(setup.hvector_func(temperatures)), pressures)
    rows = [{"temperature_K": float(t), "pressure_bar": float(p),
             "converged": bool(info["converged"][i]), "iterations": int(info["n_iter"][i]),
             "solver_residual": float(info["final_residual"][i]),
             **{k: float(v[i]) for k, v in metrics.items()}}
            for i, (t, p) in enumerate(zip(temperatures, pressures))]
    validate_diagnostics(rows)
    return ln_n, rows


def summarize(setup, cases: list, ln_n: np.ndarray, budgets: np.ndarray) -> tuple:
    """Keep logarithms before exponentiating, without concentration floors."""
    ln_x = ln_n - logsumexp(ln_n, axis=-1, keepdims=True)
    n, x = np.exp(ln_n), np.exp(ln_x)
    delta_x, delta_log10_x = x[1]-x[0], (ln_x[1]-ln_x[0])/np.log(10.)
    carbon = np.asarray(setup.formula_matrix)[setup.elements.index("C")]
    carbon_fraction = n*carbon / budgets[None, :, setup.elements.index("C"), None]
    major = json.loads((ROOT / "examples/comparisons/data/major_molecule_reference.json").read_text())
    observed = sorted({r["fastchem_species"] for r in major["species"].values()}
                      | {"C2H1", "C2H2", "e1-", "H1-"})

    def point(i, j):
        return {**cases[i], "species": setup.species[j], "x_before": float(x[0, i, j]),
                "x_after": float(x[1, i, j]), "delta_x": float(delta_x[i, j]),
                "delta_log10_x": float(delta_log10_x[i, j]),
                "carbon_share_before": float(carbon_fraction[0, i, j]),
                "carbon_share_after": float(carbon_fraction[1, i, j])}

    extrema, changes = [], []
    for name in observed:
        j = setup.species.index(name)
        significant = np.flatnonzero(np.max(x[:, :, j], axis=0) >= 1e-12)
        extrema.append({"species": name,
                        "largest_log_change": point(int(np.argmax(abs(delta_log10_x[:, j]))), j),
                        "largest_absolute_change": point(int(np.argmax(abs(delta_x[:, j]))), j),
                        "largest_log_change_above_display_threshold": None if not significant.size else
                        point(int(significant[np.argmax(abs(delta_log10_x[significant, j]))]), j)})
    # The threshold is for display only; the NPZ retains every species and case.
    mask = (np.max(x, axis=0) >= 1e-12) & (np.abs(delta_log10_x) >= np.log10(1.1))
    for i, j in zip(*np.where(mask)):
        changes.append(point(int(i), int(j)))
    arrays = dict(ln_n=ln_n, n=n, ln_x=ln_x, x=x, delta_x=delta_x,
                  delta_log10_x=delta_log10_x, carbon_fraction=carbon_fraction)
    return {"observed_species": observed, "extrema": extrema,
            "changes_above_factor_1p1_and_x_1e_minus12": changes}, arrays


def independent_check(models, budget, temperature, pressure, reference, executable) -> list:
    """Verify the highlighted point with the same h, A, b in standalone FastChem."""
    from benchmarks.fastchem4.comparison import align_species_values, occurrence_keys
    from benchmarks.fastchem4.fastchem_executable import run_fastchem_executable

    rows = []
    for mi, model in enumerate(models):
        with tempfile.TemporaryDirectory(prefix="c2h_parity_") as directory:
            directory = Path(directory)
            abundance = directory / "abundances.dat"
            hydrogen = budget[model.elements.index("H")]
            abundance.write_text("# Fixed C2H comparison budget\ne- 0.0\n" + "\n".join(
                f"{name} {12+np.log10(value/hydrogen):.17g}" for name, value in zip(model.elements, budget)
                if name != "e-") + "\n")
            constants = directory / "logK.dat"
            h = np.asarray(model.hvector_func(temperature))
            constants.write_text(fastchem_node_text(model, h, temperature))
            result = run_fastchem_executable(
                executable=executable, temperatures=[temperature], pressures=[pressure],
                element_abundance_file=abundance, gas_logk_file=constants, chemistry_mode="gas",
                chemistry_accuracy=1e-10, element_conservation_accuracy=1e-10)
        if (sorted(occurrence_keys(result.gas_names)) != sorted(occurrence_keys(model.species))
                or not np.all(result.converged & result.elements_conserved)):
            raise RuntimeError(f"Independent solver catalog or convergence mismatch: "
                               f"{len(result.gas_names)} species, converged={result.converged}, "
                               f"conserved={result.elements_conserved}")
        density = align_species_values(model.species, result.gas_names, result.gas_number_densities)
        if np.any(density <= 0) or not np.all(np.isfinite(density)):
            raise RuntimeError("Independent solver returned invalid amounts")
        ln_n = np.log(density/result.total_element_density[:, None])
        ln_x = ln_n-logsumexp(ln_n, axis=-1, keepdims=True)
        ref = reference[mi]-logsumexp(reference[mi])
        error = float(np.max(abs(ln_x[0]-ref)/np.log(10)))
        metrics = equilibrium_metrics(model, budget, ln_n, h[None, :], np.array([pressure]))
        row = {"dataset": model.metadata["dataset"], "max_abs_delta_log10_x": error,
               **{k: float(v[0]) for k, v in metrics.items()}}
        if (not all(np.isfinite(v) for k, v in row.items() if k != "dataset")
                or error > 1e-4 or row["element_relative_residual"] > 1e-5
                or row["charge_relative_residual"] > 1e-7 or row["mass_action_max_abs_ln"] > 1e-7):
            raise RuntimeError(f"Independent comparison failed: {row}")
        rows.append(row)
    return rows


def run(output: Path, executable: Path | None = None) -> None:
    """Archive endpoints, tighter cold runs and a five-point finite substitution."""
    jax.config.update("jax_enable_x64", True)
    output.mkdir(parents=True, exist_ok=False)
    models = (curated.chemsetup(), curated_c2h.chemsetup())
    base, corrected = models
    c, o = (base.elements.index(name) for name in ("C", "O"))
    ratios = [float(base.element_vector_reference[c]/base.element_vector_reference[o]), 1., 1.5]
    temperatures, pressures = np.repeat(TEMPERATURES, 5), np.tile(PRESSURES, 5)
    cases, budgets, amounts, strict_amounts, diagnostics = [], [], [[], []], [[], []], []
    for ratio in ratios:
        budget = carbon_budget(base, ratio)
        cases.extend({"C_over_O": ratio, "temperature_K": float(t), "pressure_bar": float(p)}
                     for t, p in zip(temperatures, pressures))
        budgets.extend([budget]*len(temperatures))
        for mi, model in enumerate(models):
            for tolerance, destination in ((PRIMARY_TOLERANCE, amounts), (STRICT_TOLERANCE, strict_amounts)):
                ln_n, rows = solve_cases(model, budget, temperatures, pressures, tolerance)
                destination[mi].append(ln_n)
                diagnostics.extend({"dataset": model.metadata["dataset"], "C_over_O": ratio,
                                    "epsilon_crit": tolerance, **r} for r in rows)
        print(f"C/O={ratio:.12g}: endpoints and tighter cold starts passed", flush=True)
    ln_n, strict_ln_n = [np.array([np.concatenate(r) for r in rows]) for rows in (amounts, strict_amounts)]
    summary, arrays = summarize(base, cases, ln_n, np.array(budgets))
    strict_ln_x = strict_ln_n-logsumexp(strict_ln_n, axis=-1, keepdims=True)
    tolerance_error = {"max_abs_delta_ln_x": float(np.max(abs(strict_ln_x-arrays["ln_x"]))),
                       "max_abs_delta_x": float(np.max(abs(np.exp(strict_ln_x)-arrays["x"])))}
    if tolerance_error["max_abs_delta_ln_x"] > 1e-4 or tolerance_error["max_abs_delta_x"] > 1e-9:
        raise ValueError(f"Tolerance stability failed: {tolerance_error}")
    # Highlight the largest HCN log response; trace the finite change, not a linear extrapolation.
    hcn = base.species.index("C1H1N1_hcn")
    index = int(np.argmax(abs(arrays["delta_log10_x"][:, hcn])))
    case, budget = cases[index], budgets[index]
    t, p = case["temperature_K"], case["pressure_bar"]
    lam_ln_n, lam_diagnostics = [], []
    for fraction in np.linspace(0, 1, 5):
        def hvector(temperature):
            start = base.hvector_func(temperature)
            return start+fraction*(corrected.hvector_func(temperature)-start)
        model = replace(base, hvector_func=jax.jit(hvector))
        values, rows = solve_cases(model, budget, np.array([t]), np.array([p]), STRICT_TOLERANCE)
        lam_ln_n.append(values[0])
        lam_diagnostics.append({"lambda": float(fraction), **rows[0]})
    independent = (independent_check(models, budget, t, p, strict_ln_n[:, index], executable.resolve())
                   if executable else "not_run")
    arrays.update(strict_ln_n=strict_ln_n, lambda_values=np.linspace(0, 1, 5), lambda_ln_n=np.array(lam_ln_n),
                  budget=np.array(budgets), species=base.species, elements=base.elements,
                  formula_matrix=np.asarray(base.formula_matrix),
                  temperature_K=[r["temperature_K"] for r in cases],
                  pressure_bar=[r["pressure_bar"] for r in cases], C_over_O=[r["C_over_O"] for r in cases],
                  h=np.array([m.hvector_func(np.array([r["temperature_K"] for r in cases])) for m in models]))
    np.savez_compressed(output / "equilibria.npz", **arrays)
    summary.update({"datasets": [m.metadata["dataset"] for m in models], "network_species": 537,
                    "paired_cases": len(cases), "endpoint_solves": 2*len(cases),
                    "tighter_cold_solves": 2*len(cases), "carbon_to_oxygen": ratios,
                    "temperature_K": TEMPERATURES.tolist(), "pressure_bar": PRESSURES.tolist(),
                    "tolerance_stability": tolerance_error, "highlighted_case": case,
                    "highlighted_species": [next(e for e in summary["extrema"] if e["species"] == name)
                                            for name in ("C2H1", "C2H2", "C1H4", "C1H1N1_hcn")],
                    "independent_solver": independent,
                    "scope": "Ideal, gas-only equilibrium at fixed budgets. No condensation, quenching, opacity or physical uncertainty validation."})
    for filename, data in (("summary.json", summary), ("diagnostics.json", diagnostics),
                           ("lambda_diagnostics.json", lam_diagnostics)):
        (output / filename).write_text(json.dumps(data, indent=2, allow_nan=False)+"\n")
    paths = [Path(__file__), ROOT / "mceg/reviews/build_c2h_correction.py",
             ROOT / "examples/comparisons/compare_gas_thermochemistry.py",
             ROOT / "examples/comparisons/data/major_molecule_reference.json"]
    paths += [ROOT / p for p in ("src/exogibbs/presets/curated.py", "src/exogibbs/presets/curated_c2h.py",
              "src/exogibbs/thermo/standard.py", "src/exogibbs/data/curated/gas_v1.json",
              "src/exogibbs/data/curated/c2h_v1.json", "src/exogibbs/data/FastChem4/logK/logK.dat",
              "src/exogibbs/data/FastChem4/element_abundances/asplund_2021.dat")]
    paths += sorted((ROOT / "src/exogibbs/equilibrium/gas").rglob("*.py"))
    receipt = {"created_utc": datetime.now(timezone.utc).isoformat(), "python": platform.python_version(),
               "numpy": np.__version__, "jax": jax.__version__, "backend": jax.default_backend(), "dtype": "float64",
               "command": " ".join(sys.argv), "result": "passed", "primary_tolerance": PRIMARY_TOLERANCE,
               "strict_tolerance": STRICT_TOLERANCE,
               "selected_file_sha256": {str(p.relative_to(ROOT)): file_hash(p) for p in paths},
               "output_sha256": {p.name: file_hash(p) for p in sorted(output.iterdir())},
               "fastchem_binary_sha256": file_hash(executable) if executable else None}
    (output / "receipt.json").write_text(json.dumps(receipt, indent=2)+"\n")
    print(json.dumps({"paired_cases": len(cases), "tolerance_stability": tolerance_error,
                      "highlighted_case": case, "independent_solver": independent}), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--fastchem-executable", type=Path)
    args = parser.parse_args()
    run(args.output_dir, args.fastchem_executable)
