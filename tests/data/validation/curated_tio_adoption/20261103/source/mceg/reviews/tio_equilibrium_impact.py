"""Compare two TiO candidates on the pinned, ion-inclusive gas network."""

from __future__ import annotations

import argparse
from dataclasses import replace
from datetime import datetime, timezone
import json
from pathlib import Path
import platform
import sys

import jax
import jax.numpy as jnp
import numpy as np
from scipy.special import logsumexp

from exogibbs.presets import curated

from c2h_equilibrium_impact import independent_check, solve_cases
from gas_source_comparison import ROOT, file_hash
import tio_candidate as candidate


BASELINE = ROOT / "tests/data/validation/curated_default_adoption/20261102/ions.npz"
DATA = ROOT / "tests/data/validation/tio_equilibrium_impact/20261103"
PRIMARY_TOLERANCE = 1e-13
STRICT_TOLERANCE = 5e-14
VARIANTS = ("current", "toto", "nasa_proxy")


def candidate_setup(base, thermal: str = "states", d0_eV: float = 6.824,
                    fraction: float = 1.0):
    """Override TiO alone; preserve budgets, catalog and charge convergence."""
    index = base.species.index("O1Ti1")

    def hvector(temperature):
        initial = base.hvector_func(temperature)
        target = candidate.formation_h_jax(temperature, thermal=thermal, d0_eV=d0_eV)
        return initial.at[..., index].set(initial[..., index] + fraction*(target-initial[..., index]))

    metadata = {**base.metadata, "dataset": f"review-tio-{thermal}-jain2023",
                "candidate_D0_eV": d0_eV, "substitution_fraction": fraction,
                "production_adoption": False}
    return replace(base, hvector_func=jax.jit(hvector), metadata=metadata)


def diagnostics(matrix: np.ndarray, elements: list, species: list, budgets: np.ndarray,
                ln_n: np.ndarray, h: np.ndarray, pressure: np.ndarray) -> dict:
    """Recompute conservation and every formation equation from stored amounts."""
    n = np.exp(ln_n)
    positive = budgets[0] > 0
    charge = matrix[elements.index("e-")]
    references = [species.index("e1-" if e == "e-" else e+"1") for e in elements]
    logp = ln_n-logsumexp(ln_n, axis=-1, keepdims=True)+np.log(pressure)[..., None]
    values = {
        "element_relative_residual": np.max(abs(n@matrix.T-budgets)[..., positive]/budgets[..., positive], axis=-1),
        "charge_relative_residual": abs(n@charge)/(n@abs(charge)),
        "mass_action_max_abs_ln": np.max(abs(logp-logp[..., references]@matrix+h), axis=-1),
    }
    for name, limit in (("element_relative_residual", 1e-5),
                        ("charge_relative_residual", 1e-12), ("mass_action_max_abs_ln", 1e-8)):
        if not np.all(np.isfinite(values[name])) or np.max(values[name]) > limit:
            raise ValueError(f"Failed {name}: {np.max(values[name])}")
    return values


def gibbs(ln_n: np.ndarray, h: np.ndarray, pressure: np.ndarray) -> np.ndarray:
    """Evaluate G/(RT) per normalized budget in extended summation precision."""
    values = np.asarray(ln_n, dtype=np.longdouble)
    n = np.exp(values)
    logx = values-np.log(n.sum(axis=-1))[..., None]
    return np.sum(n*(np.asarray(h, dtype=np.longdouble)+logx
                     +np.log(np.asarray(pressure, dtype=np.longdouble))[..., None]), axis=-1)


def run(output: Path, executable: Path | None = None) -> None:
    """Solve 150 new endpoints, then check selected cases and a finite path."""
    jax.config.update("jax_enable_x64", True)
    output.mkdir(parents=True, exist_ok=False)
    with np.load(BASELINE) as source:
        saved = {k: source[k].copy() for k in source.files}
    base = curated.chemsetup(ions=True)
    matrix, budgets = np.asarray(base.formula_matrix), saved["budget"]
    species, elements = list(base.species), list(base.elements)
    np.testing.assert_array_equal(species, saved["species"])
    np.testing.assert_array_equal(elements, saved["elements"])
    np.testing.assert_array_equal(matrix, saved["formula_matrix"])
    t, p = saved["temperature_K"], saved["pressure_bar"]
    np.testing.assert_allclose(base.hvector_func(t), saved["h"], rtol=0, atol=1e-12)
    assert saved["converged"].all() and len(t) == 75 and len(species) == 537
    models = [base, candidate_setup(base), candidate_setup(base, "nasa")]
    h = np.asarray([model.hvector_func(t) for model in models])
    tio, titanium, oxygen = species.index("O1Ti1"), elements.index("Ti"), elements.index("O")
    np.testing.assert_allclose(np.delete(h[1:]-h[0], tio, axis=-1), 0, rtol=0, atol=1e-12)
    amounts, rows = [saved["ln_n"]], []
    for name, model in zip(VARIANTS[1:], models[1:]):
        result = np.empty_like(saved["ln_n"])
        for ratio in np.unique(saved["C_over_O"]):
            indices = np.flatnonzero(saved["C_over_O"] == ratio)
            np.testing.assert_array_equal(budgets[indices], np.broadcast_to(budgets[indices[0]], budgets[indices].shape))
            values, checks = solve_cases(model, budgets[indices[0]], t[indices], p[indices], PRIMARY_TOLERANCE)
            result[indices] = values
            rows.extend({"variant": name, "case_index": int(i), **row} for i, row in zip(indices, checks))
        amounts.append(result)
        print(f"{name}: 75 cold endpoint solves passed", flush=True)
    ln_n = np.asarray(amounts)
    ln_x = ln_n-logsumexp(ln_n, axis=-1, keepdims=True)
    n, x = np.exp(ln_n), np.exp(ln_x)
    checks = diagnostics(matrix, elements, species, budgets, ln_n, h, p)
    titanium_fraction = n*matrix[titanium]/budgets[:, titanium, None]
    dlogx, dx = ln_x[1:]-ln_x[0], x[1:]-x[0]
    dtio = titanium_fraction[1, :, tio]-titanium_fraction[0, :, tio]
    highlight = int(np.argmax(abs(dtio)))
    displayed = np.flatnonzero(np.max(x[[0, 1], :, tio], axis=0) >= 1e-12)
    relative_case = int(displayed[np.argmax(abs(dlogx[0, displayed, tio]))])
    residual_case = int(np.argmax(checks["element_relative_residual"][1]))
    selected = sorted({highlight, relative_case, residual_case})
    tight, tight_rows = [], []
    for name, model in zip(VARIANTS[1:], models[1:]):
        subset = []
        for i in selected:
            values, extra = solve_cases(model, budgets[i], t[i:i+1], p[i:i+1], STRICT_TOLERANCE)
            subset.append(values[0])
            tight_rows.extend({"variant": name, "case_index": i, **r} for r in extra)
        tight.append(subset)
    tight = np.asarray(tight)
    tight_logx = tight-logsumexp(tight, axis=-1, keepdims=True)
    stability = float(np.max(abs(tight_logx-ln_x[1:, selected])))
    assert stability < 1e-4

    # Compare solutions under the SAME h on either side of the substitution.
    delta_current, delta_variant = [], []
    for i in (1, 2):
        delta_current.append(np.asarray(gibbs(ln_n[i], h[0], p)-gibbs(ln_n[0], h[0], p), float))
        delta_variant.append(np.asarray(gibbs(ln_n[i], h[i], p)-gibbs(ln_n[0], h[i], p), float))
    delta_current, delta_variant = np.asarray(delta_current), np.asarray(delta_variant)
    gibbs_tolerance = 2e-12
    assert np.min(delta_current) > -gibbs_tolerance
    assert np.max(delta_variant) < gibbs_tolerance

    lam, path, path_rows = np.linspace(0, 1, 5), [], []
    for fraction in lam:
        model = candidate_setup(base, fraction=float(fraction))
        values, extra = solve_cases(model, budgets[highlight], t[highlight:highlight+1],
                                   p[highlight:highlight+1], STRICT_TOLERANCE)
        path.append(values[0])
        path_rows.extend({"lambda": float(fraction), **r} for r in extra)
    energies, uncertainty, uncertainty_rows = np.array([6.814, 6.834]), [], []
    for energy in energies:
        model = candidate_setup(base, d0_eV=float(energy))
        values, extra = solve_cases(model, budgets[highlight], t[highlight:highlight+1],
                                   p[highlight:highlight+1], STRICT_TOLERANCE)
        uncertainty.append(values[0])
        uncertainty_rows.extend({"D0_eV": float(energy), **r} for r in extra)
    # Reuse the existing single-node exporter within its documented temperatures.
    exportable = np.flatnonzero(np.isin(t, [1500., 2000., 3000., 4000.]))
    independent_case = int(exportable[np.argmax(abs(dtio[exportable]))])
    independent = independent_check(models, budgets[independent_case], float(t[independent_case]),
                                    float(p[independent_case]), ln_n[:, independent_case], executable) if executable else "not_run"

    def point(variant, case, column):
        return {"case_index": int(case), "temperature_K": float(t[case]), "pressure_bar": float(p[case]),
                "C_over_O": float(saved["C_over_O"][case]), "species": species[column],
                "x_before": float(x[0, case, column]), "x_after": float(x[variant, case, column]),
                "delta_x": float(x[variant, case, column]-x[0, case, column]),
                "delta_log10_x": float((ln_x[variant, case, column]-ln_x[0, case, column])/np.log(10)),
                "titanium_share_before": float(titanium_fraction[0, case, column]),
                "titanium_share_after": float(titanium_fraction[variant, case, column])}

    observed = sorted(set(np.flatnonzero(matrix[titanium])) | {species.index(s) for s in
                       ("e1-", "H1-", "H2O1", "C1O1", "O1V1", "O1Si1")})
    summaries = []
    for i, name in enumerate(VARIANTS[1:], 1):
        k, j = np.unravel_index(np.argmax(abs(dx[i-1])), dx[i-1].shape)
        summaries.append({"variant": name, "tio_ratio_min": float(np.exp(dlogx[i-1, :, tio]).min()),
                          "tio_ratio_max": float(np.exp(dlogx[i-1, :, tio]).max()),
                          "largest_absolute_response": point(i, k, j),
                          "tio_largest_budget_response": point(i, int(np.argmax(abs(titanium_fraction[i, :, tio]-titanium_fraction[0, :, tio]))), tio),
                          "species_extrema": [point(i, int(np.argmax(abs(dx[i-1, :, j]))), j) for j in observed],
                          "max_constraint_residuals": {key: float(value[i].max()) for key, value in checks.items()}})
    proxy_difference = float(np.max(abs(ln_x[2]-ln_x[1])))
    summary = {"variants": list(VARIANTS), "cases": 75, "species": 537,
               "reused_baseline_cases": 75, "new_endpoint_solves": 150,
               "primary_tolerance": PRIMARY_TOLERANCE, "strict_tolerance": STRICT_TOLERANCE,
               "tight_case_indices": selected, "max_tight_delta_ln_x": stability,
               "highlighted_case": point(1, highlight, tio), "variants_summary": summaries,
               "max_proxy_vs_toto_delta_ln_x": proxy_difference,
               "minimum_archived_ln_n": float(ln_n.min()),
               "gibbs_exchange_tolerance": gibbs_tolerance,
               "gibbs_min_current_delta": float(delta_current.min()),
               "gibbs_max_candidate_delta": float(delta_variant.max()),
               "independent_solver": independent,
               "independent_solver_case_index": independent_case,
               "scope": "Element-aggregated ideal gas, fixed 537-species ion network and budgets; no condensation, isotope fractionation, radiative transfer or full physical uncertainty validation.",
               "energy_sensitivity": "6.824 +/- 0.010 eV only; central bound-state thermal inventory held fixed.",
               "curated_update": "none"}
    np.savez_compressed(output/"equilibria.npz", variants=VARIANTS, ln_n=ln_n, h=h,
                        budget=budgets, formula_matrix=matrix, species=species, elements=elements,
                        temperature_K=t, pressure_bar=p, C_over_O=saved["C_over_O"],
                        tight_case_indices=selected, tight_ln_n=tight, highlighted_case_index=highlight,
                        lambda_values=lam, lambda_ln_n=path,
                        uncertainty_D0_eV=energies, uncertainty_ln_n=uncertainty,
                        delta_g_in_current=delta_current, delta_g_in_variant=delta_variant)
    for filename, data in (("summary.json", summary), ("diagnostics.json", rows),
                           ("selected_diagnostics.json", {"tight": tight_rows, "lambda": path_rows,
                                                          "energy_sensitivity": uncertainty_rows})):
        (output/filename).write_text(json.dumps(data, indent=2, allow_nan=False)+"\n")
    paths = [Path(__file__), Path(candidate.__file__), BASELINE,
             ROOT/"mceg/reviews/c2h_equilibrium_impact.py",
             ROOT/"examples/comparisons/compare_gas_thermochemistry.py",
             ROOT/"examples/comparisons/data/major_molecule_reference.json"]
    paths += sorted((ROOT/"tests/data/validation/tio_candidate/20261103").glob("*.json"))
    paths += [ROOT/p for p in ("src/exogibbs/presets/curated.py", "src/exogibbs/presets/curated_c2h.py",
              "src/exogibbs/presets/curated_v1.py", "src/exogibbs/presets/fastchem4.py",
              "src/exogibbs/thermo/standard.py", "src/exogibbs/thermo/models.py",
              "src/exogibbs/data/curated/gas_v1.json", "src/exogibbs/data/curated/c2h_v1.json",
              "src/exogibbs/data/FastChem4/logK/logK.dat",
              "src/exogibbs/data/FastChem4/element_abundances/asplund_2021.dat")]
    paths += sorted((ROOT/"src/exogibbs/equilibrium/gas").rglob("*.py"))
    receipt = {"created_utc": datetime.now(timezone.utc).isoformat(), "python": platform.python_version(),
               "jax": jax.__version__, "numpy": np.__version__, "dtype": "float64", "backend": jax.default_backend(),
               "command": " ".join(sys.argv), "result": "passed",
               "selected_file_sha256": {str(p.relative_to(ROOT)): file_hash(p) for p in paths},
               "output_sha256": {p.name: file_hash(p) for p in sorted(output.iterdir())},
               "fastchem_binary_sha256": file_hash(executable) if executable else None}
    (output/"receipt.json").write_text(json.dumps(receipt, indent=2)+"\n")
    print(json.dumps({"highlight": summary["highlighted_case"], "tight_delta": stability,
                      "proxy_delta": proxy_difference, "independent_solver": independent}), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--fastchem-executable", type=Path)
    args = parser.parse_args()
    run(args.output_dir, args.fastchem_executable)
