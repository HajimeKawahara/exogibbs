"""Evaluate two H2O thermal candidates on the fixed current gas network."""

from __future__ import annotations

import argparse
from dataclasses import replace
from datetime import datetime, timezone
import json
from pathlib import Path
import platform
import sys

import jax
import numpy as np
from scipy.special import logsumexp

from exogibbs.presets import curated_tio as curated

from c2h_equilibrium_impact import independent_check, solve_cases
from gas_source_comparison import ROOT, file_hash
import h2o_candidate as candidate
from tio_equilibrium_impact import diagnostics, gibbs


BASELINE = ROOT / "tests/data/validation/curated_tio_adoption/20261103/ions.npz"
DATA = ROOT / "tests/data/validation/h2o_equilibrium_impact/20261104"
PRIMARY_TOLERANCE = 5e-14
STRICT_TOLERANCE = 1e-14
VARIANTS = ("current", "evaluated", "discrete")


def candidate_setup(base, thermal: str = "evaluated", fraction: float = 1.0):
    """Change H2O alone, preserving the catalog, charge and elemental budgets."""
    index = base.species.index("H2O1")

    def hvector(temperature):
        start = base.hvector_func(temperature)
        target = candidate.formation_h_jax(temperature, thermal=thermal)
        return start.at[..., index].set(start[..., index] + fraction*(target-start[..., index]))

    return replace(base, hvector_func=jax.jit(hvector), metadata={
        **base.metadata, "dataset": f"review-h2o-{thermal}-2025",
        "substitution_fraction": fraction, "production_adoption": False})


def log_fractions(log_amounts):
    return log_amounts-logsumexp(log_amounts, axis=-1, keepdims=True)


def run(output: Path, executable: Path | None = None) -> None:
    """Run cold endpoints, tighter controls, a finite path and solver comparisons."""
    jax.config.update("jax_enable_x64", True)
    output.mkdir(parents=True, exist_ok=False)
    with np.load(BASELINE) as archive:
        saved = {name: archive[name].copy() for name in archive.files}
    base = curated.chemsetup(ions=True)
    species, elements = list(base.species), list(base.elements)
    matrix = np.asarray(base.formula_matrix)
    budget, t, p = saved["budget"], saved["temperature_K"], saved["pressure_bar"]
    for actual, key in ((species, "species"), (elements, "elements"), (matrix, "formula_matrix")):
        np.testing.assert_array_equal(actual, saved[key])
    np.testing.assert_allclose(base.hvector_func(t), saved["h"], rtol=0, atol=1e-12)
    assert saved["converged"].all() and len(t) == 75 and len(species) == 537
    models = [base] + [candidate_setup(base, thermal) for thermal in VARIANTS[1:]]
    h = np.asarray([model.hvector_func(t) for model in models])
    water = species.index("H2O1")
    oxygen, carbon = elements.index("O"), elements.index("C")
    np.testing.assert_allclose(np.delete(h[1:]-h[0], water, axis=-1), 0, rtol=0, atol=1e-12)
    endpoints, rows = [saved["ln_n"]], []
    for name, model in zip(VARIANTS[1:], models[1:]):
        result = np.empty_like(saved["ln_n"])
        for ratio in np.unique(saved["C_over_O"]):
            indices = np.flatnonzero(saved["C_over_O"] == ratio)
            np.testing.assert_array_equal(budget[indices], np.broadcast_to(budget[indices[0]], budget[indices].shape))
            values, checks = solve_cases(model, budget[indices[0]], t[indices], p[indices], PRIMARY_TOLERANCE)
            result[indices] = values
            rows.extend({"variant": name, "case_index": int(i), **row} for i, row in zip(indices, checks))
        endpoints.append(result)
        print(f"{name}: 75 cold endpoint solves passed", flush=True)
    ln_n = np.asarray(endpoints)
    ln_x = log_fractions(ln_n)
    n, x = np.exp(ln_n), np.exp(ln_x)
    checks = diagnostics(matrix, elements, species, budget, ln_n, h, p)
    oxygen_share = n*matrix[oxygen]/budget[:, oxygen, None]
    carbon_share = n*matrix[carbon]/budget[:, carbon, None]
    dx, dlnx = x[1:]-x[0], ln_x[1:]-ln_x[0]
    highlight = int(np.argmax(abs(oxygen_share[1, :, water]-oxygen_share[0, :, water])))
    scope_case = int(np.argmax(abs(ln_x[2, :, water]-ln_x[1, :, water])))
    selected = sorted({highlight, scope_case, int(np.argmax(abs(dlnx[0, :, water]))),
                       int(np.argmax(checks["element_relative_residual"][1]))})
    tight, tight_rows = [], []
    for name, model in zip(VARIANTS, models):
        subset = []
        for i in selected:
            values, extra = solve_cases(model, budget[i], t[i:i+1], p[i:i+1], STRICT_TOLERANCE)
            subset.append(values[0])
            tight_rows.extend({"variant": name, "case_index": i, **row} for row in extra)
        tight.append(subset)
    tight = np.asarray(tight)
    tight_logx = log_fractions(tight)
    tight_delta = tight_logx-ln_x[:, selected]
    stability = float(np.max(abs(tight_delta)))
    water_stability = float(np.max(abs(tight_delta[..., water])))
    visible = np.maximum(np.exp(tight_logx), x[:, selected]) >= 1e-12
    visible_stability = float(np.max(abs(tight_delta[visible])))
    absolute_stability = float(np.max(abs(np.exp(tight_logx)-x[:, selected])))
    worst_visible = np.unravel_index(np.argmax(np.where(visible, abs(tight_delta), -1)), tight_delta.shape)
    # A ppm-scale F-budget residual affects trace fluorides; H2O and the
    # absolute abundance response must separately pass much tighter checks.
    if stability >= 1e-4 or water_stability >= 1e-7 or visible_stability >= 2e-6 or absolute_stability >= 1e-9:
        worst = np.unravel_index(np.argmax(abs(tight_delta)), tight_delta.shape)
        raise ValueError(f"Tolerance control: all={stability}, water={water_stability}, "
                         f"visible={visible_stability}, worst={(VARIANTS[worst[0]], selected[worst[1]], species[worst[2]])}")
    scope_stability = float(np.max(abs((tight_logx[2, :, water]-tight_logx[1, :, water])
                                      -(ln_x[2, selected, water]-ln_x[1, selected, water]))))

    current_g, variant_g = [], []
    for i in (1, 2):
        current_g.append(np.asarray(gibbs(ln_n[i], h[0], p)-gibbs(ln_n[0], h[0], p), float))
        variant_g.append(np.asarray(gibbs(ln_n[i], h[i], p)-gibbs(ln_n[0], h[i], p), float))
    current_g, variant_g = np.asarray(current_g), np.asarray(variant_g)
    # These are comparisons under the SAME potential, not across energy gauges.
    assert np.min(current_g) > -2e-12 and np.max(variant_g) < 2e-12
    fractions, path, path_rows = np.linspace(0, 1, 5), [], []
    for fraction in fractions:
        model = candidate_setup(base, fraction=float(fraction))
        values, extra = solve_cases(model, budget[highlight], t[highlight:highlight+1],
                                   p[highlight:highlight+1], STRICT_TOLERANCE)
        path.append(values[0])
        path_rows.extend({"lambda": float(fraction), **row} for row in extra)

    exportable = np.flatnonzero(np.isin(t, [1500., 2000., 3000., 4000.]))
    independent_cases = sorted({int(exportable[np.argmax(abs(oxygen_share[1, exportable, water]
                                                           -oxygen_share[0, exportable, water]))]),
                                int(exportable[np.argmax(abs(ln_x[2, exportable, water]
                                                            -ln_x[1, exportable, water]))])})
    independent = []
    if executable:
        for i in independent_cases:
            result = independent_check(models, budget[i], float(t[i]), float(p[i]),
                                       ln_n[:, i], executable.resolve())
            independent.append({"case_index": i, "comparisons": result})

    def point(variant, case, column):
        return {"case_index": int(case), "temperature_K": float(t[case]), "pressure_bar": float(p[case]),
                "C_over_O": float(saved["C_over_O"][case]), "species": species[column],
                "x_before": float(x[0, case, column]), "x_after": float(x[variant, case, column]),
                "delta_x": float(x[variant, case, column]-x[0, case, column]),
                "delta_log10_x": float((ln_x[variant, case, column]-ln_x[0, case, column])/np.log(10)),
                "oxygen_share_before": float(oxygen_share[0, case, column]),
                "oxygen_share_after": float(oxygen_share[variant, case, column]),
                "carbon_share_before": float(carbon_share[0, case, column]),
                "carbon_share_after": float(carbon_share[variant, case, column])}

    observed = [species.index(s) for s in ("H2O1", "H1O1", "H1", "H2", "C1O1", "C1O2",
                                           "C1H4", "C1H1N1_hcn", "O1", "O2", "e1-", "H1-")]
    summaries = []
    for i, name in enumerate(VARIANTS[1:], 1):
        case, column = np.unravel_index(np.argmax(abs(dx[i-1])), dx[i-1].shape)
        summaries.append({"variant": name, "water_ratio_min": float(np.exp(dlnx[i-1, :, water]).min()),
                          "water_ratio_max": float(np.exp(dlnx[i-1, :, water]).max()),
                          "largest_absolute_response": point(i, case, column),
                          "water_largest_oxygen_response": point(i, int(np.argmax(abs(
                              oxygen_share[i, :, water]-oxygen_share[0, :, water]))), water),
                          "species_extrema": [point(i, int(np.argmax(abs(dx[i-1, :, j]))), j) for j in observed],
                          "max_constraint_residuals": {key: float(value[i].max()) for key, value in checks.items()}})
    scope_delta = ln_x[2]-ln_x[1]
    scope_visible = np.max(x[1:], axis=0) >= 1e-12
    summary = {"variants": list(VARIANTS), "cases": 75, "species": 537,
               "reused_baseline_cases": 75, "new_endpoint_solves": 150,
               "primary_tolerance": PRIMARY_TOLERANCE, "strict_tolerance": STRICT_TOLERANCE,
               "tight_case_indices": selected, "tighter_cold_solves": 3*len(selected),
               "max_tight_delta_ln_x": stability, "max_tight_water_delta_ln_x": water_stability,
               "max_tight_visible_delta_ln_x": visible_stability,
               "max_tight_absolute_delta_x": absolute_stability,
               "worst_visible_tolerance_case": {
                   "variant": VARIANTS[worst_visible[0]], "case_index": selected[worst_visible[1]],
                   "species": species[worst_visible[2]],
                   "x_primary": float(x[worst_visible[0], selected[worst_visible[1]], worst_visible[2]]),
                   "x_tight": float(np.exp(tight_logx[worst_visible])),
                   "delta_ln_x": float(tight_delta[worst_visible])},
               "max_tight_scope_water_response_delta_ln_x": scope_stability,
               "highlighted_case": point(1, highlight, water), "variants_summary": summaries,
               "discrete_over_evaluated_water_ratio_min": float(np.exp(scope_delta[:, water]).min()),
               "discrete_over_evaluated_water_ratio_max": float(np.exp(scope_delta[:, water]).max()),
               "max_discrete_vs_evaluated_visible_abs_delta_ln_x": float(np.max(abs(scope_delta[scope_visible]))),
               "max_discrete_vs_evaluated_abs_delta_x": float(np.max(abs(x[2]-x[1]))),
               "minimum_archived_ln_n": float(ln_n.min()),
               "gibbs_exchange_tolerance": 2e-12,
               "gibbs_min_current_delta": float(current_g.min()),
               "gibbs_max_candidate_delta": float(variant_g.max()),
               "independent_solver": independent if executable else "not_run",
               "scope": "Element-aggregated ideal gas on the fixed 537-species ion network and budgets, 1500-4000 K. Thermal-source sensitivity, not condensation, transport, opacity or complete atmospheric uncertainty validation.",
               "scope_sensitivity": "Evaluated 2025 versus PW2024 discrete thermal functions at the identical ATcT energy anchor; neither source difference is a total physical uncertainty bound.",
               "curated_update": "none"}
    np.savez_compressed(output/"equilibria.npz", variants=VARIANTS, ln_n=ln_n, h=h,
                        budget=budget, formula_matrix=matrix, species=species, elements=elements,
                        temperature_K=t, pressure_bar=p, C_over_O=saved["C_over_O"],
                        tight_case_indices=selected, tight_ln_n=tight, highlighted_case_index=highlight,
                        scope_case_index=scope_case, lambda_values=fractions, lambda_ln_n=path,
                        delta_g_in_current=current_g, delta_g_in_variant=variant_g)
    for filename, data in (("summary.json", summary), ("diagnostics.json", rows),
                           ("selected_diagnostics.json", {"tight": tight_rows, "lambda": path_rows})):
        (output/filename).write_text(json.dumps(data, indent=2, allow_nan=False)+"\n")
    inputs = [Path(__file__), Path(candidate.__file__), BASELINE,
              ROOT/"mceg/reviews/c2h_equilibrium_impact.py", ROOT/"mceg/reviews/tio_equilibrium_impact.py",
              ROOT/"mceg/reviews/tio_candidate.py", ROOT/"mceg/reviews/h2o_source_review.py",
              ROOT/"mceg/reviews/gas_source_comparison.py",
              ROOT/"examples/comparisons/select_major_molecules.py",
              ROOT/"examples/comparisons/compare_gas_thermochemistry.py",
              ROOT/"examples/comparisons/data/major_molecule_reference.json"]
    inputs += sorted((ROOT/"tests/data/validation/h2o_candidate/20261104").glob("*.json"))
    inputs += [ROOT/name for name in ("src/exogibbs/presets/curated.py", "src/exogibbs/presets/curated_tio.py",
               "src/exogibbs/presets/curated_c2h.py", "src/exogibbs/presets/curated_v1.py",
               "src/exogibbs/presets/fastchem4.py", "src/exogibbs/thermo/standard.py",
               "src/exogibbs/thermo/models.py", "src/exogibbs/data/curated/gas_v1.json",
               "src/exogibbs/data/curated/c2h_v1.json", "src/exogibbs/data/curated/tio_v1.json",
               "src/exogibbs/data/FastChem4/logK/logK.dat",
               "src/exogibbs/data/FastChem4/element_abundances/asplund_2021.dat")]
    inputs += sorted((ROOT/"src/exogibbs/equilibrium/gas").rglob("*.py"))
    receipt = {"created_utc": datetime.now(timezone.utc).isoformat(), "python": platform.python_version(),
               "jax": jax.__version__, "numpy": np.__version__, "dtype": "float64", "backend": jax.default_backend(),
               "command": " ".join(sys.argv), "result": "passed",
               "selected_file_sha256": {str(p.relative_to(ROOT)): file_hash(p) for p in inputs},
               "output_sha256": {p.name: file_hash(p) for p in sorted(output.iterdir())},
               "fastchem_binary_sha256": file_hash(executable) if executable else None}
    (output/"receipt.json").write_text(json.dumps(receipt, indent=2)+"\n")
    print(json.dumps({"highlight": summary["highlighted_case"], "tight_delta": stability,
                      "scope_water_ratio": [summary["discrete_over_evaluated_water_ratio_min"],
                                            summary["discrete_over_evaluated_water_ratio_max"]],
                      "independent_solver": summary["independent_solver"]}), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--fastchem-executable", type=Path)
    args = parser.parse_args()
    run(args.output_dir, args.fastchem_executable)
