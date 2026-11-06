"""Measure a single C2H2 ATcT energy substitution on two fixed gas networks."""

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

from exogibbs.presets import curated_tio2 as curated
from c2h_equilibrium_impact import independent_check
from co2_equilibrium_impact import gibbs_ordering
from curated_h2o_adoption import solve_cases
from gas_source_comparison import ROOT, file_hash
import c2h2_candidate as candidate


BASELINE = ROOT / "tests/data/validation/curated_tio2_adoption/20261106"
PRIMARY_TOLERANCE, STRICT_TOLERANCE = 5e-14, 2e-14


def candidate_setup(base, thermal: str = "retained", fraction: float = 1.0):
    """Change C2H2 alone, leaving catalogs, budgets and atomic references fixed."""
    column = base.species.index("C2H2")

    def hvector(t):
        initial = base.hvector_func(t)
        target = candidate.formation_h_jax(t, thermal=thermal)
        return initial.at[..., column].set(initial[..., column]+fraction*(target-initial[..., column]))

    return replace(base, hvector_func=jax.jit(hvector), metadata={
        **base.metadata, "dataset": f"review-c2h2-{thermal}-atct",
        "production_adoption": False, "substitution_fraction": fraction})


def log_fractions(values):
    """Normalize stored logarithms without floors."""
    return values-logsumexp(values, axis=-1, keepdims=True)


def constraints(matrix, elements, species, budget, amounts, h, pressure):
    """Check all elements and formation equations, including neutral networks."""
    n = np.exp(amounts)
    residual = abs(n@matrix.T-budget)
    positive = budget[0] > 0
    refs = [species.index("e1-" if e == "e-" else e+"1") for e in elements]
    logp = log_fractions(amounts)+np.log(pressure)[..., None]
    values = {
        "element_relative_residual": np.max(residual[..., positive]/budget[..., positive], axis=-1),
        "mass_action_max_abs_ln": np.max(abs(logp-logp[..., refs]@matrix+h), axis=-1),
        "carbon_relative_residual": residual[..., elements.index("C")]/budget[..., elements.index("C")],
        "major_element_relative_residual": np.max(np.stack([
            residual[..., elements.index(e)]/budget[..., elements.index(e)] for e in ("H", "C", "O")]), axis=0),
    }
    if "e-" in elements:
        charge = matrix[elements.index("e-")]
        values["charge_relative_residual"] = abs(n@charge)/(n@abs(charge))
    limits = {"element_relative_residual": 1e-5, "major_element_relative_residual": 1e-9,
              "mass_action_max_abs_ln": 1e-8, "charge_relative_residual": 1e-12, "carbon_relative_residual": 1e-8}
    assert np.isfinite(amounts).all() and np.all(n > 0)
    for key, value in values.items():
        assert np.isfinite(value).all() and value.max() < limits[key], (key, value.max())
    return values


def run(output: Path, executable: Path | None = None) -> None:
    """Archive finite endpoints, cold tolerance controls and thermal controls."""
    jax.config.update("jax_enable_x64", True)
    output.mkdir(parents=True, exist_ok=False)
    summaries, rows, controls, independent = [], [], [], []
    for ions in (True, False):
        name = "ions" if ions else "neutral"
        baseline_receipt = json.loads((BASELINE/"receipt.json").read_text())
        assert file_hash(BASELINE/f"{name}.npz") == baseline_receipt["output_sha256"][f"{name}.npz"]
        with np.load(BASELINE/f"{name}.npz") as archive:
            saved = {key: archive[key].copy() for key in archive.files}
        base = curated.chemsetup(ions=ions)
        species, elements, matrix = list(base.species), list(base.elements), np.asarray(base.formula_matrix)
        for value, key in ((species, "species"), (elements, "elements"), (matrix, "formula_matrix")):
            np.testing.assert_array_equal(value, saved[key])
        t, p, budget = (saved[key] for key in ("temperature_K", "pressure_bar", "budget"))
        assert base.metadata["dataset"] == "curated-tio2-v1"
        models = [base, candidate_setup(base), candidate_setup(base, thermal="nasa")]
        h = np.asarray([model.hvector_func(t) for model in models])
        np.testing.assert_allclose(h[0], saved["h"], rtol=0, atol=1e-12)
        assert len(t) == 75 and saved["converged"].all()
        reference = np.asarray(base.element_vector_reference)
        for i, ratio in enumerate(saved["C_over_O"]):
            expected = reference.copy(); expected[elements.index("C")] = ratio*expected[elements.index("O")]
            np.testing.assert_allclose(budget[i]/budget[i,elements.index("H")], expected/expected[elements.index("H")], atol=1e-15, rtol=1e-12)
        c2h2, carbon = species.index("C2H2"), elements.index("C")
        for variant in (1, 2):
            np.testing.assert_array_equal(np.delete(h[variant]-h[0], c2h2, axis=-1), 0)
        endpoints = [saved["ln_n"]]
        for variant in (1, 2):
            endpoint = np.empty_like(saved["ln_n"])
            for ratio in np.unique(saved["C_over_O"]):
                indices = np.flatnonzero(saved["C_over_O"] == ratio)
                np.testing.assert_array_equal(budget[indices], np.broadcast_to(budget[indices[0]], budget[indices].shape))
                values, extra = solve_cases(models[variant], budget[indices[0]], t[indices], p[indices], PRIMARY_TOLERANCE)
                endpoint[indices] = values
                rows.extend({"network": name, "variant": variant, "case_index": int(i), **row} for i, row in zip(indices, extra))
            endpoints.append(endpoint)
            print(json.dumps({"network": name, "variant": variant, "cold_endpoints_passed": 75}), flush=True)
        amounts = np.asarray(endpoints)
        checks = constraints(matrix, elements, species, budget, amounts, h, p)
        ln_x, x = log_fractions(amounts), np.exp(log_fractions(amounts))
        share = np.exp(amounts)*matrix[carbon]/budget[:, carbon, None]
        delta, ratio = x[1]-x[0], np.exp(ln_x[1, :, c2h2]-ln_x[0, :, c2h2])
        highlight = int(np.argmax(abs(share[1, :, c2h2]-share[0, :, c2h2])))
        hot = np.flatnonzero(t == 4000.)
        selected = {int(np.argmax(share[0, :, c2h2])),
                    int(np.unravel_index(np.argmax(abs(ln_x[1:, :, c2h2]-ln_x[0, :, c2h2])), (2,75))[1]),
                    int(hot[np.argmax(abs(share[2, hot, c2h2]-share[0, hot, c2h2]))])}
        for i in (highlight, 4, 70):
            if len(selected) == 3: break
            selected.add(i)
        selected = sorted(selected)
        assert len(selected) == 3
        tight = []
        for variant, model in enumerate(models):
            subset = []
            for i in selected:
                values, extra = solve_cases(model, budget[i], t[i:i+1], p[i:i+1], STRICT_TOLERANCE)
                subset.append(values[0])
                controls.extend({"network": name, "kind": "tight", "variant": variant,
                                 "case_index": i, **row} for row in extra)
            tight.append(subset)
        tight = np.asarray(tight)
        difference = log_fractions(tight)-ln_x[:, selected]
        visible = np.maximum(np.exp(log_fractions(tight)), x[:, selected]) >= 1e-12
        stability = {"all_delta_ln_x": float(abs(difference).max()),
                     "C2H2_delta_ln_x": float(abs(difference[..., c2h2]).max()),
                     "visible_delta_ln_x": float(abs(difference[visible]).max()),
                     "absolute_delta_x": float(abs(np.exp(log_fractions(tight))-x[:, selected]).max())}
        print(json.dumps({"network": name, "tolerance_stability": stability}), flush=True)
        for key, limit in zip(stability, (1e-4, 1e-8, 2e-6, 1e-9)):
            assert stability[key] < limit, (name, key, stability[key])
        pairs = [gibbs_ordering(amounts[[0, variant]], h[[0, variant]], p, matrix, elements, species) for variant in (1, 2)]
        raw, corrected = np.asarray([pair[0] for pair in pairs]), np.asarray([pair[1] for pair in pairs])
        fractions, path = np.linspace(0, 1, 5), []
        for fraction in fractions:
            model = candidate_setup(base, fraction=float(fraction))
            values, extra = solve_cases(model, budget[highlight], t[highlight:highlight+1], p[highlight:highlight+1], STRICT_TOLERANCE)
            path.append(values[0])
            controls.extend({"network": name, "kind": "path", "lambda": float(fraction),
                             "case_index": highlight, **row} for row in extra)
        # Both continuous candidates have complete independent endpoint grids.
        thermal_delta = ln_x[2]-ln_x[1]
        if ions and executable:
            exportable = np.flatnonzero(np.isin(t, [1500., 2000., 3000., 4000.]))
            independent_cases = sorted({int(exportable[np.argmax(abs(share[1, exportable, c2h2]-share[0, exportable, c2h2]))]),
                                        int(exportable[np.argmax(abs(np.log(ratio[exportable])))] )})
            for i in independent_cases:
                independent.append({"network": name, "case_index": i, "comparisons": independent_check(
                    models, budget[i], float(t[i]), float(p[i]), amounts[:, i], executable.resolve())})

        def point(i, j):
            return {"case_index": int(i), "temperature_K": float(t[i]), "pressure_bar": float(p[i]),
                    "C_over_O": float(saved["C_over_O"][i]), "species": species[j],
                    "x_before": float(x[0, i, j]), "x_after": float(x[1, i, j]),
                    "delta_x": float(delta[i, j]), "ratio": float(np.exp(ln_x[1, i, j]-ln_x[0, i, j])),
                    "carbon_share_before": float(share[0, i, j]), "carbon_share_after": float(share[1, i, j])}

        observed = [s for s in ("C2H2", "C1O1", "C1O2", "C1H4", "C1H1N1", "C2", "C1H2", "C1H1", "H2O1", "H1", "H2", "e1-") if s in species]
        summary = {"network": name, "species": len(species), "cases": 75,
                   "C2H2_ratio_min": float(ratio.min()), "C2H2_ratio_max": float(ratio.max()),
                   "highlighted_case": point(highlight, c2h2),
                   "largest_absolute_response": point(*np.unravel_index(np.argmax(abs(delta)), delta.shape)),
                   "species_extrema": [point(int(np.argmax(abs(delta[:, species.index(s)]))), species.index(s)) for s in observed],
                   "max_constraint_residuals": {key: float(value.max()) for key, value in checks.items()},
                   "tight_case_indices": selected, "tolerance_stability": stability,
                   "nasa_control_C2H2_ratio_min": float(np.exp(thermal_delta[:, c2h2]).min()),
                   "nasa_over_current_ratio_min": float(np.exp(ln_x[2, :, c2h2]-ln_x[0, :, c2h2]).min()),
                   "nasa_over_current_ratio_max": float(np.exp(ln_x[2, :, c2h2]-ln_x[0, :, c2h2]).max()),
                   "nasa_control_C2H2_ratio_max": float(np.exp(thermal_delta[:, c2h2]).max()),
                   "nasa_control_max_absolute_delta_x": float(abs(x[2]-x[1]).max()),
                   "gibbs_corrected_current_min": float(corrected[:, 0].min()),
                   "gibbs_corrected_candidate_max": float(corrected[:, 1].max())}
        summary["candidate_responses"] = []
        for variant in (1, 2):
            change = x[variant]-x[0]
            carbon_records = []
            for j in np.flatnonzero(matrix[carbon] > 0):
                i = int(np.argmax(abs(share[variant,:,j]-share[0,:,j])))
                carbon_records.append({"species":species[j],"case_index":i,
                    "carbon_share_before":float(share[0,i,j]),"carbon_share_after":float(share[variant,i,j]),
                    "maximum_absolute_carbon_share_change":float(abs(share[variant,i,j]-share[0,i,j])),
                    "x_before":float(x[0,i,j]),"x_after":float(x[variant,i,j])})
            carbon_records.sort(key=lambda row:row["maximum_absolute_carbon_share_change"], reverse=True)
            i = int(np.argmax(abs(share[variant,:,c2h2]-share[0,:,c2h2])))
            summary["candidate_responses"].append({"variant":variant,"model":candidate.THERMAL[variant-1],
                "target_ratio_min":float(np.exp(ln_x[variant,:,c2h2]-ln_x[0,:,c2h2]).min()),
                "target_ratio_max":float(np.exp(ln_x[variant,:,c2h2]-ln_x[0,:,c2h2]).max()),
                "maximum_absolute_target_delta_x":float(abs(change[:,c2h2]).max()),
                "maximum_absolute_all_species_delta_x":float(abs(change).max()),
                "highlight":{"case_index":i,"temperature_K":float(t[i]),"pressure_bar":float(p[i]),"C_over_O":float(saved["C_over_O"][i]),
                    "carbon_share_before":float(share[0,i,c2h2]),"carbon_share_after":float(share[variant,i,c2h2]),
                    "x_before":float(x[0,i,c2h2]),"x_after":float(x[variant,i,c2h2])},
                "all_carbon_species_effects":carbon_records})
        summaries.append(summary)
        np.savez_compressed(output/f"{name}.npz", variants=["current", "retained_atct", "nasa_atct"], ln_n=amounts, h=h,
            formula_matrix=matrix, species=species, elements=elements, budget=budget,
            temperature_K=t, pressure_bar=p, C_over_O=saved["C_over_O"],
            tight_case_indices=selected, tight_ln_n=tight, highlighted_case_index=highlight,
            lambda_values=fractions, lambda_ln_n=path, delta_g_raw=raw, delta_g_corrected=corrected)
        print(json.dumps(summary), flush=True)
    summary = {"baseline_dataset": "curated-tio2-v1", "curated_update": "none",
               "reused_baseline_cases": 150, "new_endpoint_solves": len(rows),
               "tighter_control_solves":sum(row["kind"]=="tight" for row in controls),
               "path_control_solves":sum(row["kind"]=="path" for row in controls),
               "independent_fastchem_runs":sum(len(row["comparisons"]) for row in independent),
               "primary_tolerance": PRIMARY_TOLERANCE, "strict_tolerance": STRICT_TOLERANCE,
               "networks": summaries, "independent_solver": independent if executable else "not_run",
               "scope": "Fixed element-aggregated ideal gas, 1500-4000 K; no condensation, transport, opacity or full physical uncertainty validation.",
               "thermal_control_scope": "NASA molecular and atomic thermal functions on the full 75-case grid per network with the identical balanced ATcT D0; the primary model retains inherited atomic/thermal conventions, so this is a reference-model sensitivity, not a physical uncertainty interval."}
    for filename, data in (("summary.json", summary), ("diagnostics.json", rows), ("selected_diagnostics.json", controls)):
        (output/filename).write_text(json.dumps(data, indent=2, allow_nan=False)+"\n")
    paths = {Path(__file__), Path(candidate.__file__), BASELINE/"receipt.json"}
    paths.update(BASELINE/f"{name}.npz" for name in ("ions", "neutral"))
    paths.update(ROOT/path for path in candidate.load_candidate()["source_sha256"])
    paths.update(candidate.DATA/name for name in ("candidate.json", "validation.json", "receipt.json"))
    paths.update(Path(module.__file__).resolve() for module in tuple(sys.modules.values())
                 if getattr(module, "__file__", None)
                 and Path(module.__file__).resolve().is_relative_to(ROOT)
                 and Path(module.__file__).suffix == ".py"
                 and Path(module.__file__).name not in ("exogibbs_version.py", "curated.py"))
    paths.update((ROOT/"src/exogibbs/data/curated").glob("*.json"))
    receipt = {"created_utc": datetime.now(timezone.utc).isoformat(), "python": platform.python_version(),
               "jax": jax.__version__, "numpy": np.__version__, "dtype": "float64", "backend": jax.default_backend(),
               "command": " ".join(sys.argv), "result": "passed",
               "selected_file_sha256": {str(path.relative_to(ROOT)): file_hash(path) for path in sorted(paths) if path.is_file()},
               "output_sha256": {path.name: file_hash(path) for path in sorted(output.iterdir())},
               "fastchem_binary_sha256": file_hash(executable) if executable else None}
    (output/"receipt.json").write_text(json.dumps(receipt, indent=2)+"\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--fastchem-executable", type=Path)
    args = parser.parse_args()
    run(args.output_dir, args.fastchem_executable)
