"""Measure a single HO2 NASA/ATcT substitution on two fixed gas networks."""

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

from exogibbs.presets import curated_oh as curated
from c2h_equilibrium_impact import independent_check
from co2_equilibrium_impact import gibbs_ordering
from curated_h2o_adoption import solve_cases
from gas_source_comparison import ROOT, file_hash
import ho2_candidate as candidate


BASELINE = ROOT / "tests/data/validation/curated_oh_adoption/20261105"
PRIMARY_TOLERANCE, STRICT_TOLERANCE = 5e-14, 2e-14


def candidate_setup(base, fraction: float = 1.0):
    """Change HO2 alone, leaving catalogs, budgets and atomic references fixed."""
    column = base.species.index("H1O2")

    def hvector(t):
        initial = base.hvector_func(t)
        target = candidate.formation_h_jax(t)
        return initial.at[..., column].set(initial[..., column]+fraction*(target-initial[..., column]))

    return replace(base, hvector_func=jax.jit(hvector), metadata={
        **base.metadata, "dataset": "review-ho2-nasa-atct1172",
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
        "major_element_relative_residual": np.max(np.stack([
            residual[..., elements.index(e)]/budget[..., elements.index(e)] for e in ("H", "C", "O")]), axis=0),
    }
    if "e-" in elements:
        charge = matrix[elements.index("e-")]
        values["charge_relative_residual"] = abs(n@charge)/(n@abs(charge))
    limits = {"element_relative_residual": 1e-5, "major_element_relative_residual": 1e-9,
              "mass_action_max_abs_ln": 1e-8, "charge_relative_residual": 1e-12}
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
        with np.load(BASELINE/f"{name}.npz") as archive:
            saved = {key: archive[key].copy() for key in archive.files}
        base = curated.chemsetup(ions=ions)
        species, elements, matrix = list(base.species), list(base.elements), np.asarray(base.formula_matrix)
        for value, key in ((species, "species"), (elements, "elements"), (matrix, "formula_matrix")):
            np.testing.assert_array_equal(value, saved[key])
        t, p, budget = (saved[key] for key in ("temperature_K", "pressure_bar", "budget"))
        models = [base, candidate_setup(base)]
        h = np.asarray([model.hvector_func(t) for model in models])
        np.testing.assert_allclose(h[0], saved["h"], rtol=0, atol=1e-12)
        assert len(t) == 75 and saved["converged"].all()
        oh, oxygen = species.index("H1O2"), elements.index("O")
        np.testing.assert_array_equal(np.delete(h[1]-h[0], oh, axis=-1), 0)
        endpoint = np.empty_like(saved["ln_n"])
        for ratio in np.unique(saved["C_over_O"]):
            indices = np.flatnonzero(saved["C_over_O"] == ratio)
            np.testing.assert_array_equal(budget[indices], np.broadcast_to(budget[indices[0]], budget[indices].shape))
            values, extra = solve_cases(models[1], budget[indices[0]], t[indices], p[indices], PRIMARY_TOLERANCE)
            endpoint[indices] = values
            rows.extend({"network": name, "case_index": int(i), **row} for i, row in zip(indices, extra))
        amounts = np.asarray([saved["ln_n"], endpoint])
        checks = constraints(matrix, elements, species, budget, amounts, h, p)
        ln_x, x = log_fractions(amounts), np.exp(log_fractions(amounts))
        share = np.exp(amounts)*matrix[oxygen]/budget[:, oxygen, None]
        delta, ratio = x[1]-x[0], np.exp(ln_x[1, :, oh]-ln_x[0, :, oh])
        highlight = int(np.argmax(abs(share[1, :, oh]-share[0, :, oh])))
        selected = sorted({highlight, int(np.argmax(abs(np.log(ratio)))),
                           int(np.argmax(checks["element_relative_residual"][1]))})
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
                     "HO2_delta_ln_x": float(abs(difference[..., oh]).max()),
                     "visible_delta_ln_x": float(abs(difference[visible]).max()),
                     "absolute_delta_x": float(abs(np.exp(log_fractions(tight))-x[:, selected]).max())}
        print(json.dumps({"network": name, "tolerance_stability": stability}), flush=True)
        for key, limit in zip(stability, (1e-4, 1e-8, 2e-6, 1e-9)):
            assert stability[key] < limit, (name, key, stability[key])
        raw, corrected = gibbs_ordering(amounts, h, p, matrix, elements, species)
        fractions, path = np.linspace(0, 1, 5), []
        for fraction in fractions:
            model = candidate_setup(base, fraction=float(fraction))
            values, extra = solve_cases(model, budget[highlight], t[highlight:highlight+1], p[highlight:highlight+1], STRICT_TOLERANCE)
            path.append(values[0])
            controls.extend({"network": name, "kind": "path", "lambda": float(fraction),
                             "case_index": highlight, **row} for row in extra)
        # Original JANAF knots only, at the same ATcT D0 and NASA atoms.
        thermal_amounts, thermal_h = [], []
        for i in selected:
            control_h = candidate.janaf_formation_h_node(float(t[i]))

            def hvector(temperature, node=float(t[i]), value=control_h):
                initial = base.hvector_func(temperature)
                target = jax.numpy.where(temperature == node, value, jax.numpy.nan)
                return initial.at[..., oh].set(target)

            thermal_model = replace(base, hvector_func=jax.jit(hvector))
            values, extra = solve_cases(thermal_model, budget[i], t[i:i+1], p[i:i+1], STRICT_TOLERANCE)
            thermal_amounts.append(values[0])
            thermal_h.append(np.asarray(thermal_model.hvector_func(t[i])))
            controls.extend({"network": name, "kind": "janaf_node", "case_index": i, **row} for row in extra)
        thermal_amounts, thermal_h = np.asarray(thermal_amounts), np.asarray(thermal_h)
        thermal_delta = log_fractions(thermal_amounts)-log_fractions(tight[1])
        if ions and executable:
            exportable = np.flatnonzero(np.isin(t, [1500., 2000., 3000., 4000.]))
            independent_cases = sorted({int(exportable[np.argmax(abs(share[1, exportable, oh]-share[0, exportable, oh]))]),
                                        int(exportable[np.argmax(abs(np.log(ratio[exportable])))] )})
            for i in independent_cases:
                independent.append({"network": name, "case_index": i, "comparisons": independent_check(
                    models, budget[i], float(t[i]), float(p[i]), amounts[:, i], executable.resolve())})

        def point(i, j):
            return {"case_index": int(i), "temperature_K": float(t[i]), "pressure_bar": float(p[i]),
                    "C_over_O": float(saved["C_over_O"][i]), "species": species[j],
                    "x_before": float(x[0, i, j]), "x_after": float(x[1, i, j]),
                    "delta_x": float(delta[i, j]), "ratio": float(np.exp(ln_x[1, i, j]-ln_x[0, i, j])),
                    "oxygen_share_before": float(share[0, i, j]), "oxygen_share_after": float(share[1, i, j])}

        observed = [s for s in ("H1O2", "H1O1", "H2O1", "O1", "O2", "H1", "H2", "C1O1", "C1O2", "C1H4", "e1-") if s in species]
        summary = {"network": name, "species": len(species), "cases": 75,
                   "HO2_ratio_min": float(ratio.min()), "HO2_ratio_max": float(ratio.max()),
                   "highlighted_case": point(highlight, oh),
                   "largest_absolute_response": point(*np.unravel_index(np.argmax(abs(delta)), delta.shape)),
                   "species_extrema": [point(int(np.argmax(abs(delta[:, species.index(s)]))), species.index(s)) for s in observed],
                   "max_constraint_residuals": {key: float(value.max()) for key, value in checks.items()},
                   "tight_case_indices": selected, "tolerance_stability": stability,
                   "janaf_node_HO2_ratio_min": float(np.exp(thermal_delta[:, oh]).min()),
                   "janaf_node_HO2_ratio_max": float(np.exp(thermal_delta[:, oh]).max()),
                   "janaf_node_max_absolute_delta_x": float(abs(np.exp(log_fractions(thermal_amounts))-np.exp(log_fractions(tight[1]))).max()),
                   "gibbs_corrected_current_min": float(corrected[0].min()),
                   "gibbs_corrected_candidate_max": float(corrected[1].max())}
        summaries.append(summary)
        np.savez_compressed(output/f"{name}.npz", variants=["current", "nasa_atct"], ln_n=amounts, h=h,
            formula_matrix=matrix, species=species, elements=elements, budget=budget,
            temperature_K=t, pressure_bar=p, C_over_O=saved["C_over_O"],
            tight_case_indices=selected, tight_ln_n=tight, highlighted_case_index=highlight,
            lambda_values=fractions, lambda_ln_n=path, delta_g_raw=raw, delta_g_corrected=corrected,
            janaf_node_ln_n=thermal_amounts, janaf_node_h=thermal_h)
        print(json.dumps(summary), flush=True)
    summary = {"baseline_dataset": "curated-oh-v1", "curated_update": "none",
               "reused_baseline_cases": 150, "new_endpoint_solves": len(rows),
               "primary_tolerance": PRIMARY_TOLERANCE, "strict_tolerance": STRICT_TOLERANCE,
               "networks": summaries, "independent_solver": independent if executable else "not_run",
               "scope": "Fixed hot H-rich element-aggregated ideal gas, 1500-4000 K; trace HO2 here does not constrain low-temperature photochemistry, kinetics, condensation, transport, opacity or total physical uncertainty.",
               "thermal_control_scope": "JANAF original table knots at selected cases in each network, identical ATcT energy anchor and NASA atomic thermal functions; no continuous JANAF candidate or total uncertainty bound."}
    for filename, data in (("summary.json", summary), ("diagnostics.json", rows), ("selected_diagnostics.json", controls)):
        (output/filename).write_text(json.dumps(data, indent=2, allow_nan=False)+"\n")
    paths = {Path(__file__), Path(candidate.__file__)}
    # Include the dependencies of the reused solver and current baseline.
    previous = json.loads((BASELINE/"receipt.json").read_text())
    paths.update(ROOT/name for name in previous["selected_file_sha256"])
    paths.update(BASELINE/f"{name}.npz" for name in ("ions", "neutral"))
    paths.update(ROOT/"mceg/reviews"/f"{name}.py" for name in
                 ("ho2_source_review", "oh_source_review", "c2h_source_review", "co2_equilibrium_impact", "tio_equilibrium_impact", "tio_candidate"))
    paths.update((ROOT/"tests/data/validation/ho2_source_review/20261105").glob("*"))
    paths.update(candidate.INPUTS)
    paths.add(ROOT/"tests/data/validation/ho2_candidate/20261105/candidate.json")
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
