"""Run explicitly defined candidates on one verified named production baseline."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import importlib
import json
from pathlib import Path
import platform
import sys

import jax
import numpy as np
from scipy.special import logsumexp

from c2h_equilibrium_impact import carbon_budget, independent_check
from co2_equilibrium_impact import gibbs_ordering
from curated_h2o_adoption import solve_cases
from gas_source_comparison import ROOT, file_hash
from pending_candidate import candidate_setup

PRIMARY_TOLERANCE, STRICT_TOLERANCE = 5e-14, 2e-14


def log_fractions(amounts):
    """Normalize logarithmic amounts without concentration floors."""
    return amounts-logsumexp(amounts, axis=-1, keepdims=True)


def constraints(matrix, elements, species, budget, amounts, potentials, pressure, tracked):
    """Check every element, charge, and atomic formation equation."""
    n = np.exp(amounts)
    residual = abs(n@matrix.T-budget)
    positive = np.asarray(budget).reshape(-1, len(elements))[0] > 0
    references = [species.index("e1-" if e == "e-" else e+"1") for e in elements]
    logp = log_fractions(amounts)+np.log(pressure)[..., None]
    values = {
        "all_element_relative_residual": np.max(residual[..., positive]/budget[..., positive], axis=-1),
        "major_HCO_relative_residual": np.max(np.stack([
            residual[..., elements.index(e)]/budget[..., elements.index(e)] for e in ("H", "C", "O")]), axis=0),
        "mass_action_max_abs_ln": np.max(abs(logp-logp[..., references]@matrix+potentials), axis=-1),
    }
    for element in tracked:
        i = elements.index(element)
        values[element+"_relative_residual"] = residual[..., i]/budget[..., i]
    if "e-" in elements:
        charge = matrix[elements.index("e-")]
        values["charge_relative_residual"] = abs(n@charge)/(n@abs(charge))
    limits = {"all_element_relative_residual": 1e-5, "major_HCO_relative_residual": 1e-9,
              "mass_action_max_abs_ln": 1e-8, "charge_relative_residual": 1e-12}
    assert np.isfinite(amounts).all() and np.all(n > 0)
    for key, value in values.items():
        assert np.isfinite(value).all()
        if key in limits:
            assert np.max(value) < limits[key], (key, np.max(value))
    return values


def select_cases(logx, shares, targets, temperatures):
    """Choose allocation, relative-response and hot controls without trace floors."""
    effect = np.max(np.stack([
        abs(value[1:][..., targets]-value[0][..., targets]) for value in shares.values()]), axis=(0, 1, 3))
    highlight = int(np.argmax(effect))
    relative = abs(logx[1:][..., targets]-logx[0][..., targets])
    relative_case = int(np.unravel_index(np.argmax(relative), relative.shape)[1])
    hot = np.flatnonzero(temperatures == 4000.)
    hot_case = int(hot[np.argmax(effect[hot])])
    selected = {highlight, relative_case, hot_case}
    for i in (4, 49, 74):
        if len(selected) == 3:
            break
        selected.add(i)
    assert len(selected) == 3
    edge_cases = [highlight, hot_case]
    if highlight == hot_case:
        edge_cases[1] = next(i for i in sorted(selected) if i != highlight)
    return effect, relative, highlight, np.array(sorted(selected)), np.array(edge_cases)


def run(definition_path: Path, output: Path, executable: Path) -> None:
    """Solve150 fresh endpoints per variant plus explicit finite controls."""
    jax.config.update("jax_enable_x64", True)
    definition_path = definition_path.resolve()
    definition = json.loads(definition_path.read_text())
    assert definition["baseline_dataset"] == "curated-ch2-v1"
    assert definition["temperature_range_K"] == [1500, 4000]
    factory = importlib.import_module("exogibbs.presets.curated_ch2")
    baseline = ROOT/definition.get("baseline_archive", "tests/data/validation/curated_ch2_adoption/20261106")
    baseline_receipt = json.loads((baseline/"receipt.json").read_text())
    variants = definition["variants"]
    selected_variants = definition.get("selected_variants", [])
    assert variants and len({v["name"] for v in variants}) == len(variants)
    primary = definition.get("primary_variant", variants[0]["name"])
    primary_index = next(i+1 for i, v in enumerate(variants) if v["name"] == primary)
    tracked = definition["tracked_elements"]
    assert tracked and "e-" not in tracked
    for path, digest in definition["source_sha256"].items():
        assert file_hash(ROOT/path) == digest, path
    output.mkdir(parents=True, exist_ok=False)
    summaries, diagnostics, controls, independent = [], [], [], []
    for ions in (True, False):
        network = "ions" if ions else "neutral"
        path = baseline/f"{network}.npz"
        assert file_hash(path) == baseline_receipt["output_sha256"][path.name]
        with np.load(path) as archive:
            saved = {key: archive[key].copy() for key in archive.files}
        base = factory.chemsetup(ions=ions)
        assert base.metadata["dataset"] == definition["baseline_dataset"]
        species, elements, matrix = list(base.species), list(base.elements), np.asarray(base.formula_matrix)
        for value, key in ((species, "species"), (elements, "elements"), (matrix, "formula_matrix")):
            np.testing.assert_array_equal(value, saved[key])
        t, p, budget = (saved[key] for key in ("temperature_K", "pressure_bar", "budget"))
        assert len(t) == 75 and len(species) == (537 if ions else 421) and saved["converged"].all()
        assert np.array_equal(np.unique(t), [1500, 2000, 2500, 3000, 4000])
        for ratio in np.unique(saved["C_over_O"]):
            indices = np.flatnonzero(saved["C_over_O"] == ratio)
            expected = np.broadcast_to(carbon_budget(base, float(ratio)), budget[indices].shape)
            np.testing.assert_allclose(budget[indices], expected, rtol=1e-15, atol=0)
        models = [base]+[candidate_setup(base, v["replacements"], name=definition["id"]+"-"+v["name"]) for v in variants]
        potentials = np.asarray([model.hvector_func(t) for model in models])
        np.testing.assert_allclose(potentials[0], saved["h"], rtol=0, atol=1e-12)
        targets = sorted({s for v in variants+selected_variants for s in v["replacements"]})
        target_indices = [species.index(s) for s in targets]
        for i, v in enumerate(variants, 1):
            changed = [species.index(s) for s in v["replacements"]]
            np.testing.assert_array_equal(np.delete(potentials[i]-potentials[0], changed, axis=-1), 0)
        amounts = np.empty((len(models), *saved["ln_n"].shape))
        amounts[0] = saved["ln_n"]
        for vi, variant in enumerate(variants, 1):
            for ratio in np.unique(saved["C_over_O"]):
                indices = np.flatnonzero(saved["C_over_O"] == ratio)
                values, extra = solve_cases(models[vi], budget[indices[0]], t[indices], p[indices], PRIMARY_TOLERANCE)
                amounts[vi, indices] = values
                diagnostics.extend({"network": network, "variant": variant["name"], "case_index": int(i), **row} for i, row in zip(indices, extra))
            print(f"{definition['id']} {network} {variant['name']}:75 fresh cold endpoints passed", flush=True)
        checks = constraints(matrix, elements, species, budget, amounts, potentials, p, tracked)
        logx = log_fractions(amounts)
        x = np.exp(logx)
        shares = {e: np.exp(amounts)*matrix[elements.index(e)]/budget[:, elements.index(e), None] for e in tracked}
        # Compare allocation of all targets without depending on a trace target's
        # large relative response. Separate maximum ln ratios remain reported.
        allocation_effect, relative, highlight, selected, edge_cases = select_cases(logx, shares, target_indices, t)
        tight = np.empty((len(models), 3, len(species)))
        for vi, model in enumerate(models):
            for k, i in enumerate(selected):
                values, extra = solve_cases(model, budget[i], t[i:i+1], p[i:i+1], STRICT_TOLERANCE)
                tight[vi, k] = values[0]
                controls.extend({"network": network, "kind": "tight", "variant_index": vi, "case_index": int(i), **row} for row in extra)
        constraints(matrix, elements, species, budget[selected], tight, potentials[:, selected], p[selected], tracked)
        difference = log_fractions(tight)-logx[:, selected]
        visible = np.maximum(np.exp(log_fractions(tight)), x[:, selected]) >= 1e-12
        stability = {"all_max_abs_delta_ln_x": float(abs(difference).max()),
                     "targets_max_abs_delta_ln_x": float(abs(difference[..., target_indices]).max()),
                     "visible_max_abs_delta_ln_x": float(abs(difference[visible]).max()),
                     "max_absolute_delta_x": float(abs(np.exp(log_fractions(tight))-x[:, selected]).max())}
        for key, limit in zip(stability, (1e-4, 1e-8, 2e-6, 1e-9)):
            assert stability[key] < limit, (network, key, stability[key])
        fractions = np.linspace(0, 1, 5)
        path_amounts = []
        primary_variant = variants[primary_index-1]
        for fraction in fractions:
            model = candidate_setup(base, primary_variant["replacements"], float(fraction), definition["id"]+"-path")
            values, extra = solve_cases(model, budget[highlight], t[highlight:highlight+1], p[highlight:highlight+1], STRICT_TOLERANCE)
            path_amounts.append(values[0])
            controls.extend({"network": network, "kind": "path", "variant": primary, "lambda": float(fraction), "case_index": highlight, **row} for row in extra)
        path_amounts = np.asarray(path_amounts)
        path_h = potentials[0, highlight]+fractions[:, None]*(potentials[primary_index, highlight]-potentials[0, highlight])
        constraints(matrix, elements, species, np.broadcast_to(budget[highlight], (5, len(elements))), path_amounts, path_h, np.repeat(p[highlight], 5), tracked)
        # Joint candidates need not make each individual target monotonic; the
        # conjugate energy coordinate is monotonic for a finite linear shift.
        energy_direction = potentials[primary_index, highlight]-potentials[0, highlight]
        assert np.max(np.diff(np.exp(path_amounts)@energy_direction)) < 1e-10
        edge_amounts = np.empty((len(selected_variants), 2, len(species)))
        edge_h = np.empty_like(edge_amounts)
        edge_summary = []
        for vi, variant in enumerate(selected_variants):
            model = candidate_setup(base, variant["replacements"], name=definition["id"]+"-"+variant["name"])
            edge_h[vi] = np.asarray(model.hvector_func(t[edge_cases]))
            unchanged = np.delete(edge_h[vi]-potentials[0, edge_cases], [species.index(s) for s in variant["replacements"]], axis=-1)
            np.testing.assert_allclose(unchanged, 0., atol=1e-12, rtol=0)
            for k, i in enumerate(edge_cases):
                values, extra = solve_cases(model, budget[i], t[i:i+1], p[i:i+1], STRICT_TOLERANCE)
                edge_amounts[vi, k] = values[0]
                controls.extend({"network": network, "kind": "selected_variant", "variant": variant["name"], "case_index": int(i), **row} for row in extra)
                xx = np.exp(log_fractions(values))[0]
                edge_summary.append({"variant": variant["name"], "case_index": int(i),
                    "targets": {s: {"x": float(xx[j]), "x_over_current": float(xx[j]/x[0, i, j]),
                        "element_shares": {e: float(np.exp(values[0, j])*matrix[elements.index(e), j]/budget[i, elements.index(e)]) for e in tracked}}
                        for s, j in zip(targets, target_indices)}})
            constraints(matrix, elements, species, budget[edge_cases], edge_amounts[vi], edge_h[vi], p[edge_cases], tracked)
        pairs = [gibbs_ordering(amounts[[0, vi]], potentials[[0, vi]], p, matrix, elements, species) for vi in range(1, len(models))]
        raw = np.asarray([pair[0] for pair in pairs])
        corrected = np.asarray([pair[1] for pair in pairs])
        scale = np.asarray([100*np.finfo(np.longdouble).eps*np.max(np.sum(
            np.exp(amounts[[0, vi]])*abs(potentials[[0, vi]]+logx[[0, vi]]+np.log(p)[None, :, None]), axis=-1), axis=0) for vi in range(1, len(models))])
        resolved = (corrected[:, 0] > scale) & (corrected[:, 1] < -scale)
        if ions:
            exportable = np.flatnonzero(np.isin(t, [1500., 2000., 3000., 4000.]))
            fc_cases = {int(exportable[np.argmax(allocation_effect[exportable])]),
                        int(exportable[np.argmax(np.max(relative[:, exportable], axis=(0, 2)))])}
            for i in (4, 74):
                if len(fc_cases) == 2:
                    break
                fc_cases.add(i)
            for i in sorted(fc_cases):
                independent.append({"network": network, "case_index": i, "comparisons": independent_check(models, budget[i], float(t[i]), float(p[i]), amounts[:, i], executable.resolve())})

        def point(vi, i, j):
            return {"case_index": int(i), "temperature_K": float(t[i]), "pressure_bar": float(p[i]),
                    "C_over_O": float(saved["C_over_O"][i]), "species": species[j],
                    "x_before": float(x[0, i, j]), "x_after": float(x[vi, i, j]),
                    "ratio": float(np.exp(logx[vi, i, j]-logx[0, i, j])),
                    "delta_x": float(x[vi, i, j]-x[0, i, j]),
                    "element_shares": {e: [float(s[0, i, j]), float(s[vi, i, j])] for e, s in shares.items()}}

        responses = []
        for vi, variant in enumerate(variants, 1):
            response = {"variant": variant["name"], "targets": {}, "element_redistribution": {},
                        "largest_any_species_response": point(vi, *np.unravel_index(np.argmax(abs(x[vi]-x[0])), x[vi].shape)),
                        "resolved_Gibbs_cases": int(resolved[vi-1].sum())}
            for s, j in zip(targets, target_indices):
                ratio = np.exp(logx[vi, :, j]-logx[0, :, j])
                response["targets"][s] = {"x_ratio_min": float(ratio.min()), "x_ratio_max": float(ratio.max()),
                    "largest_absolute_response": point(vi, int(np.argmax(abs(x[vi, :, j]-x[0, :, j]))), j),
                    "maximum_element_share_before_after": {e: [float(v[0, :, j].max()), float(v[vi, :, j].max())] for e, v in shares.items()}}
            for e, values in shares.items():
                records = []
                for j in np.flatnonzero(matrix[elements.index(e)] > 0):
                    i = int(np.argmax(abs(values[vi, :, j]-values[0, :, j])))
                    records.append({**point(vi, i, j), "absolute_share_change": float(abs(values[vi, i, j]-values[0, i, j]))})
                response["element_redistribution"][e] = sorted(records, key=lambda r: r["absolute_share_change"], reverse=True)
            responses.append(response)
        summaries.append({"network": network, "species": len(species), "cases": 75,
                          "max_constraint_residuals": {k: float(v.max()) for k, v in checks.items()},
                          "tight_case_indices": selected.tolist(), "highlighted_case_index": highlight,
                          "tolerance_stability": stability, "responses": responses,
                          "selected_variant_controls": edge_summary})
        np.savez_compressed(output/f"{network}.npz", variants=["current"]+[v["name"] for v in variants],
            ln_n=amounts, h=potentials, species=species, elements=elements, formula_matrix=matrix,
            budget=budget, temperature_K=t, pressure_bar=p, C_over_O=saved["C_over_O"],
            converged=np.ones((len(models), 75), dtype=bool), tight_case_indices=selected, tight_ln_n=tight,
            path_variant_index=primary_index, path_case_index=highlight, lambda_values=fractions, lambda_ln_n=path_amounts,
            selected_variant_names=[v["name"] for v in selected_variants], selected_case_indices=edge_cases,
            selected_ln_n=edge_amounts, selected_h=edge_h,
            delta_g_raw=raw, delta_g_corrected=corrected, gibbs_numerical_scale=scale, gibbs_ordering_resolved=resolved)
    summary = {"definition": definition["id"], "baseline_dataset": definition["baseline_dataset"],
               "curated_update": "none", "new_endpoint_solves": len(diagnostics), "reused_baseline_cases": 150,
               "tighter_control_solves": sum(r["kind"] == "tight" for r in controls),
               "path_control_solves": sum(r["kind"] == "path" for r in controls),
               "selected_variant_solves": sum(r["kind"] == "selected_variant" for r in controls),
               "independent_fastchem_runs": sum(len(r["comparisons"]) for r in independent),
               "primary_tolerance": PRIMARY_TOLERANCE, "strict_tolerance": STRICT_TOLERANCE,
               "networks": summaries, "independent_solver": independent,
               "scope": "Fixed ideal-gas networks and budgets,1500-4000K; no production adoption or full physical uncertainty. Gibbs resolution is a floating-point diagnostic, not a rigorous bound or source ranking."}
    assert len(diagnostics) == 150*len(variants)
    for name, value in (("summary.json", summary), ("diagnostics.json", diagnostics), ("selected_diagnostics.json", controls)):
        (output/name).write_text(json.dumps(value, indent=2, allow_nan=False)+"\n")
    paths = {Path(__file__).resolve(), definition_path, baseline/"receipt.json", *(baseline/f"{n}.npz" for n in ("ions", "neutral"))}
    paths.update(ROOT/path for path in definition["source_sha256"])
    paths.update(Path(m.__file__).resolve() for m in tuple(sys.modules.values()) if getattr(m, "__file__", None)
                 and Path(m.__file__).resolve().is_relative_to(ROOT) and Path(m.__file__).suffix == ".py"
                 and Path(m.__file__).name not in ("curated.py", "exogibbs_version.py"))
    paths.update((ROOT/"src/exogibbs/data/curated").glob("*.json"))
    paths.update(ROOT/"src/exogibbs/data"/p for p in ("FastChem4/logK/logK.dat", "FastChem4/element_abundances/asplund_2021.dat"))
    receipt = {"created_utc": datetime.now(timezone.utc).isoformat(), "result": "passed", "command": [sys.executable, *sys.argv],
               "python": platform.python_version(), "numpy": np.__version__, "jax": jax.__version__,
               "dtype": "float64", "backend": jax.default_backend(),
               "selected_file_sha256": {str(p.relative_to(ROOT)): file_hash(p) for p in sorted(paths)},
               "output_sha256": {p.name: file_hash(p) for p in sorted(output.iterdir())},
               "fastchem_binary_sha256": file_hash(executable)}
    (output/"receipt.json").write_text(json.dumps(receipt, indent=2)+"\n")
    print(json.dumps({k: v for k, v in summary.items() if k not in ("networks", "independent_solver")}), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--definition", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--fastchem-executable", type=Path, required=True)
    args = parser.parse_args()
    run(args.definition, args.output_dir, args.fastchem_executable)
