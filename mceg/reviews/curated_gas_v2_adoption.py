"""Validate the actual combined curated gas v2 factory against frozen candidates."""

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import platform
import sys

import jax
import numpy as np

from c2h_equilibrium_impact import carbon_budget, independent_check
from co2_equilibrium_impact import gibbs_ordering
from curated_energy_adoption import archive
from curated_h2o_adoption import solve_cases
from gas_source_comparison import ROOT, file_hash
from pending_candidate import R, candidate_setup, reaction_properties
from pending_equilibrium_impact import constraints, log_fractions, select_cases

DATA = ROOT / "tests/data/validation/curated_gas_v2_adoption/20261106"
BASELINE = ROOT / "tests/data/validation/curated_ch2_adoption/20261106"


def selected_specs() -> tuple[dict, set]:
    """Assemble reviewed choices independently of the new packaged payload."""
    selection = json.loads((DATA / "selection.json").read_text())
    replacements, paths = {}, {DATA / "selection.json"}
    for row in selection["selection"]:
        path = ROOT / row["path"]
        assert file_hash(path) == row["sha256"], path
        definition = json.loads(path.read_text())
        variant = next(v for v in definition["variants"] if v["name"] == row["variant"])
        for species, spec in variant["replacements"].items():
            if species in replacements:
                assert replacements[species] == spec, species
            replacements[species] = spec
        paths.add(path)
    assert len(selection["selection"]) == 27 and len(replacements) == 28
    return replacements, paths


def dense_identity(actual, base, specs: dict) -> dict:
    """Check all columns and derivative calorics at dense and off-grid nodes."""
    temperatures = np.unique(np.r_[np.arange(1500., 4001.), np.linspace(1500.137, 3999.813, 251)])
    reference = candidate_setup(base, specs, name="combined-production-reference")
    targets = [actual.species.index(s) for s in specs]
    h = np.asarray(actual.hvector_func(temperatures))
    expected = np.asarray(reference.hvector_func(temperatures))
    error = float(abs(h - expected).max())
    assert error < 2e-12
    np.testing.assert_array_equal(np.delete(h - np.asarray(base.hvector_func(temperatures)), targets, axis=-1), 0)
    potential = lambda t: R * t * actual.hvector_func(t)[np.asarray(targets)]
    g = np.asarray(jax.jit(jax.vmap(potential))(temperatures))
    dg = np.asarray(jax.jit(jax.vmap(jax.jacfwd(potential)))(temperatures))
    ddg = np.asarray(jax.jit(jax.vmap(jax.jacfwd(jax.jacfwd(potential))))(temperatures))
    properties = np.stack((g, g-temperatures[:, None]*dg, -dg, -temperatures[:, None]*ddg), axis=-1)
    source = np.array([[list(reaction_properties(t, spec).values()) for spec in specs.values()] for t in temperatures])
    errors = abs(properties-source).max(axis=(0, 1))
    assert np.all(errors < [2e-8, 2e-8, 2e-10, 2e-10]), errors
    return {"temperature_count": len(temperatures), "offgrid_count": 251,
            "max_abs_h_difference": error, "GHSCp_max_abs_errors": errors.tolist()}


def response_summary(species, elements, matrix, budget, t, p, ratios, amounts, specs):
    """Report combined responses and all-element allocation, without trace floors."""
    logx = log_fractions(amounts)
    x = np.exp(logx)
    tracked = [e for e in elements if e != "e-"]
    shares = {e: np.exp(amounts)*matrix[elements.index(e)]/budget[:, elements.index(e), None] for e in tracked}

    def point(i, j):
        return {"case_index": int(i), "temperature_K": float(t[i]), "pressure_bar": float(p[i]),
                "C_over_O": float(ratios[i]), "species": species[j],
                "x_before": float(x[0, i, j]), "x_after": float(x[1, i, j]),
                "ratio": float(np.exp(logx[1, i, j]-logx[0, i, j])),
                "delta_x": float(x[1, i, j]-x[0, i, j]),
                "element_shares": {e: [float(v[0, i, j]), float(v[1, i, j])] for e, v in shares.items() if matrix[elements.index(e), j] > 0}}

    targets = {}
    for s in specs:
        j = species.index(s)
        ratio = np.exp(logx[1, :, j]-logx[0, :, j])
        targets[s] = {"x_ratio_min": float(ratio.min()), "x_ratio_max": float(ratio.max()),
            "largest_absolute_response": point(int(np.argmax(abs(x[1, :, j]-x[0, :, j]))), j),
            "maximum_element_share_before_after": {e: [float(v[0, :, j].max()), float(v[1, :, j].max())] for e, v in shares.items() if matrix[elements.index(e), j] > 0}}
    redistribution = {}
    for e, values in shares.items():
        rows = []
        for j in np.flatnonzero(matrix[elements.index(e)] > 0):
            i = int(np.argmax(abs(values[1, :, j]-values[0, :, j])))
            rows.append({**point(i, j), "absolute_share_change": float(abs(values[1, i, j]-values[0, i, j]))})
        redistribution[e] = sorted(rows, key=lambda r: r["absolute_share_change"], reverse=True)[:5]
    return {"targets": targets, "largest_any_species_response": point(*np.unravel_index(np.argmax(abs(x[1]-x[0])), x[1].shape)),
            "largest_five_allocation_changes_by_element": redistribution}, shares


def run(output: Path, executable: Path) -> None:
    """Solve 150 actual production endpoints, six references and ten joint controls."""
    from exogibbs.presets import curated_ch2, curated_v2
    # Import the independent solver's local code before freezing execution inputs.
    from benchmarks.fastchem4 import comparison, fastchem_executable  # noqa: F401

    jax.config.update("jax_enable_x64", True)
    specs, paths = selected_specs()
    models = [(curated_v2.chemsetup(ions=ions), curated_ch2.chemsetup(ions=ions)) for ions in (True, False)]
    dense = [dense_identity(actual, base, specs) for actual, base in models]
    paths.update((Path(__file__).resolve(), ROOT / "mceg/reviews/build_gas_v2_adoption.py", BASELINE / "receipt.json",
                  *(BASELINE / f"{n}.npz" for n in ("ions", "neutral"))))
    paths.update(Path(m.__file__).resolve() for m in tuple(sys.modules.values()) if getattr(m, "__file__", None)
                 and Path(m.__file__).resolve().is_relative_to(ROOT) and Path(m.__file__).suffix == ".py"
                 and Path(m.__file__).name not in ("curated.py", "exogibbs_version.py"))
    paths.update((ROOT / "src/exogibbs/data/curated").glob("*.json"))
    paths.update(ROOT / "src/exogibbs/data" / p for p in ("FastChem4/logK/logK.dat", "FastChem4/element_abundances/asplund_2021.dat"))
    pins = {str(p.relative_to(ROOT)): file_hash(p) for p in sorted(paths)}
    started = datetime.now(timezone.utc).isoformat()
    output.mkdir(parents=True, exist_ok=False)
    summaries, diagnostics, controls, independent = [], [], [], []
    for ni, (actual, base) in enumerate(models):
        network = "ions" if ni == 0 else "neutral"
        saved = archive(BASELINE, network)
        assert (actual.metadata["dataset"], base.metadata["dataset"]) == ("curated-gas-v2", "curated-ch2-v1")
        species, elements = list(actual.species), list(actual.elements)
        matrix = np.asarray(actual.formula_matrix)
        assert actual.species == base.species and actual.elements == base.elements
        np.testing.assert_array_equal(actual.element_vector_reference, base.element_vector_reference)
        for key, value in (("species", species), ("elements", elements), ("formula_matrix", matrix)):
            np.testing.assert_array_equal(saved[key], value)
        t, p, budget = (saved[k] for k in ("temperature_K", "pressure_bar", "budget"))
        assert len(t) == 75 and len(species) == (537 if ni == 0 else 421) and saved["converged"].all()
        h = np.asarray(actual.hvector_func(t))
        np.testing.assert_allclose(base.hvector_func(t), saved["h"], atol=1e-12, rtol=0)
        targets = [species.index(s) for s in specs]
        np.testing.assert_array_equal(np.delete(h-saved["h"], targets, axis=-1), 0)
        amounts = np.empty_like(saved["ln_n"])
        for ratio in np.unique(saved["C_over_O"]):
            indices = np.flatnonzero(saved["C_over_O"] == ratio)
            np.testing.assert_allclose(budget[indices], np.broadcast_to(carbon_budget(base, float(ratio)), budget[indices].shape), atol=0, rtol=1e-15)
            values, checks = solve_cases(actual, budget[indices[0]], t[indices], p[indices], 5e-14)
            amounts[indices] = values
            diagnostics.extend({"network": network, "case_index": int(i), **r} for i, r in zip(indices, checks))
            print(f"curated-gas-v2 {network} C/O={ratio}:25 actual production endpoints passed", flush=True)
        pair_n, pair_h = np.stack((saved["ln_n"], amounts)), np.stack((saved["h"], h))
        tracked = [e for e in elements if e != "e-"]
        closure = constraints(matrix, elements, species, budget, pair_n, pair_h, p, tracked)
        response, shares = response_summary(species, elements, matrix, budget, t, p, saved["C_over_O"], pair_n, specs)
        logx = log_fractions(pair_n)
        effect, relative, highlight, selected, _ = select_cases(logx, shares, targets, t)
        reference = candidate_setup(base, specs, name="combined-production-reference")
        tight = []
        for i in selected:
            values, checks = solve_cases(reference, budget[i], t[i:i+1], p[i:i+1], 2e-14)
            tight.append(values[0])
            controls.extend({"network": network, "kind": "tight_reference", "case_index": int(i), **r} for r in checks)
        tight = np.asarray(tight)
        constraints(matrix, elements, species, budget[selected], tight, h[selected], p[selected], tracked)
        delta = log_fractions(tight)-logx[1, selected]
        stability = {"max_abs_delta_ln_x": float(abs(delta).max()), "targets_max_abs_delta_ln_x": float(abs(delta[:, targets]).max()),
                     "max_absolute_delta_x": float(abs(np.exp(log_fractions(tight))-np.exp(logx[1, selected])).max())}
        assert all(v < limit for v, limit in zip(stability.values(), (1e-4, 1e-8, 1e-9))), stability
        fractions, path_n = np.linspace(0, 1, 5), []
        for fraction in fractions:
            model = candidate_setup(base, specs, float(fraction), "combined-joint-path")
            values, checks = solve_cases(model, budget[highlight], t[highlight:highlight+1], p[highlight:highlight+1], 2e-14)
            path_n.append(values[0])
            controls.extend({"network": network, "kind": "joint_path", "case_index": highlight, "lambda": float(fraction), **r} for r in checks)
        path_n = np.asarray(path_n)
        direction = pair_h[1, highlight]-pair_h[0, highlight]
        path_h = pair_h[0, highlight]+fractions[:, None]*direction
        constraints(matrix, elements, species, np.broadcast_to(budget[highlight], (5, len(elements))), path_n, path_h, np.repeat(p[highlight], 5), tracked)
        coordinate = np.exp(path_n)@direction
        assert np.diff(coordinate).max() < 1e-10
        raw, corrected = gibbs_ordering(pair_n, pair_h, p, matrix, elements, species)
        scale = 100*np.finfo(np.longdouble).eps*np.max(np.sum(np.exp(pair_n)*abs(pair_h+logx+np.log(p)[None, :, None]), axis=-1), axis=0)
        resolved = (corrected[0] > scale) & (corrected[1] < -scale)
        if ni == 0:
            exportable = np.flatnonzero(np.isin(t, [1500., 2000., 3000., 4000.]))
            chosen = {int(exportable[np.argmax(effect[exportable])]), int(exportable[np.argmax(np.max(relative[:, exportable], axis=(0, 2)))])}
            for i in (4, 74):
                if len(chosen) == 2:
                    break
                chosen.add(i)
            for i in sorted(chosen):
                independent.append({"network": network, "case_index": i, "comparisons": independent_check([actual], budget[i], float(t[i]), float(p[i]), amounts[i:i+1], executable.resolve())})
        summaries.append({"network": network, "species": len(species), "cases": 75, "dense_identity": dense[ni],
            "closure": {k: float(v.max()) for k, v in closure.items()}, "tight_case_indices": selected.tolist(), "tighter_reference": stability,
            "joint_path_case_index": highlight, "joint_path_coordinate": coordinate.tolist(), "resolved_Gibbs_cases": int(resolved.sum()), "combined_response": response})
        np.savez_compressed(output/f"{network}.npz", ln_n=amounts, h=h, species=species, elements=elements, formula_matrix=matrix,
            budget=budget, temperature_K=t, pressure_bar=p, C_over_O=saved["C_over_O"], converged=np.ones(75, dtype=bool),
            tight_case_indices=selected, tight_reference_ln_n=tight, path_case_index=highlight, lambda_values=fractions, lambda_ln_n=path_n,
            delta_g_raw=raw, delta_g_corrected=corrected, gibbs_numerical_scale=scale, gibbs_ordering_resolved=resolved)
        print(f"curated-gas-v2 {network}: production, tighter references and joint path passed", flush=True)
    summary = {"dataset": "curated-gas-v2", "base_dataset": "curated-ch2-v1", "targets": sorted(specs), "review_count": 27,
        "new_production_solves": len(diagnostics), "reused_baseline_cases": 150, "tighter_reference_solves": 6, "joint_path_solves": 10,
        "independent_fastchem_runs": sum(len(r["comparisons"]) for r in independent), "primary_tolerance": 5e-14, "strict_tolerance": 2e-14,
        "networks": summaries, "independent_solver": independent,
        "scope": "Actual combined production factory on two fixed ideal-gas networks and budgets,1500-4000K. These cumulative responses are distinct from the prior7350 isolated candidate endpoints. Finite Gibbs and joint-path checks are numerical diagnostics, not rigorous bounds or complete physical accuracy."}
    for filename, value in (("summary.json", summary), ("diagnostics.json", diagnostics), ("controls.json", controls)):
        (output/filename).write_text(json.dumps(value, indent=2, allow_nan=False)+"\n")
    assert all(file_hash(ROOT/p) == digest for p, digest in pins.items()), "Execution input changed after freeze"
    receipt = {"started_utc": started, "completed_utc": datetime.now(timezone.utc).isoformat(), "result": "passed",
        "command": [sys.executable, *sys.argv], "python": platform.python_version(), "numpy": np.__version__, "jax": jax.__version__,
        "backend": jax.default_backend(), "dtype": "float64", "cpu_affinity": sorted(os.sched_getaffinity(0)),
        "selected_file_sha256": pins, "fastchem_binary_sha256": file_hash(executable),
        "output_sha256": {p.name: file_hash(p) for p in sorted(output.iterdir())}}
    (output/"receipt.json").write_text(json.dumps(receipt, indent=2)+"\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--fastchem-executable", type=Path, required=True)
    args = parser.parse_args()
    run(args.output_dir, args.fastchem_executable)
