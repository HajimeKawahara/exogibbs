"""Measure the bounded CO2 candidate on saved and hydrogen-poor gas cases."""

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

from exogibbs.presets import curated_tio as curated
from c2h_equilibrium_impact import independent_check, solve_cases
from gas_source_comparison import ROOT, file_hash
from tio_equilibrium_impact import diagnostics, gibbs
import co2_candidate as candidate

BASELINE = ROOT / "tests/data/validation/curated_tio_adoption/20261103/ions.npz"
DATA = ROOT / "tests/data/validation/co2_equilibrium_impact/20261104"
PRIMARY_TOLERANCE, STRICT_TOLERANCE = 1e-13, 5e-14


def candidate_setup(base, model: str = "nist626", fraction: float = 1., delta_ln_k: float = 0.):
    """Replace only CO2; an optional local lnK shift tests thermal sensitivity."""
    index = base.species.index("C1O2")
    def hvector(t):
        old = base.hvector_func(t)
        target = candidate.formation_h_jax(t, model)-delta_ln_k
        return old.at[..., index].set(old[..., index]+fraction*(target-old[..., index]))
    return replace(base, hvector_func=jax.jit(hvector), metadata={
        **base.metadata, "dataset": "review-co2-"+model, "production_adoption": False,
        "substitution_fraction": fraction, "local_delta_ln_K": delta_ln_k})


def fractions(log_amounts):
    """Normalize without abundance floors."""
    return log_amounts-logsumexp(log_amounts, axis=-1, keepdims=True)


def gibbs_ordering(amounts, h, pressure, matrix, elements, species):
    """Remove first-order feasibility shifts before same-model G comparisons."""
    raw = np.array([gibbs(amounts[1], h[i], pressure)-gibbs(amounts[0], h[i], pressure)
                    for i in (0, 1)])
    logs = amounts.astype(np.longdouble)
    n = np.exp(logs)
    logp = logs-np.log(n.sum(axis=-1))[..., None]+np.log(pressure.astype(np.longdouble))[None, :, None]
    refs = [species.index("e1-" if e == "e-" else e+"1") for e in elements]
    potentials = logp+h.astype(np.longdouble)
    difference = (n[1]-n[0])@matrix.T
    correction = (difference[None, :, :]*potentials[..., refs]).sum(axis=-1)
    corrected = raw-correction
    assert corrected[0].min() >= -5e-14 and corrected[1].max() <= 5e-14
    return np.asarray(raw, float), np.asarray(corrected, float)


def summary_points(amounts, budget, matrix, elements, species, temperature, pressure):
    """Describe both relative CO2 response and absolute/allocation consequences."""
    logx = fractions(amounts)
    x = np.exp(logx)
    col = species.index("C1O2")
    ratio = np.exp(logx[1, :, col]-logx[0, :, col])
    dx = x[1]-x[0]
    def point(i, j):
        return {"case_index": int(i), "temperature_K": float(temperature[i]), "pressure_bar": float(pressure[i]),
                "species": species[j], "x_before": float(x[0, i, j]), "x_after": float(x[1, i, j]),
                "delta_x": float(dx[i, j]), "ratio": float(np.exp(logx[1, i, j]-logx[0, i, j])),
                "element_shares": {e: [float(np.exp(amounts[v, i, j])*matrix[elements.index(e), j]/budget[i, elements.index(e)])
                                         for v in (0, 1)] for e in ("C", "O")}}
    highlight = int(np.argmax(abs(dx[:, col])))
    overall = np.unravel_index(np.argmax(abs(dx)), dx.shape)
    return {"CO2_ratio_min": float(ratio.min()), "CO2_ratio_max": float(ratio.max()),
            "CO2_maximum_absolute_response": point(highlight, col),
            "largest_any_species_absolute_response": point(*overall),
            "selected_species_extrema": [point(int(np.argmax(abs(dx[:, species.index(s)]))), species.index(s))
                                          for s in ("C1O2", "C1O1", "H2O1", "O1", "O2", "H2", "H1O1", "C1H4", "e1-")]}


def run(output: Path, executable: Path | None = None) -> None:
    """Execute cold endpoints and independent invariant/scope checks."""
    jax.config.update("jax_enable_x64", True)
    output.mkdir(parents=True, exist_ok=False)
    with np.load(BASELINE) as archive:
        saved = {k: archive[k].copy() for k in archive.files}
    base = curated.chemsetup(ions=True)
    models = [base, candidate_setup(base)]
    species, elements = list(base.species), list(base.elements)
    matrix = np.array(base.formula_matrix)
    for actual, key in ((species, "species"), (elements, "elements"), (matrix, "formula_matrix")):
        np.testing.assert_array_equal(actual, saved[key])
    t, p, budget = saved["temperature_K"], saved["pressure_bar"], saved["budget"]
    h = np.asarray([m.hvector_func(t) for m in models])
    np.testing.assert_allclose(h[0], saved["h"], atol=1e-12, rtol=0)
    assert saved["converged"].all() and h.shape == (2, 75, 537)
    col = species.index("C1O2")
    np.testing.assert_array_equal(np.delete(h[1]-h[0], col, axis=-1), 0)
    endpoint = np.empty_like(saved["ln_n"])
    diagnostics_rows = []
    for ratio in np.unique(saved["C_over_O"]):
        indices = np.flatnonzero(saved["C_over_O"] == ratio)
        np.testing.assert_array_equal(budget[indices], np.broadcast_to(budget[indices[0]], budget[indices].shape))
        values, rows = solve_cases(models[1], budget[indices[0]], t[indices], p[indices], PRIMARY_TOLERANCE)
        endpoint[indices] = values
        diagnostics_rows.extend({"domain": "original", "variant": "nist626", "case_index": int(i), **r} for i, r in zip(indices, rows))
    amounts = np.array([saved["ln_n"], endpoint])
    checks = diagnostics(matrix, elements, species, budget, amounts, h, p)
    summary = summary_points(amounts, budget, matrix, elements, species, t, p)
    print("CO2: 75 candidate cold endpoints passed", flush=True)

    # Idealized hydrogen-poor oxidized gas, not a fitted planetary atmosphere.
    rich_b = np.array(base.element_vector_reference)
    rich_b /= rich_b[elements.index("O")]
    rich_b[elements.index("C")] = .5
    rich_b[elements.index("H")] = .01
    rich_b[elements.index("He")] = .001
    rich_b /= rich_b.sum()
    rich_t, rich_p = np.repeat([1500., 2500., 4000.], 2), np.tile([.1, 100.], 3)
    rich_budget = np.broadcast_to(rich_b, (6, len(rich_b))).copy()
    rich_h = np.asarray([m.hvector_func(rich_t) for m in models])
    rich_amounts = []
    for name, model in zip(("current", "nist626"), models):
        values, rows = solve_cases(model, rich_b, rich_t, rich_p, PRIMARY_TOLERANCE)
        rich_amounts.append(values)
        diagnostics_rows.extend({"domain": "hydrogen_poor", "variant": name, "case_index": int(i), **r} for i, r in enumerate(rows))
    rich_amounts = np.array(rich_amounts)
    rich_checks = diagnostics(matrix, elements, species, rich_budget, rich_amounts, rich_h, rich_p)
    rich_summary = summary_points(rich_amounts, rich_budget, matrix, elements, species, rich_t, rich_p)
    print("CO2: 12 hydrogen-poor cold endpoints passed", flush=True)

    highlight = summary["CO2_maximum_absolute_response"]["case_index"]
    relative = int(np.argmax(abs(fractions(amounts)[1, :, col]-fractions(amounts)[0, :, col])))
    selected = sorted({highlight, relative, int(np.argmax(checks["element_relative_residual"][1]))})
    rich_selected = sorted({rich_summary["CO2_maximum_absolute_response"]["case_index"], 5})
    tight_data, tight_rows = {}, []
    for label, indices, tt, pp, bb, initial in (("original", selected, t, p, budget, amounts),
                                               ("hydrogen_poor", rich_selected, rich_t, rich_p, rich_budget, rich_amounts)):
        tight = []
        for variant, model in enumerate(models):
            values = []
            for i in indices:
                solved, rows = solve_cases(model, bb[i], tt[i:i+1], pp[i:i+1], STRICT_TOLERANCE)
                values.append(solved[0])
                tight_rows.extend({"domain": label, "variant": variant, "case_index": i, **r} for r in rows)
            tight.append(values)
        tight = np.array(tight)
        delta = fractions(tight)-fractions(initial[:, indices])
        tight_data[label] = {"indices": indices, "ln_n": tight, "max_abs_delta_ln_x": float(abs(delta).max()),
                            "CO2_max_abs_delta_ln_x": float(abs(delta[..., col]).max()),
                            "max_abs_delta_x": float(abs(np.exp(fractions(tight))-np.exp(fractions(initial[:, indices]))).max())}
        assert abs(delta).max() < 1e-4 and abs(delta[..., col]).max() < 1e-8

    # Fixed-isotope thermal scope and quoted Q uncertainty at relevant cases.
    sensitivity, sensitivity_rows = [], []
    nodes = json.loads((candidate.DATA/"nodes.json").read_text())["nist626"]
    sensitivities = [("original", highlight, t, p, budget),
                     ("hydrogen_poor", 1, rich_t, rich_p, rich_budget),
                     ("hydrogen_poor", 5, rich_t, rich_p, rich_budget)]
    for domain, i, tt, pp, bb in sensitivities:
        u = next(r["relative_standard_u_Q"] for r in nodes if r["temperature_K"] == tt[i])
        for tag, model in (("isotope_average", candidate_setup(base, "isotope_average")),
                           ("Q_minus_1u", candidate_setup(base, delta_ln_k=float(np.log1p(-u)))),
                           ("Q_plus_1u", candidate_setup(base, delta_ln_k=float(np.log1p(u))))):
            values, rows = solve_cases(model, bb[i], tt[i:i+1], pp[i:i+1], STRICT_TOLERANCE)
            sensitivity.append({"domain": domain, "case_index": i, "variant": tag,
                                "temperature_K": float(tt[i]), "pressure_bar": float(pp[i]),
                                "relative_u_Q": u, "ln_n": values[0].tolist(),
                                "h": np.asarray(model.hvector_func(tt[i])).tolist(), "budget": bb[i].tolist()})
            sensitivity_rows.extend({"domain": domain, "case_index": i, "variant": tag, **r} for r in rows)
    path_case = rich_summary["CO2_maximum_absolute_response"]["case_index"]
    lambdas, path, path_rows = np.linspace(0, 1, 5), [], []
    for fraction in lambdas:
        model = candidate_setup(base, fraction=float(fraction))
        values, rows = solve_cases(model, rich_b, rich_t[path_case:path_case+1], rich_p[path_case:path_case+1], STRICT_TOLERANCE)
        path.append(values[0]);path_rows.extend({"lambda": fraction, **r} for r in rows)
    raw, corrected = gibbs_ordering(amounts, h, p, matrix, elements, species)
    rich_raw, rich_corrected = gibbs_ordering(rich_amounts, rich_h, rich_p, matrix, elements, species)
    allowed = np.flatnonzero(np.isin(rich_t, [1500., 4000.]))
    independent_case = int(allowed[np.argmax(abs(np.exp(fractions(rich_amounts))[1, allowed, col]-np.exp(fractions(rich_amounts))[0, allowed, col]))])
    independent = independent_check(models, rich_b, float(rich_t[independent_case]), float(rich_p[independent_case]),
                                    rich_amounts[:, independent_case], executable) if executable else "not_run"
    for name, values in (("original", amounts), ("hydrogen_poor", rich_amounts)):
        np.savez_compressed(output/(name+".npz"), ln_n=values, h=h if name=="original" else rich_h,
            budget=budget if name=="original" else rich_budget, formula_matrix=matrix, species=species, elements=elements,
            temperature_K=t if name=="original" else rich_t, pressure_bar=p if name=="original" else rich_p,
            tight_case_indices=tight_data[name]["indices"], tight_ln_n=tight_data[name]["ln_n"],
            raw_gibbs_exchange=raw if name=="original" else rich_raw,
            corrected_gibbs_exchange=corrected if name=="original" else rich_corrected)
    np.savez_compressed(output/"path.npz", lambda_values=lambdas, ln_n=path, case_index=path_case)
    output_summary = {"original": summary, "hydrogen_poor": rich_summary,
        "reused_baseline_cases": 75, "new_endpoint_solves": 87, "new_tight_solves": len(tight_rows),
        "local_sensitivity_solves": len(sensitivity_rows), "path_solves": 5,
        "primary_tolerance": PRIMARY_TOLERANCE, "strict_tolerance": STRICT_TOLERANCE,
        "constraints": {label: {k: float(v.max()) for k, v in check.items()} for label, check in (("original", checks), ("hydrogen_poor", rich_checks))},
        "tight_stability": {name: {k:v for k,v in d.items() if k!="ln_n"} for name,d in tight_data.items()},
        "independent_solver": independent, "independent_case_index": independent_case,
        "isotope_and_uncertainty": "Selected local diagnostics only: fixed-isotope ground-zero thermal average and +/-one standard Q uncertainty. No complete natural-isotope reaction or physical error band.",
        "hydrogen_poor_definition": "C/O=.5,H/O=.01,He/O=.001; every other heavy-element/O ratio retained from current preset; same537species and charge closure. Idealized controlled gas, not a planetary reconstruction.",
        "curated_update": "none"}
    for name, value in (("summary.json", output_summary), ("local_sensitivities.json", sensitivity),
                        ("diagnostics.json", {"endpoints": diagnostics_rows, "tight": tight_rows, "sensitivity": sensitivity_rows, "path": path_rows})):
        (output/name).write_text(json.dumps(value, indent=2, allow_nan=False)+"\n")
    paths = [Path(__file__), Path(candidate.__file__), BASELINE, ROOT/"mceg/reviews/tio_candidate.py",
             ROOT/"mceg/reviews/tio_equilibrium_impact.py", ROOT/"mceg/reviews/c2h_equilibrium_impact.py",
             ROOT/"examples/comparisons/compare_gas_thermochemistry.py", ROOT/"examples/comparisons/select_major_molecules.py",
             ROOT/"examples/comparisons/audit_atomic_ions.py", ROOT/"examples/comparisons/data/major_molecule_reference.json"]
    paths += list(candidate.DATA.glob("*.json"))
    paths += [ROOT/p for p in ("src/exogibbs/presets/curated_tio.py", "src/exogibbs/presets/curated_c2h.py", "src/exogibbs/presets/curated_v1.py", "src/exogibbs/presets/fastchem4.py", "src/exogibbs/thermo/standard.py", "src/exogibbs/thermo/models.py", "src/exogibbs/data/curated/gas_v1.json", "src/exogibbs/data/curated/c2h_v1.json", "src/exogibbs/data/curated/tio_v1.json", "src/exogibbs/data/FastChem4/logK/logK.dat", "src/exogibbs/data/FastChem4/element_abundances/asplund_2021.dat")]
    paths += sorted((ROOT/"src/exogibbs/equilibrium/gas").rglob("*.py"))
    receipt = {"created_utc": datetime.now(timezone.utc).isoformat(), "python": platform.python_version(),
        "jax": jax.__version__, "numpy": np.__version__, "backend": jax.default_backend(), "dtype": "float64",
        "command": " ".join(sys.argv), "result": "passed",
        "selected_file_sha256": {str(p.relative_to(ROOT)): file_hash(p) for p in paths},
        "output_sha256": {p.name: file_hash(p) for p in sorted(output.iterdir())},
        "fastchem_binary_sha256": file_hash(executable) if executable else None}
    (output/"receipt.json").write_text(json.dumps(receipt, indent=2)+"\n")
    print(json.dumps(output_summary, indent=2), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=DATA)
    parser.add_argument("--fastchem-executable", type=Path)
    args = parser.parse_args()
    run(args.output_dir, args.fastchem_executable)
