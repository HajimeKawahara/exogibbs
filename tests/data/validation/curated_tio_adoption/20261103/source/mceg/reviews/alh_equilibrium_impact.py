"""Diagnose AlH source sensitivity without adopting new runtime constants."""

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
from gas_source_comparison import DEFAULT_INPUT, file_hash, load_inputs
from exogibbs.presets import curated
from exogibbs.thermo.standard import _nasa9


TEMPERATURES = np.array([1500., 2000., 2500., 3000., 4000.])
PRESSURES = np.array([1e-6, 1e-4, .01, 1., 100.])
PRIMARY_TOLERANCE = 1e-13
STRICT_TOLERANCE = 5e-14
TRACKED = ("Al1H1", "Al1", "Al1O1", "Al1H1O1_oalh", "Al1H1O1_aloh", "Al1+", "Al1-",
           "Al2", "Al1N1", "C1Al1", "H2P1", "H3P1", "P1", "N1P1", "O1P1", "O2P1", "O6P4",
           "C1H1N1_hcn", "C2H2", "C1H4", "C1O1", "N2", "H3N1", "H2O1", "e1-")
PRIOR_BASELINE = ROOT / "tests/data/validation/curated_default_adoption/20261102/ions.npz"
SOURCE_COMPARISON = ROOT / "tests/data/validation/gas_source_comparison/baseline-001/comparison.json"


def prioritization() -> dict:
    """Screen saved abundances before treating source-discrepancy tiers as priorities."""
    comparison = json.loads(SOURCE_COMPARISON.read_text())
    candidates = (("O2P1", "P1"), ("H2Na2O2", "P1"), ("S5", "P1"), ("N3", "P1"),
                  ("Al1H1", "P2"), ("C3", "P2"), ("O1V1", "P2"), ("O1Si1", "P2"),
                  ("O1Ti1", "P2"), ("C1S1", "P2"), ("H2S1", "P2"))
    rows = []
    with np.load(PRIOR_BASELINE) as data:
        species, elements = data["species"].tolist(), data["elements"].tolist()
        n, x = np.exp(data["ln_n"]), np.exp(data["ln_x"])
        a, budget = data["formula_matrix"], data["budget"]

        def case(index):
            return {key: float(data[key][index]) for key in ("temperature_K", "pressure_bar", "C_over_O")}

        for name, tier in candidates:
            j = species.index(name)
            present = np.flatnonzero(a[:, j] > 0)
            limiting = present[np.argmin(budget[:, present]/a[present, j], axis=1)]
            # These candidate compositions keep the same limiting element over this grid.
            if not np.all(limiting == limiting[0]):
                raise ValueError("Candidate limiting element changes with composition: " + name)
            e = int(limiting[0])
            shares = n[:, j]*a[e, j]/budget[:, e]
            source = next(row for row in comparison["species"] if row["id"] == name)
            ratios = {str(int(row["temperature_K"])): float(np.exp(row["ln_Kp"]["curated"]-row["ln_Kp"]["nasa"]))
                      for row in comparison["samples"] if row["species_id"] == name and row["temperature_K"] in (1500, 3000)}
            rows.append({"species": name, "molecule": source["label"], "screening_tier": tier,
                         "curated_review_status": source["curated_provenance"]["status"],
                         "limiting_element": elements[e], "max_mole_fraction": float(x[:, j].max()),
                         "max_mole_fraction_case": case(int(np.argmax(x[:, j]))),
                         "max_element_fraction": float(shares.max()),
                         "max_element_fraction_case": case(int(np.argmax(shares))),
                         "curated_over_nasa_K": ratios})
    return {"dataset": "curated-c2h-v1", "cases": 75, "species": 537,
            "input_sha256": {str(path.relative_to(ROOT)): file_hash(path) for path in (PRIOR_BASELINE, SOURCE_COMPARISON)},
            "scope": "Selected candidate screen from saved equilibria; not an exhaustive physical importance ranking or a substitution calculation.",
            "selection": "AlH", "rationale": "AlH can hold about half of the Al budget and retains a provisional source choice. PO2 has the largest P1 candidate share here but holds only about 0.0743 percent of P; other unreviewed P1 candidates are far smaller. C3 remains a useful next candidate. VO and other major carriers already have reviewed source choices that must not be reset solely because they differ from NASA.",
            "candidates": rows}


def models() -> tuple:
    """Keep the full curated setup and replace only the balanced AlH reaction."""
    base = curated.chemsetup()
    inputs = load_inputs()
    coefficients = {}
    for name, composition in (("ALH", {"Al": 1, "H": 1}), ("AL", {"Al": 1}), ("H", {"H": 1})):
        record = inputs["nasa9_records"][name]
        if record["composition"] != composition:
            raise ValueError("Unexpected NASA species identity: " + name)
        coefficients[name] = np.array(next(row["coefficients"] for row in record["intervals"]
                                            if row["bounds_K"] == [1000., 6000.]))
    reaction = coefficients["ALH"] - coefficients["AL"] - coefficients["H"]
    index = base.species.index("Al1H1")
    coefficient_array = jnp.asarray(reaction[None, :], dtype=base.formula_matrix.dtype)

    def hvector(temperature):
        t = jnp.asarray(temperature, dtype=jnp.result_type(temperature, coefficient_array))
        valid = jnp.isfinite(t) & (t >= 1500) & (t <= 4000)
        value = _nasa9(jnp.where(valid, t, 1500), coefficient_array)[0][..., 0]
        return base.hvector_func(t).at[..., index].set(jnp.where(valid, value, jnp.nan))

    probe = replace(base, hvector_func=jax.jit(hvector), metadata={**base.metadata,
                    "dataset": "curated-c2h-v1+raw-nasa-alh-diagnostic",
                    "interpretation": "AlH-only source sensitivity; not a scientific adoption or accuracy ranking."})
    return base, probe, reaction


def run(output: Path, executable: Path | None = None) -> None:
    """Save both complete endpoints and strict cold restarts for all 75 pairs."""
    jax.config.update("jax_enable_x64", True)
    output.mkdir(parents=True, exist_ok=False)
    base, probe, reaction = models()
    pair = (base, probe)
    c, o = (base.elements.index(e) for e in ("C", "O"))
    ratios = [float(base.element_vector_reference[c]/base.element_vector_reference[o]), 1., 1.5]
    t, p = np.repeat(TEMPERATURES, 5), np.tile(PRESSURES, 5)
    cases, budgets, amounts, strict, diagnostics = [], [], [[], []], [[], []], []
    for ratio in ratios:
        budget = carbon_budget(base, ratio)
        cases.extend({"C_over_O": ratio, "temperature_K": float(ti), "pressure_bar": float(pi)}
                     for ti, pi in zip(t, p))
        budgets.extend([budget]*len(t))
        for mi, model in enumerate(pair):
            for tolerance, destination in ((PRIMARY_TOLERANCE, amounts), (STRICT_TOLERANCE, strict)):
                values, rows = solve_cases(model, budget, t, p, tolerance)
                destination[mi].append(values)
                diagnostics.extend({"dataset": model.metadata["dataset"], "C_over_O": ratio,
                                    "epsilon_crit": tolerance, **r} for r in rows)
        print(f"C/O={ratio:.12g}: 100 endpoint/strict cold solves passed", flush=True)
    ln_n, strict_ln_n = [np.array([np.concatenate(v) for v in runs]) for runs in (amounts, strict)]
    ln_x = ln_n - logsumexp(ln_n, axis=-1, keepdims=True)
    strict_ln_x = strict_ln_n - logsumexp(strict_ln_n, axis=-1, keepdims=True)
    n, x, budgets = np.exp(ln_n), np.exp(ln_x), np.array(budgets)
    stability = {"max_abs_delta_ln_x": float(np.max(abs(strict_ln_x-ln_x))),
                 "max_abs_delta_x": float(np.max(abs(np.exp(strict_ln_x)-x)))}
    if stability["max_abs_delta_ln_x"] > 1e-4 or stability["max_abs_delta_x"] > 1e-9:
        raise ValueError(f"Tolerance stability failed: {stability}")
    delta_log10_x = (ln_x[1]-ln_x[0])/np.log(10.)
    a = np.asarray(base.formula_matrix)
    shares = {e: n*a[base.elements.index(e)]/budgets[None, :, base.elements.index(e), None]
              for e in ("Al", "C", "O", "N", "P")}

    def point(i, j):
        return {**cases[i], "species": base.species[j], "x_before": float(x[0, i, j]),
                "x_after": float(x[1, i, j]), "delta_x": float(x[1, i, j]-x[0, i, j]),
                "ratio": float(np.exp(ln_x[1, i, j]-ln_x[0, i, j])),
                "delta_log10_x": float(delta_log10_x[i, j]),
                "element_fractions": {e: values[:, i, j].tolist() for e, values in shares.items()}}

    alh = base.species.index("Al1H1")
    index = int(np.argmax(shares["Al"][0, :, alh]))
    extrema = []
    for name in TRACKED:
        j = base.species.index(name)
        significant = np.flatnonzero(np.max(x[:, :, j], axis=0) >= 1e-12)
        extrema.append({"species": name, "largest_absolute_change": point(int(np.argmax(abs(x[1, :, j]-x[0, :, j]))), j),
                        "largest_log_change": point(int(np.argmax(abs(delta_log10_x[:, j]))), j),
                        "largest_log_change_above_display_threshold": None if not significant.size else
                        point(int(significant[np.argmax(abs(delta_log10_x[significant, j]))]), j)})
    # The shared single-node FastChem exporter supports these four audited temperatures.
    eligible = np.array([i for i, row in enumerate(cases) if row["temperature_K"] in (1500., 2000., 3000., 4000.)])
    parity_index = int(eligible[np.argmax(shares["Al"][0, eligible, alh])])
    parity_case = cases[parity_index]
    independent = (independent_check(pair, budgets[parity_index], parity_case["temperature_K"],
                   parity_case["pressure_bar"], strict_ln_n[:, parity_index], executable.resolve())
                   if executable else "not_run")
    arrays = dict(ln_n=ln_n, n=n, ln_x=ln_x, x=x, strict_ln_n=strict_ln_n,
                  delta_x=x[1]-x[0], delta_log10_x=delta_log10_x, budget=budgets,
                  species=base.species, elements=base.elements, formula_matrix=a,
                  temperature_K=[r["temperature_K"] for r in cases], pressure_bar=[r["pressure_bar"] for r in cases],
                  C_over_O=[r["C_over_O"] for r in cases],
                  h=np.array([model.hvector_func(np.array([r["temperature_K"] for r in cases])) for model in pair]),
                  **{e+"_fraction": values for e, values in shares.items()})
    np.savez_compressed(output / "equilibria.npz", **arrays)
    summary = {"datasets": [m.metadata["dataset"] for m in pair], "network_species": len(base.species),
               "intervention": "Replace only Al1H1 with unadjusted NASA ALH-AL-H reaction coefficients.",
               "reaction_coefficients": reaction.tolist(), "source_input_sha256": file_hash(DEFAULT_INPUT),
               "paired_cases": len(cases), "endpoint_solves": 150, "tighter_cold_solves": 150,
               "carbon_to_oxygen": ratios, "temperature_K": TEMPERATURES.tolist(), "pressure_bar": PRESSURES.tolist(),
               "tolerance_stability": stability, "highlighted_case": cases[index],
               "highlighted_species": [point(index, base.species.index(name)) for name in TRACKED],
               "highlighted_aluminum_partition": [point(index, int(j)) for j in
                    np.flatnonzero(a[base.elements.index("Al")])],
               "extrema": extrema, "independent_solver_case": parity_case, "independent_solver": independent,
               "scope": "Fixed-network ideal-gas source sensitivity, not adoption, physical validation or an uncertainty interval. "
                        "Only ions=True; no condensation, quenching or spectra. Finite amounts retained without concentration floors."}
    for filename, value in (("summary.json", summary), ("diagnostics.json", diagnostics),
                            ("prioritization.json", prioritization())):
        (output / filename).write_text(json.dumps(value, indent=2, allow_nan=False)+"\n")
    paths = [Path(__file__), DEFAULT_INPUT, PRIOR_BASELINE, SOURCE_COMPARISON]
    paths += [ROOT / name for name in ("mceg/reviews/c2h_equilibrium_impact.py", "mceg/reviews/gas_source_comparison.py",
              "mceg/reviews/curated_history.py", "examples/comparisons/compare_gas_thermochemistry.py",
              "src/exogibbs/presets/curated.py", "src/exogibbs/presets/curated_c2h.py", "src/exogibbs/presets/curated_v1.py",
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
    print(json.dumps({"highlighted_case": cases[index], "tolerance_stability": stability,
                      "independent_solver_case": parity_case, "independent_solver": independent}), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--fastchem-executable", type=Path)
    args = parser.parse_args()
    run(args.output_dir, args.fastchem_executable)
