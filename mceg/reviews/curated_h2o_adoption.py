"""Verify joint CO2/H2O adoption through the current curated factory."""

import argparse
from dataclasses import replace
from datetime import datetime, timezone
from hashlib import sha256
import json
from pathlib import Path
import platform
import sys

import jax
import jax.numpy as jnp
import numpy as np
from scipy.special import logsumexp

from exogibbs.api.gas import EquilibriumOptions, solve_profile
from exogibbs.presets import curated_h2o as curated, curated_h2o, curated_tio

from c2h_equilibrium_impact import independent_check
import co2_candidate
import h2o_candidate


ROOT = Path(__file__).resolve().parents[2]
BASELINE = ROOT / "tests/data/validation/curated_tio_adoption/20261103"


def digest(path: Path) -> str:
    return sha256(path.read_bytes()).hexdigest()


def solve_cases(setup, budget, temperatures, pressures, tolerance):
    """Require convergence and conservation in either factory mode."""
    result, info = solve_profile(
        setup, T=jnp.asarray(temperatures), P=jnp.asarray(pressures), b=jnp.asarray(budget),
        options=EquilibriumOptions(epsilon_crit=tolerance, method="vmap_cold"),
        return_diagnostics=True)
    amounts = np.asarray(result.ln_n)
    assert np.isfinite(amounts).all() and np.asarray(info["converged"]).all()
    n, matrix = np.exp(amounts), np.asarray(setup.formula_matrix)
    positive = budget > 0
    element = np.max(abs(n @ matrix.T - budget)[:, positive] / budget[positive], axis=-1)
    logp = amounts - logsumexp(amounts, axis=-1, keepdims=True) + np.log(pressures)[:, None]
    references = [setup.species.index("e1-" if name == "e-" else name+"1") for name in setup.elements]
    mass_action = np.max(abs(logp - logp[:, references] @ matrix
                             + np.asarray(setup.hvector_func(temperatures))), axis=-1)
    charge = np.zeros(len(temperatures))
    if "e-" in setup.elements:
        vector = matrix[setup.elements.index("e-")]
        charge = abs(n @ vector) / (n @ abs(vector))
    assert np.max(element) < 1e-5 and np.max(mass_action) < 1e-8 and np.max(charge) < 1e-12
    rows = [{"temperature_K": float(t), "pressure_bar": float(p), "converged": True,
             "iterations": int(info["n_iter"][i]), "solver_residual": float(info["final_residual"][i]),
             "element_relative_residual": float(element[i]),
             "charge_relative_residual": float(charge[i]), "mass_action_max_abs_ln": float(mass_action[i])}
            for i, (t, p) in enumerate(zip(temperatures, pressures))]
    return amounts, rows


def reviewed_setup(base):
    """Apply the two frozen candidate evaluators independently of new runtime."""
    carbon = base.species.index("C1O2")
    water = base.species.index("H2O1")

    def hvector(temperature):
        h = base.hvector_func(temperature)
        h = h.at[..., carbon].set(co2_candidate.formation_h_jax(temperature, model="nist626"))
        return h.at[..., water].set(h2o_candidate.formation_h_jax(temperature, thermal="evaluated"))

    return replace(base, hvector_func=jax.jit(hvector), metadata={
        **base.metadata, "dataset": "review-joint-co2-h2o"})


def run(output: Path, executable: Path | None = None) -> None:
    """Run 150 production cases and six tighter independently assembled cases."""
    jax.config.update("jax_enable_x64", True)
    assert curated.chemsetup is curated_h2o.chemsetup
    output.mkdir(parents=True, exist_ok=False)
    summaries, all_rows, independent = [], [], []
    for ions in (True, False):
        name = "ions" if ions else "neutral"
        with np.load(BASELINE / f"{name}.npz") as archive:
            previous = {key: archive[key].copy() for key in archive.files}
        setup, legacy = curated.chemsetup(ions=ions), curated_tio.chemsetup(ions=ions)
        reference = reviewed_setup(legacy)
        assert setup.metadata["dataset"] == "curated-h2o-v1"
        assert setup.species == legacy.species and setup.elements == legacy.elements
        np.testing.assert_array_equal(setup.formula_matrix, legacy.formula_matrix)
        np.testing.assert_array_equal(setup.element_vector_reference, legacy.element_vector_reference)
        np.testing.assert_array_equal(setup.species, previous["species"])
        np.testing.assert_array_equal(setup.elements, previous["elements"])
        np.testing.assert_array_equal(setup.formula_matrix, previous["formula_matrix"])
        water, carbon = (setup.species.index(s) for s in ("H2O1", "C1O2"))
        t, p, budgets = (previous[key] for key in ("temperature_K", "pressure_bar", "budget"))
        h = np.asarray(setup.hvector_func(t))
        reference_h = np.asarray(reference.hvector_func(t))
        np.testing.assert_allclose(h, reference_h, atol=1e-12, rtol=0)
        unchanged = np.delete(h - previous["h"], [water, carbon], axis=-1)
        np.testing.assert_allclose(unchanged, 0, atol=1e-12, rtol=0)
        dense_t = np.linspace(1500., 4000., 2501)
        dense_error = float(np.max(abs(np.asarray(setup.hvector_func(dense_t))
                                      - np.asarray(reference.hvector_func(dense_t)))))
        assert dense_error < 1e-12
        amounts = np.empty_like(previous["ln_n"])
        rows = []
        for ratio in np.unique(previous["C_over_O"]):
            indices = np.flatnonzero(previous["C_over_O"] == ratio)
            np.testing.assert_array_equal(budgets[indices], np.broadcast_to(
                budgets[indices[0]], budgets[indices].shape))
            values, checks = solve_cases(setup, budgets[indices[0]], t[indices], p[indices], 5e-14)
            amounts[indices] = values
            rows.extend({"network": name, "case_index": int(i), **row}
                        for i, row in zip(indices, checks))
        ln_x = amounts - logsumexp(amounts, axis=-1, keepdims=True)
        selected = np.array([0, 24, 74])
        tight = []
        for i in selected:
            values, checks = solve_cases(reference, budgets[i], t[i:i+1], p[i:i+1], 2e-14)
            tight.append(values[0])
            all_rows.extend({"network": name, "case_index": int(i), "tighter_reference": True, **row}
                            for row in checks)
        tight = np.asarray(tight)
        tight_x = tight - logsumexp(tight, axis=-1, keepdims=True)
        delta = tight_x - ln_x[selected]
        all_error = float(np.max(abs(delta)))
        targets_error = float(np.max(abs(delta[:, [water, carbon]])))
        absolute_error = float(np.max(abs(np.exp(tight_x) - np.exp(ln_x[selected]))))
        assert all_error < 1e-4 and targets_error < 1e-7 and absolute_error < 1e-9
        summary = {"ions": ions, "cases_passed": len(t), "species": len(setup.species),
                   "max_dense_h_difference_from_joint_candidates": dense_error,
                   "tighter_case_indices": selected.tolist(),
                   "max_tight_abs_delta_ln_x": all_error,
                   "max_tight_co2_h2o_abs_delta_ln_x": targets_error,
                   "max_tight_absolute_delta_x": absolute_error,
                   "max_element_relative_residual": max(r["element_relative_residual"] for r in rows),
                   "max_mass_action_abs_ln": max(r["mass_action_max_abs_ln"] for r in rows)}
        if ions:
            summary["max_charge_relative_residual"] = max(r["charge_relative_residual"] for r in rows)
            if executable:
                for i in (0, 24):
                    independent.append({"case_index": i, "comparisons": independent_check(
                        [setup], budgets[i], float(t[i]), float(p[i]), amounts[i:i+1], executable.resolve())})
        old_x = np.exp(previous["ln_x"])
        x = np.exp(ln_x)
        summary["species_response"] = [
            {"species": setup.species[j], "min_x_ratio": float(np.exp(ln_x[:, j]-previous["ln_x"][:, j]).min()),
             "max_x_ratio": float(np.exp(ln_x[:, j]-previous["ln_x"][:, j]).max()),
             "max_abs_delta_x": float(np.max(abs(x[:, j]-old_x[:, j]))),
             "case24_x_before": float(old_x[24, j]), "case24_x_after": float(x[24, j])}
            for j in (water, carbon)]
        np.savez_compressed(output / f"{name}.npz", ln_n=amounts, ln_x=ln_x, h=h,
                            reference_h=reference_h, formula_matrix=setup.formula_matrix,
                            budget=budgets, species=setup.species, elements=setup.elements,
                            temperature_K=t, pressure_bar=p, C_over_O=previous["C_over_O"],
                            converged=np.ones(len(t), dtype=bool), tighter_case_indices=selected,
                            tighter_reference_ln_n=tight)
        summaries.append(summary)
        all_rows.extend(rows)
        print(json.dumps({"network": name, **summary}), flush=True)
    summary = {"dataset": "curated-h2o-v1", "base_dataset": "curated-co2-v1",
               "comparison_dataset": "curated-tio-v1", "epsilon_crit": 5e-14,
               "strict_epsilon_crit": 2e-14, "method": "vmap_cold",
               "new_production_solves": 150, "tighter_joint_reference_solves": 6,
               "tolerance_note": "A diagnostic 1e-14 request at 1500 K, 1e-6 bar reached a float64 residual floor of 1.54e-14 without a converged flag; all recorded tighter cases require convergence at 2e-14, below the production 5e-14.",
               "networks": summaries, "independent_solver": independent if executable else "not_run",
               "scope": "Joint CO2 NIST626 and H2O evaluated-2025 production adoption, both network modes. "
                        "Independent assembly of the frozen reviewed candidates agrees in h on 2501 temperatures. "
                        "Numerical closure and solver parity do not establish complete physical accuracy."}
    for filename, value in (("summary.json", summary), ("diagnostics.json", all_rows)):
        (output / filename).write_text(json.dumps(value, indent=2, allow_nan=False)+"\n")
    paths = [Path(__file__)]
    paths += [ROOT / "mceg/reviews" / f"{name}.py" for name in (
        "co2_candidate", "co2_source_review", "h2o_candidate", "h2o_source_review",
        "c2h_equilibrium_impact", "gas_source_comparison")]
    paths += [ROOT / "src/exogibbs/presets" / f"{name}.py" for name in (
        "curated", "curated_h2o", "curated_co2", "curated_tio", "curated_c2h", "curated_v1", "fastchem4", "fastchem")]
    paths += [ROOT / "src/exogibbs/data/curated" / f"{name}.json" for name in (
        "gas_v1", "c2h_v1", "tio_v1", "co2_v1", "h2o_v1")]
    paths += [BASELINE / f"{name}.npz" for name in ("ions", "neutral")]
    paths += [ROOT / f"tests/data/validation/{name}_candidate/20261104/candidate.json"
              for name in ("co2", "h2o")]
    paths += [ROOT / name for name in ("src/exogibbs/api/gas.py", "src/exogibbs/thermo/standard.py",
        "src/exogibbs/thermo/models.py", "src/exogibbs/data/FastChem4/logK/logK.dat",
        "src/exogibbs/data/FastChem4/element_abundances/asplund_2021.dat",
        "examples/comparisons/compare_gas_thermochemistry.py", "examples/comparisons/select_major_molecules.py",
        "examples/comparisons/data/major_molecule_reference.json", "benchmarks/fastchem4/comparison.py",
        "benchmarks/fastchem4/fastchem_executable.py")]
    paths += sorted((ROOT / "src/exogibbs/equilibrium/gas").rglob("*.py"))
    receipt = {"created_utc": datetime.now(timezone.utc).isoformat(), "result": "passed",
               "python": platform.python_version(), "numpy": np.__version__, "jax": jax.__version__,
               "backend": jax.default_backend(), "dtype": "float64", "command": " ".join(sys.argv),
               "selected_file_sha256": {str(p.relative_to(ROOT)): digest(p) for p in paths},
               "independent_executable_sha256": digest(executable) if executable else None,
               "output_sha256": {p.name: digest(p) for p in sorted(output.iterdir())}}
    (output / "receipt.json").write_text(json.dumps(receipt, indent=2)+"\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--fastchem-executable", type=Path)
    arguments = parser.parse_args()
    run(arguments.output_dir, arguments.fastchem_executable)
