"""Verify the adopted HO2 factory against the frozen candidate and gas response."""

import argparse
from datetime import datetime, timezone
from hashlib import sha256
import json
from pathlib import Path
import platform
import sys

import jax
import numpy as np
from scipy.special import logsumexp

from exogibbs.presets import curated_ho2 as curated, curated_kh, curated_ho2
from c2h_equilibrium_impact import independent_check
from curated_h2o_adoption import solve_cases
from ho2_equilibrium_impact import candidate_setup, constraints


ROOT = Path(__file__).resolve().parents[2]
BASELINE = ROOT / "tests/data/validation/curated_kh_adoption/20261105"
CANDIDATE = ROOT / "tests/data/validation/ho2_equilibrium_impact/20261105"


def digest(path: Path) -> str:
    return sha256(path.read_bytes()).hexdigest()


def run(output: Path, executable: Path | None = None) -> None:
    """Require 150 fresh production solves, six tighter controls and closure."""
    jax.config.update("jax_enable_x64", True)
    assert curated.chemsetup is curated_ho2.chemsetup
    output.mkdir(parents=True, exist_ok=False)
    summaries, diagnostics, independent = [], [], []
    for ions in (True, False):
        name = "ions" if ions else "neutral"
        with np.load(BASELINE / f"{name}.npz") as archive:
            previous = {key: archive[key].copy() for key in archive.files}
        with np.load(CANDIDATE / f"{name}.npz") as archive:
            reviewed = {key: archive[key].copy() for key in archive.files}
        setup, base = curated.chemsetup(ions=ions), curated_kh.chemsetup(ions=ions)
        reference = candidate_setup(base)
        assert setup.metadata["dataset"] == "curated-ho2-v1"
        assert setup.species == base.species and setup.elements == base.elements
        np.testing.assert_array_equal(setup.formula_matrix, base.formula_matrix)
        np.testing.assert_array_equal(setup.element_vector_reference, base.element_vector_reference)
        for value, key in ((setup.species, "species"), (setup.elements, "elements"),
                           (setup.formula_matrix, "formula_matrix")):
            np.testing.assert_array_equal(value, previous[key])
            np.testing.assert_array_equal(value, reviewed[key])
        oh = setup.species.index("H1O2")
        t, p, budgets = (previous[key] for key in ("temperature_K", "pressure_bar", "budget"))
        for key in ("temperature_K", "pressure_bar", "budget", "C_over_O"):
            np.testing.assert_array_equal(previous[key], reviewed[key])
        h = np.asarray(setup.hvector_func(t))
        np.testing.assert_allclose(h[:, oh], reviewed["h"][1, :, oh], atol=1e-12, rtol=0)
        np.testing.assert_allclose(base.hvector_func(t), previous["h"], atol=1e-12, rtol=0)
        np.testing.assert_array_equal(np.delete(h - previous["h"], oh, axis=-1), 0)
        dense_t = np.linspace(1500., 4000., 2501)
        dense_error = float(np.max(abs(np.asarray(setup.hvector_func(dense_t))
                                      - np.asarray(reference.hvector_func(dense_t)))))
        assert dense_error < 1e-12
        amounts, rows = np.empty_like(previous["ln_n"]), []
        for ratio in np.unique(previous["C_over_O"]):
            indices = np.flatnonzero(previous["C_over_O"] == ratio)
            np.testing.assert_array_equal(budgets[indices], np.broadcast_to(
                budgets[indices[0]], budgets[indices].shape))
            values, checks = solve_cases(setup, budgets[indices[0]], t[indices], p[indices], 5e-14)
            amounts[indices] = values
            rows.extend({"network": name, "case_index": int(i), **row}
                        for i, row in zip(indices, checks))
        closure = constraints(np.asarray(setup.formula_matrix), list(setup.elements),
                              list(setup.species), budgets, amounts, h, p)
        ln_x = amounts - logsumexp(amounts, axis=-1, keepdims=True)
        selected = np.array([23, 50, 70] if ions else [21, 23, 50])
        tight = []
        for i in selected:
            values, checks = solve_cases(reference, budgets[i], t[i:i+1], p[i:i+1], 2e-14)
            tight.append(values[0])
            diagnostics.extend({"network": name, "case_index": int(i), "tighter_reference": True, **row}
                               for row in checks)
        tight = np.asarray(tight)
        tight_x = tight - logsumexp(tight, axis=-1, keepdims=True)
        summary = {"ions": ions, "cases_passed": len(t), "species": len(setup.species),
                   "max_dense_h_difference_from_candidate": dense_error,
                   "tighter_case_indices": selected.tolist()}
        # The frozen candidate used OH-only background; KH now changes the
        # surrounding equilibrium, so compare composition to a fresh candidate on KH.
        for label, expected, actual in (("tighter_reference", tight_x, ln_x[selected]),):
            delta = actual - expected
            all_error, oh_error = float(np.max(abs(delta))), float(np.max(abs(delta[:, oh])))
            absolute_error = float(np.max(abs(np.exp(actual) - np.exp(expected))))
            assert all_error < 1e-4 and oh_error < 1e-8 and absolute_error < 1e-9
            summary[label] = {"max_abs_delta_ln_x": all_error,
                              "max_ho2_abs_delta_ln_x": oh_error,
                              "max_absolute_delta_x": absolute_error}
        summary["closure"] = {key: float(np.max(value)) for key, value in closure.items()}
        ratio = np.exp(ln_x[:, oh] - previous["ln_x"][:, oh])
        summary["ho2_response"] = {"min_x_ratio": float(ratio.min()), "max_x_ratio": float(ratio.max()),
                                  "case23_x_before": float(np.exp(previous["ln_x"][23, oh])),
                                  "case23_x_after": float(np.exp(ln_x[23, oh]))}
        if ions and executable:
            for i in (23, 50):
                independent.append({"case_index": i, "comparisons": independent_check(
                    [setup], budgets[i], float(t[i]), float(p[i]), amounts[i:i+1], executable.resolve())})
        np.savez_compressed(output / f"{name}.npz", ln_n=amounts, ln_x=ln_x, h=h,
                            formula_matrix=setup.formula_matrix, budget=budgets, species=setup.species,
                            elements=setup.elements, temperature_K=t, pressure_bar=p,
                            C_over_O=previous["C_over_O"], converged=np.ones(len(t), dtype=bool),
                            tighter_case_indices=selected, tighter_reference_ln_n=tight)
        summaries.append(summary)
        diagnostics.extend(rows)
        print(json.dumps({"network": name, **summary}), flush=True)
    summary = {"dataset": "curated-ho2-v1", "base_dataset": "curated-kh-v1",
               "epsilon_crit": 5e-14, "strict_epsilon_crit": 2e-14, "method": "vmap_cold",
               "new_production_solves": 150, "tighter_reference_solves": 6,
               "networks": summaries, "independent_solver": independent if executable else "not_run",
               "scope": "HO2-only NASA/ATcT adoption on curated-kh-v1 at 1500-4000 K. The old candidate used curated-oh-v1, so only its HO2 thermodynamics are reused; compositions are verified with fresh candidate-on-KH controls. "
                        "Agreement with the reviewed candidate and numerical closure do not establish full physical accuracy."}
    for filename, value in (("summary.json", summary), ("diagnostics.json", diagnostics)):
        (output / filename).write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    # Record actual current dependencies; never rewrite the historical candidate receipt.
    prior = json.loads((CANDIDATE / "receipt.json").read_text())
    paths = {ROOT / name for name in prior["selected_file_sha256"]}
    paths.update((Path(__file__), ROOT / "mceg/reviews/build_ho2_adoption.py",
                  ROOT / "src/exogibbs/presets/curated_ho2.py", ROOT / "src/exogibbs/data/curated/ho2_v1.json"))
    paths.update(CANDIDATE / f"{name}.npz" for name in ("ions", "neutral"))
    paths.update(BASELINE / f"{name}.npz" for name in ("ions", "neutral"))
    paths.update((ROOT / "src/exogibbs/presets/curated_kh.py",
                  ROOT / "src/exogibbs/data/curated/kh_v1.json"))
    receipt = {"created_utc": datetime.now(timezone.utc).isoformat(), "result": "passed",
               "python": platform.python_version(), "numpy": np.__version__, "jax": jax.__version__,
               "backend": jax.default_backend(), "dtype": "float64", "command": " ".join(sys.argv),
               "selected_file_sha256": {str(p.relative_to(ROOT)): digest(p) for p in sorted(paths)},
               "independent_executable_sha256": digest(executable) if executable else None,
               "output_sha256": {p.name: digest(p) for p in sorted(output.iterdir())}}
    (output / "receipt.json").write_text(json.dumps(receipt, indent=2) + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--fastchem-executable", type=Path)
    arguments = parser.parse_args()
    run(arguments.output_dir, arguments.fastchem_executable)
