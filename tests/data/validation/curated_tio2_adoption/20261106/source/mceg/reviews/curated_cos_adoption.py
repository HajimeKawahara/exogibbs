"""Verify COS production adoption and separate its cumulative background changes."""

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

from exogibbs.presets import curated, curated_ho2, curated_cos
from c2h_equilibrium_impact import carbon_budget, independent_check
from curated_h2o_adoption import solve_cases
from cos_equilibrium_impact import candidate_setup, constraints


ROOT = Path(__file__).resolve().parents[2]
BASELINE = ROOT / "tests/data/validation/curated_ho2_adoption/20261105"
OLD_BASELINE = ROOT / "tests/data/validation/curated_oh_adoption/20261105"
CANDIDATE = ROOT / "tests/data/validation/cos_equilibrium_impact/20261105"


def digest(path: Path) -> str:
    return sha256(path.read_bytes()).hexdigest()


def run(output: Path, executable: Path | None = None) -> None:
    """Run 150 fresh production endpoints and six independent tighter controls."""
    jax.config.update("jax_enable_x64", True)
    assert curated.chemsetup is curated_cos.chemsetup
    output.mkdir(parents=True, exist_ok=False)
    summaries, diagnostics, independent = [], [], []
    for ions in (True, False):
        name = "ions" if ions else "neutral"
        archives = []
        for directory in (BASELINE, OLD_BASELINE, CANDIDATE):
            with np.load(directory/f"{name}.npz") as archive:
                archives.append({key: archive[key].copy() for key in archive.files})
        previous, original, reviewed = archives
        setup, base = curated.chemsetup(ions=ions), curated_ho2.chemsetup(ions=ions)
        reference = candidate_setup(base)
        assert setup.metadata["dataset"] == "curated-cos-v1"
        assert base.metadata["dataset"] == "curated-ho2-v1"
        assert setup.species == base.species and setup.elements == base.elements
        np.testing.assert_array_equal(setup.formula_matrix, base.formula_matrix)
        np.testing.assert_array_equal(setup.element_vector_reference, base.element_vector_reference)
        species, elements = list(setup.species), list(setup.elements)
        for value, key in ((species, "species"), (elements, "elements"),
                           (setup.formula_matrix, "formula_matrix")):
            for saved in archives:
                np.testing.assert_array_equal(value, saved[key])
        t, p, budgets = (previous[key] for key in ("temperature_K", "pressure_bar", "budget"))
        for key in ("temperature_K", "pressure_bar", "budget", "C_over_O"):
            np.testing.assert_array_equal(previous[key], original[key])
            np.testing.assert_array_equal(previous[key], reviewed[key])
        assert len(t) == 75 and previous["converged"].all()
        h = np.asarray(setup.hvector_func(t))
        np.testing.assert_allclose(base.hvector_func(t), previous["h"], atol=1e-12, rtol=0)
        cos, sulfur = species.index("C1O1S1"), elements.index("S")
        np.testing.assert_array_equal(np.delete(h-previous["h"], cos, axis=-1), 0)
        np.testing.assert_allclose(h[:, cos], reviewed["h"][1, :, cos], atol=1e-12, rtol=0)
        background_columns = [species.index(s) for s in ("H1K1", "H1O2")]
        np.testing.assert_array_equal(np.delete(h-reviewed["h"][1], background_columns+[cos], axis=-1), 0)
        dense_t = np.linspace(1500., 4000., 2501)
        dense_error = float(np.max(abs(np.asarray(setup.hvector_func(dense_t))
                                      - np.asarray(reference.hvector_func(dense_t)))))
        assert dense_error < 1e-12
        amounts, rows = np.empty_like(previous["ln_n"]), []
        for ratio in np.unique(previous["C_over_O"]):
            indices = np.flatnonzero(previous["C_over_O"] == ratio)
            np.testing.assert_allclose(budgets[indices], np.broadcast_to(carbon_budget(base, float(ratio)), budgets[indices].shape), rtol=1e-15, atol=0)
            values, checks = solve_cases(setup, budgets[indices[0]], t[indices], p[indices], 5e-14)
            amounts[indices] = values
            rows.extend({"network": name, "case_index": int(i), **row}
                        for i, row in zip(indices, checks))
        closure = constraints(np.asarray(setup.formula_matrix), elements, species, budgets, amounts, h, p)
        ln_x = amounts-logsumexp(amounts, axis=-1, keepdims=True)
        x = np.exp(ln_x)
        prior_x = previous["ln_n"]-logsumexp(previous["ln_n"], axis=-1, keepdims=True)
        selected = np.array([3, 50, 70] if ions else [3, 21, 50])
        tight = []
        for i in selected:
            values, checks = solve_cases(reference, budgets[i], t[i:i+1], p[i:i+1], 2e-14)
            tight.append(values[0])
            diagnostics.extend({"network": name, "case_index": int(i), "tighter_reference": True, **row}
                               for row in checks)
        tight = np.asarray(tight)
        tight_x = tight-logsumexp(tight, axis=-1, keepdims=True)
        delta = ln_x[selected]-tight_x
        stability = {"max_abs_delta_ln_x": float(np.max(abs(delta))),
                     "max_cos_abs_delta_ln_x": float(np.max(abs(delta[:, cos]))),
                     "max_absolute_delta_x": float(np.max(abs(x[selected]-np.exp(tight_x))))}
        assert stability["max_abs_delta_ln_x"] < 1e-4
        assert stability["max_cos_abs_delta_ln_x"] < 1e-8
        assert stability["max_absolute_delta_x"] < 1e-9
        ratio = np.exp(ln_x[:, cos]-prior_x[:, cos])
        share_before = np.exp(previous["ln_n"][:, cos])/budgets[:, sulfur]
        share_after = np.exp(amounts[:, cos])/budgets[:, sulfur]
        highlight = int(np.argmax(abs(share_after-share_before)))
        archived_x = reviewed["ln_n"][1]-logsumexp(reviewed["ln_n"][1], axis=-1, keepdims=True)
        old_x = original["ln_n"]-logsumexp(original["ln_n"], axis=-1, keepdims=True)
        cumulative = {}
        for label in ("H1K1", "H1O2", "C1O1S1"):
            j = species.index(label)
            cumulative[label] = {"production_over_old_oh_min": float(np.exp(ln_x[:, j]-old_x[:, j]).min()),
                                 "production_over_old_oh_max": float(np.exp(ln_x[:, j]-old_x[:, j]).max()),
                                 "max_absolute_delta_x": float(abs(x[:, j]-np.exp(old_x[:, j])).max())}
        old_candidate_difference = ln_x-archived_x
        summary = {"ions": ions, "cases_passed": len(t), "species": len(species),
                   "max_dense_h_difference_from_candidate_on_ho2": dense_error,
                   "tighter_case_indices": selected.tolist(), "tighter_reference": stability,
                   "closure": {key: float(np.max(value)) for key, value in closure.items()},
                   "cos_incremental_response": {"min_x_ratio": float(ratio.min()), "max_x_ratio": float(ratio.max()),
                       "highlighted_case_index": highlight, "temperature_K": float(t[highlight]),
                       "pressure_bar": float(p[highlight]), "C_over_O": float(previous["C_over_O"][highlight]),
                       "x_before": float(np.exp(prior_x[highlight, cos])), "x_after": float(x[highlight, cos]),
                       "sulfur_share_before": float(share_before[highlight]), "sulfur_share_after": float(share_after[highlight])},
                   "cumulative_response_from_curated_oh": cumulative,
                   "background_effect_relative_to_old_cos_only_candidate": {
                       "COS_ratio_min": float(np.exp(old_candidate_difference[:, cos]).min()),
                       "COS_ratio_max": float(np.exp(old_candidate_difference[:, cos]).max()),
                       "max_absolute_delta_x_all_species": float(abs(x-np.exp(archived_x)).max()),
                       "scope": "Additional KH and HO2 adoption changes the background; this is a physical model difference, not a numerical reproduction error."}}
        if ions and executable:
            for i in (3, 50):
                independent.append({"case_index": i, "comparisons": independent_check(
                    [setup], budgets[i], float(t[i]), float(p[i]), amounts[i:i+1], executable.resolve())})
        np.savez_compressed(output/f"{name}.npz", ln_n=amounts, ln_x=ln_x, h=h,
                            formula_matrix=setup.formula_matrix, budget=budgets, species=species,
                            elements=elements, temperature_K=t, pressure_bar=p,
                            C_over_O=previous["C_over_O"], converged=np.ones(len(t), dtype=bool),
                            tighter_case_indices=selected, tighter_reference_ln_n=tight)
        summaries.append(summary)
        diagnostics.extend(rows)
        print(json.dumps({"network": name, **summary}), flush=True)
    summary = {"dataset": "curated-cos-v1", "base_dataset": "curated-ho2-v1",
               "epsilon_crit": 5e-14, "strict_epsilon_crit": 2e-14, "method": "vmap_cold",
               "new_production_solves": 150, "tighter_reference_solves": 6,
               "networks": summaries, "independent_solver": independent if executable else "not_run",
               "scope": "COS-only energy adoption on the cumulative KH/HO2 base at 1500-4000 K. Target potentials match the earlier COS candidate; whole-composition differences from the OH-only candidate include KH/HO2 background changes. Numerical checks do not establish full physical accuracy."}
    for filename, value in (("summary.json", summary), ("diagnostics.json", diagnostics)):
        (output/filename).write_text(json.dumps(value, indent=2, allow_nan=False)+"\n")
    prior = json.loads((CANDIDATE/"receipt.json").read_text())
    paths = {ROOT/name for name in prior["selected_file_sha256"]}
    paths.update((Path(__file__), ROOT/"mceg/reviews/build_cos_adoption.py"))
    paths.update((ROOT/"src/exogibbs/presets").glob("curated*.py"))
    paths.update((ROOT/"src/exogibbs/data/curated").glob("*_v1.json"))
    for directory in (BASELINE, OLD_BASELINE, CANDIDATE):
        paths.update(directory/f"{name}.npz" for name in ("ions", "neutral"))
    receipt = {"created_utc": datetime.now(timezone.utc).isoformat(), "result": "passed",
               "python": platform.python_version(), "numpy": np.__version__, "jax": jax.__version__,
               "backend": jax.default_backend(), "dtype": "float64", "command": [sys.executable, *sys.argv],
               "selected_file_sha256": {str(p.relative_to(ROOT)): digest(p) for p in sorted(paths)},
               "independent_executable_sha256": digest(executable) if executable else None,
               "output_sha256": {p.name: digest(p) for p in sorted(output.iterdir())}}
    (output/"receipt.json").write_text(json.dumps(receipt, indent=2)+"\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--fastchem-executable", type=Path)
    args = parser.parse_args()
    run(args.output_dir, args.fastchem_executable)
