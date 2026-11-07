"""Compare all new cold-gas cases with the pinned independent GGChem solver.

The existing Fortran adapter evaluates NASA9 and solves pressure closure without
ExoGibbs initial guesses. Matching inputs isolate implementation agreement from
the separate question of physical thermochemical accuracy.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys

import jax
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import phosphorus_cold_validation_20261031 as study
import phosphorus_ggchem_reference as reference


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_ROOT = ROOT / "results/phosphorus_cold_validation"
GATES = {
    "max_abs_delta_lnK": 1e-8,
    "max_abs_delta_ln_mole_fraction": 1e-5,
    "element_relative_residual": 1e-8,
    "mass_action_max_abs_ln": 1e-8,
    "pressure_relative_residual": 1e-12,
}


def case_inputs(case: dict, interventions: dict, expanded: tuple) -> tuple[dict, list[str]]:
    """Reconstruct the declared intervention and reject identity/order drift."""
    if case["intervention"] == "expanded_catalog":
        data, names, _ = expanded
    else:
        definition = interventions[case["intervention"]]
        data, names = definition["data"], definition["species"]
        for key in ("kind", "shifted_species", "offset_kJ_mol", "removed"):
            if key in definition and case.get(key) != definition[key]:
                raise ValueError(f"Intervention metadata changed: {key}")
    if case["species"] != names:
        raise ValueError("Species identity or order changed")
    return data, names


def state_gates(metrics: dict, exogibbs_accepted: np.ndarray) -> np.ndarray:
    """Require every residual and every species, including tiny abundances."""
    accepted = np.asarray(exogibbs_accepted, dtype=bool).copy()
    for key, limit in GATES.items():
        values = np.asarray(metrics[key], dtype=float)
        if values.shape != accepted.shape:
            raise ValueError(f"State count changed for {key}")
        accepted &= np.isfinite(values) & (values <= limit)
    return accepted


def compare_case(case: dict, data: dict, names: list[str], binary: Path,
                 directory: Path) -> tuple[dict, np.ndarray]:
    """Run GGChem, independently check its closures, and compare every gas."""
    temperature = np.asarray(case["temperature_K"])
    pressure = np.asarray(case["pressure_bar"])
    setup = study.cold.make_setup(data, names, case["metallicity"])
    if list(setup.elements) != case["elements"]:
        raise ValueError("Element order changed")
    budget = np.asarray(setup.element_vector_reference)
    matrix = np.asarray(setup.formula_matrix)
    if not np.array_equal(budget, case["budget"]) or not np.array_equal(matrix, case["formula_matrix"]):
        raise ValueError("Element composition or abundance budget changed")
    ref = reference.run_reference(
        data, names, temperature, pressure, dict(zip(setup.elements, budget)),
        binary=binary, output_directory=directory)
    logx = np.asarray(ref["ln_mole_fraction"])
    ln_k = np.asarray(ref["ln_formation_Kp"])
    exo_logx = np.asarray(case["log_mole_fractions"])
    exo_ln_k = -np.asarray(setup.hvector_func(temperature))
    delta_x, delta_k = logx - exo_logx, ln_k - exo_ln_k

    # Re-read the primitive Fortran densities to retain closure per state.
    rows = (directory / "output.txt").read_text().splitlines()
    header = np.array([[float(v) for v in rows[1 + 4*i].split()]
                       for i in range(len(temperature))])
    density = np.array([[float(v) for line in rows[2 + 4*i:4 + 4*i]
                        for v in line.split()] for i in range(len(temperature))])
    atoms = [names.index(element) for element in setup.elements]
    order = list(setup.elements) + [name for name in names if name not in setup.elements]
    density = density[:, [order.index(name) for name in names]]
    expected = budget / budget[list(setup.elements).index("H")]
    closure = np.exp(density - np.log(header[:, 2])[:, None]) @ matrix.T
    logp = logx + np.log(pressure)[:, None]
    action = logp - logp[:, atoms] @ matrix - ln_k
    metrics = {
        "max_abs_delta_lnK": np.max(np.abs(delta_k), axis=1),
        "max_abs_delta_ln_mole_fraction": np.max(np.abs(delta_x), axis=1),
        "element_relative_residual": np.max(np.abs(closure / expected - 1), axis=1),
        "mass_action_max_abs_ln": np.max(np.abs(action), axis=1),
        "pressure_relative_residual": np.abs(header[:, 3]),
    }
    accepted = state_gates(metrics, np.asarray(case["accepted"]))
    worst = np.unravel_index(np.argmax(np.abs(delta_x)), delta_x.shape)
    report = {
        "intervention": case["intervention"], "kind": case["kind"],
        "metallicity": case["metallicity"], "species_count": len(names),
        "temperature_K": temperature.tolist(), "pressure_bar": pressure.tolist(),
        "metrics": {key: value.tolist() for key, value in metrics.items()},
        "maxima": {key: float(np.max(value)) for key, value in metrics.items()},
        "accepted": accepted.tolist(), "cases_passed": int(accepted.sum()),
        "cases_total": len(accepted), "max_pressure_iterations": int(header[:, 4].max()),
        "largest_abundance_difference": {
            "species": names[worst[1]], "temperature_K": float(temperature[worst[0]]),
            "pressure_bar": float(pressure[worst[0]]),
            "signed_delta_ln_mole_fraction": float(delta_x[worst]),
            "exogibbs_ln_mole_fraction": float(exo_logx[worst]),
            "ggchem_ln_mole_fraction": float(logx[worst]),
        },
        "primitive_output_sha256": hashlib.sha256((directory / "output.txt").read_bytes()).hexdigest(),
    }
    return report, logx


def plot_comparison(cases: dict, reference_logx: dict, output: Path) -> None:
    """Overlay the two independent solvers in the existing abundance format."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    displayed = sorted({s for z in study.METALLICITIES
                        for s, values in cases[f"full_Z{z:g}"]["P_atom_fractions"].items()
                        if max(values) > .01})
    colors = {s: study.COLORS.get(s, plt.get_cmap("tab20")(i)) for i, s in enumerate(displayed)}
    fig, axes = plt.subplots(2, 3, figsize=(13, 8), layout="constrained", sharex=True, sharey=True)
    for row, z in enumerate(study.METALLICITIES):
        key = f"full_Z{z:g}"
        case = cases[key]
        for col, pressure in enumerate(study.PRESSURES_BAR):
            ax = axes[row, col]
            selected = np.asarray(case["pressure_bar"]) == pressure
            for species in displayed:
                index = case["species"].index(species)
                values = np.asarray(case["log_mole_fractions"])[selected, index] / np.log(10.)
                if max(values) <= -18:
                    continue
                ax.plot(study.TEMPERATURES, values, color=colors[species], lw=2)
                ax.plot(study.TEMPERATURES, reference_logx[key][selected, index] / np.log(10.),
                        linestyle="--", marker="o", fillstyle="none", markersize=4,
                        color=colors[species], lw=.9)
            ax.set(xscale="log", xlim=(200, 1500), ylim=(-18, -3.5),
                   title=f"P = {pressure:g} bar | Z = {z:g}", xlabel="Temperature [K]",
                   ylabel=r"$\log_{10}(n_i/n_{\rm tot})$")
            ax.grid(alpha=.2)
    handles = [Line2D([], [], color=colors[s], label=s) for s in displayed]
    handles += [Line2D([], [], color="black", label="ExoGibbs", lw=2),
                Line2D([], [], color="black", label="GGChem", linestyle="--", marker="o", fillstyle="none")]
    fig.legend(handles=handles, loc="outside lower center", ncol=6, fontsize=8)
    fig.suptitle("Same 78-gas NASA9 inputs | independent ExoGibbs and GGChem solvers | no condensation")
    fig.savefig(output / "cold_ggchem_exogibbs_comparison.png", dpi=170)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--primitive-input", type=Path, default=DEFAULT_ROOT / "gas/primitive_cases.json")
    parser.add_argument("--candidate-input", type=Path, default=DEFAULT_ROOT / "p_candidate_records.json")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_ROOT / "ggchem")
    args = parser.parse_args()
    jax.config.update("jax_enable_x64", True)
    cases = json.loads(args.primitive_input.read_text())
    if sum(len(case["temperature_K"]) for case in cases.values()) != 448:
        raise ValueError("Expected the frozen 448-state cold gas experiment")
    data = study.cold.load_data()
    interventions = study.interventions(data)
    expanded = study.expanded_network(data, json.loads(args.candidate_input.read_text()))
    args.output_dir.mkdir(parents=True, exist_ok=True)
    binary = reference.build_reference(args.output_dir / "build")
    compared, reference_logx, errors = {}, {}, {}
    for key, case in cases.items():
        try:
            inputs, names = case_inputs(case, interventions, expanded)
            compared[key], reference_logx[key] = compare_case(
                case, inputs, names, binary, args.output_dir / key)
            print(key, compared[key]["cases_passed"], "/", compared[key]["cases_total"], flush=True)
        except (ValueError, RuntimeError, OSError, subprocess.SubprocessError) as error:
            errors[key] = f"{type(error).__name__}: {error}"
            print(key, errors[key], flush=True)
    paths = [Path(__file__), Path(study.__file__), Path(study.cold.__file__),
             Path(reference.__file__), reference.DRIVER, study.cold.FIXTURE,
             study.cold.ABUNDANCES, reference.FIXTURES / "ggchem_source.zip",
             reference.FIXTURES / "ggchem_build.json", args.primitive_input.resolve(),
             args.candidate_input.resolve(), ROOT / "src/exogibbs/thermo/standard.py",
             ROOT / "src/exogibbs/thermo/models.py"]
    report = {
        "scope": "Independent GGChem-versus-ExoGibbs comparison for all 448 new cold gas states, with identical selected thermochemistry and species identities. This does not validate the physical data or GGChem's native database selection.",
        "initialization": "GGChem receives composition, thermochemical records, temperature and pressure only; no ExoGibbs solution, seed or continuation is supplied. The existing adapter sets remember=false on each pressure iteration.",
        "gates": GATES, "cases": compared, "errors": errors,
        "cases_total": 448,
        "cases_passed": sum(case["cases_passed"] for case in compared.values()),
        "maxima": {metric: max(case["maxima"][metric] for case in compared.values())
                   for metric in GATES} if compared else {},
        "source_sha256": {str(path.relative_to(ROOT)): hashlib.sha256(path.read_bytes()).hexdigest()
                          for path in paths},
        "build": json.loads((args.output_dir / "build/build.json").read_text()),
    }
    (args.output_dir / "comparison_receipt.json").write_text(json.dumps(report, indent=2) + "\n")
    np.savez_compressed(args.output_dir / "ggchem_log_mole_fractions.npz", **reference_logx)
    if all(f"full_Z{z:g}" in reference_logx for z in study.METALLICITIES):
        plot_comparison(cases, reference_logx, args.output_dir)
    if report["cases_passed"] != report["cases_total"]:
        raise RuntimeError("Some independent comparisons failed; see comparison_receipt.json")


if __name__ == "__main__":
    main()
