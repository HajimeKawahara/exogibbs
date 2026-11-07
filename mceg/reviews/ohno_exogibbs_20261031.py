"""Separate source and species-list effects using sampled native GGChem inputs.

Only explicitly exported temperature knots are evaluated. This adapter does
not invent a new thermochemical fit or identify a mixed database as JANAF.
"""

from __future__ import annotations

import argparse
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

import jax
import jax.numpy as jnp
import numpy as np
from scipy.special import logsumexp

from exogibbs.api.gas import EquilibriumInit, EquilibriumOptions, solve, solve_profile
from exogibbs.thermo.models import ChemicalSetup


def prepare_case(case: dict) -> dict:
    """Validate and sort a neutral sampled thermochemical network."""
    result = deepcopy(case)
    names, elements = result["species"], result["elements"]
    matrix = np.asarray(result["formula_matrix"], dtype=float)
    budget = np.asarray(result["budget"], dtype=float)
    temperatures = np.asarray(result["temperature_K"], dtype=float)
    constants = np.asarray(result["lnK_bar"], dtype=float)
    if len(set(names)) != len(names) or matrix.shape != (len(elements), len(names)):
        raise ValueError("Species identities must be unique and match the stoichiometric matrix")
    if (np.any(matrix < 0) or np.any(matrix != np.round(matrix))
            or budget.shape != (len(elements),) or np.any(budget <= 0)):
        raise ValueError("This experiment requires neutral species and positive element budgets")
    if (constants.shape != (len(temperatures), len(names)) or not np.isfinite(constants).all()
            or np.any(temperatures <= 0) or len(np.unique(temperatures)) != len(temperatures)):
        raise ValueError("Invalid sampled formation constants or temperature grid")
    atoms = []
    for i, element in enumerate(elements):
        candidates = [j for j, name in enumerate(names) if name.upper() == element.upper()
                      and np.array_equal(matrix[:, j], np.eye(len(elements))[i])]
        if len(candidates) != 1:
            raise ValueError(f"Missing or ambiguous neutral atom: {element}")
        atoms.append(candidates[0])
    if np.max(np.abs(constants[:, atoms])) > 1e-10:
        raise ValueError("Formation constants must use neutral gas atoms as their reference")
    order = np.argsort(temperatures)
    result["temperature_K"] = temperatures[order].tolist()
    result["lnK_bar"] = constants[order].tolist()
    result["atom_indices"] = atoms
    result["budget"] = (budget / budget.sum()).tolist()
    if "ggchem_log_mole_fractions" in result:
        reference = np.asarray(result["ggchem_log_mole_fractions"])
        if reference.shape != constants.shape or not np.isfinite(reference).all():
            raise ValueError("Invalid independent reference composition")
        result["ggchem_log_mole_fractions"] = reference[order].tolist()
    return result


def make_setup(case: dict) -> ChemicalSetup:
    """Make atom-referenced potentials defined at the exported knots only."""
    temperatures = jnp.asarray(case["temperature_K"])
    potential = -jnp.asarray(case["lnK_bar"])

    def hvector(t):
        t = jnp.asarray(t)
        index = jnp.clip(jnp.searchsorted(temperatures, t), 0, len(temperatures) - 1)
        return jnp.where((t == temperatures[index])[..., None], potential[index], jnp.nan)

    return ChemicalSetup(jnp.asarray(case["formula_matrix"], dtype=float), jax.jit(hvector),
                         tuple(case["elements"]), tuple(case["species"]),
                         jnp.asarray(case["budget"]), {"source": "Exported native GGChem lnK at explicit knots",
                                                       "standard_pressure_bar": 1., "sampled_only": True})


def select_species(case: dict, names: list[str]) -> dict:
    """Keep named molecular identities; reference equilibria do not survive a removal."""
    result = deepcopy(case)
    columns = [case["species"].index(name) for name in names]
    result["species"] = list(names)
    result["formula_matrix"] = np.asarray(case["formula_matrix"])[:, columns].tolist()
    result["lnK_bar"] = np.asarray(case["lnK_bar"])[:, columns].tolist()
    result.pop("ggchem_log_mole_fractions", None)
    return prepare_case(result)


def controlled_cases(janaf: dict, burcat: dict) -> dict:
    """Separate matched identities, P4O6 replacement and two omitted gas species."""
    if (janaf["elements"] != burcat["elements"]
            or not np.array_equal(janaf["temperature_K"], burcat["temperature_K"])
            or not np.allclose(janaf["budget"], burcat["budget"], rtol=1e-13, atol=0)):
        raise ValueError("Controlled comparisons require identical elemental budgets and temperature knots")
    left, right = np.asarray(janaf["formula_matrix"]), np.asarray(burcat["formula_matrix"])
    common = [name for i, name in enumerate(janaf["species"]) if name in burcat["species"]
              and np.array_equal(left[:, i], right[:, burcat["species"].index(name)])]
    result = {"common_janaf": select_species(janaf, common),
              "common_burcat": select_species(burcat, common)}
    p4o6 = deepcopy(burcat)
    j, k = janaf["species"].index("P4O6"), burcat["species"].index("P4O6")
    if not np.array_equal(left[:, j], right[:, k]):
        raise ValueError("P4O6 identity differs between databases")
    constants = np.asarray(p4o6["lnK_bar"])
    constants[:, k] = np.asarray(janaf["lnK_bar"])[:, j]
    p4o6["lnK_bar"] = constants.tolist()
    p4o6.pop("ggchem_log_mole_fractions", None)
    result["burcat_with_janaf_P4O6"] = prepare_case(p4o6)
    removed = [s for s in burcat["species"] if s in ("H3PO4", "HPO2")]
    if set(removed) != {"H3PO4", "HPO2"}:
        raise ValueError("The requested Burcat gas identities are missing")
    result["burcat_without_H3PO4_HPO2"] = select_species(
        burcat, [name for name in burcat["species"] if name not in removed])
    return result


def solve_case(case: dict) -> dict:
    """Solve all samples and require primitive conservation and mass action."""
    setup = make_setup(case)
    temperatures = jnp.asarray(case["temperature_K"])
    pressure = float(case["pressure_bar"])
    options = EquilibriumOptions(epsilon_crit=1e-13, max_iter=5000, method="vmap_cold")
    result, info = solve_profile(
        setup, T=temperatures, P=jnp.full(len(temperatures), pressure), b=setup.element_vector_reference,
        options=options, return_diagnostics=True)
    cold_converged = np.asarray(info["converged"])
    # The absolute stopping test can leave small relative errors in trace elements.
    # Refine each independent solution from itself, without using the reference code.
    result, info = jax.vmap(lambda t, n, total: solve(
        setup, t, pressure, setup.element_vector_reference,
        init=EquilibriumInit(n, total), options=options, return_diagnostics=True))(
            temperatures, result.ln_n, jnp.log(result.ntot))
    amounts = np.asarray(result.ln_n)
    logx = amounts - logsumexp(amounts, axis=1)[:, None]
    matrix, budget = np.asarray(setup.formula_matrix), np.asarray(setup.element_vector_reference)
    logp = logx + np.log(pressure)
    mass_action = logp - logp[:, case["atom_indices"]] @ matrix - np.asarray(case["lnK_bar"])
    elements = np.max(np.abs(np.exp(amounts) @ matrix.T - budget) / budget, axis=1)
    residuals = np.max(np.abs(mass_action), axis=1)
    passed = (cold_converged & np.asarray(info["converged"]) & np.isfinite(logx).all(axis=1)
              & (elements < 1e-5) & (residuals < 1e-8))
    if not passed.all():
        raise RuntimeError(f"Equilibrium failed at {np.asarray(temperatures)[~passed]} K; "
                           f"element={elements[~passed]}, mass_action={residuals[~passed]}")
    output = {key: case[key] for key in ("species", "elements", "formula_matrix", "budget", "temperature_K", "pressure_bar")}
    output.update({"log_mole_fractions": logx.tolist(), "log_amounts": amounts.tolist(),
                   "element_relative_residual": elements.tolist(), "mass_action_max_abs_ln": residuals.tolist(),
                   "solver_method": "Independent cold solve, then self-seeded refinement; epsilon_crit=1e-13",
                   "cases_passed": int(passed.sum())})
    if "P" in case["elements"]:
        p = case["elements"].index("P")
        output["P_atom_fractions"] = {
            name: (np.exp(amounts[:, i]) * matrix[p, i] / budget[p]).tolist()
            for i, name in enumerate(case["species"]) if matrix[p, i]}
    if "ggchem_log_mole_fractions" in case:
        reference = np.asarray(case["ggchem_log_mole_fractions"])
        differences = np.max(np.abs(logx-reference), axis=1)
        if max(differences) > 2e-5:
            raise RuntimeError(f"Independent solver disagreement: {max(differences)} in ln mole fraction")
        output["ggchem_max_abs_delta_logx"] = differences.tolist()
    return output


def phosphorus_colors(names: list[str]) -> dict:
    """Use one deterministic species palette across native and control figures."""
    import matplotlib.pyplot as plt

    fixed = {"P4O6": "#2455b5", "PH3": "#d62728", "HPO2": "#707070",
             "H3PO4": "#7139a8", "P4O10": "#bd921d", "P2": "#299442",
             "PO": "#67bf54", "P": "#2a808c", "PN": "#b24c91",
             "PS": "#925f35", "PH2": "#d97724", "PH": "#b4a52f"}
    palettes = ("tab20", "tab20b", "tab20c")
    colors = {}
    for name in sorted(set(names)):
        index = int(hashlib.sha256(name.encode()).hexdigest()[:8], 16) % 60
        colors[name] = fixed.get(name, plt.get_cmap(palettes[index // 20])(index % 20))
    return colors


def plot_controls(cases: dict, output: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    names = sorted({name for case in cases.values() for name in case.get("P_atom_fractions", {})})
    colors = phosphorus_colors(names)
    titles = {"common_janaf": "JANAF-based baseline: common species",
              "common_burcat": "Burcat-priority: common species",
              "burcat_with_janaf_P4O6": "Burcat-priority, baseline P4O6 only",
              "burcat_without_H3PO4_HPO2": "Burcat-priority, omit H3PO4 and HPO2"}
    fig, axes = plt.subplots(2, 2, figsize=(12, 9), layout="constrained", sharex=True, sharey=True)
    for ax, (key, title) in zip(axes.flat, titles.items()):
        case = cases[key]
        values = np.asarray(case["log_mole_fractions"]) / np.log(10.)
        for name in names:
            if name not in case["species"]:
                continue
            y = values[:, case["species"].index(name)]
            if max(y) > -18:
                ax.plot(case["temperature_K"], y, label=name, color=colors[name])
        ax.set(xscale="log", xlim=(100, 3000), ylim=(-18, -5.5), title=title,
               xlabel="Temperature [K]", ylabel=r"$\log_{10}(n_i/n_{\rm tot})$")
        if min(case["temperature_K"]) > 100:
            ax.axvspan(100, min(case["temperature_K"]), color="gray", alpha=.15)
        ax.grid(alpha=.15)
        ax.legend(fontsize=7, ncol=2)
    fig.suptitle("Controlled source and species effects | 1 mbar | gas only")
    fig.savefig(output / "ohno_exogibbs_controls.png", dpi=180)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--native-input", type=Path, default=ROOT / "results/ohno_p3/native_cases.json")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "results/ohno_p3")
    parser.add_argument("--native-only", action="store_true", help="Check expanded native networks without repeating the controls")
    args = parser.parse_args()
    args.native_input = args.native_input.resolve()
    jax.config.update("jax_enable_x64", True)
    source = json.loads(args.native_input.read_text())
    native = {name: prepare_case(case) for name, case in source["cases"].items()}
    cases = native if args.native_only else {**native, **controlled_cases(native["janaf"], native["burcat"])}
    outputs = {}
    for name, case in cases.items():
        outputs[name] = solve_case(case)
        print(name, outputs[name]["cases_passed"], "passed", flush=True)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    if not args.native_only:
        plot_controls(outputs, args.output_dir)
    paths = [Path(__file__), args.native_input,
             ROOT / "src/exogibbs/equilibrium/gas/kernel/solver.py"]
    report = {"scope": "Exact sampled native inputs; common named identities and one-factor source/species controls. No new source fit or claim of reproducing Ohno's unpublished inputs.",
              "cases": outputs, "cases_passed": sum(c["cases_passed"] for c in outputs.values()),
              "selected_file_sha256": {str(p.relative_to(ROOT)) if p.is_relative_to(ROOT) else str(p):
                                       hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}}
    (args.output_dir / "exogibbs_summary.json").write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
