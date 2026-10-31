"""Stress-test a declared cold gas network without changing production data.

Species deletion, source-anchor alternatives and uniform energy probes are
separate interventions. Their spread is not a statistical uncertainty band.
"""

from __future__ import annotations

import argparse
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import sys

import jax
import jax.numpy as jnp
import numpy as np
from scipy.special import logsumexp

sys.path.insert(0, str(Path(__file__).resolve().parent))
import phosphorus_cold_20261030 as cold

from exogibbs.api.gas import EquilibriumInit, solve


TEMPERATURES = np.array([200., 300., 400., 500., 600., 800., 1000., 1500.])
PRESSURES_BAR = np.array([1e-6, 1e-3, 1.])
METALLICITIES = (1., 100.)
ENERGY_PROBE_KJ_MOL = 8.368
COLORS = {"PH3": "#d62728", "HPO2": "#707070", "H3PO4": "#7139a8",
          "P4O6": "#2455b5", "P4O9": "#1f77b4", "P4O10": "#bd921d",
          "P2": "#299442", "PO": "#67bf54", "PN": "#b24c91", "PS": "#925f35",
          "H3PO3_phosphonic": "#17becf"}


def energy_shift(data: dict, species: str, offset_kJ_mol: float) -> dict:
    """Shift one gas H and G, preserving its S, Cp, intervals and all other gases."""
    result = deepcopy(data)
    record = result["records"][species]
    if record["phase"] != "gas":
        raise ValueError("This intervention requires a gas record")
    offset = offset_kJ_mol * 1000.
    for interval in record["intervals"]:
        interval["coefficients"][7] += offset / data["source_gas_constant_J_mol_K"]
    record["hf298_J_mol"] += offset
    return result


def interventions(data: dict) -> dict:
    """Declare network removals separately from full-network energy probes."""
    names = cold.networks(data)["CNS_full_P"][0]
    result = {"full": {"kind": "baseline", "species": names, "data": data}}
    for key, removed in (("without_H3PO4", ["H3PO4"]), ("without_HOPO", ["HPO2"]),
                         ("without_both", ["H3PO4", "HPO2"]), ("without_P4O9", ["P4O9"])):
        result[key] = {"kind": "species_removal", "removed": removed,
                       "species": [s for s in names if s not in removed], "data": data}
    shifts = [(f"{s}_{sign}", s, value, "uniform_energy_probe")
              for s in ("H3PO4", "HPO2")
              for sign, value in (("lower", -ENERGY_PROBE_KJ_MOL), ("higher", ENERGY_PROBE_KJ_MOL))]
    shifts += [("H3PO4_alternative_anchor", "H3PO4", (-272.41 + 267.378) * 4.184,
                "reported_alternative_anchor_probe"),
               ("HPO2_alternative_anchor", "HPO2", (-106.2 + 112.) * 4.184,
                "reported_alternative_anchor_probe"),
               ("HPO2_experimental_anchor", "HPO2", (-110.6 + 112.) * 4.184,
                "reported_experimental_anchor_probe"),
               ("HPO_theoretical_anchor", "HPO", -93.7216 - (-56.868759),
                "reported_alternative_anchor_probe"),
               ("HPO_experimental_anchor", "HPO", -143.0928 - (-56.868759),
                "reported_experimental_anchor_probe"),
               ("P4O9_higher", "P4O9", 10., "synthetic_energy_probe")]
    for key, species, value, kind in shifts:
        result[key] = {"kind": kind, "species": names, "shifted_species": species,
                       "offset_kJ_mol": value, "data": energy_shift(data, species, value)}
    return result


def expanded_network(data: dict, catalog: dict) -> tuple[dict, list[str], list[dict]]:
    """Add identity-reviewed candidates without replacing any baseline species."""
    result = deepcopy(data)
    names = cold.networks(data)["CNS_full_P"][0].copy()
    aliases_or_isotope = {"PC", "HPO", "PH2", "PH3", "PO2", "PS", "P4", "PT"}
    candidates = [r for r in catalog["records"]
                  if not r["selected"] and r["source_name"] not in aliases_or_isotope]
    counts = {r["source_name"]: sum(s["source_name"] == r["source_name"] for s in candidates)
              for r in candidates}
    manifest = []
    for record in candidates:
        if "accepted_candidate" in record and not record["accepted_candidate"]:
            raise ValueError("Catalog inclusion decision disagrees with identity filter")
        name = record["source_name"]
        if counts[name] > 1:
            name += f"__occ{record['occurrence']}"
        if name in result["records"]:
            raise ValueError(f"Candidate would replace a baseline identity: {name}")
        if record["phase"] != "gas" or not set(record["composition"]) <= set(cold.ELEMENTS):
            raise ValueError(f"Invalid phase or elements for candidate {name}")
        result["records"][name] = deepcopy(record)
        names.append(name)
        manifest.append({"species": name, **{k: record[k] for k in
                         ("source_name", "occurrence", "composition", "source_id", "source_line",
                          "raw_record_sha256", "reference")},
                         "temperature_intervals_K": [r["bounds_K"] for r in record["intervals"]]})
    if len(candidates) != 29:
        raise ValueError(f"Expected 29 reviewed candidate identities, found {len(candidates)}")
    return result, names, manifest


def solve_case(data: dict, names: list[str], metallicity: float,
               temperatures: np.ndarray, pressures: np.ndarray) -> dict:
    """Require independent elemental and all-species mass-action closure."""
    temperatures = np.asarray(temperatures, dtype=float)
    pressures = np.asarray(pressures, dtype=float)
    if (temperatures.ndim != 1 or pressures.shape != temperatures.shape
            or not np.isfinite(temperatures).all() or not np.isfinite(pressures).all()
            or np.any(temperatures < 200.) or np.any(temperatures > 1500.)
            or np.any(pressures <= 0)):
        raise ValueError("Use finite paired temperatures in 200-1500 K and positive pressures")
    setup = cold.make_setup(data, names, metallicity)
    options = cold.EquilibriumOptions(epsilon_crit=1e-13, max_iter=5000, method="vmap_cold")
    result, initial_info = cold.solve_profile(
        setup, T=jnp.asarray(temperatures), P=jnp.asarray(pressures),
        b=setup.element_vector_reference, options=options, return_diagnostics=True)
    # Each independent cold result seeds only its own refinement; no continuation.
    result, info = jax.vmap(lambda t, p, n, total: solve(
        setup, t, p, setup.element_vector_reference, init=EquilibriumInit(n, total),
        options=options, return_diagnostics=True))(
            jnp.asarray(temperatures), jnp.asarray(pressures), result.ln_n, jnp.log(result.ntot))
    amounts = np.asarray(result.ln_n)
    logx = amounts - logsumexp(amounts, axis=1)[:, None]
    matrix, budget = np.asarray(setup.formula_matrix), np.asarray(setup.element_vector_reference)
    atoms = [names.index(e) for e in setup.elements]
    logp = logx + np.log(pressures)[:, None]
    residual = logp - logp[:, atoms] @ matrix + np.asarray(setup.hvector_func(temperatures))
    element_error = np.max(np.abs(np.exp(amounts) @ matrix.T - budget) / budget, axis=1)
    action_error = np.max(np.abs(residual), axis=1)
    initial_converged = np.asarray(initial_info["converged"])
    converged = np.asarray(info["converged"])
    accepted = (initial_converged & converged & np.isfinite(logx).all(axis=1)
                & (element_error < 1e-5) & (action_error < 1e-8))
    p = setup.elements.index("P")
    g_rt = np.asarray(cold.thermodynamics(data, names, temperatures)[0])
    return {
        "temperature_K": temperatures.tolist(), "pressure_bar": pressures.tolist(),
        "metallicity": metallicity, "species": names, "elements": list(setup.elements),
        "formula_matrix": matrix.tolist(), "budget": budget.tolist(),
        "log_amounts": amounts.tolist(), "log_mole_fractions": logx.tolist(),
        "element_chemical_potential_RT": (g_rt[:, atoms] + logp[:, atoms]).tolist(),
        "P_atom_fractions": {s: (np.exp(amounts[:, i]) * matrix[p, i] / budget[p]).tolist()
                             for i, s in enumerate(names) if matrix[p, i]},
        "element_relative_residual": element_error.tolist(),
        "mass_action_max_abs_ln": action_error.tolist(),
        "cold_converged": initial_converged.tolist(), "refined_converged": converged.tolist(),
        "accepted": accepted.tolist(), "cases_passed": int(accepted.sum()),
        "solver_method": "Independent cold starts, one self-seeded refinement, epsilon_crit=1e-13",
    }


def compact_case(case: dict) -> dict:
    """Retain every P curve and primitive numerical acceptance result, without coefficients."""
    excluded = {"log_amounts", "log_mole_fractions", "element_chemical_potential_RT", "formula_matrix"}
    result = {k: v for k, v in case.items() if k not in excluded}
    result["P_log10_mole_fractions"] = {
        s: (np.asarray(case["log_mole_fractions"])[:, case["species"].index(s)] / np.log(10.)).tolist()
        for s in case["P_atom_fractions"]}
    return result


def compare_to_baseline(cases: dict) -> dict:
    """Measure a carrier's largest absolute allocation change at identical T/P/Z."""
    differences = {}
    for key, case in cases.items():
        if case["intervention"] == "full":
            continue
        baseline = cases[f"full_Z{case['metallicity']:g}"]
        baseline_indices = {(t, p): i for i, (t, p) in
                            enumerate(zip(baseline["temperature_K"], baseline["pressure_bar"]))}
        ordered = [baseline_indices[(t, p)] for t, p in zip(case["temperature_K"], case["pressure_bar"])]
        names = sorted(set(case["P_atom_fractions"]) | set(baseline["P_atom_fractions"]))
        delta = np.array([np.asarray(case["P_atom_fractions"].get(s, np.zeros(len(ordered))))
                          - np.asarray(baseline["P_atom_fractions"].get(s, np.zeros(len(baseline_indices))))[ordered]
                          for s in names])
        species_index, sample = np.unravel_index(np.argmax(np.abs(delta)), delta.shape)
        differences[key] = {
            "max_abs_P_fraction_difference": float(abs(delta[species_index, sample])),
            "species": names[species_index], "temperature_K": case["temperature_K"][sample],
            "pressure_bar": case["pressure_bar"][sample], "metallicity": case["metallicity"],
            "signed_difference": float(delta[species_index, sample]),
        }
    return differences


def plot_results(cases: dict, output: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    fig, axes = plt.subplots(2, 3, figsize=(13, 8), layout="constrained", sharex=True, sharey=True)
    displayed = sorted({s for z in METALLICITIES for s, values in cases[f"full_Z{z:g}"]["P_atom_fractions"].items()
                        if max(values) > .01})
    colors = {s: COLORS.get(s, plt.get_cmap("tab20")(i)) for i, s in enumerate(displayed)}
    for row, z in enumerate(METALLICITIES):
        case = cases[f"full_Z{z:g}"]
        for col, pressure in enumerate(PRESSURES_BAR):
            ax = axes[row, col]
            selected = np.asarray(case["pressure_bar"]) == pressure
            for species in displayed:
                values = np.asarray(case["log_mole_fractions"])[selected, case["species"].index(species)] / np.log(10.)
                if max(values) > -18:
                    ax.plot(TEMPERATURES, values, marker=".", label=species, color=colors[species])
            ax.set(xscale="log", xlim=(200, 1500), ylim=(-18, -3.5),
                   title=f"P = {pressure:g} bar | Z = {z:g}", xlabel="Temperature [K]",
                   ylabel=r"$\log_{10}(n_i/n_{\rm tot})$")
            ax.grid(alpha=.2)
    fig.legend([Line2D([], [], color=colors[s], marker=".") for s in displayed], displayed,
               loc="outside lower center", ncol=7, fontsize=8)
    fig.suptitle("Cold P network | 78 selected gases | source intervals enforced | no condensation")
    fig.savefig(output / "cold_network_pressure_comparison.png", dpi=170)
    plt.close(fig)

    keys = [k for k in cases if k.endswith("_Z1") and not k.startswith("full_")]
    fig, axes = plt.subplots(1, 2, figsize=(13, 6), layout="constrained", sharey=True)
    labels = [k.removesuffix("_Z1").replace("_", " ") for k in keys]
    for ax, z in zip(axes, METALLICITIES):
        baseline = cases[f"full_Z{z:g}"]
        bmask = np.asarray(baseline["pressure_bar"]) == .001
        rows = []
        for solar_key in keys:
            case = cases[solar_key.removesuffix("_Z1") + f"_Z{z:g}"]
            mask = np.asarray(case["pressure_bar"]) == .001
            names = set(baseline["P_atom_fractions"]) | set(case["P_atom_fractions"])
            rows.append(np.max([
                np.abs(np.asarray(case["P_atom_fractions"].get(s, np.zeros(len(mask))))[mask]
                       - np.asarray(baseline["P_atom_fractions"].get(s, np.zeros(len(bmask))))[bmask])
                for s in names], axis=0))
        artist = ax.imshow(np.asarray(rows) * 100., aspect="auto", cmap="magma", vmin=0, vmax=100)
        ax.set(xticks=np.arange(len(TEMPERATURES)), xticklabels=[f"{t:g}" for t in TEMPERATURES],
               yticks=np.arange(len(labels)), yticklabels=labels, xlabel="Temperature [K]",
               title=f"1 mbar | Z = {z:g}")
        ax.tick_params(axis="y", labelsize=8)
    fig.colorbar(artist, ax=axes, label="Largest change in one P carrier [percentage points]", shrink=.8)
    fig.suptitle("Network and energy interventions | relative to full 78-gas baseline | not uncertainty bounds")
    fig.savefig(output / "cold_network_interventions.png", dpi=170)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=cold.ROOT / "results/phosphorus_cold_validation/gas")
    parser.add_argument("--candidate-input", type=Path, help="Optional reviewed NASA9 P-candidate catalog")
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    jax.config.update("jax_enable_x64", True)
    data = cold.load_data()
    validation = cold.validate_thermodynamics(data)
    cases, candidate_manifest, candidate_validation = {}, [], None
    for name, intervention in interventions(data).items():
        pressures = PRESSURES_BAR if intervention["kind"] in ("baseline", "species_removal") else np.array([.001])
        temperature = np.tile(TEMPERATURES, len(pressures))
        pressure = np.repeat(pressures, len(TEMPERATURES))
        for z in METALLICITIES:
            key = f"{name}_Z{z:g}"
            case = solve_case(intervention["data"], intervention["species"], z, temperature, pressure)
            case.update({k: v for k, v in intervention.items() if k not in ("data", "species")})
            case["intervention"] = name
            cases[key] = case
            (args.output_dir / "primitive_cases.json").write_text(json.dumps(cases, indent=2) + "\n")
            print(key, case["cases_passed"], "/", len(temperature), "passed", flush=True)
    if args.candidate_input:
        expanded, names, candidate_manifest = expanded_network(data, json.loads(args.candidate_input.read_text()))
        candidate_validation = cold.validate_thermodynamics(expanded)
        for z in METALLICITIES:
            key = f"expanded_catalog_Z{z:g}"
            case = solve_case(expanded, names, z, np.tile(TEMPERATURES, len(PRESSURES_BAR)),
                              np.repeat(PRESSURES_BAR, len(TEMPERATURES)))
            case.update({"intervention": "expanded_catalog", "kind": "catalog_expansion",
                         "added_species": [r["species"] for r in candidate_manifest]})
            cases[key] = case
            (args.output_dir / "primitive_cases.json").write_text(json.dumps(cases, indent=2) + "\n")
            print(key, case["cases_passed"], "/", len(case["temperature_K"]), "passed", flush=True)
    paths = [Path(__file__), Path(cold.__file__), cold.FIXTURE, cold.ABUNDANCES,
             cold.ROOT / "src/exogibbs/thermo/standard.py", cold.ROOT / "src/exogibbs/thermo/models.py",
             cold.ROOT / "src/exogibbs/api/gas.py", cold.ROOT / "src/exogibbs/equilibrium/gas/profile.py",
             cold.ROOT / "src/exogibbs/equilibrium/gas/solve.py",
             cold.ROOT / "src/exogibbs/equilibrium/gas/kernel/solver.py"]
    if args.candidate_input:
        paths.append(args.candidate_input.resolve())
    report = {"scope": "Selected 78-gas CEA/Burcat NASA9 network; pressure/composition coverage and isolated removals or energy probes. Not a complete network or a physical validation.",
              "cases_passed": sum(c["cases_passed"] for c in cases.values()),
              "cases_total": sum(len(c["temperature_K"]) for c in cases.values()),
              "cases": {key: compact_case(case) for key, case in cases.items()},
              "differences": compare_to_baseline(cases), "source_validation": validation,
              "candidate_manifest": candidate_manifest, "expanded_source_validation": candidate_validation,
              "probe_note": "Uniform +/-8.368 kJ/mol shifts are diagnostic probes, not exact source-specific uncertainty bounds. Alternative anchors retain baseline thermal functions and are not complete replacement source fits.",
              "source_gas_constant_J_mol_K": data["source_gas_constant_J_mol_K"],
              "selected_file_sha256": {str(p.relative_to(cold.ROOT)): hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}}
    (args.output_dir / "summary.json").write_text(json.dumps(report, indent=2) + "\n")
    plot_results(cases, args.output_dir)
    if report["cases_passed"] != report["cases_total"]:
        raise RuntimeError("Some samples failed numerical gates; see the saved acceptance arrays")


if __name__ == "__main__":
    main()
