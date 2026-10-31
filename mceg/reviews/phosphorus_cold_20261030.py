"""Validate a separate cold phosphorus experiment with pinned NASA9 functions.

This research model is not a production preset. All interventions share one
background; source changes are not attributed to adding missing molecules.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(Path(__file__).parent))

import jax
import jax.numpy as jnp
import numpy as np
from scipy.special import logsumexp

from exogibbs.api.gas import EquilibriumOptions, solve_profile
from exogibbs.thermo.models import ChemicalSetup
from exogibbs.thermo.standard import _minimum_cp, _nasa9


FIXTURE = ROOT / "tests/data/validation/phosphorus_cold/thermo.json"
ELEMENTS = ("H", "He", "C", "N", "O", "P", "S")
TEMPERATURES = np.array([200., 250., 300., 400., 500., 600., 700., 800.,
                         900., 999., 1000., 1001., 1100., 1200., 1400., 1500.])
PRESSURE_BAR = .001
ABUNDANCES = ROOT / "src/exogibbs/data/FastChem4/element_abundances/asplund_2021.dat"


def load_data() -> dict:
    return json.loads(FIXTURE.read_text())


def _arrays(data: dict, names: list) -> tuple:
    records = [data["records"][s] for s in names]
    width = max(len(r["intervals"]) for r in records)
    coefficients = np.zeros((width, len(names), 9))
    bounds = np.zeros((width, len(names), 2))
    bounds[..., 0], bounds[..., 1] = np.inf, -np.inf
    for j, record in enumerate(records):
        for i, interval in enumerate(record["intervals"]):
            coefficients[i, j] = interval["coefficients"]
            bounds[i, j] = interval["bounds_K"]
    lower = max(200., max(r["intervals"][0]["bounds_K"][0] for r in records))
    upper = min(1500., min(r["intervals"][-1]["bounds_K"][1] for r in records))
    return jnp.asarray(coefficients), jnp.asarray(bounds), lower, upper


def _evaluate(temperature, arrays: tuple) -> tuple:
    coefficients, bounds, lower, upper = arrays
    t = jnp.asarray(temperature, dtype=coefficients.dtype)
    safe_t = jnp.where(jnp.isfinite(t) & (t > 0), t, lower)
    shape = t.shape + (coefficients.shape[1],)
    values = [jnp.full(shape, jnp.nan) for _ in range(3)]
    for i in range(len(coefficients)):
        selected = (safe_t[..., None] >= bounds[i, :, 0]) & (safe_t[..., None] <= bounds[i, :, 1])
        values = [jnp.where(selected, new, old)
                  for new, old in zip(_nasa9(safe_t, coefficients[i]), values)]
    valid = jnp.isfinite(t) & (t >= lower) & (t <= upper)
    return tuple(jnp.where(valid[..., None], v, jnp.nan) for v in values)


def thermodynamics(data: dict, names: list, temperature) -> tuple:
    """Return species-last g/RT, s/R and cp/R within the shared cold domain."""
    return _evaluate(temperature, _arrays(data, names))


def make_setup(data: dict, names: list, metallicity: float = 1.) -> ChemicalSetup:
    """Build an atom-referenced model, retaining the source polynomials unchanged."""
    compositions = [data["records"][s]["composition"] for s in names]
    elements = tuple(e for e in ELEMENTS if any(c.get(e, 0) for c in compositions))
    matrix = jnp.asarray([[c.get(e, 0) for c in compositions] for e in elements], dtype=float)
    atom_indices = jnp.asarray([names.index(e) for e in elements])
    arrays = _arrays(data, names)
    abundance = {parts[0]: float(parts[1]) for line in ABUNDANCES.read_text().splitlines()
                 if (parts := line.split()) and not line.startswith("#")}
    budget = np.array([10.**(abundance[e]-12.) * (1. if e in ("H", "He") else metallicity)
                       for e in elements])
    budget /= budget.sum()

    def hvector(t):
        g = _evaluate(t, arrays)[0]
        return g - g[..., atom_indices] @ matrix

    return ChemicalSetup(matrix, jax.jit(hvector), elements, tuple(names), jnp.asarray(budget),
                         {"temperature_range_K": arrays[2:], "standard_pressure_bar": 1.,
                          "source": "Pinned cold phosphorus research fixture", "metallicity": metallicity})


def networks(data: dict) -> dict:
    """Name the interventions separately from the larger network checks."""
    base, extra, cns = (data["groups"][k] for k in ("base", "full_p", "cns"))
    both = base + ["HPO2", "H3PO4"]
    return {"base": (base, 1.), "add_HPO2": (base+["HPO2"], 1.),
            "add_H3PO4": (base+["H3PO4"], 1.), "add_both": (both, 1.),
            "full_P": (both+extra, 1.), "CNS_reduced_P": (both+cns, 1.),
            "CNS_full_P": (both+extra+cns, 1.), "CNS_full_P_100solar": (both+extra+cns, 100.)}


def validate_thermodynamics(data: dict) -> dict:
    """Inspect analytic heat-capacity minima and source interval discontinuities."""
    minima, boundaries, anchors = {}, [], {}
    for name, record in data["records"].items():
        interval_minima = []
        for interval in record["intervals"]:
            lo, hi = interval["bounds_K"]
            lo, hi = max(200., lo), min(1500., hi)
            if lo < hi:
                interval_minima.append(float(_minimum_cp(np.array([interval["coefficients"][:7]]), (lo, hi))[0]))
        minima[name] = min(interval_minima)
        for left, right in zip(record["intervals"][:-1], record["intervals"][1:]):
            t = right["bounds_K"][0]
            if left["bounds_K"][1] != t:
                raise ValueError(f"Noncontiguous NASA intervals: {name}")
            if 200 <= t <= 1500:
                a = np.array([float(v[0]) for v in _nasa9(t, jnp.array([left["coefficients"]]))])
                b = np.array([float(v[0]) for v in _nasa9(t, jnp.array([right["coefficients"]]))])
                boundaries.append({"species": name, "temperature_K": t,
                                   "delta_g_RT": float(b[0]-a[0]), "delta_s_R": float(b[1]-a[1]),
                                   "delta_cp_R": float(b[2]-a[2])})
        g, s, _ = thermodynamics(data, [name], 298.15)
        anchors[name] = float((g[0]+s[0])*298.15)
    if min(minima.values()) < 2.49:
        raise ValueError(f"Unphysical source heat capacity: {minima}")
    # These are gates on the source representation, not adjustments to it.
    if max(abs(r["delta_g_RT"]) for r in boundaries) > 1e-3:
        raise ValueError("Source Gibbs interval jump exceeds 1e-3 RT")
    return {"minimum_cp_R": minima, "boundaries": boundaries, "H298_over_R_K": anchors,
            "scope": "Analytic Cp minima on each interval and reported joins; no polynomial smoothing or refitting."}


def run_case(data: dict, names: list, metallicity: float) -> tuple:
    setup = make_setup(data, names, metallicity)
    lo, hi = setup.metadata["temperature_range_K"]
    temperatures = TEMPERATURES[(TEMPERATURES >= lo) & (TEMPERATURES <= hi)]
    result, info = solve_profile(setup, T=jnp.asarray(temperatures),
                                P=jnp.full(len(temperatures), PRESSURE_BAR), b=setup.element_vector_reference,
                                options=EquilibriumOptions(epsilon_crit=1e-13, max_iter=5000, method="vmap_cold"),
                                return_diagnostics=True)
    ln_n = np.asarray(result.ln_n)
    logx = ln_n - logsumexp(ln_n, axis=-1)[:, None]
    matrix, budget = np.asarray(setup.formula_matrix), np.asarray(setup.element_vector_reference)
    atom_indices = [names.index(e) for e in setup.elements]
    logp = logx + np.log(PRESSURE_BAR)
    mass_action = logp - logp[:, atom_indices] @ matrix + np.asarray(setup.hvector_func(temperatures))
    errors = np.max(np.abs(np.exp(ln_n) @ matrix.T - budget) / budget, axis=1)
    residuals = np.max(np.abs(mass_action), axis=1)
    passed = np.asarray(info["converged"]) & np.isfinite(logx).all(axis=1) & (errors < 1e-5) & (residuals < 1e-8)
    if not passed.all():
        raise RuntimeError(f"Cold model failed at {temperatures[~passed]}: element={errors}, mass_action={residuals}")
    p = setup.elements.index("P")
    fractions = np.exp(ln_n) * matrix[p] / budget[p]
    return setup, {"temperature_K": temperatures.tolist(), "metallicity": metallicity,
                   "pressure_bar": PRESSURE_BAR, "species": names, "elements": setup.elements,
                   "budget": budget.tolist(), "log_amounts": ln_n.tolist(), "log_mole_fractions": logx.tolist(),
                   "P_atom_fractions": {s: fractions[:, j].tolist() for j, s in enumerate(names) if matrix[p, j]},
                   "element_relative_residual": errors.tolist(), "mass_action_max_abs_ln": residuals.tolist()}


def plot_results(cases: dict, output: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 2, figsize=(10, 7), layout="constrained")
    displayed = sorted({s for case in cases.values() for s, f in case["P_atom_fractions"].items() if max(f) > 1e-3})
    colors = {s: plt.get_cmap("tab20")(i) for i, s in enumerate(displayed)}
    titles = ("Base H/He/O/P", "Add HOPO and H3PO4", "Add C/N/S and P competitors", "Same network, 100x solar heavy elements")
    for ax, key, title in zip(axes.flat, ("base", "add_both", "CNS_full_P", "CNS_full_P_100solar"), titles):
        case = cases[key]
        for species, fraction in case["P_atom_fractions"].items():
            if max(fraction) > 1e-3:
                ax.plot(case["temperature_K"], fraction, label=species, color=colors[species])
        ax.set(yscale="log", ylim=(1e-6, 1.2), xlabel="Temperature [K]", ylabel="Fraction of P atoms", title=title)
        ax.legend(fontsize=7, ncol=2)
        ax.grid(alpha=.2)
    fig.suptitle("Cold phosphorus | 1 mbar | gas only | pinned NASA9 sources")
    fig.savefig(output / "phosphorus_cold.png", dpi=160)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=ROOT / "results/phosphorus_cold")
    parser.add_argument("--with-ggchem", action="store_true", help="Build and run the pinned independent GGChem reference")
    args = parser.parse_args()
    jax.config.update("jax_enable_x64", True)
    data = load_data()
    validation = validate_thermodynamics(data)
    cases, reference, arrays = {}, {}, {}
    if args.with_ggchem:
        from phosphorus_ggchem_reference import build_reference, run_reference
        binary = build_reference()
    for name, (species, metallicity) in networks(data).items():
        setup, cases[name] = run_case(data, species, metallicity)
        case = cases[name]
        arrays[name + "_ln_n"] = np.asarray(case["log_amounts"])
        if args.with_ggchem:
            ref = run_reference(data, species, np.asarray(case["temperature_K"]), PRESSURE_BAR,
                                dict(zip(setup.elements, case["budget"])), binary=binary,
                                output_directory=args.output_dir / "ggchem" / name)
            delta = np.asarray(case["log_mole_fractions"]) - ref["ln_mole_fraction"]
            k_error = float(np.max(np.abs(ref["ln_formation_Kp"] + np.asarray(setup.hvector_func(np.asarray(case["temperature_K"]))))))
            logx_error = float(np.max(np.abs(delta)))
            if (k_error > 1e-8 or logx_error > 1e-5 or ref["element_relative_residual"] > 1e-8
                    or ref["mass_action_max_abs_ln"] > 1e-8 or ref["pressure_relative_residual"] > 1e-12):
                raise RuntimeError(f"Independent GGChem comparison failed: {name}, {k_error}, {logx_error}")
            arrays[name + "_ggchem_logx"] = ref.pop("ln_mole_fraction")
            arrays[name + "_ggchem_lnK"] = ref.pop("ln_formation_Kp")
            reference[name] = {**ref, "max_abs_delta_lnK": k_error, "max_abs_delta_logx": logx_error}
        print(name, "passed", len(cases[name]["temperature_K"]), flush=True)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    plot_results(cases, args.output_dir)
    paths = [Path(__file__), FIXTURE, ABUNDANCES, ROOT / "src/exogibbs/thermo/standard.py",
             ROOT / "src/exogibbs/equilibrium/gas/kernel/solver.py"]
    report = {"source_validation": validation, "cases": cases,
              "scope": "Gas-only research model on a declared subset; same-source molecular additions and network sensitivity. Not Ohno's exact inputs or a production curated extension.",
              "selected_file_sha256": {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}}
    (args.output_dir / "exogibbs.json").write_text(json.dumps(report, indent=2) + "\n")
    np.savez_compressed(args.output_dir / "equilibria.npz", **arrays)
    selected = ("PH3", "HPO2", "H3PO4", "P4O6", "P4O9", "P4O10", "P2", "PO", "PS")
    compact = {}
    for name, case in cases.items():
        compact[name] = {k: case[k] for k in ("temperature_K", "pressure_bar", "metallicity", "species", "elements", "budget")}
        compact[name].update({"max_element_relative_residual": max(case["element_relative_residual"]),
                              "max_mass_action_abs_ln": max(case["mass_action_max_abs_ln"]),
                              "P_atom_fractions": {s: case["P_atom_fractions"][s] for s in selected if s in case["P_atom_fractions"]},
                              "PH3_mole_fraction": np.exp(np.array(case["log_mole_fractions"])[:, case["species"].index("PH3")]).tolist()})
    if args.with_ggchem:
        paths.extend([Path(__file__).with_name("phosphorus_ggchem_reference.py"),
                      Path(__file__).with_name("phosphorus_ggchem_driver.f90"),
                      FIXTURE.with_name("ggchem_source.zip")])
    summary = {"cases_passed": sum(len(c["temperature_K"]) for c in cases.values()),
               "ggchem_cases_passed": sum(len(c["temperature_K"]) for c in cases.values()) if reference else 0,
               "cases": compact, "ggchem": reference, "source_validation": validation,
               "equilibria_sha256": hashlib.sha256((args.output_dir / "equilibria.npz").read_bytes()).hexdigest(),
               "selected_file_sha256": {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest() for p in paths},
               "scope": report["scope"]}
    (args.output_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")


if __name__ == "__main__":
    main()
