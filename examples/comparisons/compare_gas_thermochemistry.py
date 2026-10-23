"""Measure approved thermochemical choices on one fixed gas-only network.

This offline, table-node study does not install a runtime preset. MCEG is
optional external documentation; neither the calculation nor its tests import it.
"""

from __future__ import annotations

import argparse
from dataclasses import replace
import hashlib
import json
import os
from pathlib import Path
import platform
import subprocess
import sys
import tempfile

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))
os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.environ.setdefault("MPLCONFIGDIR", "/tmp/exogibbs_matplotlib")

import jax
import jax.numpy as jnp
import numpy as np

import audit_atomic_ions as audit
import select_atomic_ions as atomic
import select_major_molecules as molecular
from exogibbs.api.gas import EquilibriumOptions, solve_profile
from exogibbs.presets.fastchem4 import chemsetup
from exogibbs.thermo.models import ChemicalSetup
from exogibbs.utils.fastchem_parity import build_aligned_abundance_vector


TEMPERATURES_K = np.array([1500., 2000., 3000., 4000.])
PRESSURES_BAR = np.array([1.e-6, 1.e-4, 1.e-2, 1., 100.])
MODELS = ("baseline", "atoms_only", "molecules_only", "adopted", "atomic_reference")
EXCLUDED = ("N1-", "Zn1-")
ELEMENT_FILE = ROOT / "src/exogibbs/data/FastChem4/element_abundances/asplund_2021.dat"
LN10 = np.log(10.)
SOLVER_TOLERANCE = 1.e-13
BUDGET_TOLERANCE = 1.e-5
MASS_ACTION_TOLERANCE = 1.e-8
PARITY_TOLERANCE_DEX = 1.e-3
REPORT_FLOOR = 1.e-12


def build_study() -> tuple:
    """Return one catalog/budget and five exact-node h tables in the zero-atom gauge."""
    baseline = chemsetup(path="FastChem4/logK/logK.dat", silent=True)
    keep = np.array([i for i, name in enumerate(baseline.species) if name not in EXCLUDED])
    names = tuple(baseline.species[i] for i in keep)
    setup = replace(baseline, formula_matrix=baseline.formula_matrix[:, keep], species=names,
                    hvector_func=lambda t: baseline.hvector_func(t)[..., keep])
    budget = build_aligned_abundance_vector(
        setup.elements, source="fastchem_file", element_file=ELEMENT_FILE).vector
    h0 = np.asarray(baseline.hvector_func(TEMPERATURES_K))[:, keep]
    tables = np.repeat(h0[None, ...], len(MODELS), axis=0)
    atomic_rows, _ = atomic.compute_adoption()
    molecule_rows, _ = molecular.compute_adoption()
    molecular_data = json.loads(molecular.REFERENCE_PATH.read_text())
    choices = json.loads(molecular.CHOICES_PATH.read_text())["choices"]
    checks = []
    for family, rows in (("atomic", atomic_rows), ("molecular", molecule_rows)):
        for row in rows:
            t = row["temperature_K"]
            name = row.get("fastchem_species", row["species"])
            if t not in TEMPERATURES_K or name in EXCLUDED:
                continue
            ti, si = list(TEMPERATURES_K).index(t), names.index(name)
            selected = row["log10_Kp_selected"]
            if selected is None or not np.isfinite(selected):
                raise ValueError(f"No adopted value for {name} at {t} K")
            if not np.isclose(-h0[ti, si] / LN10, row["log10_Kp_fastchem"], atol=1.e-12, rtol=0):
                raise ValueError("The formation-K convention differs from the production preset")
            targets = (1, 3, 4) if family == "atomic" else (2, 3, 4)
            for model in targets:
                tables[model, ti, si] = -LN10 * selected
            reference = row.get("log10_Kp_reference", row.get("log10_Kp_bc"))
            energy_matched = None
            if family == "atomic":
                tables[4, ti, si] = -LN10 * reference
                reference_role = row["comparison_kind"]
            elif reference is not None:
                molecule = row["species"]
                old = molecular_data["barklem_collet"][molecule]["dissociation"]["dissociation_eV"]
                target = molecular.target_energy(molecule, choices[molecule], molecular_data)
                energy_matched = reference + (target - old * molecular.EV_K) / (t * LN10)
                inherited = (choices[molecule]["thermal_source"] == "fastchem"
                             and choices[molecule]["energy_source"] in ("barklem_collet", "spectroscopy"))
                reference_role = ("retained_BC_thermal_fit_check" if inherited else
                                  "BC2016_source_comparison_shared_ancestry_possible")
            else:
                reference_role = "no_BC_reference"
            checks.append({
                "family": family, "species": name, "temperature_K": t,
                "log10_Kp_baseline": row["log10_Kp_fastchem"],
                "log10_Kp_adopted": selected,
                "delta_log10_Kp": selected - row["log10_Kp_fastchem"],
                "log10_Kp_reference": reference,
                "log10_Kp_reference_at_adopted_energy": energy_matched,
                "reference_role": reference_role,
            })
    return setup, budget, tables, checks


def node_setup(setup: ChemicalSetup, table: np.ndarray) -> ChemicalSetup:
    """Supply finite h only at the four reviewed nodes; never interpolate."""
    if table.shape != (len(TEMPERATURES_K), len(setup.species)) or not np.all(np.isfinite(table)):
        raise ValueError("Expected a finite table in the fixed species order")
    values = jnp.asarray(table)
    nodes = jnp.asarray(TEMPERATURES_K)

    def hvector(temperature):
        matches = jnp.asarray(temperature)[..., None] == nodes
        return jnp.where(jnp.any(matches, axis=-1)[..., None],
                         values[jnp.argmax(matches, axis=-1)], jnp.nan)

    return replace(setup, hvector_func=jax.jit(hvector))


def equilibrium_metrics(setup: ChemicalSetup, budget: np.ndarray, ln_n: np.ndarray,
                        h: np.ndarray, pressures: np.ndarray) -> dict:
    """Check relative element/charge budgets and every formation mass-action law."""
    a = np.asarray(setup.formula_matrix)
    n = np.exp(ln_n)
    nonzero = budget > 0
    charge = a[setup.elements.index("e-")]
    reference_ids = [setup.species.index("e1-" if el == "e-" else el + "1")
                     for el in setup.elements]
    log_pressure = ln_n - np.log(n.sum(axis=-1))[:, None] + np.log(pressures)[:, None]
    mass_action = log_pressure - log_pressure[:, reference_ids] @ a + h
    return {
        "element_relative_residual": np.max(
            np.abs(n @ a.T - budget)[:, nonzero] / budget[nonzero], axis=-1),
        "charge_relative_residual": np.abs(n @ charge) / (n @ np.abs(charge)),
        "mass_action_max_abs_ln": np.max(np.abs(mass_action), axis=-1),
    }


def run_exogibbs(setup: ChemicalSetup, budget: np.ndarray, tables: np.ndarray) -> tuple:
    """Solve all cases from cold starts, without using FastChem results."""
    temperatures = np.repeat(TEMPERATURES_K, len(PRESSURES_BAR))
    pressures = np.tile(PRESSURES_BAR, len(TEMPERATURES_K))
    amounts, diagnostics = [], []
    for name, table in zip(MODELS, tables):
        model = node_setup(setup, table)
        result, info = solve_profile(
            model, T=jnp.asarray(temperatures), P=jnp.asarray(pressures),
            b=jnp.asarray(budget),
            options=EquilibriumOptions(epsilon_crit=SOLVER_TOLERANCE, method="vmap_cold"),
            return_diagnostics=True)
        jax.block_until_ready(result.n)
        ln_n = np.asarray(result.ln_n)
        metrics = equilibrium_metrics(setup, budget, ln_n,
                                      np.repeat(table, len(PRESSURES_BAR), axis=0), pressures)
        for i, (t, p) in enumerate(zip(temperatures, pressures)):
            diagnostics.append({
                "model": name, "temperature_K": t, "pressure_bar": p,
                "converged": bool(info["converged"][i]),
                "iterations": int(info["n_iter"][i]),
                "solver_residual": float(info["final_residual"][i]),
                **{k: float(v[i]) for k, v in metrics.items()},
            })
        amounts.append(ln_n)
        print(name, {k: float(np.max(v)) for k, v in metrics.items()}, flush=True)
    return np.array(amounts), diagnostics


def validate_diagnostics(rows: list) -> None:
    """Reject convergence flags unsupported by relative conservation checks."""
    for row in rows:
        passed = row["converged"] and all(np.isfinite(row[k]) for k in (
            "solver_residual", "element_relative_residual", "charge_relative_residual",
            "mass_action_max_abs_ln"))
        passed = (passed and row["element_relative_residual"] <= BUDGET_TOLERANCE
                  and row["charge_relative_residual"] <= BUDGET_TOLERANCE
                  and row["mass_action_max_abs_ln"] <= MASS_ACTION_TOLERANCE)
        if not passed:
            raise RuntimeError(f"Equilibrium verification failed: {row}")


def fastchem_node_text(setup: ChemicalSetup, h: np.ndarray, temperature: float) -> str:
    """Serialize formation ln K as constant fits valid only at this one node.

    Charge coefficients include negative electron counts for positive ions.
    This is a comparator input, never a fit for a continuous temperature range.
    """
    if temperature not in TEMPERATURES_K or h.shape != (len(setup.species),):
        raise ValueError("Invalid single-node FastChem export")
    lines = ["# Single-temperature formation ln K at standard pressure 1 bar.",
             f"# Valid only at {temperature:g} K; generated for implementation verification.",
             "# All five coefficients follow the FastChem gas convention."]
    reference_names = {"e1-" if el == "e-" else el + "1" for el in setup.elements}
    for si, name in enumerate(setup.species):
        if name in reference_names:
            if h[si] != 0:
                raise ValueError("FastChem export requires the zero-atom/electron gauge")
            continue
        composition = " ".join(f"{el} {int(n)}" for el, n in zip(
            setup.elements, np.asarray(setup.formula_matrix)[:, si]) if n != 0)
        lines.extend([f"{name} {name} : {composition}", f"0 0 {-h[si]:.17e} 0 0", ""])
    return "\n".join(lines) + "\n"


def run_fastchem(setup: ChemicalSetup, budget: np.ndarray, tables: np.ndarray,
                 ln_n: np.ndarray, executable: Path) -> tuple:
    """Compare independent FastChem runs after ExoGibbs has finished."""
    from benchmarks.fastchem4.comparison import align_species_values, occurrence_keys
    from benchmarks.fastchem4.fastchem_executable import run_fastchem_executable

    rows, all_ln_n, hashes = [], [], {}
    reference_catalog = sorted(occurrence_keys(setup.species))
    for mi, (name, table) in enumerate(zip(MODELS, tables)):
        model_amounts = []
        for ti, t in enumerate(TEMPERATURES_K):
            with tempfile.TemporaryDirectory(prefix="exogibbs_thermochemistry_") as directory:
                path = Path(directory) / "logK.dat"
                payload = fastchem_node_text(setup, table[ti], t)
                path.write_text(payload)
                hashes[f"{name}/{t:g}K"] = hashlib.sha256(payload.encode()).hexdigest()
                result = run_fastchem_executable(
                    executable=executable, temperatures=np.full(len(PRESSURES_BAR), t),
                    pressures=PRESSURES_BAR, element_abundance_file=ELEMENT_FILE,
                    gas_logk_file=path, chemistry_mode="gas", chemistry_accuracy=1.e-10,
                    element_conservation_accuracy=1.e-10)
            if sorted(occurrence_keys(result.gas_names)) != reference_catalog:
                raise RuntimeError("FastChem species catalog differs from the fixed network")
            if not np.all(result.converged & result.elements_conserved):
                raise RuntimeError(f"FastChem convergence/conservation failure for {name} at {t} K")
            density = align_species_values(setup.species, result.gas_names, result.gas_number_densities)
            if np.any(density <= 0):
                raise RuntimeError("FastChem returned nonpositive species densities")
            fc_ln_n = np.log(density / result.total_element_density[:, None])
            model_amounts.append(fc_ln_n)
            metrics = equilibrium_metrics(setup, budget, fc_ln_n,
                                          np.repeat(table[ti][None, :], len(PRESSURES_BAR), axis=0),
                                          PRESSURES_BAR)
            ex_ln_n = ln_n[mi, ti * len(PRESSURES_BAR):(ti + 1) * len(PRESSURES_BAR)]
            ex_x = np.exp(ex_ln_n) / np.exp(ex_ln_n).sum(axis=-1)[:, None]
            fc_x = density / density.sum(axis=-1)[:, None]
            delta = np.log10(ex_x / fc_x)
            for pi, p in enumerate(PRESSURES_BAR):
                significant = np.maximum(ex_x[pi], fc_x[pi]) >= REPORT_FLOOR
                maximum = float(np.max(np.abs(delta[pi, significant])))
                rows.append({
                    "model": name, "temperature_K": t, "pressure_bar": p,
                    "max_abs_dex_x_ge_1e_minus12": maximum,
                    "electron_dex": float(delta[pi, setup.species.index("e1-")]),
                    "Hminus_dex": float(delta[pi, setup.species.index("H1-")]),
                    "total_amount_relative_difference": float(
                        np.exp(ex_ln_n[pi]).sum() / np.exp(fc_ln_n[pi]).sum() - 1),
                    **{k: float(v[pi]) for k, v in metrics.items()},
                    "converged": True,
                })
                if (maximum > PARITY_TOLERANCE_DEX
                        or abs(rows[-1]["electron_dex"]) > PARITY_TOLERANCE_DEX
                        or abs(rows[-1]["Hminus_dex"]) > PARITY_TOLERANCE_DEX
                        or metrics["element_relative_residual"][pi] > BUDGET_TOLERANCE
                        or metrics["charge_relative_residual"][pi] > BUDGET_TOLERANCE
                        or abs(rows[-1]["total_amount_relative_difference"]) > BUDGET_TOLERANCE):
                    raise RuntimeError(f"FastChem implementation disagreement: {rows[-1]}")
        all_ln_n.append(np.concatenate(model_amounts))
        print(f"FastChem verified: {name}", flush=True)
    return np.array(all_ln_n), rows, hashes


def abundance_tables(setup: ChemicalSetup, ln_n: np.ndarray) -> tuple:
    """Report mole fractions and factorial source effects, retaining trace values."""
    logx = (ln_n - np.log(np.exp(ln_n).sum(axis=-1))[..., None]) / LN10
    data = json.loads(molecular.REFERENCE_PATH.read_text())
    labels = {"e-": "e1-", "H-": "H1-", "H": "H1", "OH": "H1O1", "CH": "C1H1",
              "CN": "C1N1", "C2": "C2",
              **{name: record["fastchem_species"] for name, record in data["species"].items()},
              **{name + "+": name + "1+" for name in ("Na", "K", "Mg", "Ca", "Fe", "Ni", "Co", "Cu", "Ti", "V")},
              **{name: name + "1" for name in ("Na", "K", "Mg", "Ca", "Fe", "Ni", "Co", "Cu", "Ti", "V")}}
    rows, summaries = [], []
    for label, name in labels.items():
        si = setup.species.index(name)
        samples = []
        for point, (t, p) in enumerate((t, p) for t in TEMPERATURES_K for p in PRESSURES_BAR):
            value = logx[:, point, si]
            row = {"species": label, "fastchem_species": name, "temperature_K": t, "pressure_bar": p,
                   **{model + "_x": float(10 ** v) for model, v in zip(MODELS, value)},
                   "adopted_over_baseline": float(10 ** (value[3] - value[0])),
                   "atoms_only_dex": float(value[1] - value[0]),
                   "molecules_only_dex": float(value[2] - value[0]),
                   "interaction_dex": float(value[3] - value[1] - value[2] + value[0]),
                   "atomic_reference_over_adopted": float(10 ** (value[4] - value[3]))}
            samples.append(row)
        rows.extend(samples)
        significant = [r for r in samples if min(r["baseline_x"], r["adopted_x"]) >= REPORT_FLOOR]
        worst = max(significant, key=lambda r: abs(np.log10(r["adopted_over_baseline"]))) if significant else None
        summaries.append({
            "species": label, "ratio_min_all_points": min(r["adopted_over_baseline"] for r in samples),
            "ratio_max_all_points": max(r["adopted_over_baseline"] for r in samples),
            "significant_points": len(significant),
            **{key: None if worst is None else worst[key] for key in (
                "temperature_K", "pressure_bar", "baseline_x", "adopted_x", "adopted_over_baseline")},
            "atomic_reference_max_abs_dex": max(abs(np.log10(r["atomic_reference_over_adopted"])) for r in samples),
        })
    return rows, summaries


def plot_impact(setup: ChemicalSetup, ln_n: np.ndarray, path: Path) -> None:
    """Save a pressure-temperature comparison without implying interpolation."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    logx = (ln_n - np.log(np.exp(ln_n).sum(axis=-1))[..., None]) / LN10
    fig, axes = plt.subplots(2, 3, figsize=(10.5, 6), constrained_layout=True)
    for ax, (label, name) in zip(axes.flat, (("Electron", "e1-"), ("H-", "H1-"),
                                          ("HCN", "C1H1N1_hcn"), ("H2O", "H2O1"),
                                          ("TiO", "O1Ti1"), ("MgH", "H1Mg1"))):
        si = setup.species.index(name)
        ratio = (logx[3, :, si] - logx[0, :, si]).reshape(len(TEMPERATURES_K), -1)
        image = ax.imshow(ratio, origin="lower", cmap="RdBu_r", vmin=-0.25, vmax=0.25, aspect="auto")
        for ti in range(len(TEMPERATURES_K)):
            for pi in range(len(PRESSURES_BAR)):
                ax.text(pi, ti, f"{10**ratio[ti, pi]:.3f}", ha="center", va="center", fontsize=8)
        ax.set(title=label, xticks=range(5), xticklabels=["1e-6", "1e-4", "0.01", "1", "100"],
               yticks=range(4), yticklabels=[f"{t:g}" for t in TEMPERATURES_K],
               xlabel="Pressure [bar]", ylabel="Temperature [K]")
    fig.colorbar(image, ax=axes, label="log10(adopted / baseline mole fraction)", shrink=0.8)
    fig.suptitle("Fixed 537-species solar gas network; numbers show abundance ratios")
    fig.savefig(path, dpi=180)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=ROOT / "results/gas_thermochemistry_impact")
    parser.add_argument("--fastchem-executable", type=Path)
    parser.add_argument("--fastchem-version-label", default="unspecified")
    args = parser.parse_args()
    jax.config.update("jax_enable_x64", True)
    setup, budget, tables, checks = build_study()
    ln_n, diagnostics = run_exogibbs(setup, budget, tables)
    validate_diagnostics(diagnostics)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    audit._write_csv(args.output_dir / "diagnostics.csv", diagnostics)
    audit._write_csv(args.output_dir / "equilibrium_constants.csv", checks)
    rows, summaries = abundance_tables(setup, ln_n)
    audit._write_csv(args.output_dir / "abundances.csv", rows)
    audit._write_csv(args.output_dir / "species_summary.csv", summaries)
    plot_impact(setup, ln_n, args.output_dir / "abundance_ratios.png")
    fastchem_ln_n, parity, input_hashes = np.empty(0), [], {}
    if args.fastchem_executable:
        fastchem_ln_n, parity, input_hashes = run_fastchem(
            setup, budget, tables, ln_n, args.fastchem_executable.resolve(strict=True))
        audit._write_csv(args.output_dir / "fastchem_parity.csv", parity)
    else:
        (args.output_dir / "fastchem_parity.csv").unlink(missing_ok=True)
    np.savez_compressed(args.output_dir / "equilibria.npz", ln_n=ln_n, h=tables,
                        species=setup.species, elements=setup.elements,
                        formula_matrix=np.asarray(setup.formula_matrix), budget=budget,
                        models=MODELS, temperature_K=TEMPERATURES_K, pressure_bar=PRESSURES_BAR,
                        fastchem_ln_n=fastchem_ln_n)
    paths = [Path(__file__), Path(audit.__file__), Path(atomic.__file__), Path(molecular.__file__),
             atomic.DATA_PATH, audit.REFERENCE_PATH, molecular.REFERENCE_PATH, molecular.CHOICES_PATH,
             audit.FASTCHEM_PATH, audit.NASA_PATH, ELEMENT_FILE,
             ROOT / "src/exogibbs/equilibrium/gas/kernel/solver.py",
             ROOT / "src/exogibbs/presets/fastchem4.py",
             ROOT / "benchmarks/fastchem4/fastchem_executable.py"]
    receipt = {
        "command": " ".join([sys.executable, *sys.argv]),
        "base_revision": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
        "environment": {"python": platform.python_version(), "jax": jax.__version__,
                        "numpy": np.__version__, "jax_enable_x64": jax.config.jax_enable_x64,
                        "backend": jax.default_backend()},
        "network": {"species_count": len(setup.species), "element_count": len(setup.elements) - 1,
                    "excluded_from_every_model": EXCLUDED, "charge_budget": 0.,
                    "C_over_O": float(budget[setup.elements.index("C")] / budget[setup.elements.index("O")]),
                    "elements": setup.elements, "normalized_budget": budget.tolist()},
        "temperature_K": TEMPERATURES_K.tolist(), "pressure_bar": PRESSURES_BAR.tolist(),
        "models": MODELS, "equilibrium_cases": len(diagnostics),
        "solver": {"method": "vmap_cold", "epsilon_crit": SOLVER_TOLERANCE,
                   "element_and_charge_relative_gate": BUDGET_TOLERANCE,
                   "mass_action_ln_gate": MASS_ACTION_TOLERANCE},
        "verification_maxima": {k: max(r[k] for r in diagnostics) for k in (
            "element_relative_residual", "charge_relative_residual", "mass_action_max_abs_ln")},
        "fastchem": {"status": "passed" if parity else "not_run", "cases": len(parity),
                     "version_label": args.fastchem_version_label,
                     "executable_sha256": (hashlib.sha256(args.fastchem_executable.read_bytes()).hexdigest()
                                           if args.fastchem_executable else None),
                     "single_temperature_input_sha256": input_hashes,
                     "max_abs_dex_x_ge_1e_minus12": max((r["max_abs_dex_x_ge_1e_minus12"] for r in parity), default=None),
                     "interpretation": "Implementation verification using identical node-specific thermochemistry; not physical validation."},
        "selected_file_sha256": {str(p.resolve().relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest()
                                 for p in paths},
        "output_sha256": {name: hashlib.sha256((args.output_dir / name).read_bytes()).hexdigest()
                          for name in ("diagnostics.csv", "equilibrium_constants.csv", "abundances.csv",
                                       "species_summary.csv", "abundance_ratios.png", "equilibria.npz")
                          + (("fastchem_parity.csv",) if parity else ())},
        "interpretation": "Fixed-network ideal gas sensitivity, not a production preset or an abundance accuracy certificate. Atomic-reference calculations share adopted thresholds and four partition models; they probe remaining atomic thermal differences. Molecular BC comparisons include shared ancestry and unresolved conflicts. No condensates, nonideal effects, interpolation, isotope resolution or uncertainty propagation.",
    }
    (args.output_dir / "receipt.json").write_text(json.dumps(receipt, indent=2) + "\n")


if __name__ == "__main__":
    main()
