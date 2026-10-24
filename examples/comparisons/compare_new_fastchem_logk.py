"""Compare Daniel's preliminary FastChem ion fits with the current curated preset.

Reuse the two prescribed columns and independent closure gates from PR #284.
All equilibria use ExoGibbs and the identical curated 537-species network.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
import platform
import shutil
import subprocess
import sys
import tempfile
import zipfile

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

import jax
import jax.numpy as jnp
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
import scipy
from scipy.special import logsumexp

from examples.comparisons.demo_curated_atmosphere import closure, profiles, require_closure
from exogibbs.api.gas import EquilibriumOptions, solve_profile
from exogibbs.presets import curated
from exogibbs.presets.fastchem import _parse_fastchem_coeffs, _set_element_species, logk
from exogibbs.thermo.models import ChemicalSetup
from exogibbs.thermo.stoichiometry import build_formula_matrix


MODELS = ("curated", "new", "original")
LABELS = {"curated": "ExoGibbs default (curated)", "new": "New FastChem data (preliminary)",
          "original": "Original FastChem 4 data"}
STYLES = {"curated": "-", "new": "--", "original": ":"}
SPECIES = {"Cu1": ("Cu", "#b85b00"), "Cu1+": (r"Cu$^+$", "#ccab00"),
           "Ni1": ("Ni", "#0072b2"), "Ni1+": (r"Ni$^+$", "#56b4e9"),
           "Co1": ("Co", "#009e73"), "e1-": (r"e$^-$", "#aa3377"),
           "H1-": (r"H$^-$", "#4477aa")}
SIGNIFICANT_X = 1e-12
SOURCE_URL = "https://github.com/NewStrangeWorlds/FastChem/issues/10#issuecomment-6366561070"


def comparison_setups(path: Path) -> dict:
    """Align both FastChem tables by species and check their stoichiometry."""
    default = curated.chemsetup()
    setups = {"curated": default}
    for name, source in (("new", path), ("original", ROOT / "src/exogibbs/data/FastChem4/logK/logK.dat")):
        coefficients, components = _parse_fastchem_coeffs(source.read_text())
        _, references, zeros = _set_element_species(default.elements)
        coefficients, components = {**zeros, **coefficients}, {**references, **components}
        missing = set(default.species) - coefficients.keys()
        if missing:
            raise ValueError(f"Missing species in {source}: {sorted(missing)}")
        matrix = build_formula_matrix({s: components[s] for s in default.species}, default.elements)
        np.testing.assert_array_equal(matrix, default.formula_matrix)
        fits = jnp.asarray([coefficients[s] for s in default.species])

        def hvector(t, fits=fits):
            # The tabulated polynomial is natural ln(Kp), dimensionless at 1 bar.
            return -jax.vmap(logk, in_axes=(None, 0), out_axes=-1)(jnp.asarray(t), fits)

        setups[name] = ChemicalSetup(
            default.formula_matrix, jax.jit(hvector), default.elements, default.species,
            default.element_vector_reference, {"relative_charge_convergence": True},
        )
    return setups


def audit_input_files(directory: Path) -> dict:
    """Check the archive and distinguish coefficient changes from catalog changes."""
    audits = {}
    with zipfile.ZipFile(directory / "logK.zip") as archive:
        for filename in ("logK.dat", "logK_extended.dat", "logK_hoeijmakers2019.dat"):
            path = directory / filename
            if path.read_bytes() != archive.read(filename):
                raise ValueError(f"Extracted file differs from the supplied archive: {filename}")
            new, components = _parse_fastchem_coeffs(path.read_text())
            original, old_components = _parse_fastchem_coeffs(
                (ROOT / "src/exogibbs/data/FastChem4/logK" / filename).read_text())
            changed = sorted(s for s in new.keys() & original.keys() if new[s] != original[s])
            atomic_ions = [s for s in changed if components[s].get("e-", 0)
                           and sum(v for e, v in components[s].items() if e != "e-") == 1]
            audits[filename] = {
                "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                "records": len(new), "changed_coefficients": changed,
                "changed_atomic_ions": atomic_ions,
                "changed_other_species": sorted(set(changed) - set(atomic_ions)),
                "added": sorted(new.keys() - original.keys()),
                "removed": sorted(original.keys() - new.keys()),
                "changed_compositions": sorted(s for s in components.keys() & old_components.keys()
                                               if components[s] != old_components[s]),
            }
    return audits


def write_csv(path: Path, rows: list) -> None:
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=rows[0])
        writer.writeheader()
        writer.writerows(rows)


def summarize(profile: str, species: tuple, logx: dict, temperature: np.ndarray,
              pressure: np.ndarray) -> list:
    """Rank per-species extrema only where either mole fraction reaches 1e-12."""
    rows = []
    for numerator, denominator in (("new", "curated"), ("original", "curated"), ("new", "original")):
        delta = (logx[numerator] - logx[denominator]) / np.log(10.)
        for j, name in enumerate(species):
            eligible = np.maximum(logx[numerator][:, j], logx[denominator][:, j]) >= np.log(SIGNIFICANT_X)
            if not np.any(eligible):
                continue
            i = int(np.argmax(np.where(eligible, np.abs(delta[:, j]), -np.inf)))
            rows.append({"profile": profile, "numerator": numerator, "denominator": denominator,
                         "species": name, "temperature_K": float(temperature[i]),
                         "pressure_bar": float(pressure[i]),
                         "numerator_x": float(np.exp(logx[numerator][i, j])),
                         "denominator_x": float(np.exp(logx[denominator][i, j])),
                         "ratio": float(np.exp(delta[i, j] * np.log(10.))),
                         "delta_dex": float(delta[i, j])})
    return sorted(rows, key=lambda r: abs(r["delta_dex"]), reverse=True)


def plot_profile(name: str, species: tuple, logx: dict, temperature: np.ndarray,
                 pressure: np.ndarray, output: Path) -> None:
    # Keep the translucent solid line visible around overlapping dark dashes.
    appearance = {"curated": {"lw": 4., "alpha": .3, "zorder": 2},
                  "new": {"lw": 1.8, "alpha": 1., "zorder": 4},
                  "original": {"lw": 1.8, "alpha": 1., "zorder": 3}}
    fig, axes = plt.subplots(2, 3, figsize=(13.5, 9.6), sharey=True, layout="constrained")
    fig.get_layout_engine().set(rect=(0., .13, 1., .87))
    title = "4000 K isothermal column" if name == "isothermal" else "Prescribed temperature inversion"
    fig.suptitle(title + " | solar composition, same 537 species", fontsize=16)
    for ax in axes.flat:
        ax.set(yscale="log", ylim=(100., 1e-6), ylabel="Pressure [bar]")
        ax.grid(alpha=.2)
    axes[0, 0].plot(temperature, pressure, color="#333333", lw=2.)
    axes[0, 0].set(xlabel="Temperature [K]", title="Same T(P) and elemental budget", xlim=(2400., 4100.))
    for ax, names, title in ((axes[0, 1], ("Cu1", "Cu1+"), "Copper: neutral / ion"),
                              (axes[0, 2], ("Ni1", "Ni1+"), "Nickel: neutral / ion"),
                              (axes[1, 1], ("e1-", "H1-"), "Electrons and negative hydrogen")):
        for s in names:
            label, color = SPECIES[s]
            for model in MODELS:
                ax.plot(np.exp(logx[model][:, species.index(s)]), pressure, STYLES[model],
                        color=color, **appearance[model], label=label if model == "curated" else None)
        ax.set(xscale="log", xlabel="Mole fraction", title=title)
        ax.legend(fontsize=10)
    for ax, names, percent in ((axes[1, 0], ("Cu1", "Ni1", "Co1"), False),
                               (axes[1, 2], ("e1-", "H1-"), True)):
        for s in names:
            label, color = SPECIES[s]
            for model in ("new", "original"):
                ratio = np.exp(logx[model][:, species.index(s)] - logx["curated"][:, species.index(s)])
                ax.plot(100. * (ratio - 1.) if percent else ratio, pressure, STYLES[model],
                        color=color, lw=1.8, label=label if model == "new" else None)
        ax.axvline(0. if percent else 1., color="0.6", lw=.8)
        ax.set(title="Charge carriers: relative change" if percent else "Neutral metals: abundance ratio",
               xlabel="100 × (model / default − 1) [%]" if percent else "Model / default")
        if not percent:
            ax.set_xscale("log")
        ax.legend(fontsize=10)
    fig.legend(handles=[Line2D([], [], color=".25", ls=STYLES[m], **appearance[m], label=LABELS[m]) for m in MODELS],
               loc="lower center", bbox_to_anchor=(.5, .055), ncol=3, fontsize=10)
    fig.text(.5, .015, "Ratio panels: dashed = new FastChem / default; dotted = original FastChem / default.\n"
             "ExoGibbs solver for every curve; prescribed ideal-gas columns, without condensation or photoionization.",
             ha="center", va="bottom", fontsize=10)
    for ext in ("png", "pdf"):
        fig.savefig(output / f"{name}.{ext}", dpi=150)
    plt.close(fig)


def compare_constants(setups: dict, output: Path) -> None:
    temperature = np.linspace(1500., 4000., 251)
    logk10 = {m: -np.asarray(s.hvector_func(temperature)) / np.log(10.) for m, s in setups.items()}
    species = setups["curated"].species
    rows = []
    for j, s in enumerate(species):
        for i in range(0, len(temperature), 50):
            t = temperature[i]
            rows.append({"species": s, "temperature_K": t,
                         **{f"log10_Kp_{m}": logk10[m][i, j] for m in MODELS},
                         "new_minus_curated_dex": logk10["new"][i, j] - logk10["curated"][i, j]})
    write_csv(output / "equilibrium_constants.csv", rows)
    fig, axes = plt.subplots(2, 3, figsize=(12., 7.), layout="constrained")
    for ax, s in zip(axes.flat, ("Cu1+", "Ni1+", "Co1+", "Fe1+", "Ca1+", "H1-")):
        j = species.index(s)
        for m, color in (("new", "#0072b2"), ("original", "#999999")):
            ax.plot(temperature, logk10[m][:, j] - logk10["curated"][:, j], STYLES[m],
                    color=color, label=LABELS[m])
        ax.axhline(0., color=".5", lw=.8)
        ax.set(title=s.replace("1", ""), xlabel="Temperature [K]", ylabel="log10(Kp / Kp,default)")
        ax.grid(alpha=.2)
    axes[0, 0].legend(fontsize=8)
    fig.suptitle("Atomic-ion formation constants relative to ExoGibbs default | standard pressure 1 bar")
    for ext in ("png", "pdf"):
        fig.savefig(output / f"ion_constants.{ext}", dpi=150)
    plt.close(fig)


def plot_residuals(output: Path) -> None:
    """Resolve the small new/default differences hidden by the original fits."""
    fig, axes = plt.subplots(2, 3, figsize=(13.5, 9.), sharey=True, layout="constrained")
    groups = (("Neutral metals", (("Cu1", "Cu"), ("Ni1", "Ni"), ("Co1", "Co"))),
              ("Charge carriers", (("e1-", r"e$^-$"), ("H1-", r"H$^-$"))),
              ("Selected molecules", (("C1H1N1_hcn", "HCN"), ("S2", r"S$_2$"),
                                      ("H2S1", r"H$_2$S"), ("H2O1", r"H$_2$O"), ("S1Si1", "SiS"))))
    for row, profile in enumerate(profiles()):
        with np.load(output / f"{profile}.npz") as data:
            species = list(data["species"])
            logx = {m: data[f"ln_n_{m}"] - logsumexp(data[f"ln_n_{m}"], axis=-1)[:, None]
                    for m in ("new", "curated")}
            for ax, (title, selected) in zip(axes[row], groups):
                for s, label in selected:
                    j = species.index(s)
                    change = 100. * np.expm1(logx["new"][:, j] - logx["curated"][:, j])
                    if title == "Selected molecules":
                        eligible = np.maximum(logx["new"][:, j], logx["curated"][:, j]) >= np.log(SIGNIFICANT_X)
                        change = np.where(eligible, change, np.nan)
                    color = SPECIES[s][1] if s in SPECIES else None
                    ax.plot(change, data["pressure_bar"], label=label, color=color)
                ax.axvline(0., color=".6", lw=.8)
                ax.set(yscale="log", ylim=(100., 1e-6), ylabel="Pressure [bar]",
                       xlabel="100 × (new / default − 1) [%]", title=profile.capitalize() + " | " + title)
                ax.grid(alpha=.2)
                ax.legend(fontsize=9)
    fig.suptitle("Residual differences: new FastChem data versus ExoGibbs default\n"
                 "Molecular curves shown only where either mole fraction reaches 1e−12", fontsize=14)
    for ext in ("png", "pdf"):
        fig.savefig(output / f"residuals.{ext}", dpi=150)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path,
                        help="Directory containing logK.zip and its tables; defaults to the archived ZIP")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "results/new_fastchem_logk")
    args = parser.parse_args()
    jax.config.update("jax_enable_x64", True)
    output = args.output_dir
    output.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory() as temporary:
        directory = args.input_dir or Path(temporary)
        if args.input_dir is None:
            shutil.copyfile(ROOT / "results/new_fastchem_logk/logK.zip", directory / "logK.zip")
            with zipfile.ZipFile(directory / "logK.zip") as archive:
                for filename in ("logK.dat", "logK_extended.dat", "logK_hoeijmakers2019.dat"):
                    (directory / filename).write_bytes(archive.read(filename))
        input_audit = audit_input_files(directory)
        if (directory / "logK.zip").resolve() != (output / "logK.zip").resolve():
            shutil.copyfile(directory / "logK.zip", output / "logK.zip")
        setups = comparison_setups(directory / "logK.dat")
    compare_constants(setups, output)
    summary, audits, plotted = [], [], []
    for name, (pressure, temperature) in profiles().items():
        logx, amounts = {}, {}
        for model, setup in setups.items():
            result, info = solve_profile(
                setup, T=jnp.asarray(temperature), P=jnp.asarray(pressure), b=setup.element_vector_reference,
                options=EquilibriumOptions(epsilon_crit=1e-13, method="vmap_cold"), return_diagnostics=True)
            ln_n = np.asarray(result.ln_n)
            if not np.all(info["converged"]) or not np.all(np.isfinite(ln_n)):
                raise RuntimeError(f"Nonconverged or nonfinite {name}/{model} column")
            audit = closure(setup, ln_n, temperature, pressure)
            require_closure(audit)
            audits.append({"profile": name, "model": model, "layers": len(pressure), **audit})
            amounts[model] = ln_n
            logx[model] = ln_n - logsumexp(ln_n, axis=-1)[:, None]
            for s in SPECIES:
                plotted.extend({"profile": name, "model": model, "species": s, "temperature_K": t,
                                "pressure_bar": p, "mole_fraction": x}
                               for t, p, x in zip(temperature, pressure, np.exp(logx[model][:, setup.species.index(s)])))
            print(name, model, audit, flush=True)
        setup = setups["curated"]
        np.savez_compressed(output / f"{name}.npz", temperature_K=temperature, pressure_bar=pressure,
                            species=setup.species, elements=setup.elements, formula_matrix=setup.formula_matrix,
                            budget=setup.element_vector_reference, **{f"ln_n_{m}": n for m, n in amounts.items()})
        summary.extend(summarize(name, setup.species, logx, temperature, pressure))
        plot_profile(name, setup.species, logx, temperature, pressure, output)
    plot_residuals(output)
    write_csv(output / "summary.csv", summary)
    write_csv(output / "plotted_abundances.csv", plotted)
    sources = [Path(__file__), ROOT / "examples/comparisons/demo_curated_atmosphere.py",
               ROOT / "src/exogibbs/presets/curated.py", ROOT / "src/exogibbs/presets/fastchem.py",
               ROOT / "src/exogibbs/data/curated/gas_v1.json",
               ROOT / "src/exogibbs/data/FastChem4/element_abundances/asplund_2021.dat",
               ROOT / "src/exogibbs/equilibrium/gas/kernel/solver.py"]
    sources.extend(sorted((ROOT / "src/exogibbs/data/FastChem4/logK").glob("logK*.dat")))
    receipt = {
        "source_url": SOURCE_URL, "source_status": "preliminary; comment dated 2026-10-23",
        "git_head": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
        "dataset": "curated-gas-v1", "species_count": len(setups["curated"].species),
        "scope": "Same 537 species, formula matrix, normalized Asplund 2021 budget and ExoGibbs solver. "
                 "N1- and Zn1- omitted from both FastChem tables to match curated. "
                 "Only logK.dat used in equilibria; other tables receive a file audit only.",
        "environment": {"python": platform.python_version(), "jax": jax.__version__,
                        "numpy": np.__version__, "scipy": scipy.__version__, "matplotlib": matplotlib.__version__},
        "backend": jax.default_backend(), "dtype": "float64", "epsilon_crit": 1e-13,
        "relative_charge_convergence": True, "cases_passed": sum(r["layers"] for r in audits),
        "audits": audits, "input_audit": input_audit,
        "ranking_mole_fraction_floor": SIGNIFICANT_X,
        "constants_csv_temperatures_K": [1500., 2000., 2500., 3000., 3500., 4000.],
        "selected_file_sha256": {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest() for p in sources},
        "output_sha256": {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(output.iterdir())
                          if p.suffix in (".npz", ".png", ".pdf", ".csv", ".zip")},
    }
    (output / "receipt.json").write_text(json.dumps(receipt, indent=2) + "\n")


if __name__ == "__main__":
    main()
