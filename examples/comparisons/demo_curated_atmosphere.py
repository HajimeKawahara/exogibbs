"""Compare curated and legacy gas thermochemistry in prescribed atmosphere columns.

The 4000 K column samples a large-effect regime found in the earlier fixed-grid
study. The inverted profile also exercises continuous temperature evaluation.
Neither profile is a radiative-convective solution or a fit to an observed planet.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

import jax
import jax.numpy as jnp
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np

from exogibbs.api.gas import EquilibriumOptions, solve_profile
from exogibbs.presets import curated, fastchem, fastchem4
from exogibbs.thermo.models import ChemicalSetup


LAYERS = 161
SIGNIFICANT_X = 1e-12
MODELS = ("curated", "fastchem4", "fastchem3")
STYLES = {"curated": "-", "fastchem4": "--", "fastchem3": ":"}
LABELS = {"curated": "Curated: 537 species", "fastchem4": "FastChem 4 data: same 537 species",
          "fastchem3": "Legacy FastChem 3 preset: 523 species"}
SPECIES = {"Cu1": ("Cu", "#b85b00"), "Cu1+": (r"Cu$^+$", "#ccab00"),
           "Ni1": ("Ni", "#0072b2"), "Ni1+": (r"Ni$^+$", "#56b4e9"),
           "Co1": ("Co", "#009e73"), "e1-": (r"e$^-$", "#aa3377"),
           "H1-": (r"H$^-$", "#4477aa")}


def profiles() -> dict:
    """Return fixed pressure coordinates and two explicitly prescribed T(P) curves."""
    logp = np.linspace(-6., 2., LAYERS)
    return {
        "isothermal": (10.**logp, np.full(LAYERS, 4000.)),
        "inverted": (10.**logp, np.interp(logp, [-6., -4., -2., 0., 2.],
                                         [4000., 3500., 2500., 3000., 3500.])),
    }


def comparison_setups() -> dict:
    """Fix the elemental budget and numerical options while retaining labeled catalogs."""
    adopted = curated.chemsetup()
    setups = {"curated": adopted}
    bases = {"fastchem4": fastchem4.chemsetup(path="FastChem4/logK/logK.dat", silent=True),
             "fastchem3": fastchem.chemsetup(silent=True)}
    for name, base in bases.items():
        species = adopted.species if name == "fastchem4" else base.species
        columns = jnp.array([base.species.index(s) for s in species])
        rows = jnp.array([base.elements.index(el) for el in adopted.elements])
        matrix = base.formula_matrix[rows[:, None], columns]

        def hvector(t, source=base.hvector_func, indices=columns):
            return source(t)[..., indices]

        setups[name] = ChemicalSetup(
            matrix, jax.jit(hvector), adopted.elements, species,
            adopted.element_vector_reference,
            {**base.metadata, "relative_charge_convergence": True},
        )
    np.testing.assert_array_equal(setups["fastchem4"].formula_matrix, adopted.formula_matrix)
    return setups


def closure(setup: ChemicalSetup, ln_n: np.ndarray, temperature: np.ndarray,
            pressure: np.ndarray) -> dict:
    """Audit primitive amounts, independently of the solver stopping residual."""
    n = np.exp(ln_n)
    a, b = np.asarray(setup.formula_matrix), np.asarray(setup.element_vector_reference)
    charge = a[setup.elements.index("e-")]
    logp = ln_n - np.log(n.sum(axis=-1))[:, None] + np.log(pressure)[:, None]
    references = [setup.species.index("e1-" if el == "e-" else el + "1") for el in setup.elements]
    residual = logp - logp[:, references] @ a + np.asarray(setup.hvector_func(temperature))
    return {
        "element_relative_residual": float(np.max(np.abs(n @ a.T - b)[:, b > 0] / b[b > 0])),
        "charge_relative_residual": float(np.max(np.abs(n @ charge) / (n @ np.abs(charge)))),
        "mass_action_max_abs_ln": float(np.max(np.abs(residual))),
    }


def require_closure(metrics: dict) -> None:
    gates = {"element_relative_residual": 1e-5, "charge_relative_residual": 1e-12,
             "mass_action_max_abs_ln": 1e-8}
    if any(not np.isfinite(metrics[k]) or metrics[k] > limit for k, limit in gates.items()):
        raise RuntimeError(f"Atmosphere column failed independent closure: {metrics}")


def summarize(profile: str, setups: dict, logx: dict, temperature: np.ndarray,
              pressure: np.ndarray) -> list:
    """Rank only shared species that reach the stated abundance floor in either model."""
    rows = []
    adopted = setups["curated"]
    for baseline in MODELS[1:]:
        for name in adopted.species:
            if name not in setups[baseline].species:
                continue
            new = logx["curated"][:, adopted.species.index(name)]
            old = logx[baseline][:, setups[baseline].species.index(name)]
            significant = np.maximum(new, old) >= np.log(SIGNIFICANT_X)
            if not np.any(significant):
                continue
            delta = (new - old) / np.log(10.)
            i = int(np.argmax(np.where(significant, np.abs(delta), -np.inf)))
            rows.append({"profile": profile, "baseline": baseline, "species": name,
                         "temperature_K": float(temperature[i]), "pressure_bar": float(pressure[i]),
                         "baseline_x": float(np.exp(old[i])), "curated_x": float(np.exp(new[i])),
                         "curated_over_baseline": float(np.exp(new[i] - old[i])),
                         "delta_dex": float(delta[i])})
    return sorted(rows, key=lambda r: abs(r["delta_dex"]), reverse=True)


def plot_profile(name: str, setups: dict, logx: dict, temperature: np.ndarray,
                 pressure: np.ndarray, output: Path) -> None:
    fig, axes = plt.subplots(2, 3, figsize=(13.5, 9.6), sharey=True, layout="constrained")
    fig.get_layout_engine().set(rect=(0., .13, 1., .87))
    title = "4000 K isothermal column" if name == "isothermal" else "Prescribed temperature inversion"
    fig.suptitle(title + " | solar composition, ideal gas only", fontsize=16)
    for ax in axes.flat:
        ax.set_yscale("log")
        ax.set_ylim(100., 1e-6)
        ax.set_ylabel("Pressure [bar]")
        ax.grid(alpha=.2)
    axes[0, 0].plot(temperature, pressure, color="#333333", lw=2.)
    axes[0, 0].set_xlabel("Temperature [K]")
    axes[0, 0].set_title("Same T(P) and elemental budget")
    axes[0, 0].set_xlim(2400., 4100.)

    def amounts(ax, names, title):
        for species in names:
            label, color = SPECIES[species]
            for model in MODELS:
                values = np.exp(logx[model][:, setups[model].species.index(species)])
                ax.plot(values, pressure, STYLES[model], color=color, lw=1.8,
                        label=label if model == "curated" else None)
        ax.set_xscale("log")
        ax.set_xlabel("Mole fraction")
        ax.set_title(title)
        ax.legend(fontsize=10, loc="best")

    amounts(axes[0, 1], ("Cu1", "Cu1+"), "Copper: neutral / ion redistribution")
    amounts(axes[0, 2], ("Ni1", "Ni1+"), "Nickel: neutral / ion redistribution")
    amounts(axes[1, 1], ("e1-", "H1-"), "Electrons and negative hydrogen")
    for ax, names, percent in ((axes[1, 0], ("Cu1", "Ni1", "Co1"), False),
                               (axes[1, 2], ("e1-", "H1-"), True)):
        for species in names:
            label, color = SPECIES[species]
            new = logx["curated"][:, setups["curated"].species.index(species)]
            for baseline, style in (("fastchem4", "-"), ("fastchem3", ":")):
                old = logx[baseline][:, setups[baseline].species.index(species)]
                ratio = np.exp(new - old)
                ax.plot(100. * (ratio - 1.) if percent else ratio, pressure, style,
                        color=color, lw=1.8, label=label if baseline == "fastchem4" else None)
        ax.axvline(0. if percent else 1., color="0.6", lw=.8)
        ax.set_title("Charge carriers: relative change" if percent else "Neutral metals: abundance ratio")
        ax.set_xlabel("100 × (curated / baseline − 1) [%]" if percent else "Curated / baseline")
        if not percent:
            ax.set_xscale("log")
        ax.legend(fontsize=10, loc="best")
    handles = [Line2D([], [], color="0.25", ls=STYLES[m], label=LABELS[m]) for m in MODELS]
    fig.legend(handles=handles, loc="lower center", bbox_to_anchor=(.5, .055), ncol=3, fontsize=10)
    fig.text(.5, .015, "Ratio panels: solid = relative to FastChem 4; dotted = relative to FastChem 3.\n"
             "Prescribed columns; no condensation, transport, photoionization, or spectral calculation.",
             ha="center", va="bottom", fontsize=10)
    for extension in ("png", "pdf"):
        fig.savefig(output / f"{name}.{extension}", dpi=180)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=Path("/tmp/exogibbs_curated_atmosphere"))
    args = parser.parse_args()
    jax.config.update("jax_enable_x64", True)
    output = args.output_dir
    output.mkdir(parents=True, exist_ok=True)
    setups = comparison_setups()
    summary, audits, files = [], [], []
    for name, (pressure, temperature) in profiles().items():
        logx = {}
        for model, setup in setups.items():
            result, info = solve_profile(
                setup, T=jnp.asarray(temperature), P=jnp.asarray(pressure),
                b=setup.element_vector_reference,
                options=EquilibriumOptions(epsilon_crit=1e-13, method="vmap_cold"),
                return_diagnostics=True)
            if not np.all(info["converged"]) or not np.all(np.isfinite(result.ln_n)):
                raise RuntimeError(f"Nonconverged or nonfinite {name}/{model} column")
            ln_n = np.asarray(result.ln_n)
            audit = closure(setup, ln_n, temperature, pressure)
            require_closure(audit)
            audits.append({"profile": name, "model": model, "layers": len(pressure), **audit})
            logx[model] = ln_n - np.log(np.exp(ln_n).sum(axis=-1))[:, None]
            filename = f"{name}_{model}.npz"
            np.savez_compressed(output / filename, temperature_K=temperature, pressure_bar=pressure,
                                ln_n=ln_n, logx=logx[model], h=setup.hvector_func(temperature),
                                species=setup.species, elements=setup.elements,
                                formula_matrix=setup.formula_matrix, budget=setup.element_vector_reference)
            files.append(filename)
            print(name, model, audit, flush=True)
        summary.extend(summarize(name, setups, logx, temperature, pressure))
        plot_profile(name, setups, logx, temperature, pressure, output)
        files.extend([f"{name}.png", f"{name}.pdf"])
    with (output / "summary.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=summary[0])
        writer.writeheader()
        writer.writerows(summary)
    files.append("summary.csv")
    paths = [Path(__file__), ROOT / "src/exogibbs/presets/curated.py",
             ROOT / "src/exogibbs/data/curated/gas_v1.json",
             ROOT / "src/exogibbs/data/fastchem/logK/logK.dat",
             ROOT / "src/exogibbs/data/FastChem4/logK/logK.dat",
             ROOT / "src/exogibbs/data/FastChem4/element_abundances/asplund_2021.dat",
             ROOT / "src/exogibbs/equilibrium/gas/kernel/solver.py"]
    receipt = {
        "dataset": "curated-gas-v1", "backend": jax.default_backend(), "dtype": "float64",
        "epsilon_crit": 1e-13, "relative_charge_convergence": "enabled for every model",
        "cases_passed": sum(r["layers"] for r in audits), "audits": audits,
        "species_counts": {name: len(s.species) for name, s in setups.items()},
        "temperature_profiles": {"isothermal": {"temperature_K": 4000.},
                                 "inverted": {"log10_pressure_bar": [-6., -4., -2., 0., 2.],
                                              "temperature_K": [4000., 3500., 2500., 3000., 3500.],
                                              "interpolation": "linear in log10 pressure"}},
        "ranking": {"statistic": "maximum absolute log10 abundance ratio over shared species",
                    "minimum_mole_fraction_in_either_model": SIGNIFICANT_X},
        "comparison_scope": {
            "fastchem4": "Thermochemistry-only comparison on the identical 537-species network; N- and Zn- removed.",
            "fastchem3": "Legacy 523-species catalog versus curated 537: includes catalog and thermochemistry changes.",
            "shared": "Same normalized Asplund 2021 file budget, T(P), solver, convergence mode and tolerance.",
        },
        "interpretation": "Illustrative ideal-gas source sensitivity; no atmospheric retrieval, radiative-convective solution or physical accuracy claim.",
        "selected_file_sha256": {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest() for p in paths},
        "output_sha256": {name: hashlib.sha256((output / name).read_bytes()).hexdigest() for name in files},
    }
    (output / "receipt.json").write_text(json.dumps(receipt, indent=2) + "\n")


if __name__ == "__main__":
    main()
