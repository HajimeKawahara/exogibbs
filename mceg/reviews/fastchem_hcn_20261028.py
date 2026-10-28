"""Compare the preliminary HCN refit with #298 and current curated chemistry.

Run offline; save generated columns and figures under results/. This is a
provider-owned experiment, not a thermochemistry feature in MCEG itself.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import platform
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
import numpy as np
from scipy.special import logsumexp

from examples.comparisons.compare_new_fastchem_logk import comparison_setups, write_csv
from examples.comparisons.demo_curated_atmosphere import closure, profiles, require_closure
from exogibbs.api.gas import EquilibriumOptions, solve_profile
from exogibbs.presets.fastchem import _parse_fastchem_coeffs


PREVIOUS = ROOT / "tests/data/validation/new_fastchem_logk/logK.zip"
UPDATED = ROOT / "tests/data/validation/new_fastchem_hcn/logK_hcn.zip"
SOURCE = "https://github.com/NewStrangeWorlds/FastChem/issues/10#issuecomment-6439534232"
HCN = "C1H1N1_hcn"
MODELS = ("curated", "previous", "updated")
LABELS = ("ExoGibbs curated", "FastChem ion update (#298)", "FastChem HCN update")
APPEARANCE = ({"color": "#0072b2", "lw": 4, "alpha": .3, "ls": "-"},
              {"color": "#777777", "lw": 1.5, "ls": ":"},
              {"color": "#0072b2", "lw": 1.6, "ls": "--"})


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def audit_archives(previous: Path, updated: Path) -> dict:
    """Retain raw identifiers before comparing the declared HCN alias."""
    result = {}
    with zipfile.ZipFile(previous) as old, zipfile.ZipFile(updated) as new:
        for filename in ("logK.dat", "logK_extended.dat", "logK_hoeijmakers2019.dat"):
            a, ac = _parse_fastchem_coeffs(old.read(filename).decode())
            b, bc = _parse_fastchem_coeffs(new.read(filename).decode())
            added, removed = sorted(b.keys() - a.keys()), sorted(a.keys() - b.keys())
            aliases = {}
            if filename == "logK_hoeijmakers2019.dat" and "C1H1N1_1" in a and HCN in b:
                aliases["C1H1N1_1"] = HCN
                a[HCN], ac[HCN] = a.pop("C1H1N1_1"), ac.pop("C1H1N1_1")
            changes = sorted(s for s in a.keys() & b.keys() if a[s] != b[s])
            composition = sorted(s for s in ac.keys() & bc.keys() if ac[s] != bc[s])
            if a.keys() != b.keys() or changes != [HCN] or composition:
                raise ValueError(f"Unexpected update scope in {filename}")
            result[filename] = {"previous_sha256": hashlib.sha256(old.read(filename)).hexdigest(),
                                "updated_sha256": hashlib.sha256(new.read(filename)).hexdigest(),
                                "records": len(b), "raw_added": added, "raw_removed": removed,
                                "explicit_aliases": aliases, "changed_coefficients": changes,
                                "changed_compositions": composition}
    return result


def setups() -> dict:
    """Use the standard table only, with the same nuclear and charge budget."""
    with tempfile.TemporaryDirectory() as directory:
        path = Path(directory) / "logK.dat"
        values = {}
        for key, archive_path in (("previous", PREVIOUS), ("updated", UPDATED)):
            with zipfile.ZipFile(archive_path) as archive:
                path.write_bytes(archive.read("logK.dat"))
            compared = comparison_setups(path)
            values[key] = compared["new"]
        return {"curated": compared["curated"], **values}


def decompose(setup: object, delta_logx: np.ndarray, delta_logk: np.ndarray) -> dict:
    """Close the fixed-T,P HCN ratio against Kp and the atomic response."""
    j = setup.species.index(HCN)
    atoms = [setup.species.index(e + "1") for e in ("H", "C", "N")]
    atom_term = delta_logx[:, atoms].sum(axis=1)
    error = delta_logx[:, j] - delta_logk[:, j] - atom_term
    if not np.all(np.isfinite(error)) or np.max(np.abs(error)) > 1e-8:
        raise ValueError("HCN mass-action factorization failed")
    return {"K_ratio": np.exp(delta_logk[:, j]), "atom_product": np.exp(atom_term),
            "abundance_ratio": np.exp(delta_logx[:, j]), "residual": error}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=ROOT / "results/new_fastchem_hcn")
    args = parser.parse_args()
    output = args.output_dir
    output.mkdir(parents=True, exist_ok=True)
    jax.config.update("jax_enable_x64", True)
    audit = audit_archives(PREVIOUS, UPDATED)
    models = setups()
    default = models["curated"]
    hcn = default.species.index(HCN)
    audits, rows, maxima, samples = [], [], [], []
    fig, axes = plt.subplots(2, 3, figsize=(13, 9), layout="constrained")
    for row, (profile, (pressure, temperature)) in enumerate(profiles().items()):
        amounts, logx, h = {}, {}, {}
        for name, setup in models.items():
            result, info = solve_profile(setup, T=jnp.asarray(temperature), P=jnp.asarray(pressure),
                                        b=setup.element_vector_reference,
                                        options=EquilibriumOptions(epsilon_crit=1e-13, method="vmap_cold"),
                                        return_diagnostics=True)
            n = np.asarray(result.ln_n)
            if not np.all(info["converged"]) or not np.all(np.isfinite(n)):
                raise ValueError(f"Failed convergence: {profile}/{name}")
            gate = closure(setup, n, temperature, pressure)
            require_closure(gate)
            audits.append({"profile": profile, "model": name, "layers": len(pressure), **gate})
            print(profile, name, gate, flush=True)
            amounts[name] = n
            logx[name] = n - logsumexp(n, axis=-1)[:, None]
            h[name] = np.asarray(setup.hvector_func(temperature))
        np.savez_compressed(output / f"{profile}.npz", temperature_K=temperature, pressure_bar=pressure,
                            species=default.species, budget=default.element_vector_reference,
                            formula_matrix=default.formula_matrix, elements=default.elements,
                            **{f"ln_n_{m}": n for m, n in amounts.items()})
        for numerator, denominator in (("previous", "curated"), ("updated", "curated"), ("updated", "previous")):
            delta = logx[numerator] - logx[denominator]
            fact = decompose(default, delta, h[denominator] - h[numerator])
            for i, (t, p) in enumerate(zip(temperature, pressure)):
                rows.append({"profile": profile, "numerator": numerator, "denominator": denominator,
                             "temperature_K": float(t), "pressure_bar": float(p),
                             "numerator_x": float(np.exp(logx[numerator][i, hcn])),
                             "denominator_x": float(np.exp(logx[denominator][i, hcn])),
                             **{k: float(v[i]) for k, v in fact.items()}})
            if profile == "inverted":
                samples.append(next(r for r in rows[-len(pressure):] if np.isclose(r["pressure_bar"], .01)))
            for j, species in enumerate(default.species):
                eligible = np.maximum(logx[numerator][:, j], logx[denominator][:, j]) >= np.log(1e-12)
                if not np.any(eligible):
                    continue
                i = int(np.argmax(np.where(eligible, np.abs(delta[:, j]), -np.inf)))
                maxima.append({"profile": profile, "numerator": numerator, "denominator": denominator,
                               "species": species, "ratio": float(np.exp(delta[i, j])),
                               "temperature_K": float(temperature[i]), "pressure_bar": float(pressure[i]),
                               "numerator_x": float(np.exp(logx[numerator][i, j])),
                               "denominator_x": float(np.exp(logx[denominator][i, j]))})
        for m, label, style in zip(MODELS, LABELS, APPEARANCE):
            axes[row, 0].plot(np.exp(logx[m][:, hcn]), pressure, label=label, **style)
        for m, style in zip(MODELS[1:], APPEARANCE[1:]):
            axes[row, 1].plot(100 * np.expm1(logx[m][:, hcn] - logx["curated"][:, hcn]), pressure, **style)
        for s, color in (("S2", "#009e73"), ("H2S1", "#cc7900"), ("S1Si1", "#aa3377")):
            j = default.species.index(s)
            for m, ls in (("previous", ":"), ("updated", "--")):
                delta = 100 * np.expm1(logx[m][:, j] - logx["curated"][:, j])
                eligible = np.maximum(logx[m][:, j], logx["curated"][:, j]) >= np.log(1e-12)
                axes[row, 2].plot(np.where(eligible, delta, np.nan), pressure, ls=ls, color=color,
                                  label=s if m == "updated" else None)
        axes[row, 0].set(xscale="log", xlabel="HCN mole fraction", title=profile)
        axes[row, 1].set(xlabel="100 (HCN / curated - 1) [%]", title="HCN difference")
        axes[row, 2].set(xlabel="100 (molecule / curated - 1) [%]", title="Other molecular differences")
        for ax in axes[row]:
            ax.set(yscale="log", ylim=(100, 1e-6), ylabel="Pressure [bar]")
            ax.grid(alpha=.2)
        for ax in axes[row, 1:]:
            ax.axvline(0, color=".6", lw=.8)
    axes[0, 0].legend(fontsize=8)
    axes[0, 2].legend(fontsize=8)
    fig.suptitle("HCN refit | same 537 species and solar budget | ExoGibbs solver\n"
                 "dashed: HCN update; dotted: previous ion update; translucent solid: curated")
    for ext in ("png", "pdf"):
        fig.savefig(output / f"comparison.{ext}", dpi=150)
    plt.close(fig)
    write_csv(output / "hcn_columns.csv", rows)
    write_csv(output / "species_extrema.csv", maxima)
    temperatures = np.array([1500., 2000., 2500., 3000., 3500., 4000.])
    potentials = {m: np.asarray(s.hvector_func(temperatures))[:, hcn] for m, s in models.items()}
    constants = [{"temperature_K": float(t),
                  **{m + "_over_curated": float(np.exp(potentials["curated"][i] - potentials[m][i]))
                     for m in MODELS[1:]}} for i, t in enumerate(temperatures)]
    sources = [Path(__file__), PREVIOUS, UPDATED, ROOT / "examples/comparisons/compare_new_fastchem_logk.py",
               ROOT / "examples/comparisons/demo_curated_atmosphere.py", ROOT / "src/exogibbs/presets/curated.py",
               ROOT / "src/exogibbs/presets/fastchem.py", ROOT / "src/exogibbs/data/curated/gas_v1.json",
               ROOT / "src/exogibbs/data/FastChem4/logK/logK.dat",
               ROOT / "src/exogibbs/data/FastChem4/element_abundances/asplund_2021.dat",
               ROOT / "src/exogibbs/equilibrium/gas/kernel/solver.py"]
    report = {"source_url": SOURCE, "source_status": "preliminary; 2026-10-27 comment; no release adoption",
              "git_head": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
              "environment": {"python": platform.python_version(), "jax": jax.__version__, "numpy": np.__version__,
                              "backend": jax.default_backend(), "dtype": "float64"},
              "scope": "Two 161-layer profiles from #284/#298; fixed 537-species network and solar budget. "
                       "Only standard logK.dat solved; other archives audited. No spectra or retrieval.",
              "cases_passed": sum(a["layers"] for a in audits), "input_audit": audit, "closure": audits,
              "constants": constants, "HCN_at_2500K_0p01bar": samples,
              "HCN_extrema": [r for r in maxima if r["species"] == HCN],
              "largest_updated_over_curated": sorted([r for r in maxima if r["numerator"] == "updated" and r["denominator"] == "curated"],
                                                      key=lambda r: abs(np.log(r["ratio"])), reverse=True)[:15],
              "max_HCN_factorization_residual": max(abs(r["residual"]) for r in rows),
              "selected_file_sha256": {str(p.relative_to(ROOT)): digest(p) for p in sources},
              "output_sha256": {p.name: digest(p) for p in sorted(output.iterdir()) if p.suffix in (".npz", ".csv", ".png", ".pdf")}}
    (output / "summary.json").write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
