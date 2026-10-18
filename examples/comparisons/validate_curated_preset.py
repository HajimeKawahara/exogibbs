"""Check continuous curated thermochemistry and both fixed gas networks offline.

An optional existing FastChem executable supplies independent equilibrium
comparisons. It is never used for initialization or required by the preset.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
import platform
import sys
import tempfile

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

import jax
import jax.numpy as jnp
import numpy as np

import build_curated_preset as build
from exogibbs.api.gas import EquilibriumOptions, solve_profile
from exogibbs.io.load_data import get_data_filepath
from exogibbs.presets.curated import chemsetup


TEMPERATURES = np.array([1500., 1735., 2000., 2517., 3000., 3973., 4000.])
PRESSURES = np.array([1e-6, 1e-4, .01, 1., 100.])


def thermal_checks(data: dict) -> dict:
    reference = json.loads(build.REFERENCE_PATH.read_text())
    janaf, partitions = {}, {}
    for name, source in reference["janaf"].items():
        record = next(r for r in data["janaf_reactions"].values() if r["table_id"] == source["table_id"])
        errors = []
        knots = np.array(record["knots_K"])
        for key, (cp, entropy, sensible) in source["rows"].items():
            t = float(key)
            if t in knots:
                continue
            i = np.searchsorted(knots, t) - 1
            width = knots[i+1] - knots[i]
            u, c = (t-knots[i])/width, np.array(record["coefficients"][i])
            g = np.polynomial.polynomial.polyval(u, c)
            dg = np.polynomial.polynomial.polyval(u, np.polynomial.polynomial.polyder(c))/width
            d2g = np.polynomial.polynomial.polyval(u, np.polynomial.polynomial.polyder(c, 2))/width**2
            r = build.molecular.R
            enthalpy = 1000*(sensible - source["h0_minus_h298_kJ_mol"])
            errors.append([abs((g-(enthalpy-t*entropy)/r)/t/np.log(10)),
                           abs(r*(g-t*dg)-enthalpy), abs(-r*dg-entropy), abs(-r*t*d2g-cp)])
        maxima = np.max(errors, axis=0)
        janaf[name] = dict(zip(("max_abs_log10_K_error", "max_abs_H_error_J_mol",
                               "max_abs_S_error_J_mol_K", "max_abs_Cp_error_J_mol_K"), maxima.tolist()))
        janaf[name]["withheld_nodes"] = len(errors)
        if maxima[0] > 6e-5 or maxima[3] > .04:
            raise RuntimeError("JANAF representation fails withheld-node gates")
    t = np.array(reference["partition_temperature_K"])
    for name, source in reference["partitions"].items():
        record = data["partition_functions"][name]
        energy, weights = np.array(record["excitation_K"]), np.array(record["weights"])
        truth = np.array(source["values"])
        actual = np.exp(-energy[None, :]/t[:, None]) @ weights
        leave_out = []
        for temperature in build.KNOTS:
            keep = t != temperature
            fit = build.partition_fit(t[keep], truth[keep], source["ground_weight"])
            predicted = np.exp(-np.array(fit["excitation_K"])/temperature) @ np.array(fit["weights"])
            leave_out.append(abs(np.log10(predicted/truth[t == temperature][0])))
        partitions[name] = {"max_relative_table_error": float(np.max(np.abs(actual/truth-1))),
                            "max_leave_one_out_log10_Q_error": float(max(leave_out)),
                            "positive_weights": bool(np.all(weights > 0))}
        if np.max(np.abs(actual/truth-1)) > 4e-6 or max(leave_out) > 6e-5:
            raise RuntimeError("Partition representation fails source/leave-one-out gates")
    return {"janaf": janaf, "partitions": partitions,
            "interpretation": "Representation checks against adopted sources; leave-one-out fits probe interpolation stability, not physical accuracy."}


def metrics(setup, ln_n: np.ndarray, h: np.ndarray, pressure: np.ndarray) -> dict:
    n, a, b = np.exp(ln_n), np.asarray(setup.formula_matrix), np.asarray(setup.element_vector_reference)
    reference = [setup.species.index("e1-" if name == "e-" else name + "1") for name in setup.elements]
    logp = ln_n - np.log(n.sum(axis=-1))[:, None] + np.log(pressure)[:, None]
    out = {"element_relative_residual": np.max(np.abs(n@a.T-b)[:, b>0]/b[b>0], axis=-1),
           "mass_action_max_abs_ln": np.max(np.abs(logp-logp[:, reference]@a+h), axis=-1)}
    if "e-" in setup.elements:
        charge = a[setup.elements.index("e-")]
        out["charge_relative_residual"] = np.abs(n@charge)/(n@np.abs(charge))
    else:
        out["charge_relative_residual"] = np.zeros(len(ln_n))
    return out


def compare_fastchem(setup, h: np.ndarray, ln_n: np.ndarray, executable: Path) -> tuple:
    from benchmarks.fastchem4.comparison import align_species_values, occurrence_keys
    from benchmarks.fastchem4.fastchem_executable import run_fastchem_executable

    all_amounts, rows, hashes = [], [], {}
    reference = {"e1-" if el == "e-" else el + "1" for el in setup.elements}
    a = np.asarray(setup.formula_matrix)
    for ti, temperature in enumerate(TEMPERATURES):
        lines = ["# Single-temperature curated formation constants at 1 bar.",
                 f"# Valid only at {temperature:g} K.", "# Independent solver verification only."]
        for i, name in enumerate(setup.species):
            if name in reference:
                continue
            composition = " ".join(f"{el} {int(count)}" for el, count in zip(setup.elements, a[:, i]) if count)
            lines.extend([f"{name} {name} : {composition}", f"0 0 {-h[ti, i]:.17e} 0 0", ""])
        payload = "\n".join(lines) + "\n"
        hashes[f"{temperature:g}K"] = hashlib.sha256(payload.encode()).hexdigest()
        with tempfile.TemporaryDirectory(prefix="exogibbs_curated_") as directory:
            path = Path(directory)/"logK.dat"
            path.write_text(payload)
            result = run_fastchem_executable(
                executable=executable, temperatures=np.full(5, temperature), pressures=PRESSURES,
                element_abundance_file=Path(get_data_filepath(setup.metadata["abundances"]["path"])),
                gas_logk_file=path, chemistry_mode="gas", chemistry_accuracy=1e-10,
                element_conservation_accuracy=1e-10)
        catalog = occurrence_keys(result.gas_names)
        if not setup.metadata["ions"]:
            # FastChem retains a zero-electron output column in neutral mode.
            electron = [i for i, key in enumerate(catalog) if key == ("e-", 0)]
            if np.any(result.gas_number_densities[:, electron] / result.ideal_gas_density[:, None] > 1e-30):
                raise RuntimeError("Neutral comparison has a nonzero electron population")
            catalog = tuple(key for key in catalog if key != ("e-", 0))
        if sorted(catalog) != sorted(occurrence_keys(setup.species)):
            raise RuntimeError("Independent solver catalog differs")
        if not np.all(result.converged & result.elements_conserved):
            raise RuntimeError("Independent solver did not converge and conserve elements")
        density = align_species_values(setup.species, result.gas_names, result.gas_number_densities)
        if np.any(density <= 0):
            raise RuntimeError("Nonpositive comparison density")
        amounts = np.log(density/result.total_element_density[:, None])
        all_amounts.append(amounts)
        ex = ln_n[ti*5:(ti+1)*5]
        logx = ex - np.log(np.exp(ex).sum(axis=-1))[:, None]
        fc_logx = amounts - np.log(np.exp(amounts).sum(axis=-1))[:, None]
        delta = (logx-fc_logx)/np.log(10)
        closure = metrics(setup, amounts, np.repeat(h[ti][None, :], 5, axis=0), PRESSURES)
        for pi, pressure in enumerate(PRESSURES):
            significant = np.maximum(np.exp(logx[pi]), np.exp(fc_logx[pi])) >= 1e-12
            row = {"temperature_K": temperature, "pressure_bar": pressure,
                   "max_abs_dex_all_species": float(np.max(np.abs(delta[pi]))),
                   "max_abs_dex_x_ge_1e_minus12": float(np.max(np.abs(delta[pi, significant]))),
                   **{k: float(v[pi]) for k, v in closure.items()}}
            for name in ("e1-", "H1-"):
                row[name + "_abs_dex"] = float(abs(delta[pi, setup.species.index(name)])) if name in setup.species else 0.
            if (row["max_abs_dex_all_species"] > 1e-3 or row["element_relative_residual"] > 1e-5
                    or row["charge_relative_residual"] > 1e-5 or row["mass_action_max_abs_ln"] > 1e-8):
                raise RuntimeError(f"Independent comparison failed: {row}")
            rows.append(row)
    return np.concatenate(all_amounts), rows, hashes


def write_csv(path: Path, rows: list) -> None:
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=rows[0])
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=Path("/tmp/exogibbs_curated_validation"))
    parser.add_argument("--fastchem-executable", type=Path)
    args = parser.parse_args()
    jax.config.update("jax_enable_x64", True)
    output = args.output_dir
    output.mkdir(parents=True, exist_ok=True)
    data = json.loads(build.OUTPUT_PATH.read_text())
    (output/"thermal_checks.json").write_text(json.dumps(thermal_checks(data), indent=2) + "\n")
    diagnostics, parity, input_hashes = [], [], {}
    temperatures = np.repeat(TEMPERATURES, len(PRESSURES))
    pressures = np.tile(PRESSURES, len(TEMPERATURES))
    for ions in (True, False):
        name = "ions" if ions else "neutral"
        setup = chemsetup(ions=ions)
        if ions:
            core = np.isin(temperatures, build.KNOTS)
            default_result, default_info = solve_profile(
                setup, T=jnp.array(temperatures[core]), P=jnp.array(pressures[core]),
                b=setup.element_vector_reference,
                options=EquilibriumOptions(method="vmap_cold"), return_diagnostics=True)
            default_metrics = metrics(setup, np.asarray(default_result.ln_n),
                                      np.asarray(setup.hvector_func(temperatures[core])), pressures[core])
            if (not np.all(default_info["converged"])
                    or np.max(default_metrics["charge_relative_residual"]) > 1e-10):
                raise RuntimeError("Default precision fails relative charge closure")
            default_rows = [{"temperature_K": t, "pressure_bar": p,
                             "charge_relative_residual": float(error)} for t, p, error in zip(
                                 temperatures[core], pressures[core], default_metrics["charge_relative_residual"])]
            write_csv(output/"default_charge.csv", default_rows)
        result, info = solve_profile(
            setup, T=jnp.array(temperatures), P=jnp.array(pressures), b=setup.element_vector_reference,
            options=EquilibriumOptions(epsilon_crit=1e-13, method="vmap_cold"), return_diagnostics=True)
        ln_n, h = np.asarray(result.ln_n), np.asarray(setup.hvector_func(TEMPERATURES))
        closure = metrics(setup, ln_n, np.repeat(h, len(PRESSURES), axis=0), pressures)
        if (not np.all(info["converged"]) or np.max(closure["element_relative_residual"]) > 1e-5
                or np.max(closure["charge_relative_residual"]) > 1e-12
                or np.max(closure["mass_action_max_abs_ln"]) > 1e-8):
            raise RuntimeError(f"ExoGibbs conservation/convergence failed for {name}")
        for i, (temperature, pressure) in enumerate(zip(temperatures, pressures)):
            diagnostics.append({"network": name, "temperature_K": temperature, "pressure_bar": pressure,
                                "iterations": int(info["n_iter"][i]), "solver_residual": float(info["final_residual"][i]),
                                **{k: float(v[i]) for k, v in closure.items()}})
        fc_ln_n = np.empty((0, len(setup.species)))
        if args.fastchem_executable:
            fc_ln_n, rows, hashes = compare_fastchem(setup, h, ln_n, args.fastchem_executable.resolve())
            parity.extend({"network": name, **row} for row in rows)
            input_hashes[name] = hashes
        np.savez_compressed(output/(name + ".npz"), ln_n=ln_n, fastchem_ln_n=fc_ln_n, h=h,
                            species=setup.species, elements=setup.elements, formula_matrix=setup.formula_matrix,
                            budget=setup.element_vector_reference, temperature_K=temperatures, pressure_bar=pressures)
        print(name, {k: float(np.max(v)) for k, v in closure.items()}, flush=True)
    write_csv(output/"diagnostics.csv", diagnostics)
    if parity:
        write_csv(output/"fastchem_parity.csv", parity)
    paths = [Path(__file__), Path(build.__file__), build.OUTPUT_PATH, build.REFERENCE_PATH,
             ROOT/"src/exogibbs/presets/curated.py", ROOT/"src/exogibbs/equilibrium/gas/kernel/solver.py",
             ROOT/"src/exogibbs/equilibrium/gas/solve.py", ROOT/"src/exogibbs/equilibrium/gas/types.py"]
    artifacts = ["thermal_checks.json", "diagnostics.csv", "default_charge.csv", "ions.npz", "neutral.npz"]
    if parity:
        artifacts.append("fastchem_parity.csv")
    receipt = {"dataset": data["id"], "python": platform.python_version(), "jax": jax.__version__,
               "dtype": "float64", "backend": jax.default_backend(), "epsilon_crit": 1e-13,
               "temperature_K": TEMPERATURES.tolist(), "pressure_bar": PRESSURES.tolist(),
               "cases_passed": len(diagnostics), "fastchem": "passed" if parity else "not_run",
               "default_charge_cases_passed": len(default_rows),
               "fastchem_cases_passed": len(parity), "fastchem_input_sha256": input_hashes,
               "fastchem_binary_sha256": (hashlib.sha256(args.fastchem_executable.read_bytes()).hexdigest()
                                           if args.fastchem_executable else None),
               "selected_file_sha256": {str(p.resolve().relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest() for p in paths},
               "output_sha256": {name: hashlib.sha256((output/name).read_bytes()).hexdigest() for name in artifacts},
               "interpretation": "Continuous implementation and representation checks, not independent physical accuracy of the entire mixture."}
    (output/"receipt.json").write_text(json.dumps(receipt, indent=2) + "\n")


if __name__ == "__main__":
    main()
