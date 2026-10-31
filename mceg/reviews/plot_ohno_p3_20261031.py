"""Plot native database selections in Ohno's axes and publish compact results.

The compact receipt contains calculated abundances, not third-party
thermochemical coefficients or sampled equilibrium-constant tables.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from ohno_exogibbs_20261031 import ROOT, phosphorus_colors


def compact_case(case: dict) -> dict:
    """Retain all P-species curves and declared numerical diagnostics."""
    names = case["species"]
    matrix = np.asarray(case["formula_matrix"])
    phosphorus = matrix[case["elements"].index("P")]
    indices = np.flatnonzero(phosphorus)
    logx = np.asarray(case.get("ggchem_log_mole_fractions", case.get("log_mole_fractions")))
    x = np.exp(logx)
    fractions = x * phosphorus / (x @ phosphorus)[:, None]
    result = {key: case[key] for key in ("elements", "temperature_K", "pressure_bar", "budget")}
    result.update({"species_count": len(names), "P_species_count": len(indices),
                   "P_log10_mole_fractions": {names[i]: (logx[:, i] / np.log(10.)).tolist() for i in indices},
                   "P_atom_fractions": {names[i]: fractions[:, i].tolist() for i in indices}})
    for key in ("native_closure", "cases_passed"):
        if key in case:
            result[key] = case[key]
    for key in ("element_relative_residual", "mass_action_max_abs_ln", "ggchem_max_abs_delta_logx"):
        if key in case:
            result["max_" + key] = max(case[key])
    if "effective_records" in case:
        result["P_record_sources"] = {
            names[i]: {k: record[k] for k in ("source", "fit") if k in record}
            for i in indices if (record := case["effective_records"].get(names[i])) is not None}
    return result


def plot_native(cases: dict, output: Path, title: str) -> None:
    """Use log temperature and log10 molecular number fraction for both panels."""
    colors = phosphorus_colors([s for c in cases.values() for s in c["P_log10_mole_fractions"]])
    fig, axes = plt.subplots(1, 2, figsize=(12, 5.4), layout="constrained", sharex=True, sharey=True)
    for ax, name, panel in zip(axes, ("janaf", "burcat"),
                               ("JANAF-based baseline", "Burcat-priority")):
        case = cases[name]
        for species, values in case["P_log10_mole_fractions"].items():
            if max(values) > -18:
                ax.plot(case["temperature_K"], values, label=species, color=colors[species], lw=1.5)
        ax.set(xscale="log", xlim=(100, 3000), ylim=(-18, -5.5),
               xlabel="Temperature [K]", ylabel=r"$\log_{10}(n_i/n_{\rm tot})$",
               title=f"{panel}\n{case['species_count']} neutral species; {case['P_species_count']} contain P")
        ax.axvspan(100, 200, color="gray", alpha=.15)
        ax.text(142, -12, "Not evaluated", rotation=90, ha="center", va="center", fontsize=8, color="gray")
        ax.grid(alpha=.15)
        ax.legend(fontsize=7, ncol=2, loc="lower left")
    fig.suptitle(title)
    fig.savefig(output, dpi=180)
    plt.close(fig)


def carrier_difference(first: dict, second: dict) -> dict:
    """Locate the largest P-budget redistribution on a shared sampled grid."""
    np.testing.assert_allclose(first["temperature_K"], second["temperature_K"], rtol=0, atol=1e-10)
    names = sorted(set(first["P_atom_fractions"]) | set(second["P_atom_fractions"]))
    zero = np.zeros(len(first["temperature_K"]))
    differences = np.array([np.asarray(first["P_atom_fractions"].get(s, zero))
                            - np.asarray(second["P_atom_fractions"].get(s, zero)) for s in names])
    i, j = np.unravel_index(np.argmax(np.abs(differences)), differences.shape)
    return {"max_abs_P_atom_fraction_difference": float(abs(differences[i, j])),
            "species": names[i], "temperature_K": first["temperature_K"][j]}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=ROOT / "results/ohno_p3")
    args = parser.parse_args()
    native_path = args.output_dir / "native_cases.json"
    exo_path = args.output_dir / "exogibbs_summary.json"
    native = json.loads(native_path.read_text())
    exo = json.loads(exo_path.read_text())
    cases = {name: compact_case(case) for name, case in native["cases"].items()}
    plot_native(cases, args.output_dir / "ohno_p3_database_comparison.png",
                "Public GGChem comparison | H/He/C/N/O/P/S | 1 mbar | gas only")
    paths = [Path(__file__), Path(__file__).with_name("ohno_exogibbs_20261031.py"),
             native_path, exo_path]
    report = {"scope": "Ohno-style axes under declared public inputs; not a reconstruction of unpublished original inputs. Temperature sampling is not a validation of all source fits.",
              "ggchem_revision": native["revision"], "network": native["network"],
              "data_priority": native["data_priority"], "filters": native["filters"],
              "native": cases, "exogibbs": {name: compact_case(case) for name, case in exo["cases"].items()},
              "exogibbs_cases_passed": exo["cases_passed"],
              "selected_file_sha256": {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}}
    report["native_manifests"] = {filename: hashlib.sha256((args.output_dir / filename).read_bytes()).hexdigest()
                                  for filename in ("native_manifest.json", "native27_manifest.json")}
    report["two_gas_vs_common_species"] = carrier_difference(
        report["exogibbs"]["burcat_without_H3PO4_HPO2"], report["exogibbs"]["common_burcat"])
    expanded_path = args.output_dir / "native27_cases.json"
    if expanded_path.exists():
        expanded = json.loads(expanded_path.read_text())
        expanded_cases = {name: compact_case(case) for name, case in expanded["cases"].items()}
        report["expanded_native"] = expanded_cases
        report["element_scope_effect"] = {name: carrier_difference(cases[name], expanded_cases[name]) for name in cases}
        plot_native(expanded_cases, args.output_dir / "ohno_p3_expanded_elements.png",
                    f"Public GGChem comparison | {len(next(iter(expanded_cases.values()))['elements'])} elements | 1 mbar | gas only")
        report["selected_file_sha256"][str(expanded_path.relative_to(ROOT))] = hashlib.sha256(expanded_path.read_bytes()).hexdigest()
    expanded_exo_path = args.output_dir / "exogibbs27/exogibbs_summary.json"
    if expanded_exo_path.exists():
        expanded_exo = json.loads(expanded_exo_path.read_text())
        report["expanded_exogibbs"] = {name: compact_case(case) for name, case in expanded_exo["cases"].items()}
        report["exogibbs_cases_passed"] += expanded_exo["cases_passed"]
        report["selected_file_sha256"][str(expanded_exo_path.relative_to(ROOT))] = hashlib.sha256(expanded_exo_path.read_bytes()).hexdigest()
    (args.output_dir / "summary.json").write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
