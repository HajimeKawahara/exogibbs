"""Inspect pinned native phase runs and optionally prepare a function-export input.

The optional input contains algebraically reconstructed phase functions only as
a cross-check. Run phosphorus_native_phase_export_20261031.py on that input to
replace them with independent direct SUPERSAT function evaluations before using
them for an ExoGibbs comparison.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import re

import numpy as np
from scipy.special import logsumexp


ROOT = Path(__file__).resolve().parents[2]
REVISION = "65bbb8f9ca3d6c0b4e58cb7dd3862322452d0d4b"
BASE = ROOT / "results/ohno_p3"
OUT = ROOT / "results/phosphorus_cold_validation"
UPSTREAM = BASE / f"GGchem-{REVISION}"
THRESHOLDS = {
    "element_relative_max": 1e-7,
    "maximum_log10_saturation": 1e-7,
    "active_max_abs_log10_saturation": 1e-7,
    "pressure_relative_max": 1e-8,
}
ACTIVE_CUTOFF = 1e-100


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_json(path: Path, data: dict) -> None:
    path.write_text(json.dumps(data, indent=2, allow_nan=False) + "\n")


def dust_metadata() -> dict:
    """Read names and integer compositions, without copying fit coefficients."""
    lines = (UPSTREAM / "data/DustChem.dat").read_text().splitlines()
    dust = {}
    for i, line in enumerate(lines):
        if not re.match(r"^\S+\[[sl]\]\s", line):
            continue
        count = int(lines[i + 2])
        composition = {
            lines[i + 3 + j].split()[1]: float(lines[i + 3 + j].split()[0])
            for j in range(count)
        }
        name = line.split()[0]
        dust[name.replace("[s]", "")] = {
            "native_name": name, "composition": composition, "source_line": i + 1,
        }
    return dust


def analyze(export_input: Path | None = None) -> None:
    dust = dust_metadata()
    case = json.loads((BASE / "native_cases.json").read_text())["cases"]["burcat"]
    elements = case["elements"]
    matrix = np.asarray(case["formula_matrix"])
    budget = np.asarray(case["budget"])
    run = OUT / "native7_condensed"
    gasrun = OUT / "native7_gas"
    rows = (run / "Static_Conc.dat").read_text().splitlines()
    n_el, n_mol, n_dust, n_points = map(int, rows[1].split())
    values = np.loadtxt(run / "Static_Conc.dat", skiprows=3)
    gas_values = np.loadtxt(gasrun / "Static_Conc.dat", skiprows=3)
    assert n_el == len(elements) and n_mol + n_el == len(case["species"])
    assert n_points == len(values) == len(gas_values) == 31
    assert np.array_equal(values[:, 0], gas_values[:, 0])
    assert (run / "dispol.out").read_bytes() == (BASE / "native_burcat/dispol.out").read_bytes()
    assert (gasrun / "dispol.out").read_bytes() == (run / "dispol.out").read_bytes()
    assert (gasrun / "input.in").read_text().replace(
        ".false. ! model_eqcond", ".true. ! model_eqcond"
    ) == (run / "input.in").read_text()
    start = 4 + matrix.shape[1]
    names = [name[1:] for name in rows[2].split()[start:start + n_dust]]
    phase_matrix = np.array([
        [dust[name]["composition"].get(element, 0.0) for name in names]
        for element in elements
    ])
    gas_counts = 10.0 ** (values[:, 4:start] - np.log10(values[:, 1])[:, None])
    phase_counts = 10.0 ** values[:, start + n_dust:start + 2 * n_dust]
    saturation = values[:, start:start + n_dust]
    active = phase_counts > ACTIVE_CUTOFF
    conservation = (gas_counts @ matrix.T + phase_counts @ phase_matrix.T) / budget - 1.0
    gates = {
        "element_relative_max": float(abs(conservation).max()),
        "maximum_log10_saturation": float(saturation.max()),
        "active_max_abs_log10_saturation": float(abs(saturation[active]).max()),
        "pressure_relative_max": float(abs(values[:, 2] / 1000.0 - 1.0).max()),
    }
    assert all(gates[name] < threshold for name, threshold in THRESHOLDS.items())
    gas_n = 10.0 ** (gas_values[:, 4:start] - np.log10(gas_values[:, 1])[:, None])
    gas_closure = float(abs(gas_n @ matrix.T / budget - 1.0).max())
    assert gas_closure < THRESHOLDS["element_relative_max"]
    phosphorus = elements.index("P")
    gas_p = gas_counts @ matrix[phosphorus] / budget[phosphorus]
    phase_p = phase_counts * phase_matrix[phosphorus] / budget[phosphorus]
    inputs = [UPSTREAM / "ggchem16_original", UPSTREAM / "data/DustChem.dat", Path(__file__)]
    for directory in (run, gasrun):
        inputs.extend(directory / name for name in ("input.in", "abundances.dat", "dispol.out", "Static_Conc.dat"))
    report = {
        "scope": "Completed native 7-element equilibrium-condensation sensitivity; original source temperature intervals are not certified by native dust fits.",
        "gas_species_count": matrix.shape[1], "condensed_species_count": n_dust,
        "temperature_K": values[:, 0].tolist(), "pressure_bar": 0.001,
        "states_passed": {"gas_only": 31, "gas_and_condensates": 31},
        "low_temperature_samples_le_1500K": int((values[:, 0] <= 1500).sum()),
        "gas_P_fraction": gas_p.tolist(),
        "condensed_P_fractions": {
            name: phase_p[:, j].tolist() for j, name in enumerate(names) if phase_matrix[phosphorus, j]
        },
        "gas_input_bytes_match_previous": True,
        "gates": gates, "acceptance_thresholds": THRESHOLDS,
        "active_phase_criterion": {"amount_per_H_greater_than": ACTIVE_CUTOFF},
        "saturation_convention": "Reported gates use native log10(S); NH4SH uses a square-root reaction ratio, so its formula-unit affinity is 2 ln(S).",
        "source_revision": REVISION,
        "input_sha256": {str(path.resolve().relative_to(ROOT)): digest(path) for path in inputs},
        "matched_gas_only": {
            "temperature_grid_exactly_matches": True,
            "only_input_change": "model_eqcond false -> true",
            "implicit_effect": "Condensation mode also enables native CHECK_MELTING corrections, including H2O[s] and P4S3[l].",
            "element_relative_max": gas_closure,
            "gas_P_fraction": (gas_n @ matrix[phosphorus] / budget[phosphorus]).tolist(),
            "input_text": (gasrun / "input.in").read_text(),
        },
        "condensed_input_text": (run / "input.in").read_text(),
        "original_abundances_file_text": (run / "abundances.dat").read_text(),
        "reproduction": {
            "binary_build": "Run mceg/reviews/ohno_ggchem_20261031.py --run --check-unmodified to build the original pinned GGChem executable from the verified archive.",
            "run": "For each mode, create a separate directory, link pinned upstream data/, copy the abundances text to abundances.dat, write the archived input text to input.in, and run ggchem16_original input.in.",
            "analysis_script": str(Path(__file__).relative_to(ROOT)),
            "archive_sha256": digest(BASE / "ggchem_upstream.zip"),
            "function_export": "Use this script --export-input PATH, then phosphorus_native_phase_export_20261031.py --input PATH to evaluate direct phase functions for an independent ExoGibbs solve.",
        },
        "limitations": [
            "Native dust fits do not enforce all original temperature intervals; this is a source-dependent numerical scenario.",
            "The 7-element catalog excludes metals and their phosphate phases, and lacks NH4H2PO4.",
            "No rainout or condensed solutions are included.",
        ],
    }
    write_json(OUT / "native7_phase_receipt.json", report)
    partial = OUT / "native27_condensed/Static_Conc.dat"
    if partial.exists():
        partial_values = np.loadtxt(partial, skiprows=3)
        write_json(OUT / "native27_phase_attempt.json", {
            "status": "incomplete_interrupted",
            "reason": "The native 27-element condensation trajectory was interrupted after partial high-temperature output; excluded from validated low-temperature results.",
            "expected_samples": 121, "completed_samples": len(partial_values),
            "completed_temperature_range_K": [float(partial_values[:, 0].min()), float(partial_values[:, 0].max())],
            "termination_exit_code": 130, "output_sha256": digest(partial),
        })
    if export_input is not None:
        export_bootstrap(export_input, case, names, phase_matrix, values, gas_values)
    print("Native gas/condensed states passed: 31 + 31; gates:", gates)


def export_bootstrap(path: Path, case: dict, names: list, matrix: np.ndarray,
                     condensed: np.ndarray, gas: np.ndarray) -> None:
    """Prepare a comparison input; direct exporter must replace phase functions."""
    ngas = len(case["species"])
    start = 4 + ngas
    exact_t = np.asarray(case["temperature_K"])[::4]
    np.testing.assert_allclose(exact_t, condensed[:, 0], rtol=0, atol=1e-6)
    atom_count = len(case["elements"])
    potentials = []
    for values in (gas, condensed):
        density = values[:, 4:start] * np.log(10.0)
        logx = density - logsumexp(density, axis=1)[:, None]
        atom_logp = logx[:, :atom_count] + np.log(values[:, 2] / 1e6)[:, None]
        log_s = values[:, start:start + len(names)] * np.log(10.0)
        assert abs(log_s).max() < 299 * np.log(10.0)
        log_s[:, names.index("NH4SH")] *= 2.0
        potentials.append(atom_logp @ matrix - log_s)
    verified = [i for i, name in enumerate(names) if name not in ("H2O", "P4S3[l]")]
    error = float(abs(potentials[0][:, verified] - potentials[1][:, verified]).max())
    assert error < 1e-7
    density = condensed[:, 4:start] * np.log(10.0)
    result = {
        "scope": "Bootstrap: phase functions reconstructed from native condensation output; direct function export is required before comparison.",
        "source_revision": REVISION,
        "temperature_K": exact_t.tolist(), "pressure_bar": 0.001,
        "elements": case["elements"], "gas_species": case["species"],
        "gas_formula_matrix": case["formula_matrix"],
        "gas_lnK_bar": np.asarray(case["lnK_bar"])[::4].tolist(),
        "budget": case["budget"], "condensate_species": names,
        "condensate_formula_matrix": matrix.tolist(),
        "condensate_hvector_RT": potentials[1].tolist(),
        "native_condensed_gas_logx": (density - logsumexp(density, axis=1)[:, None]).tolist(),
        "native_condensed_amounts_per_H": (10.0 ** condensed[:, start + len(names):start + 2 * len(names)]).tolist(),
        "native_condensed_gas_amounts_per_H": (10.0 ** (condensed[:, 4:start] - np.log10(condensed[:, 1])[:, None])).tolist(),
        "max_abs_function_difference_without_melting_corrections_RT": error,
        "saturation_normalization": {"NH4SH": "Native S is sqrt(reaction activity ratio); formula-unit affinity is 2 ln(S)."},
        "input_sha256": {
            str(source.relative_to(ROOT)): digest(source)
            for source in (BASE / "native_cases.json", OUT / "native7_gas/Static_Conc.dat", OUT / "native7_condensed/Static_Conc.dat")
        },
    }
    write_json(path, result)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--export-input", type=Path)
    analyze(parser.parse_args().export_input)
