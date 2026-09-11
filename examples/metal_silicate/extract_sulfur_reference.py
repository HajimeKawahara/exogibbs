"""Extract full GCE sulfur/nitrogen and carbon references offline.
================================================================

Run the hash-verified author equations in an isolated temporary directory.
SymPy is needed only for this optional extraction, never ordinary imports.
"""

from __future__ import annotations

import argparse
import ast
from contextlib import redirect_stdout
import hashlib
import io
import json
import os
from pathlib import Path
import re
import runpy
import shutil
import sys
from tempfile import TemporaryDirectory
from typing import Any

import numpy as np
import scipy
from scipy.optimize import least_squares


COMMIT = "31558873d8da460c3cd11986b574cb347621b43d"
SOURCE_HASHES = {
    "Gibbs.py": "472809c09f215989e8bf294b4cbce6f008b7522eebee55a4593f16110808d390",
    "Sulfur_Nitrogen_Version/Equations.py": "5b76c1498dd1470c7a5a1b7deb24b9eb2a296493d31b5e6f04c33684d7e061e3",
    "Sulfur_Nitrogen_Version/chem_input.dat": "f2b7cd7eb4307ee265d3a2be1f4b77ff002b92566d11af6ad7b6d57e1cb691b0",
    "Carbon_Version/Equations.py": "71ba973fed7b865a3a9fa1908b7c331997675650d6b48b7ad27a484aeb54f8f6",
    "Carbon_Version/chem_input.dat": "00bc68f2d744539b9077de9c7309d43cb1760c5b77e5fe941a1475a151873cdd",
    "src/read_Molecular_Weight.py": "dcd0430f625835c6420042b2e9a9887a9ed39d5b0886d375ed998f20e5f4496d",
    "Molecular_Weight.dat": "eec73b5d3187e53f09a43d1716d0bc066c199a3845399a93e316a93809c1c025",
}


def _network(checkout: Path, version: str, thermochemistry: dict) -> dict[str, Any]:
    import sympy as sy

    namespace = runpy.run_path(str(checkout / version / "Equations.py"))
    species = namespace["species"]
    sulfur = version == "Sulfur_Nitrogen_Version"
    count = 28 if sulfur else 19
    reactions = namespace["ff"][:count]
    variables = namespace["var"]
    grt = namespace["GRT_T"]
    pressure = next(symbol for symbol in reactions[7].free_symbols if str(symbol) == "P")
    # Replace the source's explicitly named corrections to recover only the
    # balanced reaction coefficients, including the separate host correction.
    corrections = [sy.Integer(0)] * count
    corrections[1] = namespace["lngSi"] / 2
    corrections[3] = -namespace["lngSi"] / 2 - namespace["lngO"]
    corrections[6] = -namespace["lngSi"]
    corrections[19 if sulfur else 18] = namespace["lngCmetal"]
    if sulfur:
        corrections[19] += namespace["lngO"]
        corrections[21] = namespace["lngS"]
    rows = []
    for expression, correction in zip(reactions, corrections):
        ordinary = sy.expand(expression - correction)
        rows.append({name: float(ordinary.coeff(sy.log(variables[name])))
                     for name in species if ordinary.coeff(sy.log(variables[name]))})
    formula = {
        name: {element: float(number or 1)
               for element, number in re.findall(r"([A-Z][a-z]?)(\d*)", name.split("_")[0])}
        for name in species
    }
    if sulfur:
        formula["FeO15_silicate"] = {"Fe": 1.0, "O": 1.5}
    elements = ["Si", "Mg", "O", "Fe", "H", "Na", "C"] + (["S", "N"] if sulfur else [])
    phases = {phase: [name for name in species if name.endswith("_" + phase)]
              for phase in ("silicate", "metal", "gas")}
    groups = [np.array([species.index(name) for name in names]) for names in phases.values()]
    formula_matrix = np.array([[formula[name].get(element, 0) for name in species]
                               for element in elements])
    inputs = {node.targets[0].id: ast.literal_eval(node.value)
              for node in ast.parse((checkout / version / "chem_input.dat").read_text()).body
              if isinstance(node, ast.Assign)}
    budget = np.array([inputs["n" + element] for element in elements])
    original = sy.lambdify(
        (sy.IndexedBase("var", len(variables), real=True), namespace["T_SME"],
         namespace["T_AMOI"], pressure, sy.Symbol("Pstd", real=True), grt), reactions, "numpy",
    )
    cases = []
    for case_index, temperature in enumerate(thermochemistry["TK"]):
        constants = np.array([thermochemistry[f"GRT{i}"][case_index] for i in range(28)])
        reference_pressure = 1.0e4 if sulfur else 1000.0
        initial = np.full(len(species), 0.002)
        guesses = {"MgO_silicate": 0.20, "SiO2_silicate": 0.04,
                   "MgSiO3_silicate": 0.50, "FeO_silicate": 0.10,
                   "FeSiO3_silicate": 0.02, "H2_silicate": 0.02,
                   "H2O_silicate": 0.08, "Fe_metal": 0.84,
                   "Si_metal": 0.06, "O_metal": 0.03, "H_metal": 0.06,
                   "H2_gas": 0.80, "H2O_gas": 0.15}
        for name, value in guesses.items():
            initial[species.index(name)] = value
        for group, phase_amount in zip(groups, (budget[0] + budget[1], budget[3], budget[4] / 2)):
            initial[group] *= phase_amount / initial[group].sum()
        scale = budget.sum()

        def quantities(log_amounts):
            amounts = scale * np.exp(log_amounts)
            fractions = np.empty_like(amounts)
            for group in groups:
                fractions[group] = amounts[group] / amounts[group].sum()
            chemistry = np.array(original(np.r_[fractions, 1., 1., 1.], temperature,
                                          temperature, reference_pressure, 1.0, constants))
            balance = formula_matrix @ amounts / budget - 1
            return amounts, fractions, chemistry, balance

        result = least_squares(lambda y: np.r_[quantities(y)[2], quantities(y)[3]],
                               np.log(initial / scale), bounds=(-650, 10),
                               xtol=1e-13, ftol=1e-13, gtol=1e-13, max_nfev=2000)
        amounts, fractions, chemistry, balance = quantities(result.x)
        if not result.success or max(np.max(np.abs(chemistry)), np.max(np.abs(balance))) > 1e-9:
            raise RuntimeError(f"Unaccepted {version} reference at {temperature}: {result.message}; "
                               f"chemistry={np.max(np.abs(chemistry))}, balance={np.max(np.abs(balance))}")
        constants_used = constants if sulfur else constants[np.r_[np.arange(18), 19]]
        probes = []
        for exponent, probe_pressure in ((1.0, 1.0), (0.7, 1000.0)):
            probe = initial**exponent
            for group in groups:
                probe[group] /= probe[group].sum()
            probes.append({"P_bar": probe_pressure, "mole_fractions": probe.tolist(),
                           "source_reaction_residual": np.asarray(original(
                               np.r_[probe, 1., 1., 1.], temperature, temperature,
                               probe_pressure, 1.0, constants)).tolist()})
        cases.append({"id": f"{version.lower()}_{int(temperature)}K",
                      "T_K": float(temperature), "P_bar": reference_pressure,
                      "source_delta_g_over_rt": constants_used.tolist(),
                      "initial_component_amounts_mol": initial.tolist(),
                      "component_amounts_mol": amounts.tolist(),
                      "source_reaction_residual": chemistry.tolist(),
                      "relative_element_residual": balance.tolist(),
                      "scipy_evaluations": result.nfev, "probes": probes})
    return {"model_id": f"gce_{version.lower()}_source_local_v1", "version": version,
            "phases": phases, "elements": elements, "component_formulas": formula,
            "element_amounts_mol": budget.tolist(),
            "original_conditions": {name: inputs[name]
                                    for name in ("T_AMOI", "T_SME", "Mplanet_Mearth")},
            "reactions": [{"id": f"R{i}", "stoichiometry": row} for i, row in enumerate(rows)],
            "cases": cases}


def extract_reference(checkout: Path) -> dict[str, Any]:
    """Evaluate the exact pinned full networks without downloading resources."""
    import sympy

    checkout = checkout.resolve()
    for name, expected in SOURCE_HASHES.items():
        if hashlib.sha256((checkout / name).read_bytes()).hexdigest() != expected:
            raise ValueError(f"Source hash mismatch: {name}")
    previous_cwd, previous_path, previous_argv = Path.cwd(), sys.path[:], sys.argv[:]
    previous_reader = sys.modules.pop("read_Molecular_Weight", None)
    try:
        with TemporaryDirectory(prefix="exogibbs-sulfur-source-") as temporary:
            root = Path(temporary)
            shutil.copyfile(checkout / "Molecular_Weight.dat", root / "Molecular_Weight.dat")
            (root / "work").mkdir()
            os.chdir(root / "work")
            sys.path.insert(0, str(checkout / "src"))
            sys.argv = [str(checkout / "Gibbs.py"), "2350", "3000"]
            with redirect_stdout(io.StringIO()):
                thermochemistry = runpy.run_path(str(checkout / "Gibbs.py"))
                networks = {name: _network(checkout, version, thermochemistry)
                            for name, version in (("sulfur_nitrogen", "Sulfur_Nitrogen_Version"),
                                                  ("carbon", "Carbon_Version"))}
    finally:
        os.chdir(previous_cwd)
        sys.path, sys.argv = previous_path, previous_argv
        sys.modules.pop("read_Molecular_Weight", None)
        if previous_reader is not None:
            sys.modules["read_Molecular_Weight"] = previous_reader
    return {
        "schema_version": 1,
        "extraction_environment": {"python": sys.version.split()[0],
                                   "numpy": np.__version__, "scipy": scipy.__version__,
                                   "sympy": sympy.__version__, "dtype": "float64"},
        "source": {"repository": "https://github.com/ExoInteriors/GlobalChemicalEquilibrium_Release",
                   "commit": COMMIT, "files": SOURCE_HASHES,
                   "license": "GPL-3.0; unmodified code evaluated offline, not vendored",
                   "log10_to_ln": 2.302585093, "standard_pressure_bar": 1.0,
                   "gas_constant_J_mol_K": 8.314462618153,
                   "nitrogen_gas_constant_J_mol_K": 8.314462618},
        "evidence_level": "source reproduction",
        "domain": {"mathematical": "Positive finite amounts, T and P; Fe-rich interior alloy.",
                   "calibration": "No joint calibration or phase stability established.",
                   "reference_temperatures_K": [2350.0, 3000.0],
                   "extrapolation": "Frozen source cases; both extrapolate the MgO liquid fit."},
        "audit": [
            "All original components and reactions retained: S/N 37/28, Carbon 26/19.",
            "Source molecular/formula-unit moles in gas/silicate; atomic moles in metal; FeO15 means FeO1.5.",
            "One supplied isothermal T and local P replace the source two-temperature and planetary-pressure closure.",
            "Both versions use ln(P/Pstd) for R14; the Young-version fixed 1e4 bar offset does not apply.",
            "The Calvo correction uses all silicate association-component mole fractions, not an oxide-renormalized composition.",
            "Preserve lngS = -2.302585093*(-logC_S + ln(x_FeO)); the mixed log-base convention requires original-Calvo audit.",
            "logC_S = -5.704 + 3.15*x_FeO + 0.12*x_MgO + 0.75*x_Na2O; pressure, CaO, TiO2 and K2O terms are omitted.",
            "GRT21 = mu_Fe_source/(RT) - 2.302585093*(-9+14530/T); do not interpret the host correction as intrinsic alloy excess energy.",
            "C correction is -2.303*19.5*ln(1-x_O); GRT19 includes -2.303*(0.3+3822/T)+mu_O_source/(RT).",
            "Carbon R18 uses GRT19 without lngO; S/N R19 includes lngO. These are distinct pinned source prescriptions.",
            "N dissolution preserves the negative GRT25 and the source's two gas constants; its sign and Bernadou basis are not independently calibrated here.",
            "No SCSS law, sulfide phase, graphite/carbide, metal N/nitride, common multicomponent excess energy or stability selection is supplied.",
            "Original Calvo/Blanchard/Fischer calibration data and SCSS data are absent from this fixture; source agreement is not empirical validation.",
        ],
        "networks": networks,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkout", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.write_text(json.dumps(extract_reference(args.checkout), indent=2,
                                     allow_nan=False) + "\n", encoding="utf-8")
