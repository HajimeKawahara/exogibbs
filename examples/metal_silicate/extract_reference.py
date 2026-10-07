"""Extract the source thermochemistry
========================================

Extract frozen thermochemistry from a verified local GCE checkout, offline.
"""

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
import sys
from tempfile import TemporaryDirectory


REFERENCE_PATH = Path(__file__).with_name("reference.json")
COMMIT = "31558873d8da460c3cd11986b574cb347621b43d"
SOURCE_HASHES = {
    "Young_2023_Version/Equations.py": "9ee5b8c5da458325df9a4dccbf121b972d59821421b96a1fdc500a7f0dbef191",
    "Young_2023_Version/Gibbs_Young_Version.py": "de26349e1b71c60b7f5fb60ba35a8f11d04d69e811795e7dd880be0370e91d03",
    "Young_2023_Version/chem_input.dat": "7b7927bd8cf6923274624ebce1fa79259d6a5777b4f49c7a23d5f658e4084aef",
}
ELEMENTS = ["Si", "Mg", "O", "Fe", "H", "Na", "C"]
# Frozen subset of the independently generated ExoEOS PR1 standard-state record.
EXOEOS_STANDARD_REFERENCE = {
    "repository": "https://github.com/HajimeKawahara/exoeos",
    "commit": "39176c423e9e7783dd956e1abe75c9b610a81864",
    "path": "tests/reference/fe_si_o_ma2001.json",
    "sha256": "cf59178a3cf6622b924cd8d8ac1d03e7e5feb65c6f2e687dcb3b66820b601e0f",
    "model_id": "ma2001_fe_si_o_young2023_printed_v1",
    "components": ["Fe", "Si", "O"],
    "states": [
        {
            "id": "ternary_2350", "T_K": 2350.0, "P_Pa": 100000.0,
            "x": [0.85, 0.1, 0.05],
            "ln_gamma_source": [-0.020569452065669763, -4.506406117248992, -3.555211577385589],
            "ln_gamma_formal": [-0.020569452065669763, -9.097248670440482, 6.197341614103773],
            "mu0_formal_minus_source_over_RT": [0.0, 4.590842553191489, -9.752553191489362],
        },
        {
            "id": "ternary_3000", "T_K": 3000.0, "P_Pa": 100000.0,
            "x": [0.8, 0.15, 0.05],
            "ln_gamma_source": [-0.060760074213998684, -3.1426088880910044, -2.089261590132516],
            "ln_gamma_formal": [-0.060760074213998684, -6.738768888091005, 4.620738409867484],
            "mu0_formal_minus_source_over_RT": [0.0, 3.59616, -6.71],
        },
    ],
}
# Each row follows the species order in the pinned Young-version source.
REACTION_ROWS = [
    {6: -1, 5: 1, 1: 1},
    {11: -1, 1: -0.5, 12: 0.5, 3: 1},
    {2: -1, 0: 1, 1: 1},
    {12: -0.5, 13: -1, 1: 0.5},
    {14: -2, 7: 1},
    {4: -1, 3: 1, 1: 1},
    {8: -2, 12: -1, 1: 1, 7: 2},
    {16: -1, 19: -0.5, 17: 1},
    {18: -1, 19: -0.5, 15: 2, 16: 1},
    {19: -0.5, 15: -1, 20: 1},
    {3: -1, 19: 0.5, 21: 1},
    {0: -1, 19: 0.5, 22: 1},
    {1: -1, 19: 0.5, 23: 1},
    {5: -1, 19: 0.5, 24: 2},
    {15: -1, 7: 1},
    {20: -1, 8: 1},
    {16: -1, 9: 1},
    {17: -1, 10: 1},
]


def _source_case(script: Path, temperature: float, phases: dict) -> dict:
    """Run the unmodified source in a temporary directory at one temperature."""
    previous_cwd, previous_argv, previous_path = Path.cwd(), sys.argv, sys.path[:]
    try:
        with TemporaryDirectory(prefix="exogibbs-thermochemistry-") as directory:
            os.chdir(directory)
            sys.argv = [str(script), str(temperature), str(temperature)]
            with redirect_stdout(io.StringIO()):
                namespace = runpy.run_path(str(script), run_name="__main__")
            gibbs = namespace["Gibbs"]
            shomate, standards = {}, {}
            components = phases["silicate"][:4] + ["Na2O_silicate", "Fe_metal"] + phases["gas"]
            for species in components:
                formula, phase = species.split("_")
                function = namespace[f"Gibbs{'melt' if phase == 'silicate' else phase}{formula}"]
                captured = []

                def capture(state_temperature, *coefficients):
                    captured.extend(float(value) for value in coefficients)
                    return gibbs(state_temperature, *coefficients)

                function.__globals__["Gibbs"] = capture
                standards[species] = float(function(temperature))
                shomate[species] = captured
            return {
                "id": f"isothermal_{int(temperature)}K_1bar",
                "T_K": temperature,
                "P_bar": 1.0,
                "shomate": shomate,
                "source_mu0_J_mol": standards,
                "source_delta_g_over_rt": [float(namespace[f"GRT{i}"][0]) for i in range(18)],
            }
    finally:
        os.chdir(previous_cwd)
        sys.argv, sys.path = previous_argv, previous_path


def extract_reference(checkout: Path) -> dict:
    """Return a reproducible reference from the exact pinned source files."""
    checkout = checkout.resolve()
    for relative_path, expected in SOURCE_HASHES.items():
        actual = hashlib.sha256((checkout / relative_path).read_bytes()).hexdigest()
        if actual != expected:
            raise ValueError(f"Source hash mismatch: {relative_path}")
    source = checkout / "Young_2023_Version"
    assignments = {
        node.targets[0].id: node.value
        for node in ast.parse((source / "Equations.py").read_text()).body
        if isinstance(node, ast.Assign) and isinstance(node.targets[0], ast.Name)
    }
    species = ast.literal_eval(assignments["species"])
    phases = {phase: [name for name in species if name.endswith(f"_{phase}")]
              for phase in ("silicate", "metal", "gas")}
    inputs = {
        node.targets[0].id: ast.literal_eval(node.value)
        for node in ast.parse((source / "chem_input.dat").read_text()).body
        if isinstance(node, ast.Assign)
    }
    return {
        "schema_version": 1,
        "status": "Frozen standard thermochemistry; no equilibrium solve or phase-stability claim.",
        "source": {
            "repository": "https://github.com/ExoInteriors/GlobalChemicalEquilibrium_Release",
            "commit": COMMIT,
            "version": "Young_2023_Version",
            "license": "GPL-3.0; source used for offline reference evaluation only",
            "files": SOURCE_HASHES,
            "gas_constant_J_mol_K": 8.314462618153,
            "log10_to_ln": 2.302585093,
            "standard_pressure_bar": 1.0,
            "shomate_coefficient_order": ["DH", "A", "B", "C", "D", "E", "F", "G", "H"],
            "shomate_units": "t=T_K/1000; enthalpy in kJ/mol, entropy in J/(mol K), output mu0 in J/mol",
            "original_conditions": {name: inputs[name] for name in ("T_AMOI", "T_SME", "Mplanet_Mearth")},
            "thermochemistry_provenance": {
                "Shomate": "Chase (1998) / NIST, as transcribed in the pinned source; no independent NIST download",
                "metal_Si": "Corgne et al. (2008) reaction fit, as used by the pinned source",
                "metal_O": "Badro et al. (2015) reaction fit with the sign selected in the pinned source",
                "metal_H": "Okuchi (1997) fit, as used by the pinned source",
                "silicate_H2": "Hirschmann fit evaluated at 1 bar in the pinned source",
                "silicate_H2O": "Moore et al. (1998) fit, as used by the pinned source",
                "silicate_Na_Fe": "Empirical Magma-code reactions R0/R5 selected by the pinned source",
                "silicate_CO_CO2": "G17=5200+119.77*T J/mol; G16=G17+RT*log_to_ln*log10(3), as used by the pinned source",
            },
            "notes": [
                "Preserve all 25 components and 18 reactions of this version; the general GCE network also includes SiH4.",
                "Source reaction R14 uses -ln(1e4/Pstd), not the ordinary -ln(P/Pstd); this is an additional closure prescription.",
                "Source reactions R1, R3, R4 and R6 use T_SME; the remaining reactions use T_AMOI.",
                "The two frozen cases set both source temperatures equal and evaluate 1 bar standards; they are not source planetary solutions.",
                "No calibrated joint T/P/composition domain is established. Liquid standard continuations do not establish liquid-phase stability.",
                "The source MgO liquid coefficients list 3105-5000 K; both frozen temperatures extrapolate these formal endmember coefficients.",
                "GRT is dimensionless Delta G/(R*T), despite the source output-file header saying J/mol. Natural-log equilibrium constants equal -GRT.",
                "R3 follows Equations.py: 0.5 Si(metal)+O(metal) -> 0.5 SiO2(silicate); the Gibbs script's reaction comment points the other way.",
                "The source activities use gamma_Fe=1 and the GCE O coefficient variant; the completed ExoEOS Fe-Si-O model is a distinct model.",
                "Na2SiO3 and FeSiO3 standards must be reconstructed from the selected empirical reactions, not substituted from the unused source standard formulas.",
            ],
        },
        "phases": phases,
        "elements": ELEMENTS,
        "component_formulas": {
            name: {element: int(count or 1) for element, count in re.findall(r"([A-Z][a-z]?)(\d*)", name.split("_")[0])}
            for name in species
        },
        "original_element_amounts_mol": {element: inputs[f"n{element}"] for element in ELEMENTS},
        "local_contract": {
            "temperature": "one supplied temperature, in K, for every local reaction",
            "pressure": "supplied independently in bar; no planetary pressure law",
            "standard_pressure_bar": 1.0,
            "component_amount_unit": "mol of the declared component formula units; atomic mol for metal",
            "element_amount_unit": "mol of atoms, ordered by elements",
            "phase_amount_definition": "N_phase=sum(n_i in phase); n_i=N_phase*x_i; sum(x_i)=1 for a present phase",
            "element_closure": "b=sum(A_phase@n_phase); finite local oxygen budget, no external oxygen buffer",
            "example_budget": "original_element_amounts_mol interpreted as a supplied local parcel; no planetary mass is inferred",
            "zero_budget": "exact zero; remove unsupported components in a future solve instead of inserting traces",
            "phase_assumptions": "declared silicate liquid, metal liquid, gas; no phase selection or liquid-stability validation",
        },
        "exoeos_standard_reference": EXOEOS_STANDARD_REFERENCE,
        "reactions": [
            {"id": f"R{i}", "stoichiometry": {species[j]: value for j, value in row.items()},
             "source_temperature": "T_SME" if i in (1, 3, 4, 6) else "T_AMOI"}
            for i, row in enumerate(REACTION_ROWS)
        ],
        "cases": [_source_case(source / "Gibbs_Young_Version.py", temperature, phases)
                  for temperature in (2350.0, 3000.0)],
    }


def main() -> None:
    """Regenerate or check the fixture without downloading external resources."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gce-checkout", type=Path, required=True)
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--check", action="store_true")
    group.add_argument("--output", type=Path, default=REFERENCE_PATH)
    args = parser.parse_args()
    reference = extract_reference(args.gce_checkout)
    if args.check:
        if json.loads(REFERENCE_PATH.read_text()) != reference:
            raise SystemExit("The committed reference differs from the pinned source.")
        print("The committed reference matches the pinned source.")
    else:
        args.output.write_text(json.dumps(reference, indent=2, allow_nan=False) + "\n")


if __name__ == "__main__":
    main()
