"""Milestone 1 local source-to-atmosphere chemistry diagnostics.

Compare the frozen 2350 K source with packaged neutral FastChem4 data using
ExoGibbs solvers. These parcels do not close a planetary pressure or inventory.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import subprocess
import time
from typing import Any, Sequence

import jax
import jax.numpy as jnp
import numpy as np

import exogibbs
from exogibbs.api.condensate import (
    CondensateChemicalSetup,
    CondensateEquilibriumOptions,
    build_condensate_chemical_setup,
    solve as solve_condensate,
)
from exogibbs.api.gas import EquilibriumOptions, solve as solve_gas
from exogibbs.io.load_data import get_data_filepath
from exogibbs.presets.fastchem4_cond import condensate_chemical_setup
from exogibbs.thermo.models import ChemicalSetup
from exogibbs.utils.elements import element_mass


_SPEC = importlib.util.spec_from_file_location(
    "m1_source", Path(__file__).with_name("sulfur_source.py"),
)
SOURCE = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(SOURCE)

MODEL_ID = "m1_source_upper_chemistry_v1"
ELEMENTS = ("H", "He", "O", "Mg", "Si", "Fe", "Na")
SHARED_SOURCE_SPECIES = (
    "H2_gas", "O2_gas", "H2O_gas", "Fe_gas", "Mg_gas", "SiO_gas",
    "Na_gas", "SiH4_gas", "He_gas",
)
SHARED_GAS_SPECIES = (
    "H2", "O2", "H2O1", "Fe1", "Mg1", "O1Si1", "Na1", "H4Si1", "He1",
)
EXPANDED_GAS_SPECIES = (
    "Fe1", "H1", "He1", "Mg1", "Na1", "O1", "Si1", "Fe1H1",
    "Fe1H2O2", "Fe1O1", "H1Mg1", "H1Mg1O1", "H1Na1", "H1Na1O1",
    "H1O1", "H1O2", "H1Si1", "H2", "H2Mg1O2", "H2Na2O2", "H2O1",
    "H2O2", "H2Si1", "H3Si1", "H4Si1", "Mg1O1", "Mg2", "Na1O1",
    "Na2", "O1Si1", "O2", "O2Si1", "O3", "Si2", "Si3",
)
CONDENSATE_SPECIES = (
    "Fe(s,l)", "Fe(OH)2(s)", "Fe(OH)3(s)", "FeO(s,l)", "Fe2O3(s)",
    "Fe3O4(s)", "NaH(s)", "NaOH(s,l)", "MgH2(s)", "Mg(OH)2(s)",
    "Mg(s,l)", "MgO(s,l)", "MgSiO3(s,l)", "Mg2SiO4(s,l)", "Mg2Si(s,l)",
    "Na(s,l)", "NaO2(s)", "Na2O(s,l)", "Na2O2(s)", "Na2SiO3(s,l)",
    "Na2Si2O5(s,l)", "SiO2(s,l)", "Si(s,l)", "Fe2SiO4(s)", "H2O(s,l)",
    "SiO(s)",
)
ELEMENT_TOLERANCE = 1.0e-9
CHEMICAL_TOLERANCE = 1.0e-8
BOUNDARY_CONTRACT = {
    "id": "m1_fixed_source_retained_parcel_v1",
    "kind": "approximate",
    "continuous": ["temperature_K", "total_pressure_bar", "element_amounts_mol"],
    "amount_basis": "Source gas atom moles equal upper gas plus retained cloud atom moles; no rainout.",
    "mass_basis": "Atom moles are authoritative; each consumer recomputes kg with its own atomic masses.",
    "contact_phases": "Source silicate and metal stay fixed; upper pure condensates do not contact them.",
    "unconstrained": ["gas_partial_pressures", "species_amounts", "source_reaction_equilibrium",
                      "enthalpy", "entropy", "energy_balance"],
    "acceptance": "Local audits and transfer closure do not certify common equilibrium or accept M1-A.",
}


def subset_setup(setup: ChemicalSetup, species: Sequence[str]) -> ChemicalSetup:
    """Select named species and seven element rows without changing standards."""
    names = tuple(species)
    if len(set(names)) != len(names):
        raise ValueError("Species must be unique.")
    columns = [setup.species.index(name) for name in names]
    rows = [setup.elements.index(element) for element in ELEMENTS]
    excluded = [i for i, element in enumerate(setup.elements) if element not in ELEMENTS]
    matrix = np.asarray(setup.formula_matrix)
    if np.any(matrix[np.ix_(excluded, columns)] != 0):
        raise ValueError("Selected species contain an excluded element or charge.")
    indices = jnp.asarray(columns)

    def hvector(temperature):
        return jnp.take(setup.hvector_func(temperature), indices, axis=-1)

    upper = setup.temperature_validity_upper
    upper = None if upper is None else tuple(upper[i] for i in columns)
    metadata = dict(setup.metadata or {})
    if upper is not None:
        metadata["temperature_validity_upper"] = upper
    return ChemicalSetup(
        formula_matrix=jnp.asarray(matrix[np.ix_(rows, columns)]),
        hvector_func=hvector, elements=ELEMENTS, species=names,
        temperature_validity_upper=upper, metadata=metadata,
    )


def build_setups() -> tuple[ChemicalSetup, CondensateChemicalSetup]:
    """Build the shared gas control and the explicit neutral upper catalog."""
    full = condensate_chemical_setup(silent=True)
    shared = subset_setup(full.gas_setup, SHARED_GAS_SPECIES)
    upper = build_condensate_chemical_setup(
        gas_setup=subset_setup(full.gas_setup, EXPANDED_GAS_SPECIES),
        condensate_setup=subset_setup(full.condensate_setup, CONDENSATE_SPECIES),
    )
    return shared, upper


def source_inputs(oxygen_factor: float = 1.0) -> tuple[dict, dict, np.ndarray]:
    """Use pinned ratios, exact-zero CNS and He/H=0.1 at fixed local mass."""
    if not np.isfinite(oxygen_factor) or oxygen_factor <= 0:
        raise ValueError("oxygen_factor must be finite and positive.")
    network = SOURCE.build_helium_network(SOURCE.load_reference()["networks"]["sulfur_nitrogen"])
    case = next(case for case in network["cases"] if case["T_K"] == 2350.0)
    elements = network["elements"]
    budget = np.asarray(network["element_amounts_mol"], dtype=float)
    budget[[elements.index(element) for element in ("C", "N", "S")]] = 0.0
    budget[elements.index("He")] = 0.1 * budget[elements.index("H")]
    weights = np.array([element_mass[element] * 1.0e-3 for element in elements])
    reference_mass = weights @ budget
    budget[elements.index("O")] *= oxygen_factor
    budget *= reference_mass / (weights @ budget)
    return network, case, budget


def source_gas_inventory(network: dict, result: dict) -> np.ndarray:
    """Recount source gas atoms in ELEMENTS order, excluding both deep phases."""
    species, formula, _ = SOURCE.component_matrices(network)
    columns = [species.index(name) for name in network["phases"]["gas"]]
    gas_atoms = formula[:, columns] @ np.asarray(result["component_amounts_mol"])[columns]
    absent = [network["elements"].index(element) for element in ("C", "N", "S")]
    if np.any(gas_atoms[absent] != 0):
        raise ValueError("The milestone 1 parcel requires exact-zero C/N/S.")
    return gas_atoms[[network["elements"].index(element) for element in ELEMENTS]]


def audit_parcel(
    gas_setup: ChemicalSetup, temperature: float, pressure_bar: float,
    budget: np.ndarray, gas_amounts: np.ndarray, *,
    condensate_setup: ChemicalSetup | None = None,
    condensate_amounts: np.ndarray | None = None,
) -> dict[str, Any]:
    """Recount atoms/mass and independently test ideal-gas/pure-phase KKT.

    Input amounts share one mol basis. Output mol/kg refers to gas+cloud;
    cloud has mass but contributes neither gas mole fractions nor pressure.
    """
    b, gas = np.asarray(budget, dtype=float), np.asarray(gas_amounts, dtype=float)
    if (not np.isfinite(temperature) or temperature <= 0 or
            not np.isfinite(pressure_bar) or pressure_bar <= 0):
        raise ValueError("Temperature and pressure must be finite and positive.")
    if b.shape != (len(gas_setup.elements),) or np.any(~np.isfinite(b)) or np.any(b <= 0):
        raise ValueError("The active element budget must be finite and positive.")
    if gas.shape != (len(gas_setup.species),) or np.any(~np.isfinite(gas)) or np.any(gas <= 0):
        raise ValueError("Gas amounts must be finite and positive in species order.")
    a_g = np.asarray(gas_setup.formula_matrix)
    if condensate_setup is None:
        if condensate_amounts is not None:
            raise ValueError("Condensate amounts require a condensate setup.")
        cloud, a_c, h_c, eligible = np.zeros(0), np.zeros((len(b), 0)), np.zeros(0), np.zeros(0, bool)
        condensate_names = ()
    else:
        if condensate_setup.elements != gas_setup.elements:
            raise ValueError("Gas and condensate element orders must match.")
        cloud = np.asarray(condensate_amounts, dtype=float)
        condensate_names = condensate_setup.species
        if cloud.shape != (len(condensate_names),) or np.any(~np.isfinite(cloud)) or np.any(cloud < 0):
            raise ValueError("Condensate amounts must be finite and nonnegative.")
        a_c = np.asarray(condensate_setup.formula_matrix)
        h_c = np.asarray(condensate_setup.hvector_func(temperature))
        bounds = condensate_setup.temperature_validity_upper
        eligible = np.ones(len(cloud), bool) if bounds is None else temperature <= np.asarray(bounds)
    gas_atoms, cloud_atoms = a_g @ gas, a_c @ cloud
    relative_elements = (gas_atoms + cloud_atoms) / b - 1.0
    weights = np.array([element_mass[element] * 1.0e-3 for element in gas_setup.elements])
    gas_mass, cloud_mass, target_mass = weights @ gas_atoms, weights @ cloud_atoms, weights @ b
    x = gas / gas.sum()
    mu = np.asarray(gas_setup.hvector_func(temperature)) + np.log(x) + np.log(pressure_bar)
    if not np.all(np.isfinite(mu)) or not np.all(np.isfinite(h_c[eligible])):
        raise ValueError("Nonfinite thermochemical potentials at the requested point.")
    potentials = np.linalg.lstsq(a_g.T, mu, rcond=None)[0]
    gas_residual = mu - a_g.T @ potentials
    insertion = h_c - a_c.T @ potentials
    present = cloud > 0
    max_gas = float(np.max(np.abs(gas_residual)))
    max_present = float(np.max(np.abs(insertion[present]), initial=0.0))
    max_absent = float(np.max(-insertion[eligible & ~present], initial=0.0))
    mass_residual = float((gas_mass + cloud_mass) / target_mass - 1.0)
    accepted = bool(
        np.all(np.isfinite(mu)) and np.all(np.isfinite(insertion[eligible]))
        and np.max(np.abs(relative_elements)) < ELEMENT_TOLERANCE
        and abs(mass_residual) < ELEMENT_TOLERANCE
        and max_gas < CHEMICAL_TOLERANCE and max_present < CHEMICAL_TOLERANCE
        and max_absent < CHEMICAL_TOLERANCE and not np.any(present & ~eligible)
    )
    names = list(gas_setup.species)
    return {
        "accepted": accepted, "elements": list(gas_setup.elements),
        "gas_species": names, "condensate_species": list(condensate_names),
        "element_amounts_mol": b.tolist(), "gas_amounts_mol": gas.tolist(),
        "condensate_amounts_mol": cloud.tolist(),
        "gas_element_amounts_mol": gas_atoms.tolist(),
        "cloud_element_amounts_mol": cloud_atoms.tolist(),
        "relative_element_residual": relative_elements.tolist(),
        "relative_mass_residual": mass_residual,
        "gas_stationarity_max_abs": max_gas,
        "present_condensate_residual_max_abs": max_present,
        "absent_condensate_violation_max_abs": max_absent,
        "condensate_temperature_eligible": eligible.tolist(),
        "condensate_insertion_rt": [float(value) if valid else None for value, valid in zip(insertion, eligible)],
        "gas_mass_kg": float(gas_mass), "cloud_mass_kg": float(cloud_mass),
        "gas_mass_fraction": float(gas_mass / (gas_mass + cloud_mass)),
        "total_element_mol_per_kg": ((gas_atoms + cloud_atoms) / (gas_mass + cloud_mass)).tolist(),
        "gas_mole_fractions": x.tolist(), "partial_pressures_bar": (x * pressure_bar).tolist(),
        "mean_gas_molar_mass_kg_per_mol": float(gas_mass / gas.sum()),
        "h2o_h2": float(x[names.index("H2O1")] / x[names.index("H2")]) if {"H2O1", "H2"} <= set(names) else None,
    }


def solve_parcel(
    setup: ChemicalSetup | CondensateChemicalSetup, temperature: float,
    pressure_bar: float, element_amounts_mol: np.ndarray,
) -> dict[str, Any]:
    """Solve and audit one retained parcel, restoring the caller's mol basis."""
    b = np.asarray(element_amounts_mol, dtype=float)
    if b.shape != (len(ELEMENTS),) or np.any(~np.isfinite(b)) or np.any(b <= 0):
        raise ValueError("Provide seven finite positive element amounts in ELEMENTS order.")
    if not np.isfinite(temperature) or temperature <= 0 or not np.isfinite(pressure_bar) or pressure_bar <= 0:
        raise ValueError("Temperature and pressure must be finite and positive.")
    if not jax.config.x64_enabled:
        raise ValueError("Milestone 1 tolerances require JAX_ENABLE_X64=1.")
    scale = float(b.sum())
    if not np.isfinite(scale):
        raise ValueError("The total atom amount must be finite.")
    started = time.perf_counter()
    if isinstance(setup, CondensateChemicalSetup):
        result = solve_condensate(
            setup, temperature, pressure_bar, jnp.asarray(b / scale),
            options=CondensateEquilibriumOptions(
                return_diagnostics=True, rainout=False,
                full_condensate_budget_relative_tolerance=ELEMENT_TOLERANCE,
            ),
        )
        gas = np.asarray(result.gas_n) * scale
        cloud = np.asarray(result.condensate_amounts) * scale
        report = audit_parcel(setup.gas_setup, temperature, pressure_bar, b, gas,
                              condensate_setup=setup.condensate_setup, condensate_amounts=cloud)
        report.update(solver_status=result.status, solver_converged=bool(result.converged))
        report["accepted"] = report["accepted"] and bool(result.converged)
    else:
        result, diagnostics = solve_gas(
            setup, temperature, pressure_bar, jnp.asarray(b / scale),
            options=EquilibriumOptions(epsilon_crit=1.0e-14), return_diagnostics=True,
        )
        report = audit_parcel(setup, temperature, pressure_bar, b, np.asarray(result.n) * scale)
        report["solver_diagnostics"] = {key: np.asarray(value).tolist() for key, value in diagnostics.items()}
        report["solver_converged"] = bool(diagnostics["converged"])
        report["accepted"] = report["accepted"] and report["solver_converged"]
    report.update(T_K=float(temperature), P_bar=float(pressure_bar),
                  normalization_atom_mol=scale, elapsed_seconds=time.perf_counter() - started)
    return report


def shared_reaction_comparison(network: dict, case: dict, report: dict) -> dict[str, Any]:
    """Test upper gas against gas-only source reactions, separate from its KKT."""
    species, _, reactions = SOURCE.component_matrices(network)
    columns = [species.index(name) for name in SHARED_SOURCE_SPECIES]
    other = [i for i in range(len(species)) if i not in columns]
    rows = np.flatnonzero(np.all(reactions[:, other] == 0, axis=1) & np.any(reactions[:, columns] != 0, axis=1))
    names = report["gas_species"]
    x = np.asarray(report["gas_mole_fractions"])[[names.index(name) for name in SHARED_GAS_SPECIES]]
    residual = np.asarray(case["source_delta_g_over_rt"])[rows] + reactions[np.ix_(rows, columns)] @ (np.log(x) + np.log(report["P_bar"]))
    return {"source_reaction_indices": rows.tolist(), "source_reaction_residual": residual.tolist()}


def audit_boundary(
    network: dict, case: dict, source: dict,
    setup: ChemicalSetup | CondensateChemicalSetup, report: dict,
) -> dict[str, Any]:
    """Audit the approximate boundary at common T/P with frozen deep phases.

    Use a source result from ``SOURCE.solve_reduced_source`` and a parcel
    result from ``solve_parcel``. Recount actual component arrays; saved parcel
    summaries and acceptance flags cannot establish atom or chemical closure.
    Source reaction shifts are model discrepancies, not upper KKT residuals.
    """
    if source["T_K"] != case["T_K"] or report["T_K"] != source["T_K"] or report["P_bar"] != source["P_bar"]:
        raise ValueError("A boundary comparison requires the same source and upper temperature and pressure.")
    species, formula, reactions = SOURCE.component_matrices(network)
    if tuple(source["species"]) != species or source["elements"] != network["elements"]:
        raise ValueError("Source component and element orders must match the network.")
    condensed = isinstance(setup, CondensateChemicalSetup)
    gas_setup = setup.gas_setup if condensed else setup
    if gas_setup.elements != ELEMENTS or tuple(report["gas_species"]) != gas_setup.species:
        raise ValueError("Boundary gas species and element orders must match the setup.")
    if condensed and tuple(report["condensate_species"]) != setup.condensate_species:
        raise ValueError("Boundary condensate species must match the setup.")
    if not condensed and (report["condensate_species"] or report["condensate_amounts_mol"]):
        raise ValueError("A gas-only boundary cannot include condensates.")
    budget = source_gas_inventory(network, source)
    audited = audit_parcel(
        gas_setup, report["T_K"], report["P_bar"], budget, report["gas_amounts_mol"],
        condensate_setup=setup.condensate_setup if condensed else None,
        condensate_amounts=report["condensate_amounts_mol"] if condensed else None,
    )
    amounts = np.asarray(source["component_amounts_mol"], dtype=float)
    source_budget = np.asarray(source["element_amounts_mol"], dtype=float)
    supported = np.all(formula[source_budget == 0] == 0, axis=0)
    if np.any(~np.isfinite(amounts)) or np.any(amounts[supported] <= 0) or np.any(amounts[~supported] != 0):
        raise ValueError("Source amounts must be positive on active components and exactly zero elsewhere.")
    active = np.flatnonzero(np.all(reactions[:, ~supported] == 0, axis=1))
    fractions = amounts.copy()
    for phase in network["phases"].values():
        columns = [species.index(name) for name in phase]
        fractions[columns] /= amounts[columns].sum()
    chemistry = SOURCE._make_reaction_residual(network, case, np.flatnonzero(supported), active)
    original = np.asarray(chemistry(np.log(fractions[supported]), source["P_bar"]))
    source_balance = formula @ amounts - source_budget
    source_balance[source_budget > 0] /= source_budget[source_budget > 0]
    columns = [species.index(name) for name in SHARED_SOURCE_SPECIES]
    upper_columns = [gas_setup.species.index(name) for name in SHARED_GAS_SPECIES]
    upper_x = np.asarray(audited["gas_mole_fractions"])[upper_columns]
    delta_log_p = np.log(upper_x) - np.log(fractions[columns])
    shifts = reactions[np.ix_(active, columns)] @ delta_log_p
    replaced = original + shifts
    local_accepted = bool(source["accepted"] and report["accepted"])
    return {
        "contract": BOUNDARY_CONTRACT.copy(),
        "contract_met": bool(local_accepted and report["solver_converged"] and audited["accepted"]
                             and np.max(np.abs(original)) < CHEMICAL_TOLERANCE
                             and np.max(np.abs(source_balance)) < ELEMENT_TOLERANCE),
        "local_accepted": local_accepted,
        "common_equilibrium_certified": False, "m1_a_accepted": False,
        "T_K": report["T_K"], "P_bar": report["P_bar"], "elements": list(ELEMENTS),
        "source_gas_element_amounts_mol": budget.tolist(),
        "relative_element_residual": audited["relative_element_residual"],
        "relative_mass_residual": audited["relative_mass_residual"],
        "cloud_element_fraction": (np.asarray(audited["cloud_element_amounts_mol"]) / budget).tolist(),
        "cloud_mass_fraction": 1.0 - audited["gas_mass_fraction"],
        "shared_gas_species": list(SHARED_GAS_SPECIES),
        "delta_log_partial_pressure": delta_log_p.tolist(),
        "source_reaction_indices": active.tolist(),
        "source_elements": list(network["elements"]),
        "source_relative_element_residual": source_balance.tolist(),
        "source_reaction_residual_rt": original.tolist(),
        "delta_source_reaction_residual_rt": shifts.tolist(),
        "fixed_deep_source_reaction_residual_rt": replaced.tolist(),
        "max_abs_delta_source_reaction_residual_rt": float(np.max(np.abs(shifts))),
        "unmet_source_reaction_indices": active[np.abs(replaced) >= CHEMICAL_TOLERANCE].tolist(),
    }


def provenance() -> dict[str, Any]:
    """Record actual imports, local git state, and source/data hashes offline."""
    root = Path(__file__).resolve().parents[2]
    package = Path(exogibbs.__file__).resolve().parent

    def git(directory, *args):
        result = subprocess.run(["git", "-C", str(directory), *args], text=True,
                                capture_output=True, check=False)
        return result.stdout.strip() if result.returncode == 0 else None

    paths = [Path(__file__).resolve(), Path(SOURCE.__file__).resolve(), SOURCE.REFERENCE_PATH,
             Path(get_data_filepath("FastChem4/logK/logK_wo_ions.dat")),
             Path(get_data_filepath("FastChem4/logK/logK_condensates.dat"))]
    digest = hashlib.sha256()
    for path in sorted(package.rglob("*.py")):
        digest.update(str(path.relative_to(package)).encode())
        digest.update(path.read_bytes())
    return {
        "example_git_commit": git(root, "rev-parse", "HEAD"),
        "example_git_status": git(root, "status", "--short"),
        "provider_git_commit": git(package, "rev-parse", "HEAD"),
        "provider_git_status": git(package, "status", "--short"),
        "exogibbs_import_path": str(package), "package_python_sha256": digest.hexdigest(),
        "jax_version": jax.__version__, "numpy_version": np.__version__,
        "jax_backend": jax.default_backend(), "x64_enabled": jax.config.x64_enabled,
        "files": [{"path": str(path), "sha256": hashlib.sha256(path.read_bytes()).hexdigest()} for path in paths],
    }


def run_diagnostic(
    pressure_bar: float, *, oxygen_factor: float = 1.0,
    points: Sequence[tuple[float, float]] = (),
) -> dict[str, Any]:
    """Compare one source boundary and optional independent upper T/P points."""
    if not np.isfinite(pressure_bar) or pressure_bar <= 0:
        raise ValueError("pressure_bar must be finite and positive.")
    if any(not np.isfinite(value) or value <= 0 for point in points for value in point):
        raise ValueError("Upper point temperatures and pressures must be finite and positive.")
    network, case, budget = source_inputs(oxygen_factor)
    record = {
        "model_id": MODEL_ID, "provenance": provenance(),
        "scope": "Local chemical diagnostic; no planetary pressure closure or M1-A acceptance.",
        "oxygen_factor": oxygen_factor, "helium_h_atom_ratio": 0.1,
        "inputs": {"source_T_K": case["T_K"], "source_P_bar": pressure_bar,
                   "source_model_id": network["model_id"], "source_case_id": case["id"],
                   "elements": network["elements"], "element_amounts_mol": budget.tolist(),
                   "upper_points_T_K_P_bar": [list(point) for point in points]},
        "source": None, "parcels": [], "failures": [],
        "boundary_contract": BOUNDARY_CONTRACT.copy(),
        "atomic_molar_masses_kg_per_mol": {e: element_mass[e] * 1.0e-3 for e in network["elements"]},
    }
    started = time.perf_counter()
    try:
        source = SOURCE.solve_reduced_source(network, case, element_amounts_mol=budget,
                                              pressure_bar=pressure_bar)
    except (ValueError, RuntimeError) as error:
        record["failures"].append({"stage": "source", "reason": str(error), "elapsed_seconds": time.perf_counter() - started})
        return record
    record["source"] = source
    record["source_elapsed_seconds"] = time.perf_counter() - started
    b = source_gas_inventory(network, source)
    source_columns = [source["species"].index(name) for name in SHARED_SOURCE_SPECIES]
    source_gas = np.asarray(source["component_amounts_mol"])[source_columns]
    source_x = source_gas / source_gas.sum()
    weights = np.array([element_mass[element] * 1.0e-3 for element in ELEMENTS])
    record["source_gas"] = {
        "species": list(SHARED_GAS_SPECIES), "element_amounts_mol": b.tolist(),
        "gas_amounts_mol": source_gas.tolist(), "gas_mole_fractions": source_x.tolist(),
        "partial_pressures_bar": (source_x * pressure_bar).tolist(),
        "mean_gas_molar_mass_kg_per_mol": float(weights @ b / source_gas.sum()),
        "h2o_h2": float(source_x[2] / source_x[0]),
    }
    started = time.perf_counter()
    shared, upper = build_setups()
    record["setup_elapsed_seconds"] = time.perf_counter() - started
    record["catalog"] = {
        "elements": list(ELEMENTS), "shared_source_species": list(SHARED_SOURCE_SPECIES),
        "shared_gas_species": list(shared.species), "expanded_gas_species": list(upper.gas_species),
        "condensate_species": list(upper.condensate_species),
        "condensate_temperature_validity_upper_K": list(upper.condensate_setup.temperature_validity_upper),
        "validity_note": "Upper temperature bounds only; no joint source/upper T/P calibration is asserted.",
    }
    jobs = [("shared_gas", shared, 2350.0, pressure_bar),
            ("expanded_gas", upper.gas_setup, 2350.0, pressure_bar),
            ("expanded_condensed", upper, 2350.0, pressure_bar)]
    for temperature, pressure in points:
        jobs.extend([("expanded_gas", upper.gas_setup, temperature, pressure),
                     ("expanded_condensed", upper, temperature, pressure)])
    for stage, setup, temperature, pressure in jobs:
        started = time.perf_counter()
        try:
            report = solve_parcel(setup, temperature, pressure, b)
            report["stage"] = stage
            if temperature == 2350.0:
                report["source_comparison"] = shared_reaction_comparison(network, case, report)
            if temperature == source["T_K"] and pressure == source["P_bar"]:
                report["boundary_comparison"] = audit_boundary(network, case, source, setup, report)
            record["parcels"].append(report)
            if not report["accepted"]:
                record["failures"].append({"stage": stage, "T_K": temperature, "P_bar": pressure,
                                           "reason": "Independent local acceptance failed; see parcel residuals."})
        except (ValueError, RuntimeError, np.linalg.LinAlgError) as error:
            record["failures"].append({"stage": stage, "T_K": temperature, "P_bar": pressure,
                                       "reason": str(error), "elapsed_seconds": time.perf_counter() - started})
    return record


def main() -> None:
    """Write a new diagnostic record without overwriting a previous run."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pressure-bar", type=float, required=True)
    parser.add_argument("--oxygen-factor", type=float, default=1.0)
    parser.add_argument("--point", type=float, nargs=2, action="append", default=[], metavar=("T_K", "P_BAR"))
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error("Output already exists; choose a new record path.")
    record = run_diagnostic(args.pressure_bar, oxygen_factor=args.oxygen_factor, points=args.point)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x", encoding="utf-8") as stream:
        json.dump(record, stream, indent=2, allow_nan=False)
        stream.write("\n")
    print(json.dumps({"output": str(args.output), "parcels": len(record["parcels"]), "failures": record["failures"]}))
    if record["failures"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
