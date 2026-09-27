"""Bound the full declared finite-source Gibbs gap from saved phase proofs.

No native evaluation, equilibrium solve, or archived status is modified.
"""
import argparse
from decimal import Decimal
from fractions import Fraction
import hashlib
import json
from pathlib import Path
import subprocess
from types import SimpleNamespace

import numpy as np

from hydrogen import dissolved_h2_standard_rt, hirschmann2012_ln_solubility
from m2_common_gas import anchored_standards_rt, build_common_gas_setup
from m2_common_plane import (_I, _dot, feasible_primal, ideal_energy, ideal_minimum,
                             interval_json, liquid_common_plane, liquid_mixing, liquid_standard_intervals,
                             primal_dual_certificate, rational_solve, solution_common_plane)
from m2_finite_gas import atmosphere_gauge_rt, build_atmosphere_setup
from m2_liquid_global import require_saved_liquid_expression
from run_bse_common_gibbs import source_standards_rt
from run_m2_alloy_insertion_bound import saved_inputs, sha256, verified_recipe


PURE_PHASES = frozenset(("sphene", "aenigmatite", "muscovite", "quartz", "tridymite",
                         "cristobalite", "corundum", "sillimanite", "rutile", "perovskite",
                         "whitlockite", "apatite", "water"))


def saved_solution_provider(summary: dict, eos_checkout: Path) -> tuple:
    """Load the exact immutable EOS declaration from its recorded git blobs."""
    revision = summary["exoeos_commit"]
    blobs = {}
    for relative in ("examples/melts_solid_mixing.py", "examples/m2_solid_mixing/parameters.json"):
        raw = subprocess.check_output(["git", "-C", str(eos_checkout), "show", revision+":"+relative])
        if hashlib.sha256(raw).hexdigest() != summary["provider_hashes"][relative]:
            raise ValueError("The recorded EOS solution declaration hash is inconsistent.")
        blobs[relative] = raw
    namespace = {"__file__": str(eos_checkout/"examples/melts_solid_mixing.py")}
    exec(compile(blobs["examples/melts_solid_mixing.py"], namespace["__file__"], "exec"), namespace)
    data = blobs["examples/m2_solid_mixing/parameters.json"]
    namespace["PARAMETER_PATH"] = SimpleNamespace(read_bytes=lambda: data, read_text=lambda: data.decode())
    receipt = {"git_commit": revision, "checkout": str(eos_checkout),
               "blob_sha256": {name: hashlib.sha256(raw).hexdigest() for name, raw in blobs.items()}}
    return namespace["solid_mixing_parameters"], set(json.loads(data)["models"]), receipt


def require_solution_proof(proof: dict, row: dict, parameters: dict, standard_states: dict) -> None:
    """Bind the full domain/expression and native standards, not just a label."""
    if proof["parameters"] != parameters or proof["native_standard_states"] != standard_states:
        raise ValueError("The solution bound changed its declared expression, domain or native standards.")
    if proof["phase"] != parameters["phase"] or any(proof[key] != row[key] for key in (
            "lower_bound_rt_per_formula_unit", "unresolved_box_count", "formal_global_insertion_bound_accepted")):
        raise ValueError("The solution proof differs from its bound summary.")


def assess(binding_path: Path, alloy_path: Path, eos_checkout: Path, *, case=None, water_path=None,
           alloy_tolerance_rt=1e-10, alloy_max_nodes=20000) -> dict:
    here = Path(__file__).resolve().parent
    code_names = ("run_m2_common_plane.py", "m2_common_plane.py", "m2_liquid_global.py",
                  "m2_extended_common_plane.py", "m2_extended_alloy.py", "m2_associated_global.py",
                  "m2_water_global.py", "run_m2_water_global.py", "m2_host_standards.py",
                  "run_m2_alloy_insertion_bound.py", "m2_helium_global.py", "m2_helium.py")
    files = {str(here/name): sha256(here/name) for name in code_names}
    files[str(eos_checkout/"src/exoeos/ma_interval.py")] = sha256(eos_checkout/"src/exoeos/ma_interval.py")

    def read(path, expected=None):
        path = Path(path).resolve()
        digest = sha256(path)
        if expected is not None and digest != expected:
            raise ValueError("Changed bound input: "+str(path))
        files[str(path)] = digest
        return json.loads(path.read_text())

    binding = read(binding_path)
    if binding.get("schema") == "m2_five_root_evidence_aggregate_v1":
        if binding.get("status") != "all_five_artifacts_verified" or not binding.get("all_five_cases_complete"):
            raise ValueError("Require a verified complete aggregate.")
        matches = [row for row in binding["cases"] if row["case"] == case]
        if len(matches) != 1:
            raise ValueError("Select exactly one recorded aggregate case.")
        row = matches[0]
    else:
        if not binding.get("binding_verified"):
            raise ValueError("Require an exact final-root evidence binding.")
        row = binding["row"]
        if case is not None and case != row["case"]:
            raise ValueError("The case selector does not match the binding.")
    read(row["source"]["path"], row["source"]["sha256"])
    read(row["physical_audit"]["path"], row["physical_audit"]["sha256"])
    report, audit, state, scenario, _, _ = saved_inputs(Path(row["source"]["path"]), Path(row["physical_audit"]["path"]), allow_extended=True)
    source = state["source"]
    recipe_receipt = {}
    files.update(verified_recipe(source, eos_checkout,allow_historical_builders=True,receipt=recipe_receipt))
    elements = source["source_internal_record"]["elements"]
    budget = list(map(Fraction, source["source_metadata"]["input"]["element_amounts_mol"]))
    if elements != report["inventory"]["elements"] or list(map(Fraction, report["inventory"]["total_element_amounts_mol"])) != budget:
        raise ValueError("The exact source budget must equal the recorded finite inventory.")
    potentials = source["source_internal_result"]["elemental_potentials_rt"]
    plane = dict(zip(elements, potentials))
    temperature, pressure = source["temperature_K"], source["pressure_bar"]
    host = audit["host_stability"]
    properties = host["provider_properties"]
    is_water = source["source_metadata"].get("liquid_model") == "published_water"
    if 'helium_dissolution' in source['source_metadata'] and not is_water:
        raise ValueError('The common-plane He extension currently requires reconstructed water.')
    expression = properties.get("mixing_expression")
    if is_water != (water_path is not None):
        raise ValueError("The reconstructed-water source requires its explicit matching water proof.")
    phase_bounds = []

    def bound(name, value, atoms, **metadata):
        phase_bounds.append({"phase": name, "bound": _I(value), "minimum_atoms": _I(atoms), **metadata})

    proofs = row["declared_expression_phase_evidence"]["proofs"]
    proof_dirs = {}
    for kind in (("solids",) if is_water else ("solids", "liquid")):
        entry = proofs[kind]["file"]
        summary = read(entry["path"], entry["sha256"])
        proof_dirs[kind] = Path(entry["path"]).parent
        old_audit = read(proof_dirs[kind]/"source_physical_audit.json", summary["source_sha256"])
        old_host = old_audit["host_stability"]
        # This also permits explicitly saved M1 proof reuse, solely when all
        # mathematical inputs coincide, independently of root filenames.
        if ({k: v for k, v in old_host["provider_properties"].items() if k != "provenance"}
                != {k: v for k, v in properties.items() if k != "provenance"}
                or old_host["native_dissolved_h2_moles"] != host["native_dissolved_h2_moles"]):
            raise ValueError("Saved proof inputs differ from the actual selected host.")
    liquid = read(water_path) if is_water else read(proof_dirs["liquid"]/"assessment.json")
    solid_summary = read(proof_dirs["solids"]/"summary.json")
    solid_host = read(proof_dirs["solids"]/"host_properties.json")
    require_saved_liquid_expression(properties, solid_host)
    if (properties["component_moles"] != solid_host["component_moles"]
            or properties["mu_RT"] != solid_host["mu_RT"]):
        raise ValueError("The solution proof tangent data must exactly match the saved host.")
    reference, _ = source_standards_rt(temperature, pressure)
    setup = build_atmosphere_setup(source["gas_model"])
    parcel = source["source_atmosphere_parcel"]
    gauge = atmosphere_gauge_rt(setup, temperature, reference)
    gas, cloud = setup.gas_setup, setup.condensate_setup
    ag, ac = np.asarray(gas.formula_matrix), np.asarray(cloud.formula_matrix)
    hg, hc = np.asarray(gas.hvector_func(temperature)), np.asarray(cloud.hvector_func(temperature))
    if (list(gas.species) != parcel["gas_species"] or list(cloud.species) != parcel["condensate_species"]
            or list(setup.elements) != parcel["elements"]
            or not np.array_equal(gauge, parcel["element_gauge_rt"])
            or not np.array_equal(hg+ag.T@gauge, parcel["gas_standard_potentials_rt"])
            or not np.array_equal(hc+ac.T@gauge, parcel["condensate_standard_potentials_rt"])):
        raise ValueError("The unchanged source gas recipe must reproduce all saved standard arrays exactly.")
    common = build_common_gas_setup(expanded=True)
    h2_gas = anchored_standards_rt(common, temperature, reference)[0][common.species.index("H2")]
    h2_base = float(dissolved_h2_standard_rt(h2_gas, hirschmann2012_ln_solubility(pressure)))
    offsets = scenario["standard_offsets_rt"]
    allowed_shifts = {"H2_dissolved", "Fe_metal", "Si_metal", "O_metal", "H_metal"}
    if is_water or not offsets.get("h2o_melts", 0.):
        allowed_shifts.add("h2o_melts")
    if set(offsets)-allowed_shifts:
        raise ValueError("Unsupported extra standard shifts in the saved recipe.")
    h2_standard = _I(h2_base)+_I(offsets.get("H2_dissolved", 0.))
    if is_water:
        from run_m2_water_global import saved_water_problem
        from m2_extended_common_plane import water_common_plane, water_primal_energy
        water_parameters, _, water_binding = saved_water_problem(report, audit, row["source"]["sha256"],exoeos_checkout=eos_checkout)
        if 'helium_elimination' in water_binding:
            files.update({str(eos_checkout/name):digest for name,digest in
                          water_binding['helium_elimination']['receipt']['provider_recipe_file_sha256'].items()})
        if (liquid["input_and_code_sha256"].get(str(Path(row["source"]["path"]).resolve())) != row["source"]["sha256"]
                or liquid["input_and_code_sha256"].get(str(Path(row["physical_audit"]["path"]).resolve())) != row["physical_audit"]["sha256"]):
            raise ValueError("The water proof is not bound to these exact root/audit bytes.")
        lo, atoms, details = water_common_plane(water_parameters, water_binding, liquid)
    else:
        lo, atoms, details = liquid_common_plane(properties, liquid, plane, h2_standard)
    bound("H2_augmented_declared_liquid", lo, atoms, **details)
    solution_parameters, expected_solutions, solution_provider_receipt = saved_solution_provider(solid_summary, eos_checkout)
    native_standards = read(proof_dirs["solids"]/"native_standard_state_receipt.json")
    standards_by_phase = {item["phase"]: item for item in native_standards["candidate_standard_states"]}
    solid_names = []
    for phase in solid_summary["phases"]:
        name = phase["phase"]
        proof = read(proof_dirs["solids"]/(name+".json"))
        if proof["phase"] != name or proof["parameters"]["T_K"] != temperature or proof["parameters"]["P_Pa"] != properties["P_Pa"]:
            raise ValueError("A solution bound has a different phase or state.")
        require_solution_proof(proof, phase, solution_parameters(name, temperature, properties["P_Pa"]), standards_by_phase[name])
        lo, atoms, details = solution_common_plane(proof, solid_host, host["native_dissolved_h2_moles"], plane)
        bound(name, lo, atoms, **details)
        solid_names.append(name)
    if len(solid_names) != 20 or set(solid_names) != expected_solutions:
        raise ValueError("The complete twenty-model solution catalog is required.")
    rt = _I(properties["basis"]["common_R_J_mol_K"])*_I(temperature)
    host_columns = properties["basis"]["component_element_matrix"]
    host_elements = properties["basis"]["element_order"]
    oxide_matrix = list(map(list, zip(*properties["basis"]["component_oxide_matrix"])))
    masses = list(map(Fraction, properties["oxide_molar_masses_g_mol"]))
    native = host["native_candidate_properties"]
    pure_names = []
    for candidate in native["saturation"]["candidates"]:
        name = candidate["phase"]
        if name in solid_names:
            continue
        if name not in PURE_PHASES or candidate["status"] != "ok_candidate_properties":
            raise ValueError("The native candidate catalog has an unsupported pure phase.")
        coefficients = rational_solve(oxide_matrix, [Fraction(v)/m for v, m in zip(candidate["oxide_mass_g"], masses)])
        atom_column = [sum(c*Fraction(col[j]) for c, col in zip(coefficients, host_columns)) for j in range(len(host_elements))]
        if any(v < 0 for v in atom_column) or any(v and e not in elements for e, v in zip(host_elements, atom_column)):
            raise ValueError("The supplied pure-phase amount requires an unsupported element.")
        cost = _I(candidate["gibbs_J"])/rt-sum((_I(v)*_I(plane.get(e, 0.)) for e, v in zip(host_elements, atom_column)), _I(0))
        bound("native_pure:"+name, cost, sum(atom_column), amount_unit="one recorded native oxide-mass/Gibbs unit; linear scaling only")
        pure_names.append(name)
    if set(pure_names) != PURE_PHASES:
        raise ValueError("All thirteen remaining native pure candidates are required.")
    extended_alloy = source["source_metadata"].get("metal_model", "ma") != "ma"
    if extended_alloy:
        if alloy_path is not None:
            raise ValueError("Extended alloys require a fresh declared-expression replay, not a four-component bound.")
        from m2_extended_alloy import load_saved_alloy, certify_saved_alloy
        from m2_associated_global import alloy_energy_interval
        alloy_context = load_saved_alloy(source, eos_checkout)
        files.update({str(eos_checkout/name): digest for name,digest
                      in alloy_context["row"]["provider_recipe_file_sha256"].items()})
        alloy = certify_saved_alloy(source, alloy_context,tolerance=alloy_tolerance_rt,max_nodes=alloy_max_nodes)
        alloy_components = alloy["component_order"]
        alloy_plane = [sum((_I(float(col.get(e, 0.)))*_I(float(plane[e])) for e in elements),_I(0))
                       for col in alloy["component_formulas"]]
        replayed_bound = Decimal(alloy["lower_bound_rt"])
        bound("source:"+alloy["model_id"], replayed_bound,
              min(sum(col.values()) for col in alloy["component_formulas"]),
              amount_unit="mole of declared chemical components",
              original_strict_nonnegative_bound=replayed_bound >= 0)
    else:
        if alloy_path is None:
            raise ValueError("The four-component source requires its original saved alloy bound.")
        alloy = read(alloy_path)
        if (alloy["source"]["sha256"] != row["source"]["sha256"]
                or alloy["physical_audit"]["sha256"] != row["physical_audit"]["sha256"]
                or alloy["supporting_potentials_rt"] != [plane[e] for e in alloy["component_order"]]
                or not alloy["completed"] or alloy["reduced_curvature_lower_bound_rt"] <= 0):
            raise ValueError("The alloy bound must use this exact source, common plane and full box.")
        from exoeos.ma_fe_si_o_h import MaFeSiOHLiquid
        from exoeos.ma_interval import ma_alloy_insertion_lower_bound
        model = MaFeSiOHLiquid()
        alloy_names = [element+"_metal" for element in alloy["component_order"]]
        standards = (np.array([reference[name] for name in alloy_names])
                     + np.asarray(model.standard_state_shift_RT(temperature)))
        alloy_offsets = [offsets.get(name, 0.) for name in alloy_names]
        if (standards.tolist() != alloy["base_standard_potentials_rt"]
                or alloy_offsets != alloy["linear_standard_offsets_rt"]
                or alloy["lower_atomic_fractions"] != scenario["metal_bounds"]["lower"]
                or alloy["upper_atomic_fractions"] != scenario["metal_bounds"]["upper"]
                or alloy["interaction_K"] != np.asarray(model.dry_model.interaction_K).tolist()
                or alloy["temperature_K"] != temperature or alloy["pressure_bar"] != pressure):
            raise ValueError("The alloy bound changed the actual source model or domain.")
        replayed_bound = ma_alloy_insertion_lower_bound(
            model, temperature, np.array(alloy["lower_atomic_fractions"]), np.array(alloy["upper_atomic_fractions"]),
            np.array(alloy["evaluation_point"]["solute_fractions"]), standards,
            np.array(alloy["supporting_potentials_rt"]), standard_offsets_rt=np.array(alloy_offsets))
        if replayed_bound != alloy["insertion_lower_bound_rt_per_mol_atoms"]:
            raise ValueError("The archived alloy interval bound was not reproduced exactly.")
        bound("source_Fe_Si_O_H_alloy", alloy["insertion_lower_bound_rt_per_mol_atoms"], 1,
              original_strict_nonnegative_bound=alloy["strict_nonnegative_bound"])
        alloy_components = alloy["component_order"]
        alloy_plane = list(map(_I, alloy["supporting_potentials_rt"]))
    atom_columns = lambda matrix: [[Fraction(float(matrix[setup.elements.index(e), i])) if e in setup.elements else Fraction(0)
                                   for e in elements] for i in range(matrix.shape[1])]
    gas_columns, cloud_columns = atom_columns(ag), atom_columns(ac)
    gauge_by_element = dict(zip(setup.elements, map(float, gauge)))
    gas_standard = [_I(float(v))+sum((_I(gauge_by_element.get(e, 0.))*_I(c) for e, c in zip(elements, col)), _I(0))
                    +_I(float(np.log(pressure))) for v, col in zip(hg, gas_columns)]
    cloud_standard = [_I(float(v))+sum((_I(gauge_by_element.get(e, 0.))*_I(c) for e, c in zip(elements, col)), _I(0))
                      for v, col in zip(hc, cloud_columns)]
    gas_costs = [st-_dot(col, potentials) for st, col in zip(gas_standard, gas_columns)]
    bound("ideal_atmosphere", ideal_minimum(gas_costs), min(map(sum, gas_columns)), species=len(gas_costs))
    eligible = [temperature <= limit for limit in cloud.temperature_validity_upper]
    if eligible != parcel["condensate_temperature_eligible"]:
        raise ValueError("The condensate validity catalog has changed.")
    for name, st, col, valid in zip(cloud.species, cloud_standard, cloud_columns, eligible):
        if valid:
            bound("cloud:"+name, st-_dot(col, potentials), sum(col))
    record = source["source_internal_record"]
    names = [name for phase in record["phases"].values() for name in phase]
    ledger = dict(zip(names, source["source_internal_result"]["component_amounts_mol"]))
    lower_names = record["phases"]["silicate"]+record["phases"]["metal"]
    columns = [[Fraction(record["component_formulas"][name].get(e, 0)) for e in elements] for name in lower_names]
    amounts = [Fraction(ledger[name]) for name in lower_names]
    columns += gas_columns+cloud_columns
    amounts += list(map(Fraction, parcel["gas_amounts_mol"]))+list(map(Fraction, parcel["condensate_amounts_mol"]))
    repaired, repair = feasible_primal(columns, amounts, budget)
    repair["basis_names"] = [(lower_names+list(gas.species)+list(cloud.species))[i] for i in repair["basis_indices"]]
    repaired_lower = dict(zip(lower_names, repaired))
    melt_amounts = [repaired_lower.get(name+"_melts", Fraction(0)) for name in properties["component_order"]]
    for name, col in zip(properties["component_order"], host_columns):
        if name+"_melts" in repaired_lower:
            if record["component_formulas"][name+"_melts"] != {e: c for e, c in zip(host_elements, col) if c}:
                raise ValueError("The source and liquid component formulas differ.")
    h2 = repaired_lower["H2_dissolved"]
    if is_water:
        primal_liquid = water_primal_energy(water_parameters, water_binding, properties,
                                           plane, melt_amounts, h2, repaired_lower.get('He_dissolved',0))
    else:
        primal_liquid = liquid_mixing(expression, melt_amounts)
        primal_liquid += sum((_I(n)*mu for n, mu in zip(melt_amounts, liquid_standard_intervals(properties)) if n), _I(0))
        primal_liquid += ideal_energy([sum(melt_amounts), h2])+_I(h2)*h2_standard
    metal = [repaired_lower[e+"_metal"] for e in alloy_components]
    total = sum(metal)
    if extended_alloy:
        primal_alloy = alloy_energy_interval(alloy_context, metal)
    else:
        fraction = [v/total for v in metal]
        if any(x < Fraction(low) or x > Fraction(high) for x, low, high in zip(fraction, alloy["lower_atomic_fractions"], alloy["upper_atomic_fractions"])):
            raise ValueError("The feasible primal leaves the declared alloy box.")
        from exoeos.ma_interval import _ma_excess
        primal_alloy = ideal_energy(metal)+_I(total)*_ma_excess(temperature, alloy["interaction_K"], list(map(_I, fraction[1:])))
        primal_alloy += sum((_I(n)*(_I(st)+_I(off)) for n, st, off in zip(metal, alloy["base_standard_potentials_rt"], alloy["linear_standard_offsets_rt"])), _I(0))
    count = len(lower_names)
    repaired_gas = repaired[count:count+len(gas.species)]
    repaired_cloud = repaired[count+len(gas.species):]
    if any(n and not valid for n, valid in zip(repaired_cloud, eligible)):
        raise ValueError("The feasible primal uses an ineligible condensate.")
    primal_atmosphere = ideal_energy(repaired_gas)+_dot(repaired_gas, gas_standard)+_dot(repaired_cloud, cloud_standard)
    primal = primal_liquid+primal_alloy+primal_atmosphere
    result = primal_dual_certificate(budget, potentials, primal, phase_bounds)
    recorded_energy = source["source_internal_result"]["gibbs_rt"]
    energy_difference = (primal-_I(recorded_energy))/_I(sum(budget))
    energy_error = max(energy_difference.lo.copy_abs(), energy_difference.hi.copy_abs())
    chemistry_error = max(row["numerical_closure"]["local_kkt_max_abs_rt"],
                          *(parcel[key] for key in ("gas_stationarity_max_abs", "present_condensate_residual_max_abs",
                                                     "absent_condensate_violation_max_abs")))
    numerical_prerequisites = (row["numerical_closure"]["source_relative_element_residual"] <= 1e-9
                               and chemistry_error <= 1e-8 and energy_error <= Decimal.from_float(1e-9))
    result["numerical_prerequisites"] = {"source_atoms_tolerance": 1e-9, "chemical_tolerance_rt": 1e-8,
                                          "maximum_chemical_residual_rt": chemistry_error,
                                          "independent_primal_vs_saved_energy_error_rt_per_inventory_atom": str(energy_error),
                                          "accepted": numerical_prerequisites}
    alloy_point = (primal_alloy-_dot(metal, alloy_plane))/_I(total)
    alloy_lower = _I(replayed_bound)
    alloy_minimum = _I(alloy_lower.lo, alloy_point.hi)
    search_error = alloy_point-alloy_lower
    complementarity = _I(total/sum(budget))*_I(max(alloy_minimum.lo.copy_abs(), alloy_minimum.hi.copy_abs()))
    metal_condition = (alloy_minimum.lo >= Decimal.from_float(1e-8).copy_negate()
                       and alloy_minimum.hi <= Decimal.from_float(1e-8)
                       and search_error.hi <= Decimal.from_float(1e-8)
                       and complementarity.hi <= Decimal.from_float(1e-8))
    result["present_alloy_condition"] = {
        "global_minimum_interval_rt_per_mol_components" if extended_alloy else "global_minimum_interval_rt_per_mol_atoms": interval_json(alloy_minimum),
        "composition_search_uncertainty_upper_rt": str(search_error.hi),
        "complementarity_upper_rt_per_inventory_atom": str(complementarity.hi),
        "tolerance_rt": 1e-8, "accepted": metal_condition,
        "scope": "Positive selected alloy only; amount normalized by total inventory atoms. The original plane and negative bounds are retained."}
    result["declared_finite_source_numerically_accepted"] = (
        result["declared_model_gap_accepted"] and numerical_prerequisites and metal_condition)
    result.update(schema="m2_declared_finite_source_common_plane_gap_v1", case=row["case"],
                  source=row["source"], physical_audit=row["physical_audit"],
                  source_recipe_verification=recipe_receipt,
                  temperature_K=temperature, pressure_bar=pressure, exact_primal_repair=repair,
                  solution_provider=solution_provider_receipt,
                  phase_bounds=[{**{k: v for k, v in item.items() if k not in ("bound", "minimum_atoms")},
                                 "global_insertion_lower_bound_rt": str(item["bound"].lo),
                                 "minimum_atoms_per_unit_lower_bound": str(item["minimum_atoms"].lo)} for item in phase_bounds],
                  phase_energy_intervals_rt_mol={"liquid": interval_json(primal_liquid), "alloy": interval_json(primal_alloy),
                                                "atmosphere": interval_json(primal_atmosphere)},
                  h2_standard_recipe={"base_rt": h2_base, "scenario_offset_rt": offsets.get("H2_dissolved", 0.)},
                  excluded_temperature_ineligible_clouds=[name for name, valid in zip(cloud.species, eligible) if not valid],
                  scope="Global Gibbs bounds at this source T/P, finite 13-element budget and declared phase domains. Exact atom repair supplies a feasible upper bound; a uniform shift of the common elemental plane supplies a global lower bound. This is error-bounded model acceptance, not exact KKT, empirical validity, unbounded Fe alloy composition, metal-free boundary certification, native-build equivalence, or a Gibbs minimum of the nonisothermal planet.")
    root = Path(__file__).resolve().parents[2]
    if any(sha256(path) != digest for path, digest in files.items()):
        raise ValueError("An input changed during the assessment.")
    result["input_and_code_sha256"] = files
    result["alloy_interval_bound_replayed_exactly"] = not extended_alloy
    if extended_alloy:
        result["fresh_extended_alloy_bound"] = alloy
    if is_water:
        result["water_proof_bound_to_same_source_and_plane"] = True
    result["proof_commit"] = subprocess.check_output(["git", "-C", str(root), "rev-parse", "HEAD"], text=True).strip()
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--evidence-binding", type=Path, required=True)
    parser.add_argument("--case", help="Required when selecting a case from the five-root aggregate.")
    parser.add_argument("--alloy-bound", type=Path)
    parser.add_argument("--water-proof", type=Path)
    parser.add_argument("--alloy-tolerance-rt", type=float, default=1e-10)
    parser.add_argument("--alloy-max-nodes", type=int, default=20000)
    parser.add_argument("--exoeos-checkout", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error("Use a new output file; original evidence is never overwritten.")
    result = assess(args.evidence_binding, args.alloy_bound, args.exoeos_checkout.resolve(), case=args.case,
                    water_path=args.water_proof,alloy_tolerance_rt=args.alloy_tolerance_rt,alloy_max_nodes=args.alloy_max_nodes)
    with args.output.open("x") as stream:
        json.dump(result, stream, indent=2, allow_nan=False)
        stream.write("\n")
    print(json.dumps({key: result[key] for key in ("case", "declared_model_gap_accepted", "normalized_gap_per_inventory_atom")}))


if __name__ == "__main__":
    main()
