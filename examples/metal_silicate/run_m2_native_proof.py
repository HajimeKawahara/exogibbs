"""Fresh independent common-plane proof for the native M2 wheel forward path.

This separate example may use pinned provider examples and the native binary.
It consumes primitive amounts and material coefficients, never saved statuses.
"""

import argparse
from decimal import Decimal
from fractions import Fraction
import hashlib
import importlib.metadata
import importlib.util
import json
from pathlib import Path
import subprocess
import time

import numpy as np

from exoeos import associated_metal, magma_volatiles
from m2_associated_global import _scalar_gradient
from m2_common_plane import (_I, _dot, _exp, feasible_primal, ideal_energy, ideal_minimum,
                             interval_json, primal_dual_certificate, rational_solve, solution_common_plane)
from m2_sodium_global import certify_sodium_insertion, parent_domain
from m2_solid_global import certify_solid_insertion
from m2_water_global import certify_water_common_plane, water_insertion_value
from melts_coupled import load_melts_evaluator
from run_m2_common_plane import PURE_PHASES

R = 8.31446261815324


def _module(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def assess(request, eos, runtime, python, output, *, max_nodes=100000):
    """Recompute all phase bounds and an exactly feasible primal energy."""
    if request["schema"] != "m2_native_proof_request_v1":
        raise ValueError("Unsupported primitive source request.")
    output.mkdir(parents=True, exist_ok=False)
    def write(name, value):
        (output/name).write_text(json.dumps(value, indent=2, allow_nan=False)+"\n")
    t, p = request["temperature_K"], request["pressure_Pa"]
    elements, names = request["elements"], request["component_order"]
    atoms = np.asarray(request["atom_matrix"])
    budget, dual = request["element_amounts_mol"], request["element_potentials_rt"]
    plane = dict(zip(elements, dual))
    phase_indices = {phase: [names.index(n) for n in selected] for phase, selected in request["phases"].items()}
    liquid, metal, atmosphere = request["liquid"], request["metal"], request["atmosphere"]
    if (liquid["provider_model_id"] != magma_volatiles.MODEL_ID
            or metal["provider_model_id"] != associated_metal.MODEL_ID):
        raise ValueError("Unsupported material scalar.")
    water = liquid["parameters"]
    rows = []
    mi, li = phase_indices["metal"], phase_indices["silicate"]
    for i, (lo, hi) in enumerate(zip(metal["lower"], metal["upper"])):
        for sign, bound in ((1, lo), (-1, hi)):
            coefficients = np.zeros(len(names)); coefficients[mi] = -sign*bound
            coefficients[mi[i]] += sign
            rows.append(dict(name=f"metal_{i}_{sign}", coefficients=coefficients.tolist()))
    coefficients = np.zeros(len(names)); coefficients[li[:11]] = atoms[elements.index("O"), li[:11]]
    coefficients[li[11]] = -1
    rows.append(dict(name="water_capacity", coefficients=coefficients.tolist()))
    primal, repair = feasible_primal(atoms.T.tolist(), request["component_amounts_mol"], budget, composition_rows=rows)
    write("exact_primal.json", dict(amounts=[str(v) for v in primal], repair=repair))
    costs = lambda standard, indices: [_I(s)-_dot(atoms[:, i].tolist(), dual) for s, i in zip(standard, indices)]
    parameters = dict(dry_standard_costs_rt=costs(liquid["dry_standards_rt"], li[:11]),
        water_gas_cost_rt=_I(liquid["gas_standards_rt"][0])-2*_I(plane["H"])-_I(plane["O"]),
        dissolved_h2_cost_rt=_I(liquid["gas_standards_rt"][1])+_I(11.403)+_I(.76)*_I(p)/_I(1e9)
                                +_I(liquid["h2_offset_rt"])-2*_I(plane["H"]),
        log_capacity_prefactor=water["log_capacity_prefactor"], temperature_K=t,
        predictor_oxide_counts=water["predictor_oxide_counts"], predictor_temperature_weights_K=water["predictor_temperature_weights_K"],
        dry_masses_kg_mol=water["dry_molar_masses_kg_mol"], water_mass_kg_mol=water["water_molar_mass_kg_mol"],
        dry_oxygen_counts=atoms[elements.index("O"), li[:11]].tolist(), entropy_coefficient=water["entropy_coefficient"],
        quadratic_matrix_rt=(np.asarray(water["quadratic_matrix_J_mol"])/(R*t)).tolist())
    # Independent interval arithmetic eliminates all nonnegative dissolved He.
    he_cost = _I(liquid["gas_standards_rt"][2])-_I(plane["He"])
    expected_mass = np.asarray(request["atomic_masses_kg_mol"]) @ atoms[:, li[:11]]
    if not np.array_equal(expected_mass, liquid["helium_dry_masses_kg_mol"]):
        raise ValueError("The He host masses differ from the primitive elemental ledger.")
    rate = _I(liquid["helium_capacity"])*_exp(-he_cost)
    eliminated = {**parameters, "dry_standard_costs_rt": [c-_I(m)*rate for c, m in
                   zip(parameters["dry_standard_costs_rt"], liquid["helium_dry_masses_kg_mol"])]}
    n_liquid = [primal[i] for i in li]
    liquid_proof = certify_water_common_plane(eliminated, list(map(float, n_liquid[:11])),
                                              tolerance_rt=1e-10, max_nodes=max_nodes)
    write("liquid.json", liquid_proof)
    bounds = [dict(phase="liquid_H2O_H2_He", bound=_I(Decimal(liquid_proof["lower_bound_rt_per_dry_component"])),
                   minimum_atoms=_I(float(np.min(atoms[:, li[:11]].sum(axis=0)))))]
    insertion = water_insertion_value(parameters, n_liquid[:11], n_liquid[11], n_liquid[12])
    he = _I(n_liquid[13]); dry_mass = _dot(liquid["helium_dry_masses_kg_mol"], n_liquid[:11])
    if n_liquid[13]:
        insertion += he*(he_cost+(he/(dry_mass*_I(liquid["helium_capacity"]))).log()-1)
    print("liquid", liquid_proof["lower_bound_rt_per_dry_component"], flush=True)

    provider = _module(eos/"examples/m2_material/sodium_metal.py", "native_proof_sodium")
    model, _ = provider.make_associated_model(t, temperature_policy=metal["temperature_policy"],
                                             hydrogen_oxygen_model=metal["hydrogen_oxygen_model"])
    parent = provider._potassium(); host = model.host.host
    dry = np.asarray(host.host.host.dry_model.interaction_K).tolist()
    epsilon, matrix = np.asarray(host.host.epsilon), np.asarray(host.additional_matrix)
    if not np.array_equal(epsilon, metal["phosphorus_epsilon"]) or not np.array_equal(matrix, metal["additional_matrix"]):
        raise ValueError("The native and independently declared alloy coefficients differ.")
    excess = lambda xs: parent.associated_excess(t, dry, epsilon.tolist(), matrix.tolist(), xs)
    lo, hi = parent_domain(metal["lower"], metal["upper"])
    kappa = parent.associated_curvature_lower_bound(model.host, t, lo, hi)
    if kappa <= 0 or matrix[2, 3] != 0:
        raise ValueError("This native proof requires the convex H--O-free reference.")
    alloy_costs = costs(metal["standards_rt"], mi)
    alloy = certify_sodium_insertion(excess, alloy_costs, metal["lower"], metal["upper"], kappa,
                                     [_I(0)]*18, tolerance=1e-10, max_nodes=max_nodes)
    write("alloy.json", alloy)
    bounds.append(dict(phase="associated20", bound=_I(Decimal(alloy["lower_bound_rt"])), minimum_atoms=_I(1)))
    n = [primal[i] for i in mi]; total = sum(n)
    value, _ = _scalar_gradient(lambda xs: excess(xs[:-1]), alloy_costs, [v/total for v in n[1:]], intervals=True)
    insertion += _I(total)*value
    print("alloy", alloy["lower_bound_rt"], flush=True)

    gi, ci = phase_indices["gas"], phase_indices["cloud"]
    gas_cost = [v+_I(p/1e5).log() for v in costs(atmosphere["gas_standard_potentials_rt"], gi)]
    bounds.append(dict(phase="ideal_gas", bound=ideal_minimum(gas_cost), minimum_atoms=_I(1)))
    ng = [primal[i] for i in gi]
    insertion += _dot(gas_cost, ng)+ideal_energy(ng)
    for index, standard, eligible in zip(ci, atmosphere["condensate_standard_potentials_rt"], atmosphere["condensate_temperature_eligible"]):
        if eligible:
            cost = costs([standard], [index])[0]
            bounds.append(dict(phase=names[index], bound=cost, minimum_atoms=_I(float(atoms[:, index].sum()))))
            insertion += cost*_I(primal[index])
        elif primal[index] != 0:
            raise ValueError("An unavailable cloud has a nonzero amount.")

    # The retained native competitor catalog is evaluated afresh at this root.
    native = load_melts_evaluator(eos)
    gas_water = liquid["gas_standards_rt"][0]
    evaluator = load_melts_evaluator(eos, liquid_model="published_water", runtime=runtime,
                                    python_executable=python, gas_water_standard_rt=lambda t, p: gas_water)
    native_n = np.zeros(len(native.COMPONENTS))
    scale = 1/float(sum(n_liquid[:12]))
    for name, value in zip(request["phases"]["silicate"][:12], n_liquid[:12]):
        native_n[native.COMPONENTS.index(name[:-6])] = float(value)*scale
    h2 = float(n_liquid[12])*scale
    kwargs = dict(runtime=runtime, python_executable=python, common_R=R)
    properties = evaluator.evaluate_liquid(t, p, native_n, **kwargs)
    solid_provider = _module(eos/"examples/melts_solid_mixing.py", "native_proof_solids")
    models = json.loads(solid_provider.PARAMETER_PATH.read_text())["models"]
    native_properties = native.evaluate_liquid(t, p, native_n, include_saturation=True,
                                               candidate_standard_states=list(models), **kwargs)
    write("host_properties.json", properties)
    write("native_competitors.json", native_properties)
    for standard in native_properties["candidate_standard_states"]:
        phase = standard["phase"]
        params = solid_provider.solid_mixing_parameters(phase, t, p)
        proof = certify_solid_insertion(params, standard, properties, h2, max_nodes=max_nodes)
        write(phase+".json", proof)
        bound, minimum_atoms, _ = solution_common_plane(proof, properties, h2, plane)
        bounds.append(dict(phase=phase, bound=bound, minimum_atoms=minimum_atoms))
        print(phase, str(bound.lo), flush=True)
    if len(native_properties["candidate_standard_states"]) != 20:
        raise ValueError("Require all twenty native solution competitors.")
    oxide_matrix = list(map(list, zip(*properties["basis"]["component_oxide_matrix"])))
    masses = list(map(Fraction, properties["oxide_molar_masses_g_mol"]))
    seen = set()
    for candidate in native_properties["saturation"]["candidates"]:
        phase = candidate["phase"]
        if phase in models:
            continue
        if phase not in PURE_PHASES or candidate["status"] != "ok_candidate_properties":
            raise ValueError("Unexpected or unavailable native pure competitor.")
        weights = rational_solve(oxide_matrix, [Fraction(v)/m for v, m in zip(candidate["oxide_mass_g"], masses)])
        columns = properties["basis"]["component_element_matrix"]
        atoms_pure = [sum(c*Fraction(col[j]) for c, col in zip(weights, columns)) for j in range(len(columns[0]))]
        if any(v < 0 for v in atoms_pure):
            raise ValueError("A pure competitor has an invalid atom ledger.")
        bound = _I(candidate["gibbs_J"])/(_I(R)*_I(t))-_dot(atoms_pure, [plane.get(e, 0.) for e in properties["basis"]["element_order"]])
        bounds.append(dict(phase="native_pure:"+phase, bound=bound, minimum_atoms=_I(sum(atoms_pure))))
        seen.add(phase)
    if seen != PURE_PHASES:
        raise ValueError("Require all thirteen native pure competitors.")
    energy = _dot(budget, dual)+insertion
    difference = (energy-_I(request["gibbs_rt"]))/sum(map(Fraction, budget))
    if max(abs(difference.lo), abs(difference.hi)) > Decimal("1e-9"):
        raise ValueError("Independent primitive energy differs from the executed native scalar.")
    certificate = primal_dual_certificate(budget, dual, energy, bounds)
    report = dict(schema="m2_native_common_plane_v1", certificate=certificate,
                  phase_count=len(bounds), phase_bounds=[{**row, "bound": interval_json(row["bound"]),
                  "minimum_atoms": interval_json(row["minimum_atoms"])} for row in bounds],
                  primitive_gibbs_difference_rt_per_atom=interval_json(difference),
                  scope="Declared scalar and complete competitor catalog; empirical material accuracy remains unestablished.")
    write("assessment.json", report)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("request", "exoeos-checkout", "runtime", "output"):
        parser.add_argument("--"+name, type=Path, required=True)
    parser.add_argument("--python", required=True)
    args = parser.parse_args()
    started = time.perf_counter()
    report = assess(json.loads(args.request.read_text()), args.exoeos_checkout, args.runtime, args.python, args.output)
    proof_root = Path(__file__).parent
    proof_files = list(proof_root.glob("*global.py"))+[
        Path(__file__), proof_root/"m2_common_plane.py", proof_root/"run_m2_common_plane.py",
        proof_root/"melts_coupled.py"]
    protocol = dict(request_sha256=hashlib.sha256(args.request.read_bytes()).hexdigest(), elapsed_seconds=time.perf_counter()-started,
        installed_versions={name: importlib.metadata.version(name) for name in ("exoeos", "exogibbs", "jax", "numpy", "scipy")},
        provider_commits={name: subprocess.check_output(["git", "-C", str(root), "rev-parse", "HEAD"], text=True).strip()
                          for name, root in (("exoeos", args.exoeos_checkout), ("exogibbs", Path(__file__).parents[2]))},
        proof_code_sha256={p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(set(proof_files))})
    (args.output/"protocol.json").write_text(json.dumps(protocol, indent=2)+"\n")
    if not report["certificate"]["declared_model_gap_accepted"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
