"""Independent outward bounds for the full-catalog second-virial gas.

The receipt's binary64 pair coefficients are exact constants of the declared
fixed-T/P model. Neither the provider's rounded convexity diagnostic nor its
automatic derivatives are used as proof. Zero physical amounts remain zero;
a positive reference point is needed only for the entropy lower bound.
"""
from decimal import Decimal
from fractions import Fraction
import hashlib
from pathlib import Path

import numpy as np

from m2_common_plane import _I, _dot, ideal_energy, ideal_minimum, interval_json


SCHEMA = "m2_full_catalog_major_gas_second_virial_v1"


def _inputs(parameters):
    if parameters.get("schema") != SCHEMA:
        raise ValueError("Require the declared full-catalog second-virial gas model.")
    species = parameters["species"]
    matrix = np.asarray(parameters["coefficients_m3_mol"], dtype=float)
    if (not species or len(set(species)) != len(species)
            or matrix.shape != (len(species), len(species))
            or not np.all(np.isfinite(matrix)) or not np.array_equal(matrix, matrix.T)):
        raise ValueError("Require a finite symmetric pair matrix on the complete species basis.")
    constants = [float(parameters[key]) for key in
                 ("temperature_K", "pressure_Pa", "gas_constant_J_mol_K")]
    if any(not np.isfinite(value) or value <= 0 for value in constants):
        raise ValueError("Gas T, P and R must be finite and strictly positive.")
    temperature, pressure, gas_constant = map(_I, constants)
    ideal_density = pressure/(gas_constant*temperature)
    return matrix, ideal_density


def gas_convexity_certificate(parameters: dict) -> dict:
    """Prove H >= alpha*diag(1/x) on the entire nonnegative simplex.

    For a simplex tangent v, Cauchy gives ||v||_1^2 <= sum(v_i^2/x_i).
    The residual Hessian is 2*rho*B minus a rank-one term with coefficient
    4*rho^2/(1+2*rho*Bmix). Pair extrema bound both terms for every x,
    independently of the saved composition and including its zero faces.
    """
    matrix, ideal_density = _inputs(parameters)
    maximum = _I(float(np.max(np.abs(matrix))))
    minimum = _I(float(np.min(matrix)))
    discriminant = 1+4*ideal_density*minimum
    result = {"schema": "m2_gas_interval_convexity_v1", "accepted": False,
              "arithmetic": "50-digit outward Decimal intervals",
              "scope": "Entire nonnegative full-catalog simplex at the recorded T/P",
              "minimum_pair_coefficient_m3_mol": interval_json(minimum),
              "maximum_absolute_pair_coefficient_m3_mol": interval_json(maximum),
              "global_discriminant": interval_json(discriminant),
              "entropy_curvature_lower_bound": None}
    if discriminant.lo <= 0:
        return {**result, "reason": "No strictly positive global mechanical denominator."}
    denominator = discriminant.sqrt()
    density = 2*ideal_density/(1+denominator)
    alpha = 1-2*density*maximum-4*density*density*maximum*maximum/denominator
    result.update(mechanical_denominator_lower_bound=str(denominator.lo),
                  density_upper_bound_mol_m3=str(density.hi),
                  entropy_curvature_interval=interval_json(alpha),
                  entropy_curvature_lower_bound=str(alpha.lo),
                  accepted=alpha.lo > 0)
    if alpha.lo <= 0:
        result["reason"] = "The global entropy-curvature lower bound is not positive."
    return result


def _residual_state(parameters, fractions):
    matrix, ideal_density = _inputs(parameters)
    x = list(map(Fraction, fractions))
    if len(x) != len(matrix) or any(value < 0 for value in x) or sum(x) != 1:
        raise ValueError("Require exact normalized nonnegative full-catalog fractions.")
    bx = [_dot(row, x) for row in matrix.tolist()]
    bmix = _dot(x, bx)
    discriminant = 1+4*ideal_density*bmix
    if discriminant.lo <= 0:
        raise ValueError("The gas point has no certified stable density root.")
    density = 2*ideal_density/(1+discriminant.sqrt())
    z = 1+density*bmix
    logarithm = z.log()
    return {"residual_gibbs_rt": 2*density*bmix-logarithm,
            "log_fugacity_coefficients": [2*density*value-logarithm for value in bx],
            "density_mol_m3": density, "compressibility_factor": z}


def gas_residual_energy(parameters: dict, amounts: list) -> _I:
    """Enclose G^r/(RT) at the unreduced exact-rational feasible primal."""
    values = list(map(Fraction, amounts))
    if len(values) != len(parameters["species"]) or any(value < 0 for value in values):
        raise ValueError("Require nonnegative gas amounts on the complete species basis.")
    total = sum(values)
    if not total:
        _inputs(parameters)
        return _I(0)
    state = _residual_state(parameters, [value/total for value in values])
    return _I(total)*state["residual_gibbs_rt"]


def gas_insertion_lower_bound(parameters: dict, costs: list, amounts: list,
                              supported: list) -> tuple:
    """Use entropy strong convexity to bound every supported gas composition.

    ``costs`` includes each species' standard, log pressure and the negative
    common elemental plane. Unsupported species have exact zero fractions,
    while their positions remain in the full EOS matrix and amount vector.
    """
    certificate = gas_convexity_certificate(parameters)
    if not certificate["accepted"]:
        raise ValueError("The gas lacks a positive global entropy-curvature proof.")
    size = len(parameters["species"])
    if (len(costs) != size or len(amounts) != size or len(supported) != size
            or any(type(value) is not bool for value in supported) or not any(supported)):
        raise ValueError("Require full-catalog costs, amounts and explicit elemental support.")
    costs = list(map(_I, costs))
    if any(not value.lo.is_finite() or not value.hi.is_finite()
           or value.lo > value.hi for value in costs):
        raise ValueError("Gas costs must be finite ordered intervals.")
    values = list(map(Fraction, amounts))
    if any(value < 0 or (value and not active) for value, active in zip(values, supported)):
        raise ValueError("The gas reference contains a negative or unsupported physical amount.")
    total = sum(values)
    if total <= 0:
        raise ValueError("Require a positive gas reference amount.")
    # This trial regularization is solely a proof reference, never a physical
    # amount floor or a restriction on the certified nonnegative simplex.
    trial = [value if value else total/Fraction(10**100) if active else Fraction(0)
             for value, active in zip(values, supported)]
    normalization = sum(trial)
    x = [value/normalization for value in trial]
    state = _residual_state(parameters, x)
    active = [i for i, value in enumerate(supported) if value]
    energy = ideal_energy(x)+_dot(x, costs)+state["residual_gibbs_rt"]
    chemical = [costs[i]+_I(x[i]).log()+state["log_fugacity_coefficients"][i]
                for i in active]
    alpha = _I(Decimal(certificate["entropy_curvature_lower_bound"]))
    lower = energy-_dot([x[i] for i in active], chemical)
    lower += alpha*ideal_minimum([mu/alpha-_I(x[i]).log()
                                  for i, mu in zip(active, chemical)])
    details = {"convexity": certificate, "supported_species_indices": active,
               "unsupported_species_indices": [i for i in range(size) if not supported[i]],
               "exact_reference_fractions": [str(value) for value in x],
               "reference_regularized_zero_indices": [i for i in active if not values[i]],
               "reference_regularization_changes_primal": False,
               "reference_gibbs_insertion_interval_rt": interval_json(energy),
               "global_insertion_lower_interval_rt": interval_json(lower),
               "entire_nonnegative_supported_simplex": True}
    return lower, details


def gas_element_support(formula_matrix, gas_elements, inventory_elements, budget):
    """Keep exact zero faces for every unavailable elemental inventory."""
    matrix = np.asarray(formula_matrix, dtype=float)
    if (matrix.ndim != 2 or matrix.shape[0] != len(gas_elements)
            or len(inventory_elements) != len(budget)
            or np.any(~np.isfinite(matrix)) or np.any(matrix < 0)):
        raise ValueError("Require the full nonnegative gas formula matrix and atom budget.")
    available = dict(zip(inventory_elements, map(Fraction, budget)))
    if any(value < 0 for value in available.values()):
        raise ValueError("A gas proof requires nonnegative elemental inventories.")
    unavailable = [i for i, element in enumerate(gas_elements) if not available.get(element, 0)]
    return [not any(matrix[j, i] for j in unavailable) for i in range(matrix.shape[1])]


def require_saved_gas_eos(source: dict, eos_checkout: Path):
    """Replay and bind the actual options, species, T/P and provider bytes."""
    from m2_finite_gas import build_atmosphere_setup
    from m2_gas_eos import build_gas_eos

    metadata = source["source_metadata"].get("gas_eos")
    parcel = source["source_atmosphere_parcel"]
    saved = parcel.get("gas_eos")
    options = source.get("gas_eos_options")
    if options is None:
        if metadata is not None or saved is not None:
            raise ValueError("An ideal source cannot carry an undeclared nonideal gas receipt.")
        return None
    if metadata is None or saved is None or metadata != saved:
        raise ValueError("The source and parcel require the identical nonideal gas receipt.")
    if (source["source_metadata"]["atmosphere"]["gas_model"] != source["gas_model"]
            or source["source_internal_record"]["atmosphere_gas_model"] != source["gas_model"]):
        raise ValueError("The gas model differs from the executed source declaration.")
    setup = build_atmosphere_setup(source["gas_model"])
    species = list(setup.gas_species)
    if (species != parcel["gas_species"] or metadata["species"] != species
            or metadata["options"] != options
            or metadata["temperature_K"] != source["temperature_K"]
            or metadata["pressure_Pa"] != source["pressure_bar"]*1e5):
        raise ValueError("The gas receipt differs from the actual source model, options or T/P.")
    for path, digest in metadata["file_sha256"].items():
        if hashlib.sha256(Path(path).read_bytes()).hexdigest() != digest:
            raise ValueError("A pinned gas provider file changed: "+path)
    for name, digest in metadata["provider_recipe_file_sha256"].items():
        if hashlib.sha256((Path(eos_checkout)/name).read_bytes()).hexdigest() != digest:
            raise ValueError("A pinned gas provider recipe changed: "+name)
    model = build_gas_eos(eos_checkout, species, options)
    replayed = model.parameters(source["temperature_K"], source["pressure_bar"]*1e5)
    if replayed != metadata:
        raise ValueError("The complete gas EOS receipt differs from the provider replay.")
    certificate = gas_convexity_certificate(replayed)
    if not certificate["accepted"]:
        raise ValueError("The saved gas domain lacks an independent global convexity proof.")
    if parcel.get("gas_eos_convexity") != certificate:
        raise ValueError("The saved gas convexity receipt differs from its independent replay.")
    return replayed
