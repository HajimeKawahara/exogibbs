"""Retained gas/cloud Gibbs energy in atmospheric atom coordinates
===============================================================

The existing parcel solver owns species equilibrium and condensate selection.
The callback owns no reservoir ledger: each call supplies absolute atom moles,
and ``parcel(T, P, b)`` returns the corresponding primitive species amounts.
"""

from __future__ import annotations

from copy import deepcopy
from functools import lru_cache

import jax
import jax.numpy as jnp
from jax.scipy.special import xlogy
import numpy as np

from exogibbs.api.condensate import (
    CondensateEquilibriumOptions, build_condensate_chemical_setup,
    solve as solve_condensate,
)
from exogibbs.api.gas import EquilibriumOptions, solve as solve_gas
from exogibbs.thermo.models import ChemicalSetup
from exogibbs.utils.elements import element_mass

from common_gibbs import PhaseEvaluationError
from full_potential import PhaseState
from m1_chemistry import audit_parcel
from m2_gas_global import gas_convexity_certificate


def _primitive_gibbs(gas, cloud, gas_standard, cloud_standard, log_pressure):
    """Explicit ideal-gas/pure-cloud scalar, independent of atom potentials."""
    total = jnp.sum(gas)
    return (jnp.dot(gas, gas_standard + log_pressure)
            + jnp.sum(xlogy(gas, gas)) - xlogy(total, total)
            + jnp.dot(cloud, cloud_standard))


_primitive_value_and_grad = jax.jit(jax.value_and_grad(_primitive_gibbs, argnums=(0, 1)))


def _restrict(setup, rows, columns):
    """Remove unsupported species and element rows without changing standards."""
    selected = jnp.asarray(columns)
    validity = setup.temperature_validity_upper
    return ChemicalSetup(
        formula_matrix=jnp.asarray(np.asarray(setup.formula_matrix)[np.ix_(rows, columns)]),
        hvector_func=lambda t: jnp.take(setup.hvector_func(t), selected, axis=-1),
        elements=tuple(setup.elements[i] for i in rows),
        species=tuple(setup.species[i] for i in columns),
        temperature_validity_upper=None if validity is None else tuple(validity[i] for i in columns),
    )


def audit_atmosphere(setup, temperature, pressure, amounts, gas_amounts, condensate_amounts,
                     element_gauge_rt=None, *, gas_eos=None):
    """Audit supplied primitive amounts, without solving or trusting saved fields.

    Exact-zero elements and their containing species stay excluded. The full
    catalog is returned, with unsupported insertion potentials reported as
    unavailable. ``accepted`` describes numerical equilibrium only.
    """
    gas_setup, cloud_setup = setup.gas_setup, setup.condensate_setup
    elements = tuple(setup.elements)
    if tuple(gas_setup.elements) != elements or tuple(cloud_setup.elements) != elements:
        raise ValueError("Atmospheric gas and condensate element orders must agree.")
    b = np.asarray(amounts, dtype=float)
    ng, nc = np.asarray(gas_amounts, dtype=float), np.asarray(condensate_amounts, dtype=float)
    if (b.shape != (len(elements),) or ng.shape != (len(gas_setup.species),)
            or nc.shape != (len(cloud_setup.species),)
            or any(not np.all(np.isfinite(n)) or np.any(n < 0) for n in (b, ng, nc))
            or not np.isfinite(b.sum())):
        raise ValueError("Atmospheric atoms and species amounts must be finite matching nonnegative vectors.")
    if not all(np.isfinite(x) and x > 0 for x in (temperature, pressure)):
        raise ValueError("Atmospheric temperature and pressure must be positive and finite.")
    q = np.zeros_like(b) if element_gauge_rt is None else np.asarray(
        element_gauge_rt(temperature, pressure) if callable(element_gauge_rt) else element_gauge_rt,
        dtype=float,
    )
    if q.shape != b.shape or not np.all(np.isfinite(q)):
        raise ValueError("The atmospheric elemental gauge must be a finite matching vector.")
    ag, ac = np.asarray(gas_setup.formula_matrix), np.asarray(cloud_setup.formula_matrix)
    hg, hc = np.asarray(gas_setup.hvector_func(temperature)), np.asarray(cloud_setup.hvector_func(temperature))
    validity = cloud_setup.temperature_validity_upper
    eligible = np.ones(len(hc), bool) if validity is None else temperature <= np.asarray(validity)
    if not np.all(np.isfinite(hg)) or not np.all(np.isfinite(hc[eligible])):
        raise ValueError("Atmospheric standards are unavailable at the requested temperature.")
    gas_supported = np.all(ag[b == 0] == 0, axis=0)
    cloud_supported = np.all(ac[b == 0] == 0, axis=0)
    if np.any(ng[~gas_supported] != 0) or np.any(nc[~cloud_supported] != 0):
        raise ValueError("Species containing exact-zero elements must have exact-zero amounts.")
    lnphi, residual_energy = np.zeros_like(ng), 0.
    if gas_eos is not None:
        if tuple(gas_eos.species) != tuple(gas_setup.species):
            raise ValueError("The gas EOS must preserve the full atmospheric species order.")
        if ng.sum():
            state = gas_eos.state(temperature, pressure * 1e5, ng / ng.sum())
            lnphi = np.asarray(state.lnphi, dtype=float)
            residual_energy = float(gas_eos.gibbs_residual_rt(temperature, pressure * 1e5, ng))
            if lnphi.shape != ng.shape or not np.all(np.isfinite(lnphi)) or not np.isfinite(residual_energy):
                raise ValueError("The gas EOS returned nonfinite or mismatched thermodynamic values.")
    potential = np.full(len(b), -np.inf)
    insertion = np.full(len(nc), np.nan)
    audit = {"accepted": True, "gas_stationarity_max_abs": 0.,
             "present_condensate_residual_max_abs": 0., "absent_condensate_violation_max_abs": 0.}
    if np.any(b > 0):
        rows, gas_columns, cloud_columns = np.flatnonzero(b > 0), np.flatnonzero(gas_supported), np.flatnonzero(cloud_supported)
        gas, cloud = _restrict(gas_setup, rows, gas_columns), _restrict(cloud_setup, rows, cloud_columns)
        if np.linalg.matrix_rank(np.asarray(gas.formula_matrix)) != len(rows):
            raise ValueError("The supported gas catalog cannot span the atmospheric atoms.")
        if gas_eos is not None:
            # The independent KKT audit uses the EOS at the supplied composition.
            # Its shifted vector is local to this audit; standards stay unchanged.
            gas = ChemicalSetup(
                formula_matrix=gas.formula_matrix,
                hvector_func=lambda t: jnp.asarray(hg[gas_columns] + lnphi[gas_columns]),
                elements=gas.elements, species=gas.species,
                temperature_validity_upper=gas.temperature_validity_upper)
        audit = audit_parcel(
            gas, temperature, pressure, b[rows], ng[gas_columns],
            condensate_setup=cloud if len(cloud_columns) else None,
            condensate_amounts=nc[cloud_columns] if len(cloud_columns) else None,
        )
        chemical = hg[gas_columns] + np.log(ng[gas_columns] / ng.sum()) + np.log(pressure) + lnphi[gas_columns]
        potential[rows] = np.linalg.lstsq(np.asarray(gas.formula_matrix).T, chemical, rcond=None)[0]
        insertion[cloud_columns] = hc[cloud_columns] - ac[np.ix_(rows, cloud_columns)].T @ potential[rows]
    active_gas, active_cloud = ng > 0, nc > 0
    fraction = ng / ng.sum() if ng.sum() else np.zeros_like(ng)
    # This is an independent primitive species sum, not Euler's identity.
    raw_energy = float(ng[active_gas] @ (hg[active_gas] + np.log(fraction[active_gas])
                                        + np.log(pressure)) + nc[active_cloud] @ hc[active_cloud]) + residual_energy
    gas_atoms, cloud_atoms = ag @ ng, ac @ nc
    weights = np.array([element_mass[name] * 1e-3 for name in elements])
    gas_mass, cloud_mass, target_mass = float(weights @ gas_atoms), float(weights @ cloud_atoms), float(weights @ b)
    relative = np.zeros_like(b)
    relative[b > 0] = (gas_atoms + cloud_atoms)[b > 0] / b[b > 0] - 1.
    accepted = bool(audit["accepted"] and np.max(np.abs(relative), initial=0.) < 1e-9
                    and np.all((gas_atoms + cloud_atoms)[b == 0] == 0))
    report = {
        "accepted": accepted, "T_K": float(temperature), "P_bar": float(pressure),
        "elements": list(elements), "gas_species": list(gas_setup.species),
        "condensate_species": list(cloud_setup.species), "element_amounts_mol": b.tolist(),
        "gas_amounts_mol": ng.tolist(), "condensate_amounts_mol": nc.tolist(),
        "gas_element_amounts_mol": gas_atoms.tolist(), "cloud_element_amounts_mol": cloud_atoms.tolist(),
        "relative_element_residual": relative.tolist(),
        "relative_mass_residual": (gas_mass + cloud_mass) / target_mass - 1. if target_mass else 0.,
        **{key: audit[key] for key in ("gas_stationarity_max_abs", "present_condensate_residual_max_abs",
                                     "absent_condensate_violation_max_abs")},
        "condensate_temperature_eligible": eligible.tolist(),
        "condensate_element_supported": cloud_supported.tolist(),
        "condensate_insertion_rt": [float(x) if valid and supported else None
                                    for x, valid, supported in zip(insertion, eligible, cloud_supported)],
        "gas_mass_kg": gas_mass, "cloud_mass_kg": cloud_mass,
        "gas_mass_fraction": gas_mass / (gas_mass + cloud_mass) if target_mass else None,
        "total_element_mol_per_kg": ((gas_atoms + cloud_atoms) / (gas_mass + cloud_mass)).tolist() if target_mass else None,
        "gas_mole_fractions": fraction.tolist(), "partial_pressures_bar": (pressure * fraction).tolist(),
        "mean_gas_molar_mass_kg_per_mol": gas_mass / ng.sum() if ng.sum() else None,
        "gibbs_rt": raw_energy + float(q @ (gas_atoms + cloud_atoms)), "unshifted_gibbs_rt": raw_energy,
        "elemental_potentials_rt": (potential + q).tolist(), "element_gauge_rt": q.tolist(),
        "gas_standard_potentials_rt": (hg + ag.T @ q).tolist(),
        "condensate_standard_potentials_rt": (hc + ac.T @ q).tolist(),
    }
    if gas_eos is not None:
        report.update(gas_eos=gas_eos.parameters(temperature, pressure * 1e5),
                      gas_lnphi=lnphi.tolist(), gas_residual_gibbs_rt=residual_energy)
    return report


def make_atmosphere_phase(setup, element_gauge_rt, *, gas_eos=None):
    """Return a full-potential callback in ``setup.elements`` atom-mol order.

    ``element_gauge_rt`` is a vector or ``q(T_K, P_bar)`` callback. The same
    elemental shift applies to every gas and pure condensate. Exact-zero atom
    budgets remove their containing species before solving; their one-sided
    insertion potentials are -inf. An empty atmosphere has zero energy. No
    atom floor or condensate mixing entropy is introduced. Failed parcels
    raise ``PhaseEvaluationError``. ``parcel`` returns a full-catalog copy.
    An optional ``gas_eos`` supplies residual G and full-catalog fugacity;
    each T/P requires global convexity and each parcel requires a converged
    composition-dependent fugacity iteration with fresh independent KKT.
    """
    gas_setup, cloud_setup = setup.gas_setup, setup.condensate_setup
    elements = tuple(setup.elements)
    ag, ac = np.asarray(gas_setup.formula_matrix), np.asarray(cloud_setup.formula_matrix)
    if gas_eos is not None and tuple(gas_eos.species) != tuple(gas_setup.species):
        raise ValueError("The gas EOS must preserve the full atmospheric species order.")
    residual_value_and_grad = (None if gas_eos is None else jax.jit(jax.value_and_grad(
        lambda t, p, n: gas_eos.gibbs_residual_rt(t, p * 1e5, n), argnums=2)))

    @lru_cache(maxsize=16)
    def convexity(temperature, pressure):
        proof = gas_convexity_certificate(gas_eos.parameters(temperature, pressure * 1e5))
        if not proof["accepted"]:
            raise PhaseEvaluationError("The gas EOS lacks a positive global entropy curvature bound.")
        return proof

    @lru_cache(maxsize=16)
    def support(mask):
        active = np.array(mask, dtype=bool)
        rows = np.flatnonzero(active)
        gas_columns = np.flatnonzero(np.all(ag[~active] == 0, axis=0))
        cloud_columns = np.flatnonzero(np.all(ac[~active] == 0, axis=0))
        gas = _restrict(gas_setup, rows, gas_columns)
        cloud = _restrict(cloud_setup, rows, cloud_columns)
        if np.linalg.matrix_rank(np.asarray(gas.formula_matrix)) != len(rows):
            raise PhaseEvaluationError("The supported gas catalog cannot span the atmospheric atoms.")
        combined = (build_condensate_chemical_setup(gas_setup=gas, condensate_setup=cloud)
                    if len(cloud_columns) else None)
        return rows, gas_columns, cloud_columns, gas, combined

    @lru_cache(maxsize=1)
    def solve(temperature, pressure, atom_tuple):
        b = np.array(atom_tuple)
        ng, nc = np.zeros(len(gas_setup.species)), np.zeros(len(cloud_setup.species))
        iteration_report = {}
        try:
            if gas_eos is not None:
                iteration_report["gas_eos_convexity"] = convexity(temperature, pressure)
            if np.any(b > 0):
                rows, gas_columns, cloud_columns, gas, combined = support(tuple(b > 0))
                scale = float(b.sum())
                reduced_budget = jnp.asarray(b[rows] / scale)
                correction = np.zeros(len(gas_columns))
                for iteration in range(1, 65 if gas_eos is not None else 2):
                    options = ({} if gas_eos is None else {
                        "lnphi_func": lambda t, p, x: jnp.asarray(correction)})
                    if len(cloud_columns):
                        result = solve_condensate(
                            combined, temperature, pressure, reduced_budget,
                            options=CondensateEquilibriumOptions(
                                return_diagnostics=True, rainout=False,
                                full_condensate_budget_relative_tolerance=1e-9,
                            ), **options,
                        )
                        gas_n, cloud_n = np.asarray(result.gas_n), np.asarray(result.condensate_amounts)
                        converged = bool(result.converged)
                    else:
                        result, diagnostics = solve_gas(
                            gas, temperature, pressure, reduced_budget,
                            options=EquilibriumOptions(epsilon_crit=1e-14), return_diagnostics=True,
                            **options,
                        )
                        gas_n, cloud_n = np.asarray(result.n), np.zeros(0)
                        converged = bool(diagnostics["converged"])
                    if not converged:
                        raise PhaseEvaluationError("The atmospheric parcel solver did not converge.")
                    ng[gas_columns], nc[cloud_columns] = scale * gas_n, scale * cloud_n
                    if gas_eos is None:
                        break
                    state = gas_eos.state(temperature, pressure * 1e5, ng / ng.sum())
                    actual = np.asarray(state.lnphi)[gas_columns]
                    error = float(np.max(np.abs(actual - correction)))
                    if not np.all(np.isfinite(actual)):
                        raise PhaseEvaluationError("The gas EOS returned nonfinite fugacity coefficients.")
                    if error <= 1e-11:
                        iteration_report["gas_eos_iteration"] = {
                            "iterations": iteration, "lnphi_fixed_point_max_abs": error,
                            "lnphi_tolerance": 1e-11, "converged": True}
                        break
                    correction = actual
                else:
                    raise PhaseEvaluationError("The composition-dependent gas fugacity iteration did not converge.")
            report = audit_atmosphere(setup, temperature, pressure, b, ng, nc, element_gauge_rt,
                                      **({} if gas_eos is None else {"gas_eos": gas_eos}))
            if not report["accepted"]:
                raise PhaseEvaluationError("The atmospheric parcel failed independent equilibrium acceptance.")
        except (ValueError, RuntimeError, FloatingPointError, np.linalg.LinAlgError) as error:
            raise PhaseEvaluationError(str(error)) from error
        return {**report, **iteration_report, "solver_converged": True}

    def evaluate(temperature, pressure, amounts):
        b = np.asarray(amounts, dtype=float)
        if (b.shape != (len(elements),) or not np.all(np.isfinite(b)) or np.any(b < 0)
                or not np.isfinite(b.sum())):
            raise ValueError("Atmospheric atom amounts must be finite and nonnegative in element order.")
        if not all(np.isfinite(x) and x > 0 for x in (temperature, pressure)):
            raise ValueError("Atmospheric temperature and pressure must be positive and finite.")
        if not jax.config.x64_enabled:
            raise ValueError("Atmospheric equilibrium requires JAX_ENABLE_X64=1.")
        return solve(float(temperature), float(pressure), tuple(b))

    def callback(temperature, pressure, amounts):
        report = evaluate(temperature, pressure, amounts)
        return PhaseState(np.asarray(report["elemental_potentials_rt"]), report["gibbs_rt"])

    def energy_value_and_grad_rt(temperature, pressure, amounts):
        """Differentiate primitive G and project its stationary atom envelope.

        At an accepted parcel, dG/dn = A.T lambda. Differentiating the explicit
        primitive scalar with AD and solving this full-rank system gives
        dG_min/db = lambda without subtracting trace-sized energy changes.
        Neither saved elemental potentials nor callback mu are reused.
        """
        report = evaluate(temperature, pressure, amounts)
        b = np.asarray(amounts, dtype=float)
        potential = np.full(len(b), -np.inf)
        if not np.any(b > 0):
            return 0., potential
        ng, nc = np.asarray(report["gas_amounts_mol"]), np.asarray(report["condensate_amounts_mol"])
        rows, gas_columns, cloud_columns = np.flatnonzero(b > 0), np.flatnonzero(ng > 0), np.flatnonzero(nc > 0)
        q = np.asarray(element_gauge_rt(temperature, pressure) if callable(element_gauge_rt)
                       else element_gauge_rt, dtype=float)
        hg = np.asarray(gas_setup.hvector_func(temperature)) + ag.T @ q
        hc = np.asarray(cloud_setup.hvector_func(temperature)) + ac.T @ q
        energy, (gas_gradient, cloud_gradient) = _primitive_value_and_grad(
            ng[gas_columns], nc[cloud_columns], hg[gas_columns], hc[cloud_columns], np.log(pressure))
        if gas_eos is not None:
            residual, correction = residual_value_and_grad(temperature, pressure, ng)
            energy += residual
            gas_gradient += correction[gas_columns]
        matrix = np.r_[ag[np.ix_(rows, gas_columns)].T, ac[np.ix_(rows, cloud_columns)].T]
        gradient = np.r_[np.asarray(gas_gradient), np.asarray(cloud_gradient)]
        if np.linalg.matrix_rank(matrix) != len(rows) or not np.all(np.isfinite(gradient)):
            raise PhaseEvaluationError("The primitive scalar does not identify finite atom derivatives.")
        value = np.linalg.lstsq(matrix, gradient, rcond=None)[0]
        if np.max(np.abs(matrix @ value - gradient), initial=0.) >= 1e-8:
            raise PhaseEvaluationError("The primitive scalar is not stationary on the accepted support.")
        potential[rows] = value
        return float(energy), potential

    callback.parcel = lambda t, p, b: deepcopy(evaluate(t, p, b))
    callback.energy_value_and_grad_rt = energy_value_and_grad_rt
    callback.elements = elements
    callback.setup = setup
    callback.element_gauge_rt = element_gauge_rt
    callback.gas_eos = gas_eos
    return callback
