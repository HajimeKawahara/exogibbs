"""Independent ideal-gas/pure-condensate KKT audit on an absolute mol basis."""

from typing import Any, Optional

import numpy as np
from numpy.typing import ArrayLike

from .setup import CondensateChemicalSetup


def audit(setup: CondensateChemicalSetup, temperature_k: float, pressure_bar: float,
          element_amounts_mol: ArrayLike, gas_amounts_mol: ArrayLike,
          condensate_amounts_mol: ArrayLike, *, element_gauge_rt: Optional[ArrayLike] = None,
          gas_log_amounts_mol: Optional[ArrayLike] = None) -> dict[str, Any]:
    """Recompute conservation, chemical potentials and phase insertion.

    This host audit assumes ideal gas and pure condensates, with no rainout.
    Exact-zero element budgets exclude their species. The common elemental
    gauge changes absolute G/duals but cannot change isolated equilibrium.
    Solver flags and saved duals are deliberately not accepted as inputs.
    Chemical residuals are dimensionless, G/(R*T) is in mol.

    Optional natural logarithms of absolute gas mol amounts retain chemical
    potentials when a positive amount underflows float64. Reachable logs must
    be finite, excluded logs must be -inf, and linear/log amounts must agree.
    A linear zero is allowed only when exp(log amount) also underflows; no
    composition floor or arbitrary truncation is accepted. The omitted atom
    bound is included in the unchanged elemental tolerance.
    """
    t, p = float(temperature_k), float(pressure_bar)
    gas, cloud = setup.gas_setup, setup.condensate_setup
    b, ng, nc = map(lambda v: np.asarray(v, dtype=float),
                    (element_amounts_mol, gas_amounts_mol, condensate_amounts_mol))
    if (not np.isfinite(t) or not np.isfinite(p) or t <= 0 or p <= 0
            or b.shape != (len(gas.elements),) or ng.shape != (len(gas.species),)
            or nc.shape != (len(cloud.species),) or gas.elements != cloud.elements
            or any(np.any(~np.isfinite(v)) or np.any(v < 0) for v in (b, ng, nc))
            or not np.isfinite(b.sum()) or not np.isfinite(ng.sum())
            or b.sum() <= 0 or ng.sum() <= 0):
        raise ValueError("Require positive T/P and matching finite nonnegative amounts.")
    ag, ac = np.asarray(gas.formula_matrix), np.asarray(cloud.formula_matrix)
    rows = b > 0
    supported_g = np.all(ag[~rows] == 0, axis=0)
    supported_c = np.all(ac[~rows] == 0, axis=0)
    if np.any(ng[~supported_g] != 0) or np.any(nc[~supported_c] != 0):
        raise ValueError("Species containing absent elements must stay exactly zero.")
    omitted = np.zeros_like(b)
    underflow = np.zeros_like(ng, dtype=bool)
    logs = None
    if gas_log_amounts_mol is None:
        if np.any(ng[supported_g] <= 0):
            raise ValueError("Reachable gas species must be positive unless consistent finite logs are supplied.")
        log_fractions = np.log(ng[supported_g]/ng.sum())
    else:
        logs = np.asarray(gas_log_amounts_mol, dtype=float)
        if (logs.shape != ng.shape or not np.all(np.isfinite(logs[supported_g]))
                or not np.all(np.isneginf(logs[~supported_g]))):
            raise ValueError("Reachable gas logs must be finite; excluded gas logs must be -inf.")
        with np.errstate(over="ignore", under="ignore"):
            represented = np.exp(logs)
        if not np.all(np.isfinite(represented)) or np.any((ng == 0) != (represented == 0)):
            raise ValueError("Linear gas zeros must correspond exactly to exponential underflow or exclusion.")
        normal = represented >= np.finfo(float).tiny
        tolerance = 64*np.finfo(float).eps*np.maximum(1., np.abs(logs[normal]))
        subnormal = (represented > 0) & ~normal
        quantum = np.nextafter(0., 1.)
        if (np.any(np.abs(np.log(ng[normal])-logs[normal]) > tolerance)
                or np.any(np.abs(ng[subnormal]-represented[subnormal]) > quantum)):
            raise ValueError("Linear gas amounts and their natural logarithms are inconsistent.")
        log_fractions = logs[supported_g]-np.logaddexp.reduce(logs[supported_g])
        underflow = supported_g & (ng == 0)
        # Each omitted amount is less than one float64 subnormal quantum.
        # Round products and sums outward, preserving exact-zero atom rows.
        for column in np.abs(ag[:, underflow]).T:
            term = np.where(column > 0, np.nextafter(column*quantum, np.inf), 0.)
            omitted = np.where(omitted+term > 0, np.nextafter(omitted+term, np.inf), 0.)
    gauge = np.zeros_like(b) if element_gauge_rt is None else np.asarray(element_gauge_rt, dtype=float)
    if gauge.shape != b.shape or not np.all(np.isfinite(gauge)):
        raise ValueError("The elemental gauge must be finite and follow the element order.")
    hg = np.asarray(gas.hvector_func(t))+ag.T @ gauge
    hc = np.asarray(cloud.hvector_func(t))+ac.T @ gauge
    valid_t = (np.ones(len(nc), bool) if cloud.temperature_validity_upper is None
               else t <= np.asarray(cloud.temperature_validity_upper))
    eligible = valid_t & supported_c
    mu = hg[supported_g]+log_fractions+np.log(p)
    matrix = ag[np.ix_(rows, supported_g)]
    if np.linalg.matrix_rank(matrix) != rows.sum():
        raise ValueError("The reachable gas must span the active elements.")
    dual = np.zeros_like(b)
    dual[rows] = np.linalg.lstsq(matrix.T, mu, rcond=None)[0]
    insertion = hc-ac.T @ dual
    residual = ag @ ng+ac @ nc-b
    relative = residual/np.where(rows, b, b.sum())
    relative_bound = (np.abs(residual)+omitted)/np.where(rows, b, b.sum())
    bounded = omitted > 0
    relative_bound[bounded] = np.nextafter(
        np.nextafter(np.abs(residual[bounded])+omitted[bounded], np.inf)/b[bounded], np.inf)
    stationarity = float(np.max(np.abs(mu-matrix.T @ dual[rows])))
    present = nc > 0
    present_error = float(np.max(np.abs(insertion[present]), initial=0.))
    absent_error = float(np.max(-insertion[eligible & ~present], initial=0.))
    accepted = bool(np.all(np.isfinite(mu)) and np.all(np.isfinite(hc[eligible]))
                    and np.max(relative_bound) <= 1e-9 and np.all(residual[~rows] == 0)
                    and stationarity <= 1e-8 and present_error <= 1e-8 and absent_error <= 1e-8
                    and not np.any(present & ~eligible))
    return dict(accepted=accepted, elements=list(gas.elements), T_K=t, P_bar=p,
                gas_species=list(gas.species), condensate_species=list(cloud.species),
                element_amounts_mol=b.tolist(), gas_amounts_mol=ng.tolist(), condensate_amounts_mol=nc.tolist(),
                relative_element_residual=relative.tolist(), gas_stationarity_max_abs=stationarity,
                gas_log_amounts_mol=None if logs is None else logs.tolist(),
                gas_underflow_species=[name for name, flag in zip(gas.species, underflow) if flag],
                gas_underflow_element_bound_mol=omitted.tolist(),
                relative_element_residual_upper_bound=relative_bound.tolist(),
                present_condensate_residual_max_abs=present_error, absent_condensate_violation_max_abs=absent_error,
                condensate_temperature_eligible=valid_t.tolist(), condensate_element_supported=supported_c.tolist(),
                condensate_insertion_rt=[float(v) if ok else None for v, ok in zip(insertion, eligible)],
                elemental_potentials_rt=[float(v) if ok else None for v, ok in zip(dual, rows)],
                element_gauge_rt=gauge.tolist(), gas_standard_potentials_rt=hg.tolist(),
                condensate_standard_potentials_rt=[float(v) if ok else None for v, ok in zip(hc, valid_t)],
                gibbs_rt=float(ng[supported_g] @ mu+nc[present] @ hc[present]))
