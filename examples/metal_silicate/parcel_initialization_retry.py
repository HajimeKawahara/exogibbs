"""Optional one-shot condensate initialization retry for retained M2 parcels.

This numerical helper changes initial amounts only. The original local solver,
its catalog-wide support search, and every downstream acceptance audit remain
authoritative. Consumers own persistence of the diagnostic receipts.
"""

from dataclasses import fields, is_dataclass
from functools import wraps
import math

import jax.numpy as jnp
import numpy as np

from exogibbs.api.condensate import CondensateEquilibriumInit


def _plain(value):
    """Copy diagnostics into strict-JSON values, retaining nonfinite evidence."""
    if is_dataclass(value):
        return {field.name: _plain(getattr(value, field.name)) for field in fields(value)}
    if isinstance(value, dict):
        return {str(key): _plain(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_plain(item) for item in value]
    if isinstance(value, np.generic):
        return _plain(value.item())
    if isinstance(value, float) and not math.isfinite(value):
        return {"nonfinite_float": repr(value)}
    if value is None or isinstance(value, (str, bool, int, float)):
        return value
    if hasattr(value, "shape") and hasattr(value, "dtype"):
        return _plain(np.asarray(value).tolist())
    raise TypeError("Unsupported parcel diagnostic value: " + type(value).__name__)


def _seed(setup, budget, result):
    """Retain only finite, positive-gas, conserved caller-gauge amounts."""
    gas = np.asarray(result.gas_n, dtype=float)
    cloud = np.asarray(result.condensate_amounts, dtype=float)
    ag = np.asarray(setup.gas_setup.formula_matrix, dtype=float)
    ac = np.asarray(setup.condensate_setup.formula_matrix, dtype=float)
    if (budget.ndim != 1 or np.any(~np.isfinite(budget)) or np.any(budget < 0)
            or gas.shape != (len(setup.gas_species),)
            or cloud.shape != (len(setup.condensate_species),)
            or np.any(~np.isfinite(gas)) or np.any(gas <= 0)
            or np.any(~np.isfinite(cloud)) or np.any(cloud < 0)):
        return None
    actual = ag @ gas + ac @ cloud
    positive = budget > 0
    if (np.any(actual[~positive] != 0)
            or np.max(np.abs(actual[positive] / budget[positive] - 1.), initial=0.) > 1e-9):
        return None
    return {"gas_amounts": gas.copy(), "condensate_amounts": cloud.copy()}


def make_previous_parcel_retry(solver, *, enabled=False, record=None):
    """Wrap a condensate solver with an explicitly enabled, single warm retry.

    The cold call is always first and is unchanged. Only a returned
    ``converged=False`` may trigger a retry. Its donor is the latest
    solver-converged result for the identical setup object, byte-identical
    caller budget and pressure standard. All donor amounts use the solver's
    caller gauge; the helper never normalizes them a second time.

    Explicit initializers/support, rainout and nonideal fugacity callbacks
    are outside this retry's scope. Temperature-ineligible donor condensates
    also prevent a retry. There is no final-support constraint or tolerance
    override. Failed calls are sent to ``record(receipt)`` before returning or
    propagating an exception, so a surrounding generic exception cannot erase
    their raw solver diagnostics. This callback must persist receipts itself.

    A solver-converged donor is not a claim that a consumer's independent
    parcel, whole-column or pressure-root audit has passed.
    """
    if type(enabled) is not bool:
        raise TypeError("enabled must be a bool.")
    if not callable(solver) or (record is not None and not callable(record)):
        raise TypeError("solver and any record callback must be callable.")
    previous = None

    @wraps(solver)
    def wrapped(setup, temperature, pressure, budget, **kwargs):
        nonlocal previous
        if not enabled:
            return solver(setup, temperature, pressure, budget, **kwargs)
        b = np.asarray(budget, dtype=float)
        options = kwargs.get("options")
        unsupported = None
        if options is None or options.rainout or not options.return_diagnostics:
            unsupported = "requires explicit retained-parcel diagnostic options"
        if any(kwargs.get(key) is not None for key in (
                "init", "initializer", "support_indices", "support_amounts_init", "lnphi_func")):
            unsupported = "explicit initialization/support or fugacity callback"
        point = {"temperature_K": float(temperature), "pressure_bar": float(pressure),
                 "pressure_standard_bar": float(kwargs.get("Pref", 1.)),
                 "elements": list(setup.elements), "gas_species": list(setup.gas_species),
                 "condensate_species": list(setup.condensate_species),
                 "caller_element_amounts": b.tolist()}
        receipt = {"schema": "m2_previous_parcel_retry_v1", "point": point,
                   "options": _plain(options), "retry_attempted": False,
                   "scope": "Initial amounts only; original solver and downstream audits are unchanged."}

        def emit():
            if record is not None:
                record(_plain(receipt))

        try:
            cold = solver(setup, temperature, pressure, budget, **kwargs)
        except Exception as error:
            receipt.update(cold_exception=type(error).__name__ + ": " + str(error),
                           skip_reason="the cold solver raised instead of returning a result")
            emit()
            raise
        if bool(cold.converged):
            seed = None if unsupported else _seed(setup, b, cold)
            previous = (None if seed is None else
                        {"setup": setup, "budget": b.copy(), "point": point, **seed})
            return cold
        receipt["cold_result"] = _plain(cold)
        reason = unsupported
        if reason is None and previous is None:
            reason = "no eligible solver-converged donor"
        if reason is None and (previous["setup"] is not setup
                or previous["budget"].shape != b.shape
                or previous["budget"].tobytes() != b.tobytes()
                or previous["point"]["pressure_standard_bar"] != point["pressure_standard_bar"]):
            reason = "the donor setup, caller budget or pressure standard differs"
        if reason is None:
            upper = setup.condensate_setup.temperature_validity_upper
            if upper is not None and np.any((previous["condensate_amounts"] > 0)
                                           & (float(temperature) > np.asarray(upper))):
                reason = "a donor condensate is temperature-ineligible"
        if reason is not None:
            receipt["skip_reason"] = reason
            emit()
            return cold
        initial = CondensateEquilibriumInit(
            gas_ln_n=jnp.asarray(np.log(previous["gas_amounts"])),
            gas_ntot=jnp.asarray(previous["gas_amounts"].sum()),
            condensate_amounts=jnp.asarray(previous["condensate_amounts"]),
        )
        receipt.update(retry_attempted=True,
                       donor={"point": previous["point"],
                              "gas_amounts": previous["gas_amounts"].tolist(),
                              "condensate_amounts": previous["condensate_amounts"].tolist(),
                              "solver_converged": True},
                       initial=_plain(initial))
        try:
            warm = solver(setup, temperature, pressure, budget, **{**kwargs, "init": initial})
        except Exception as error:
            receipt["warm_exception"] = type(error).__name__ + ": " + str(error)
            emit()
            raise
        receipt["warm_result"] = _plain(warm)
        emit()
        if bool(warm.converged):
            seed = _seed(setup, b, warm)
            previous = (None if seed is None else
                        {"setup": setup, "budget": b.copy(), "point": point, **seed})
        return warm

    return wrapped
