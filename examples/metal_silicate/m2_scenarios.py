"""Explicit, uncalibrated standard-energy and alloy-domain sensitivity cases."""

from __future__ import annotations

import numpy as np

from full_potential import PhaseState
from run_metal_selection import METAL_LOWER, METAL_UPPER


def _finite_number(value):
    if type(value) not in (int, float):
        return False
    try:
        return np.isfinite(float(value))
    except (ValueError, OverflowError):
        return False


def normalize_scenario(scenario=None):
    """Validate a JSON scenario without inventing physical uncertainty bounds."""
    if scenario is None:
        scenario = {}
    if not isinstance(scenario, dict) or set(scenario) - {"standard_offsets_rt", "metal_bounds"}:
        raise ValueError("Unknown or invalid M2 provider scenario.")
    offsets = scenario.get("standard_offsets_rt", {})
    names = ("H2_dissolved", "H_metal")
    if not isinstance(offsets, dict) or set(offsets) - set(names):
        raise ValueError("Standard offsets may name only H2_dissolved and H_metal.")
    for value in offsets.values():
        if not _finite_number(value):
            raise ValueError("Standard offsets must be finite JSON numbers, not booleans.")
    bounds = scenario.get("metal_bounds", {})
    if not isinstance(bounds, dict) or set(bounds) - {"lower", "upper"}:
        raise ValueError("Metal bounds may contain only lower and upper atomic fractions.")
    values = []
    for name, default in (("lower", METAL_LOWER), ("upper", METAL_UPPER)):
        raw = bounds.get(name, default.tolist())
        if (not isinstance(raw, list) or len(raw) != 4
                or any(not _finite_number(value) for value in raw)):
            raise ValueError("Metal bounds require four finite JSON numbers in Fe, Si, O, H order.")
        values.append(np.asarray(raw, dtype=float))
    lower, upper = values
    if (np.any(lower < 0) or np.any(upper > 1) or np.any(lower > upper)
            or lower.sum() > 1 or upper.sum() < 1 or lower[0] <= 0):
        raise ValueError("The scenario requires a feasible Fe-rich alloy composition box.")
    return {"standard_offsets_rt": {name: float(offsets.get(name, 0.)) for name in names},
            "metal_bounds": {"lower": lower.tolist(), "upper": upper.tolist()}}


def metal_selection_domain(scenario=None):
    """Return bounds whose curvature must be re-evaluated by the provider."""
    bounds = normalize_scenario(scenario)["metal_bounds"]
    return np.asarray(bounds["lower"]), np.asarray(bounds["upper"])


def apply_standard_offsets(record, callbacks, scenario):
    """Add the same linear term to extensive G and its component potentials."""
    normalized = normalize_scenario(scenario)
    result = dict(callbacks)
    for phase, names in record["phases"].items():
        shift = np.asarray([normalized["standard_offsets_rt"].get(name, 0.) for name in names])
        if not np.any(shift):
            continue
        original = callbacks[phase]

        def shifted(t, p, n, original=original, shift=shift):
            state = original(t, p, n)
            return PhaseState(np.asarray(state.mu_rt) + shift,
                              float(state.gibbs_rt + np.asarray(n) @ shift))

        if hasattr(original, "energy_value_and_grad_rt"):
            def gradient(t, p, n, original=original, shift=shift):
                energy, mu = original.energy_value_and_grad_rt(t, p, n)
                return energy + np.asarray(n) @ shift, mu + shift
            shifted.energy_value_and_grad_rt = gradient
        result[phase] = shifted
    return result
