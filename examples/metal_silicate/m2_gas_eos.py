"""Connect the declared full-catalog M2 gas to its ExoEOS provider."""

from __future__ import annotations

import importlib.util
from pathlib import Path

import jax
import numpy as np

from full_potential import PhaseState


def build_gas_eos(exoeos_checkout, gas_species, gas_eos_options=None):
    """Return the explicit EOS recipe, or retain the historical ideal gas.

    The provider owns coefficients, their temperature policies, residual G,
    fugacity and density. This adapter preserves the complete species order.
    """
    if gas_eos_options is None:
        return None
    import exoeos

    checkout = Path(exoeos_checkout).resolve()
    if Path(exoeos.__file__).resolve() != checkout / "src/exoeos/__init__.py":
        raise ValueError("Import the explicitly selected ExoEOS checkout.")
    path = checkout / "examples/m2_material/major_gas_eos.py"
    spec = importlib.util.spec_from_file_location("_exoeos_m2_major_gas", path)
    provider = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(provider)
    return provider.make_major_gas_eos(tuple(gas_species), gas_eos_options)


def with_gas_residual(ideal_callback, gas_eos):
    """Add the same provider scalar and fugacity to a primitive gas phase."""
    if gas_eos is None:
        return ideal_callback

    derivative = jax.jit(jax.value_and_grad(
        lambda t, p, n: gas_eos.gibbs_residual_rt(t, p * 1e5, n), argnums=2))

    def evaluate(temperature, pressure, amounts):
        state = ideal_callback(temperature, pressure, amounts)
        n = np.asarray(amounts, dtype=float)
        if not np.any(n):
            return state
        eos_state = gas_eos.state(temperature, pressure * 1e5, n / n.sum())
        residual = float(gas_eos.gibbs_residual_rt(temperature, pressure * 1e5, n))
        return PhaseState(state.mu_rt + np.asarray(eos_state.lnphi), state.gibbs_rt + residual)

    def energy_value_and_grad_rt(temperature, pressure, amounts):
        energy, gradient = ideal_callback.energy_value_and_grad_rt(temperature, pressure, amounts)
        if not np.any(np.asarray(amounts)):
            return energy, gradient
        residual, correction = derivative(temperature, pressure, np.asarray(amounts))
        return energy + residual, gradient + correction

    evaluate.energy_value_and_grad_rt = energy_value_and_grad_rt
    evaluate.gas_eos = gas_eos
    return evaluate
