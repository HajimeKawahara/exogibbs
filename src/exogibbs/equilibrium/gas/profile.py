"""Profile scheduling for gas-only equilibrium."""

from typing import Callable, Dict, Literal, Mapping, Optional, Tuple, Union

import jax
import jax.numpy as jnp
from jax import lax

from exogibbs.equilibrium.gas.initialization import (
    DEFAULT_INITIALIZER,
    prepare_init,
    resolve_initial_guess,
)
from exogibbs.equilibrium.gas.solve import equilibrium
from exogibbs.equilibrium.gas.types import (
    Array,
    EquilibriumInit,
    EquilibriumInitializer,
    EquilibriumInitRequest,
    EquilibriumOptions,
    EquilibriumResult,
)
from exogibbs.thermo.models import ChemicalSetup


ProfileMethod = Literal[
    "vmap_cold",
    "scan_hot_from_top",
    "scan_hot_from_bottom",
]

_PROFILE_SCAN_BODY_CACHE: Dict[
    Tuple[int, int, float, int, int, bool],
    Callable,
] = {}


def resolve_profile_method(
    method: Optional[ProfileMethod],
    initializer: Optional[EquilibriumInitializer],
) -> ProfileMethod:
    """Resolve the explicit or default profile scheduling method."""

    if method is not None:
        return method
    if initializer is not None:
        return "vmap_cold"
    return "scan_hot_from_bottom"


def _get_profile_scan_body(
    setup: ChemicalSetup,
    b: Array,
    reference_pressure: float,
    options: EquilibriumOptions,
    initializer: Optional[EquilibriumInitializer],
    return_diagnostics: bool,
) -> Callable:
    key = (
        id(setup),
        id(b),
        float(reference_pressure),
        id(options),
        id(initializer or DEFAULT_INITIALIZER),
        return_diagnostics,
    )
    cached = _PROFILE_SCAN_BODY_CACHE.get(key)
    if cached is not None:
        return cached

    species_count = int(setup.formula_matrix.shape[1])

    if return_diagnostics:

        def scan_body(carry, tp_pair):
            ln_nk_previous, ln_ntot_previous = carry
            temperature, pressure = tp_pair
            solver_init = resolve_initial_guess(
                initializer,
                EquilibriumInitRequest(
                    setup=setup,
                    T=temperature,
                    P=pressure,
                    b=b,
                    K=species_count,
                    previous_solution=EquilibriumInit(
                        ln_nk=ln_nk_previous,
                        ln_ntot=ln_ntot_previous,
                    ),
                ),
            )
            result, diagnostics = equilibrium(
                setup,
                temperature,
                pressure,
                b,
                Pref=reference_pressure,
                init=solver_init,
                options=options,
                return_diagnostics=True,
            )
            next_total = jnp.log(jnp.clip(result.ntot, 1.0e-300))
            return (result.ln_n, next_total), (result, diagnostics)

    else:

        def scan_body(carry, tp_pair):
            ln_nk_previous, ln_ntot_previous = carry
            temperature, pressure = tp_pair
            solver_init = resolve_initial_guess(
                initializer,
                EquilibriumInitRequest(
                    setup=setup,
                    T=temperature,
                    P=pressure,
                    b=b,
                    K=species_count,
                    previous_solution=EquilibriumInit(
                        ln_nk=ln_nk_previous,
                        ln_ntot=ln_ntot_previous,
                    ),
                ),
            )
            result = equilibrium(
                setup,
                temperature,
                pressure,
                b,
                Pref=reference_pressure,
                init=solver_init,
                options=options,
                return_diagnostics=False,
            )
            next_total = jnp.log(jnp.clip(result.ntot, 1.0e-300))
            return (result.ln_n, next_total), result

    _PROFILE_SCAN_BODY_CACHE[key] = scan_body
    return scan_body


def equilibrium_profile(
    setup: ChemicalSetup,
    T: Array,
    P: Array,
    b: Array,
    *,
    Pref: float = 1.0,
    initializer: Optional[EquilibriumInitializer] = None,
    options: Optional[EquilibriumOptions] = None,
    return_diagnostics: bool = False,
) -> Union[EquilibriumResult, Tuple[EquilibriumResult, Mapping[str, Array]]]:
    """Compute gas equilibrium along a one-dimensional profile."""

    temperatures = jnp.asarray(T)
    pressures = jnp.asarray(P)
    if temperatures.ndim != 1 or pressures.ndim != 1:
        raise ValueError("T and P must be 1D arrays of equal length.")
    if temperatures.shape[0] != pressures.shape[0]:
        raise ValueError("T and P must have the same length.")
    if b.ndim != 1:
        raise ValueError("b must be a 1D array shared across layers.")

    active_options = options or EquilibriumOptions()
    method = resolve_profile_method(active_options.method, initializer)
    valid_methods = (
        "vmap_cold",
        "scan_hot_from_top",
        "scan_hot_from_bottom",
    )
    if method not in valid_methods:
        raise ValueError(
            f"Unknown solve method {method!r}. Expected one of {valid_methods}."
        )

    if method == "vmap_cold":
        if return_diagnostics:
            layer_function = jax.vmap(
                lambda temperature, pressure: equilibrium(
                    setup,
                    temperature,
                    pressure,
                    b,
                    Pref=Pref,
                    initializer=initializer,
                    options=active_options,
                    return_diagnostics=True,
                ),
                in_axes=(0, 0),
            )
            return layer_function(temperatures, pressures)
        layer_function = jax.vmap(
            lambda temperature, pressure: equilibrium(
                setup,
                temperature,
                pressure,
                b,
                Pref=Pref,
                initializer=initializer,
                options=active_options,
                return_diagnostics=False,
            ),
            in_axes=(0, 0),
        )
        return layer_function(temperatures, pressures)

    if method == "scan_hot_from_bottom":
        temperatures_input = jnp.flip(temperatures, axis=0)
        pressures_input = jnp.flip(pressures, axis=0)
    else:
        temperatures_input = temperatures
        pressures_input = pressures

    species_count = int(setup.formula_matrix.shape[1])
    first_init = resolve_initial_guess(
        initializer,
        EquilibriumInitRequest(
            setup=setup,
            T=temperatures_input[0],
            P=pressures_input[0],
            b=b,
            K=species_count,
        ),
    )
    ln_nk_init, ln_ntot_init = prepare_init(first_init, b, species_count)
    scan_body = _get_profile_scan_body(
        setup,
        b,
        Pref,
        active_options,
        initializer,
        return_diagnostics,
    )
    if return_diagnostics:
        _, (result_sequence, diagnostic_sequence) = lax.scan(
            scan_body,
            (ln_nk_init, ln_ntot_init),
            (temperatures_input, pressures_input),
        )
    else:
        _, result_sequence = lax.scan(
            scan_body,
            (ln_nk_init, ln_ntot_init),
            (temperatures_input, pressures_input),
        )
        diagnostic_sequence = None

    if method == "scan_hot_from_bottom":
        result_sequence = EquilibriumResult(
            ln_n=jnp.flip(result_sequence.ln_n, axis=0),
            n=jnp.flip(result_sequence.n, axis=0),
            x=jnp.flip(result_sequence.x, axis=0),
            ntot=jnp.flip(result_sequence.ntot, axis=0),
            iterations=result_sequence.iterations,
            metadata=result_sequence.metadata,
        )
        if diagnostic_sequence is not None:
            diagnostic_sequence = {
                key: jnp.flip(value, axis=0)
                for key, value in diagnostic_sequence.items()
            }

    if return_diagnostics:
        return result_sequence, diagnostic_sequence
    return result_sequence


solve_profile = equilibrium_profile


__all__ = ("equilibrium_profile", "solve_profile")
