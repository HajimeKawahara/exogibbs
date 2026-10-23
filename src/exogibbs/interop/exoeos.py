"""Optional ExoEOS adapters for gas fugacity, activities and full phase G."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Callable, Mapping, Optional, Sequence

import jax.numpy as jnp
import numpy as np

from exogibbs.thermo.fugacity import LogFugacityCoefficientFunction
from exogibbs.utils.units import convert_pressure

if TYPE_CHECKING:
    from exogibbs.api.multiphase import Phase


_UNSPECIFIED_SPECIES_POLICIES = ("error", "ideal")


def _magma_basis(elements, components):
    """Load the optional provider and map its atom rows by name, never position."""
    try:
        from exoeos import magma
    except ImportError as exc:
        raise ImportError("Native multiphase adapters require ExoEOS magma.") from exc
    elements = _validate_species(elements, label="elements")
    matrix = np.asarray(magma.element_matrix(components))
    omitted = [name for name, row in zip(magma.ELEMENTS, matrix)
               if name not in elements and np.any(row != 0)]
    if omitted:
        raise ValueError(f"elements omits atoms in the phase: {omitted}.")
    mapped = np.zeros((len(elements), len(components)))
    for index, name in enumerate(elements):
        if name in magma.ELEMENTS:
            mapped[index] = matrix[magma.ELEMENTS.index(name)]
    return magma, mapped


def make_magma_liquid_phase(
    *, elements: Sequence[str], components: Sequence[str], name: str = "liquid",
) -> Phase:
    """Return a multiphase Phase using native full liquid G, unchanged in SI.

    Supply one complete native liquid basis, optionally reordered. This does
    not add ideal/excess mixing or the old empirical water correction. The
    twelve-component hydrous basis does not include H2 or He dissolution.
    ExoEOS owns all standards and its positive-composition derivative domain.
    No material coefficient is exposed by this fixed provider callback.
    """
    from exogibbs.api.multiphase import Phase

    components = _validate_species(components, label="components")
    magma, matrix = _magma_basis(elements, components)
    bases = (magma.COMPONENTS, magma.TERNARY_COMPONENTS, magma.CMAS_COMPONENTS,
             magma.CMFAS_COMPONENTS, magma.FERRIC_COMPONENTS,
             magma.DRY11_COMPONENTS, magma.HYDROUS_COMPONENTS)
    matches = [basis for basis in bases if set(basis) == set(components)]
    if not matches:
        raise ValueError("components must be a complete supported native liquid basis.")
    order = np.asarray([components.index(component) for component in matches[0]])

    def gibbs_j(temperature_k, pressure_pa, amounts_mol, parameters):
        return magma.liquid_gibbs(temperature_k, pressure_pa, amounts_mol[order])

    def derivative_domain(temperature_k, pressure_pa, amounts_mol, parameters):
        return (temperature_k != 1480.) & jnp.all(amounts_mol > 0)

    return Phase(name, components, matrix, gibbs_j,
                 domain="ExoEOS native liquid; positive component derivative domain",
                 standard_state="ExoEOS magma native MELTS full liquid G",
                 derivative_domain=derivative_domain)


def make_magma_pure_phase(
    *, elements: Sequence[str], component: str, molar_gibbs_j: Callable,
    name: str,
) -> Phase:
    """Wrap a caller-selected native molar g(T_K, P_Pa) as n*g in joules.

    The caller owns selection of the thermodynamic phase matching component.
    This accepts pure phases, not signed-coordinate native solid solutions.
    """
    from exogibbs.api.multiphase import Phase

    _, matrix = _magma_basis(elements, (component,))
    if not callable(molar_gibbs_j):
        raise ValueError("molar_gibbs_j must be callable.")

    def gibbs_j(temperature_k, pressure_pa, amounts_mol, parameters):
        return amounts_mol[0] * molar_gibbs_j(temperature_k, pressure_pa)

    return Phase(name, (component,), matrix, gibbs_j,
                 standard_state="caller-selected ExoEOS native molar G")


def _load_state_tp() -> Any:
    try:
        from exoeos import state_tp
    except ImportError as exc:
        raise ImportError(
            "make_pure_lnphi_func requires a current ExoEOS checkout with "
            "the state_tp API."
        ) from exc
    return state_tp


def _validate_species(
    source_species: Sequence[str], *, label: str = "source_species",
) -> tuple[str, ...]:
    if source_species is None or isinstance(source_species, (str, bytes)):
        raise ValueError(f"{label} must be a sequence of species names.")
    species = tuple(source_species)
    if not species:
        raise ValueError(f"{label} must contain at least one species.")
    if not all(isinstance(name, str) and name for name in species):
        raise ValueError(f"{label} must contain non-empty species names.")
    if len(set(species)) != len(species):
        duplicates = sorted({name for name in species if species.count(name) > 1})
        raise ValueError(
            f"{label} names must be unique; duplicates: {duplicates}."
        )
    return species


def _state_inputs(
    temperature: Any,
    pressure_bar: Any,
    *dtype_values: jnp.ndarray,
) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Normalize scalar provider inputs and convert pressure to Pa."""

    temperature_array = jnp.asarray(temperature)
    pressure_array = jnp.asarray(pressure_bar)
    if temperature_array.ndim != 0 or pressure_array.ndim != 0:
        raise ValueError(
            "Temperature and pressure must be scalars; use jax.vmap for batches."
        )
    dtype = jnp.result_type(
        temperature_array, pressure_array, *dtype_values, jnp.float32,
    )
    return temperature_array.astype(dtype), convert_pressure(
        pressure_array.astype(dtype), from_unit="bar", to_unit="Pa",
    )


def make_pure_lnphi_func(
    *,
    source_species: Sequence[str],
    eos_by_species: Mapping[str, Any],
    unspecified_species: str = "error",
    phase: str = "vapor",
) -> LogFugacityCoefficientFunction:
    """Return an ExoGibbs callback backed by one-component ExoEOS models.

    ``eos_by_species`` maps names in ``source_species`` to one-component
    ExoEOS models. Missing models are rejected unless
    ``unspecified_species="ideal"``, which returns ``ln(phi) = 0`` for those
    species. ExoGibbs pressure in bar is converted to Pa before each ExoEOS
    state evaluation. ExoEOS retains ownership of state dtype promotion and
    does not expose species labels, so model identity is a caller contract.
    Temperature and pressure must be scalars; use ``jax.vmap`` for batches.
    """

    species = _validate_species(source_species)
    models_by_species = dict(eos_by_species)
    if not all(isinstance(name, str) and name for name in models_by_species):
        raise ValueError("eos_by_species keys must be non-empty species names.")

    unknown_species = sorted(set(models_by_species) - set(species))
    if unknown_species:
        raise ValueError(
            "eos_by_species contains names absent from source_species: "
            f"{unknown_species}."
        )
    if unspecified_species not in _UNSPECIFIED_SPECIES_POLICIES:
        raise ValueError(
            "unspecified_species must be 'error' or 'ideal'; "
            f"got {unspecified_species!r}."
        )
    if not isinstance(phase, str) or not phase:
        raise ValueError("phase must be a non-empty string.")

    invalid_models = [
        name for name, model in models_by_species.items() if model is None
    ]
    if invalid_models:
        raise ValueError(
            "eos_by_species values must be ExoEOS models; got None for "
            f"{sorted(invalid_models)}."
        )

    missing_species = [name for name in species if name not in models_by_species]
    if missing_species and unspecified_species == "error":
        raise ValueError(
            "eos_by_species is missing source species: "
            f"{missing_species}. Set unspecified_species='ideal' to use "
            "ln(phi) = 0 for them."
        )

    models = tuple(models_by_species.get(name) for name in species)
    for name, model in zip(species, models):
        if model is None:
            continue
        component_count = getattr(model, "component_count", None)
        if component_count is not None and component_count != 1:
            raise ValueError(
                f"eos_by_species[{name!r}] must be a one-component EOS; "
                f"got component_count={component_count}."
            )

    state_tp = _load_state_tp() if models_by_species else None

    def lnphi_func(
        temperature: Any,
        pressure_bar: Any,
        mole_fractions: Optional[jnp.ndarray],
    ) -> jnp.ndarray:
        if mole_fractions is not None:
            raise ValueError(
                "make_pure_lnphi_func supports pure-component fugacity only; "
                "mole_fractions must be None."
            )

        temperature_array, pressure_pa = _state_inputs(temperature, pressure_bar)
        dtype = temperature_array.dtype
        pure_composition = jnp.ones((1,), dtype=dtype)
        ideal_lnphi = jnp.zeros((), dtype=dtype)

        values = []
        for name, model in zip(species, models):
            if model is None:
                values.append(ideal_lnphi)
                continue
            state = state_tp(
                model,
                temperature_array,
                pressure_pa,
                pure_composition,
                phase=phase,
            )
            model_lnphi = jnp.asarray(state.lnphi)
            if model_lnphi.shape != (1,):
                raise ValueError(
                    f"ExoEOS state for {name!r} must return one fugacity "
                    f"coefficient; got shape {model_lnphi.shape}."
                )
            values.append(model_lnphi[0])
        return jnp.stack(values)

    return lnphi_func


def make_solution_lngamma_func(
    *,
    source_components: Sequence[str],
    model: Any,
    model_components: Optional[Sequence[str]] = None,
) -> Callable[[Any, Any, jnp.ndarray], jnp.ndarray]:
    """Adapt one homogeneous ExoEOS solution to ``lngamma(T_K, P_bar, x)``.

    Input fractions and output natural-log coefficients follow
    ``source_components``. Model order is read from ``model.components``;
    models without labels, such as ``IdealSolution``, require explicit
    ``model_components``. Component sets must match exactly. The model must
    declare ``activity_basis='mole_fraction'`` and
    ``standard_state_convention='symmetric'``.

    The callback converts bar to Pa once, and returns only ``ln(gamma)``:
    ideal mixing, standard potentials, and gas pressure terms are excluded.
    It preserves JAX tracing. Callers must supply normalized nonnegative
    fractions and enforce the provider's physical domain before tracing;
    this adapter checks shapes without clipping or normalizing compositions.
    """
    components = _validate_species(source_components, label="source_components")
    declared = getattr(model, "components", None)
    if model_components is None:
        if declared is None:
            raise ValueError(
                "model_components is required when model.components is absent."
            )
        model_components = declared
    provider_components = _validate_species(model_components, label="model_components")
    if declared is not None and tuple(declared) != provider_components:
        raise ValueError("model_components must agree with model.components in order.")
    if set(components) != set(provider_components):
        raise ValueError("source_components and model_components must have identical sets.")
    if getattr(model, "activity_basis", None) != "mole_fraction":
        raise ValueError("The solution model must declare activity_basis='mole_fraction'.")
    if getattr(model, "standard_state_convention", None) != "symmetric":
        raise ValueError(
            "The solution model must declare standard_state_convention='symmetric'."
        )
    if not callable(getattr(model, "gex_RT", None)):
        raise ValueError("The solution model must implement gex_RT(T, P, x).")
    try:
        from exoeos import solution_state
    except ImportError as exc:
        raise ImportError(
            "make_solution_lngamma_func requires ExoEOS with the solution_state API."
        ) from exc

    to_provider = jnp.asarray([components.index(name) for name in provider_components])
    to_consumer = jnp.asarray([provider_components.index(name) for name in components])
    expected_shape = (len(components),)

    def lngamma_func(
        temperature: Any, pressure_bar: Any, mole_fractions: jnp.ndarray,
    ) -> jnp.ndarray:
        if mole_fractions is None:
            raise ValueError("Solution activities require phase mole_fractions, not None.")
        composition = jnp.asarray(mole_fractions)
        temperature_array, pressure_pa = _state_inputs(
            temperature, pressure_bar, composition,
        )
        if composition.shape != expected_shape:
            raise ValueError(f"mole_fractions must have shape {expected_shape}.")
        state = solution_state(
            model,
            temperature_array,
            pressure_pa,
            composition.astype(temperature_array.dtype)[to_provider],
        )
        values = jnp.asarray(state.lngamma)
        if values.shape != expected_shape:
            raise ValueError(
                f"ExoEOS lngamma must have shape {expected_shape}; got {values.shape}."
            )
        return values[to_consumer]

    return lngamma_func


__all__ = ["make_pure_lnphi_func", "make_solution_lngamma_func"]
