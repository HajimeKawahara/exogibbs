"""Adapt pure-component ExoEOS states to the ExoGibbs fugacity port."""

from __future__ import annotations

from typing import Any, Mapping, Optional, Sequence

import jax.numpy as jnp

from exogibbs.thermo.fugacity import LogFugacityCoefficientFunction


_BAR_TO_PA = 1.0e5
_UNSPECIFIED_SPECIES_POLICIES = ("error", "ideal")


def _load_state_tp() -> Any:
    try:
        from exoeos import state_tp
    except ImportError as exc:
        raise ImportError(
            "make_pure_lnphi_func requires a current ExoEOS checkout with "
            "the state_tp API."
        ) from exc
    return state_tp


def _validate_species(source_species: Sequence[str]) -> tuple[str, ...]:
    if source_species is None or isinstance(source_species, (str, bytes)):
        raise ValueError("source_species must be a sequence of species names.")
    species = tuple(source_species)
    if not species:
        raise ValueError("source_species must contain at least one species.")
    if not all(isinstance(name, str) and name for name in species):
        raise ValueError("source_species must contain non-empty species names.")
    if len(set(species)) != len(species):
        duplicates = sorted({name for name in species if species.count(name) > 1})
        raise ValueError(
            f"source_species names must be unique; duplicates: {duplicates}."
        )
    return species


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

        temperature_array = jnp.asarray(temperature)
        pressure_bar_array = jnp.asarray(pressure_bar)
        dtype = jnp.result_type(
            temperature_array,
            pressure_bar_array,
            jnp.float32,
        )
        temperature_array = temperature_array.astype(dtype)
        pressure_pa = pressure_bar_array.astype(dtype) * jnp.asarray(
            _BAR_TO_PA,
            dtype=dtype,
        )
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


__all__ = ["make_pure_lnphi_func"]
