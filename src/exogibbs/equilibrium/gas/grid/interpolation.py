"""Validation, composition coordinates, and interpolation for gas grids."""

from __future__ import annotations

from typing import Optional

import jax.numpy as jnp
from jax import core

from exogibbs.equilibrium.gas.grid.types import (
    _COMPOSITION_AXIS_NAME,
    _setup_metadata_matches,
    Array,
    EquilibriumGrid,
    EquilibriumGridInterpolationOptions,
    EquilibriumGridInterpolationResult,
    EquilibriumGridMetadata,
)
from exogibbs.thermo.models import ChemicalSetup
from exogibbs.utils.elements import element_mass
from exogibbs.utils.interpolation import Interpolator3D


def _verification_dtype_warning() -> str:
    if jnp.asarray(1.0).dtype != jnp.float64:
        return " JAX is currently running in float32; enable jax_enable_x64=True for tighter FastChem comparisons."
    return ""


def _require_h_he_reference_abundance_setup(setup: ChemicalSetup) -> Array:
    if setup.element_vector_reference is None:
        raise ValueError("setup.element_vector_reference is required for grid generation.")
    if setup.elements is None:
        raise ValueError("setup.elements is required for H/He metallicity grid generation.")
    if "H" not in setup.elements or "He" not in setup.elements:
        raise ValueError("H/He metallicity grid generation requires H and He in setup.elements.")
    return jnp.asarray(setup.element_vector_reference)


def _element_mass_vector(setup: ChemicalSetup, dtype: jnp.dtype) -> Array:
    if setup.elements is None:
        raise ValueError("setup.elements is required for H/He metallicity grid generation.")
    try:
        masses = [0.0 if element == "e-" else element_mass[element] for element in setup.elements]
    except KeyError as exc:
        raise KeyError(f"Missing elemental mass for '{exc.args[0]}'.") from exc
    return jnp.asarray(masses, dtype=dtype)


def _validated_element_vector(setup: ChemicalSetup, element_vector: Array) -> Array:
    if setup.elements is None:
        raise ValueError("setup.elements is required for metallicity calculations.")

    b = jnp.asarray(element_vector)
    if b.ndim != 1:
        raise ValueError(
            f"element_vector must be one-dimensional with shape ({len(setup.elements)},), got {b.shape}."
        )
    if b.shape[0] != len(setup.elements):
        raise ValueError(
            f"element_vector length must match setup.elements ({len(setup.elements)}), got {b.shape[0]}."
        )
    return b


def compute_physical_metal_mass_fraction(setup: ChemicalSetup, element_vector: Array) -> Array:
    """Return the physical metal mass fraction Z for an elemental abundance vector."""
    b = _validated_element_vector(setup, element_vector)
    masses = _element_mass_vector(setup, b.dtype)
    non_electron_mask = jnp.asarray([element != "e-" for element in setup.elements], dtype=b.dtype)
    metal_mask = jnp.asarray(
        [element not in {"H", "He", "e-"} for element in setup.elements],
        dtype=b.dtype,
    )
    weighted_abundances = masses * b
    total_mass = jnp.sum(weighted_abundances * non_electron_mask)
    metal_mass = jnp.sum(weighted_abundances * metal_mask)
    return metal_mass / jnp.clip(total_mass, 1e-300)


def compute_reference_physical_metal_mass_fraction(setup: ChemicalSetup) -> Array:
    """Return the reference physical metal mass fraction Zsun for a setup."""
    return compute_physical_metal_mass_fraction(
        setup,
        _require_h_he_reference_abundance_setup(setup),
    )


def compute_physical_log10_z_over_z_sun(
    setup: ChemicalSetup,
    element_vector: Array,
) -> Array:
    """Return physical ``log10(Z/Zsun)`` for an elemental abundance vector."""
    z = compute_physical_metal_mass_fraction(setup, element_vector)
    if not isinstance(z, core.Tracer) and float(z) <= 0.0:
        raise ValueError(
            "Physical log10(Z/Zsun) is undefined when the elemental abundance vector has Z <= 0."
        )

    z_sun = compute_reference_physical_metal_mass_fraction(setup)
    if not isinstance(z_sun, core.Tracer) and float(z_sun) <= 0.0:
        raise ValueError(
            "Physical log10(Z/Zsun) is undefined when setup.element_vector_reference has Zsun <= 0."
        )

    return jnp.log10(z / z_sun)


def _h_he_metallicity_scale_from_log10_z_over_z_sun(
    setup: ChemicalSetup,
    log10_z_over_z_sun: float,
) -> Array:
    b_ref = _require_h_he_reference_abundance_setup(setup)
    z_sun = compute_reference_physical_metal_mass_fraction(setup)
    target_z = z_sun * jnp.asarray(10.0**log10_z_over_z_sun, dtype=b_ref.dtype)
    if float(target_z) >= 1.0:
        raise ValueError(
            "Target physical metallicity requires Z >= 1, which is not valid for an H/He atmosphere."
        )
    return (target_z * (1.0 - z_sun)) / jnp.clip(z_sun * (1.0 - target_z), 1e-300)


def validate_equilibrium_grid_compatibility(
    grid: EquilibriumGrid,
    setup: ChemicalSetup,
    preset_name: str,
    *,
    expected_composition_axis_name: str = _COMPOSITION_AXIS_NAME,
) -> None:
    """Validate that a loaded grid is compatible with a runtime preset/setup.

    This checks preset identity, setup metadata/species/elements signature via
    ``grid.metadata.matches_setup(...)``, and the stored composition-axis name.
    It raises ``ValueError`` on the first mismatch and returns ``None`` on success.
    Verification-related metadata is intentionally not part of compatibility.
    """
    _validate_equilibrium_grid_metadata_compatibility(
        grid.metadata,
        setup,
        preset_name,
        expected_composition_axis_name=expected_composition_axis_name,
    )


def _validate_equilibrium_grid_metadata_compatibility(
    metadata: EquilibriumGridMetadata,
    setup: ChemicalSetup,
    preset_name: str,
    *,
    expected_composition_axis_name: str = _COMPOSITION_AXIS_NAME,
    grid_label: str = "Equilibrium grid",
) -> None:
    """Validate grid metadata against a runtime chemical setup."""

    if metadata.preset_name != preset_name:
        raise ValueError(
            f"{grid_label} preset mismatch: grid uses '{metadata.preset_name}' "
            f"but runtime requested '{preset_name}'."
        )
    if metadata.preset_elements != (
        tuple(setup.elements) if setup.elements is not None else None
    ):
        raise ValueError(
            f"{grid_label} elements mismatch: grid metadata does not match "
            "the runtime setup.elements ordering/content."
        )
    if metadata.preset_species != (
        tuple(setup.species) if setup.species is not None else None
    ):
        raise ValueError(
            f"{grid_label} species mismatch: grid metadata does not match "
            "the runtime setup.species ordering/content."
        )
    if not _setup_metadata_matches(
        metadata.preset_setup_metadata,
        setup.metadata,
    ):
        raise ValueError(
            f"{grid_label} setup metadata mismatch: a field stored by the "
            "grid is missing or different in the runtime setup.metadata."
        )
    if metadata.composition_axis_name != expected_composition_axis_name:
        raise ValueError(
            f"{grid_label} composition axis mismatch: "
            f"expected '{expected_composition_axis_name}' but grid stores "
            f"'{metadata.composition_axis_name}'."
        )
    if not metadata.matches_setup(setup, preset_name):
        raise ValueError(
            f"{grid_label} preset signature mismatch: the stored preset/setup "
            "signature is not compatible with the runtime setup."
        )


def _as_scalar_query(value: float, name: str) -> Array:
    scalar = jnp.asarray(value)
    if scalar.ndim != 0:
        raise NotImplementedError(
            "EquilibriumGrid interpolation currently supports only scalar queries; "
            f"got {name} with shape {scalar.shape}."
        )
    return scalar


def _interpolate_grid_field(
    grid: EquilibriumGrid,
    field: Array,
    temperature: Array,
    pressure: Array,
    log10_z_over_z_sun: Array,
    options: EquilibriumGridInterpolationOptions,
) -> Array:
    interpolator_kwargs = dict(options.interpolator_kwargs or {})
    if "period" in interpolator_kwargs:
        raise NotImplementedError(
            "EquilibriumGrid interpolation does not expose periodic interpolation yet."
        )
    interpolator = Interpolator3D(
        grid.temperature_axis,
        grid.pressure_axis,
        grid.log10_z_over_z_sun_axis,
        field,
        method=options.method,
        extrap=options.extrap,
        **interpolator_kwargs,
    )
    interpolated = jnp.asarray(interpolator(temperature, pressure, log10_z_over_z_sun))
    if options.extrap is False:
        has_nan = jnp.any(jnp.isnan(interpolated))
        if not isinstance(has_nan, core.Tracer) and bool(has_nan):
            raise ValueError(
                "Interpolation query lies outside the stored equilibrium grid bounds. "
                "Pass EquilibriumGridInterpolationOptions(extrap=...) to opt into extrapolation."
            )
    return interpolated


def _interpolate_equilibrium_grid_field(
    grid: EquilibriumGrid,
    field: Array,
    temperature: float,
    pressure: float,
    log10_z_over_z_sun: float,
    *,
    options: Optional[EquilibriumGridInterpolationOptions] = None,
) -> Array:
    """Interpolate one field on an equilibrium grid."""

    active_options = options or EquilibriumGridInterpolationOptions()
    return _interpolate_grid_field(
        grid,
        field,
        _as_scalar_query(temperature, "temperature"),
        _as_scalar_query(pressure, "pressure"),
        _as_scalar_query(log10_z_over_z_sun, "log10_z_over_z_sun"),
        active_options,
    )


def interpolate_equilibrium_grid(
    grid: EquilibriumGrid,
    temperature: float,
    pressure: float,
    log10_z_over_z_sun: float,
    *,
    options: Optional[EquilibriumGridInterpolationOptions] = None,
) -> EquilibriumGridInterpolationResult:
    """Interpolate one equilibrium state from a stored grid.

    The query must currently be scalar in all three coordinates.
    """
    active_options = options or EquilibriumGridInterpolationOptions()
    temperature_query = _as_scalar_query(temperature, "temperature")
    pressure_query = _as_scalar_query(pressure, "pressure")
    composition_query = _as_scalar_query(
        log10_z_over_z_sun,
        "log10_z_over_z_sun",
    )
    return EquilibriumGridInterpolationResult(
        ln_n=_interpolate_grid_field(
            grid,
            grid.outputs.ln_n,
            temperature_query,
            pressure_query,
            composition_query,
            active_options,
        ),
        x=_interpolate_grid_field(
            grid,
            grid.outputs.x,
            temperature_query,
            pressure_query,
            composition_query,
            active_options,
        ),
        ntot=_interpolate_grid_field(
            grid,
            grid.outputs.ntot,
            temperature_query,
            pressure_query,
            composition_query,
            active_options,
        ),
    )
