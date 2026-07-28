"""Typed gas-equilibrium grid models."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal, Mapping, Optional, Tuple, Union

import jax
import jax.numpy as jnp
import numpy as np

from exogibbs.equilibrium.gas.types import EquilibriumInit
from exogibbs.thermo.models import ChemicalSetup


Array = jax.Array
EquilibriumGridSource = Literal["exogibbs", "fastchem"]
_COMPOSITION_AXIS_NAME = "log10(Z/Zsun)"
_COMPOSITION_AXIS_DEFINITION = (
    "H/He atmosphere metallicity axis. For each grid value m = log10(Z/Zsun), "
    "the solver input element vector is built from the preset reference abundance "
    "vector by solving for the uniform non-H, non-He, non-electron abundance scaling "
    "that yields the target physical metal mass fraction Z = Zsun * 10**m, while "
    "keeping H and He at their preset reference abundances and setting electrons to 0."
)
_FASTCHEM_COMPARISON_ABUNDANCE_FLOOR = 1.0e-10
_FASTCHEM_COMPARISON_TOLERANCE_PERCENT = 0.5
_NONE_ATTR_SENTINEL = "__none__"
_GRID_DIM_TEMPERATURE = "temperature"
_GRID_DIM_PRESSURE = "pressure"
_GRID_DIM_COMPOSITION = "log10_z_over_z_sun"
_GRID_DIM_SPECIES = "species"
_GRID_SPECIES_DIMS = (
    _GRID_DIM_TEMPERATURE,
    _GRID_DIM_PRESSURE,
    _GRID_DIM_COMPOSITION,
    _GRID_DIM_SPECIES,
)
_GRID_SCALAR_DIMS = (
    _GRID_DIM_TEMPERATURE,
    _GRID_DIM_PRESSURE,
    _GRID_DIM_COMPOSITION,
)
_LEGACY_RUNTIME_ADDITIONAL_SETUP_METADATA_KEYS = frozenset(
    {
        "fastchem_hvector_logk_source_trace",
        "fastchem_hvector_logk_source_trace_function",
        "fastchem_logk_source_records",
    }
)
def _metadata_json_safe(value: Any) -> Any:
    """Return a deterministic JSON-safe metadata value."""

    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, np.bool_):
        return bool(value)
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        return float(value)
    if callable(value):
        module = getattr(value, "__module__", value.__class__.__module__)
        qualname = getattr(value, "__qualname__", value.__class__.__qualname__)
        return f"{module}.{qualname}"
    if isinstance(value, Mapping):
        return {str(key): _metadata_json_safe(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_metadata_json_safe(item) for item in value]
    return repr(value)


def _freeze_setup_metadata(
    metadata: Optional[Mapping[str, Any]],
) -> Optional[Mapping[str, Any]]:
    if metadata is None:
        return None
    return {str(key): _metadata_json_safe(value) for key, value in metadata.items()}


def _setup_metadata_matches(
    stored_metadata: Optional[Mapping[str, Any]],
    runtime_metadata: Optional[Mapping[str, Any]],
) -> bool:
    """Match every stored setup field while allowing newer runtime fields.

    Older packaged grids predate detailed source-record trace metadata.  A
    runtime setup may therefore add those known trace fields that the grid
    could not store, but it may not add other fields or remove or change any
    field that the grid did store.  Newly generated grids retain the detailed
    trace fields and consequently compare them as part of this same contract.
    """

    stored = _freeze_setup_metadata(stored_metadata)
    runtime = _freeze_setup_metadata(runtime_metadata)
    if stored is None or runtime is None:
        return stored == runtime
    stored_fields_match = all(
        key in runtime and runtime[key] == value
        for key, value in stored.items()
    )
    additional_runtime_keys = set(runtime) - set(stored)
    return stored_fields_match and additional_runtime_keys.issubset(
        _LEGACY_RUNTIME_ADDITIONAL_SETUP_METADATA_KEYS
    )


@dataclass(frozen=True)
class EquilibriumGridOutputs:
    """Equilibrium fields stored on a T/P/composition grid.

    ``ln_n``, ``n``, and ``x`` are stored per species on the full
    ``(temperature, pressure, log10(Z/Zsun), species)`` grid.
    ``ntot`` is stored on the corresponding ``(temperature, pressure, log10(Z/Zsun))``
    grid.
    """

    ln_n: Array
    n: Array
    x: Array
    ntot: Array


@dataclass(frozen=True)
class EquilibriumGridMetadata:
    """Host-side provenance for a generated equilibrium grid."""

    preset_name: str
    preset_setup_metadata: Optional[Mapping[str, str]]
    preset_elements: Optional[Tuple[str, ...]]
    preset_species: Optional[Tuple[str, ...]]
    source: EquilibriumGridSource
    composition_axis_name: str = _COMPOSITION_AXIS_NAME
    composition_axis_definition: str = _COMPOSITION_AXIS_DEFINITION
    exogibbs_epsilon_crit: Optional[float] = None
    exogibbs_max_iter: Optional[int] = None
    verify_exogibbs_against_fastchem: bool = True
    verification_abundance_floor: Optional[float] = None
    verification_tolerance_percent: Optional[float] = None
    verification_points_checked: Optional[int] = None
    verification_species_compared: Optional[int] = None
    verification_max_abs_percent_deviation: Optional[float] = None
    verification_worst_temperature: Optional[float] = None
    verification_worst_pressure: Optional[float] = None
    verification_worst_log10_z_over_z_sun: Optional[float] = None
    verification_worst_species_index: Optional[int] = None
    verification_worst_species_name: Optional[str] = None
    verification_passed: Optional[bool] = None

    @classmethod
    def from_setup(
        cls,
        setup: ChemicalSetup,
        preset_name: str,
        source: EquilibriumGridSource,
        *,
        exogibbs_epsilon_crit: Optional[float] = None,
        exogibbs_max_iter: Optional[int] = None,
        verify_exogibbs_against_fastchem: bool = True,
        verification_abundance_floor: Optional[float] = None,
        verification_tolerance_percent: Optional[float] = None,
        verification_points_checked: Optional[int] = None,
        verification_species_compared: Optional[int] = None,
        verification_max_abs_percent_deviation: Optional[float] = None,
        verification_worst_temperature: Optional[float] = None,
        verification_worst_pressure: Optional[float] = None,
        verification_worst_log10_z_over_z_sun: Optional[float] = None,
        verification_worst_species_index: Optional[int] = None,
        verification_worst_species_name: Optional[str] = None,
        verification_passed: Optional[bool] = None,
    ) -> "EquilibriumGridMetadata":
        return cls(
            preset_name=preset_name,
            preset_setup_metadata=_freeze_setup_metadata(setup.metadata),
            preset_elements=tuple(setup.elements) if setup.elements is not None else None,
            preset_species=tuple(setup.species) if setup.species is not None else None,
            source=source,
            exogibbs_epsilon_crit=exogibbs_epsilon_crit,
            exogibbs_max_iter=exogibbs_max_iter,
            verify_exogibbs_against_fastchem=verify_exogibbs_against_fastchem,
            verification_abundance_floor=verification_abundance_floor,
            verification_tolerance_percent=verification_tolerance_percent,
            verification_points_checked=verification_points_checked,
            verification_species_compared=verification_species_compared,
            verification_max_abs_percent_deviation=verification_max_abs_percent_deviation,
            verification_worst_temperature=verification_worst_temperature,
            verification_worst_pressure=verification_worst_pressure,
            verification_worst_log10_z_over_z_sun=verification_worst_log10_z_over_z_sun,
            verification_worst_species_index=verification_worst_species_index,
            verification_worst_species_name=verification_worst_species_name,
            verification_passed=verification_passed,
        )

    def matches_setup(self, setup: ChemicalSetup, preset_name: str) -> bool:
        """Return True when a runtime setup matches this grid's preset signature."""
        return (
            self.preset_name == preset_name
            and _setup_metadata_matches(
                self.preset_setup_metadata,
                setup.metadata,
            )
            and self.preset_elements == (tuple(setup.elements) if setup.elements is not None else None)
            and self.preset_species == (tuple(setup.species) if setup.species is not None else None)
        )


@dataclass(frozen=True)
class EquilibriumGrid:
    """Minimal in-memory equilibrium grid container."""

    temperature_axis: Array
    pressure_axis: Array
    log10_z_over_z_sun_axis: Array
    outputs: EquilibriumGridOutputs
    metadata: EquilibriumGridMetadata

    def interpolate(
        self,
        temperature: float,
        pressure: float,
        log10_z_over_z_sun: float,
        *,
        options: Optional["EquilibriumGridInterpolationOptions"] = None,
    ) -> "EquilibriumGridInterpolationResult":
        """Interpolate stored equilibrium fields at one grid point."""

        from exogibbs.equilibrium.gas.grid.interpolation import (
            interpolate_equilibrium_grid,
        )

        return interpolate_equilibrium_grid(
            self,
            temperature,
            pressure,
            log10_z_over_z_sun,
            options=options,
        )


@dataclass(frozen=True)
class EquilibriumGridInterpolationOptions:
    """Minimal interpolation controls for ``EquilibriumGrid`` lookups."""

    method: str = "linear"
    extrap: Union[bool, float, Tuple[object, ...]] = False
    interpolator_kwargs: Optional[Mapping[str, object]] = None


@dataclass(frozen=True)
class EquilibriumGridInterpolationResult:
    """Interpolated equilibrium state at one ``(T, P, log10(Z/Zsun))`` query."""

    ln_n: Array
    x: Array
    ntot: Array

    @property
    def ln_ntot(self) -> Array:
        return jnp.log(jnp.clip(self.ntot, 1e-300))

    def to_equilibrium_init(self) -> EquilibriumInit:
        return EquilibriumInit(
            ln_nk=jnp.asarray(self.ln_n),
            ln_ntot=jnp.asarray(self.ln_ntot),
        )
