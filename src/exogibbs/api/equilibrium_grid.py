from __future__ import annotations

from dataclasses import dataclass
from dataclasses import fields
import json
from pathlib import Path
from typing import Callable, Literal, Mapping, Optional, Sequence, Tuple, Union

import jax
import jax.numpy as jnp
from jax import core
import numpy as np
from interpax import Interpolator3D

from exogibbs.api.chemistry import ChemicalSetup
from exogibbs.api.chemistry import update_element_vector
from exogibbs.api.equilibrium import EquilibriumOptions, equilibrium
from exogibbs.api.equilibrium import EquilibriumInit
from exogibbs.io.load_data import get_data_filepath
from exogibbs.utils.elements import element_mass
from exogibbs.utils.nameparser import strip_trailing_one

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


def _freeze_setup_metadata(
    metadata: Optional[Mapping[str, str]],
) -> Optional[Mapping[str, str]]:
    if metadata is None:
        return None
    return dict(metadata)


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
            and self.preset_setup_metadata == _freeze_setup_metadata(setup.metadata)
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
    if grid.metadata.preset_name != preset_name:
        raise ValueError(
            f"Equilibrium grid preset mismatch: grid uses '{grid.metadata.preset_name}' "
            f"but runtime requested '{preset_name}'."
        )
    if grid.metadata.preset_elements != (
        tuple(setup.elements) if setup.elements is not None else None
    ):
        raise ValueError(
            "Equilibrium grid elements mismatch: grid metadata does not match "
            "the runtime setup.elements ordering/content."
        )
    if grid.metadata.preset_species != (
        tuple(setup.species) if setup.species is not None else None
    ):
        raise ValueError(
            "Equilibrium grid species mismatch: grid metadata does not match "
            "the runtime setup.species ordering/content."
        )
    if grid.metadata.preset_setup_metadata != _freeze_setup_metadata(setup.metadata):
        raise ValueError(
            "Equilibrium grid setup metadata mismatch: grid metadata does not match "
            "the runtime setup.metadata."
        )
    if grid.metadata.composition_axis_name != expected_composition_axis_name:
        raise ValueError(
            "Equilibrium grid composition axis mismatch: "
            f"expected '{expected_composition_axis_name}' but grid stores "
            f"'{grid.metadata.composition_axis_name}'."
        )
    if not grid.metadata.matches_setup(setup, preset_name):
        raise ValueError(
            "Equilibrium grid preset signature mismatch: the stored preset/setup "
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
            "EquilibriumGrid interpolation does not expose interpax periodic interpolation yet."
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
    composition_query = _as_scalar_query(log10_z_over_z_sun, "log10_z_over_z_sun")
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


def _serialize_metadata_attr(value):
    if value is None:
        return _NONE_ATTR_SENTINEL
    if isinstance(value, (tuple, list, dict)):
        return json.dumps(value)
    if isinstance(value, np.bool_):
        return bool(value)
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        return float(value)
    return value


def _parse_metadata_attr(value, field_name: str):
    if value == _NONE_ATTR_SENTINEL:
        return None
    if field_name == "preset_setup_metadata":
        return json.loads(value)
    if field_name in {"preset_elements", "preset_species"}:
        return tuple(json.loads(value))
    return value


def equilibrium_grid_to_dataset(grid: EquilibriumGrid) -> "xr.Dataset":
    """Convert an in-memory equilibrium grid into an xarray Dataset.

    The Dataset uses the dimension and coordinate names ``temperature``,
    ``pressure``, ``log10_z_over_z_sun``, and ``species``.
    """
    try:
        import xarray as xr
    except ImportError as exc:
        raise ImportError("Equilibrium grid serialization requires the optional 'xarray' package.") from exc

    species = grid.metadata.preset_species
    if species is None:
        raise ValueError("grid.metadata.preset_species is required for xarray serialization.")
    if len(species) != int(grid.outputs.ln_n.shape[-1]):
        raise ValueError(
            "grid.metadata.preset_species length must match the stored species dimension."
        )

    attrs = {
        field.name: _serialize_metadata_attr(getattr(grid.metadata, field.name))
        for field in fields(EquilibriumGridMetadata)
    }
    dataset = xr.Dataset(
        data_vars={
            "ln_n": (_GRID_SPECIES_DIMS, np.asarray(grid.outputs.ln_n)),
            "n": (_GRID_SPECIES_DIMS, np.asarray(grid.outputs.n)),
            "x": (_GRID_SPECIES_DIMS, np.asarray(grid.outputs.x)),
            "ntot": (_GRID_SCALAR_DIMS, np.asarray(grid.outputs.ntot)),
        },
        coords={
            _GRID_DIM_TEMPERATURE: np.asarray(grid.temperature_axis),
            _GRID_DIM_PRESSURE: np.asarray(grid.pressure_axis),
            _GRID_DIM_COMPOSITION: np.asarray(grid.log10_z_over_z_sun_axis),
            _GRID_DIM_SPECIES: np.asarray(species, dtype=object),
        },
        attrs=attrs,
    )
    return dataset


def _require_dataset_coord(dataset: "xr.Dataset", coord_name: str) -> None:
    if coord_name not in dataset.coords:
        raise ValueError(f"Dataset is missing required coordinate '{coord_name}'.")


def _require_dataset_var_dims(dataset: "xr.Dataset", var_name: str, expected_dims: Tuple[str, ...]) -> None:
    if var_name not in dataset.data_vars:
        raise ValueError(f"Dataset is missing required data variable '{var_name}'.")
    actual_dims = tuple(dataset[var_name].dims)
    if actual_dims != expected_dims:
        raise ValueError(
            f"Dataset variable '{var_name}' must have dims {expected_dims}, got {actual_dims}."
        )


def equilibrium_grid_from_dataset(dataset: "xr.Dataset") -> EquilibriumGrid:
    """Convert an xarray Dataset into an in-memory equilibrium grid."""
    try:
        import xarray as xr
    except ImportError as exc:
        raise ImportError("Equilibrium grid deserialization requires the optional 'xarray' package.") from exc

    if not isinstance(dataset, xr.Dataset):
        raise TypeError("dataset must be an xarray.Dataset.")

    for coord_name in (
        _GRID_DIM_TEMPERATURE,
        _GRID_DIM_PRESSURE,
        _GRID_DIM_COMPOSITION,
        _GRID_DIM_SPECIES,
    ):
        _require_dataset_coord(dataset, coord_name)

    _require_dataset_var_dims(dataset, "ln_n", _GRID_SPECIES_DIMS)
    _require_dataset_var_dims(dataset, "n", _GRID_SPECIES_DIMS)
    _require_dataset_var_dims(dataset, "x", _GRID_SPECIES_DIMS)
    _require_dataset_var_dims(dataset, "ntot", _GRID_SCALAR_DIMS)

    species_labels = tuple(str(species) for species in dataset.coords[_GRID_DIM_SPECIES].values.tolist())
    if len(species_labels) != int(dataset.sizes[_GRID_DIM_SPECIES]):
        raise ValueError("Dataset species labels must align with the stored species dimension.")
    if len(species_labels) == 0:
        raise ValueError("Dataset species coordinate must not be empty.")

    metadata_values = {}
    for field in fields(EquilibriumGridMetadata):
        if field.name not in dataset.attrs:
            raise ValueError(f"Dataset attrs are missing required metadata field '{field.name}'.")
        metadata_values[field.name] = _parse_metadata_attr(dataset.attrs[field.name], field.name)

    if tuple(metadata_values["preset_species"]) != species_labels:
        raise ValueError(
            "Dataset species coordinate does not match metadata field 'preset_species'."
        )
    if metadata_values["composition_axis_name"] != _COMPOSITION_AXIS_NAME:
        raise ValueError(
            f"Dataset attr 'composition_axis_name' must be '{_COMPOSITION_AXIS_NAME}'."
        )

    metadata = EquilibriumGridMetadata(**metadata_values)
    return EquilibriumGrid(
        temperature_axis=jnp.asarray(dataset.coords[_GRID_DIM_TEMPERATURE].values),
        pressure_axis=jnp.asarray(dataset.coords[_GRID_DIM_PRESSURE].values),
        log10_z_over_z_sun_axis=jnp.asarray(dataset.coords[_GRID_DIM_COMPOSITION].values),
        outputs=EquilibriumGridOutputs(
            ln_n=jnp.asarray(dataset["ln_n"].values),
            n=jnp.asarray(dataset["n"].values),
            x=jnp.asarray(dataset["x"].values),
            ntot=jnp.asarray(dataset["ntot"].values),
        ),
        metadata=metadata,
    )


def save_equilibrium_grid_netcdf(grid: EquilibriumGrid, path: str) -> None:
    """Save an equilibrium grid to NetCDF via xarray."""
    dataset = equilibrium_grid_to_dataset(grid)
    dataset.to_netcdf(Path(path), engine="scipy")


def load_equilibrium_grid_netcdf(path: str) -> EquilibriumGrid:
    """Load an equilibrium grid from a NetCDF file via xarray."""
    try:
        import xarray as xr
    except ImportError as exc:
        raise ImportError("Equilibrium grid deserialization requires the optional 'xarray' package.") from exc

    with xr.open_dataset(Path(path), engine="scipy") as dataset:
        return equilibrium_grid_from_dataset(dataset.load())


def _resolve_preset_builder(preset_name: str) -> Callable[[], ChemicalSetup]:
    if preset_name == "ykb4":
        from exogibbs.presets.ykb4 import chemsetup

        return chemsetup
    if preset_name == "fastchem":
        from exogibbs.presets.fastchem import chemsetup

        return lambda: chemsetup(silent=True)
    raise ValueError(f"Unknown preset '{preset_name}'. Expected one of ('ykb4', 'fastchem').")


def _build_grid_outputs(
    temperature_axis: Array,
    pressure_axis: Array,
    log10_z_over_z_sun_axis: Array,
    solve_point: Callable[[float, float, float], Tuple[Array, Array, Array, Array]],
) -> EquilibriumGridOutputs:
    ln_n_slices = []
    n_slices = []
    x_slices = []
    ntot_slices = []
    for temperature in temperature_axis:
        ln_n_pressure = []
        n_pressure = []
        x_pressure = []
        ntot_pressure = []
        for pressure in pressure_axis:
            ln_n_composition = []
            n_composition = []
            x_composition = []
            ntot_composition = []
            for log10_z_over_z_sun in log10_z_over_z_sun_axis:
                ln_n, n, x, ntot = solve_point(
                    float(temperature),
                    float(pressure),
                    float(log10_z_over_z_sun),
                )
                ln_n_composition.append(ln_n)
                n_composition.append(n)
                x_composition.append(x)
                ntot_composition.append(ntot)
            ln_n_pressure.append(jnp.stack(ln_n_composition, axis=0))
            n_pressure.append(jnp.stack(n_composition, axis=0))
            x_pressure.append(jnp.stack(x_composition, axis=0))
            ntot_pressure.append(jnp.stack(ntot_composition, axis=0))
        ln_n_slices.append(jnp.stack(ln_n_pressure, axis=0))
        n_slices.append(jnp.stack(n_pressure, axis=0))
        x_slices.append(jnp.stack(x_pressure, axis=0))
        ntot_slices.append(jnp.stack(ntot_pressure, axis=0))

    return EquilibriumGridOutputs(
        ln_n=jnp.stack(ln_n_slices, axis=0),
        n=jnp.stack(n_slices, axis=0),
        x=jnp.stack(x_slices, axis=0),
        ntot=jnp.stack(ntot_slices, axis=0),
    )


def _build_fastchem_species_indices(
    setup: ChemicalSetup,
    fastchem: "pyfastchem.FastChem",
) -> Sequence[int]:
    import pyfastchem

    if setup.species is None:
        raise ValueError("setup.species is required for FastChem-backed grid generation.")

    species_indices = []
    for species in setup.species:
        fastchem_index = fastchem.getGasSpeciesIndex(species)
        if fastchem_index == pyfastchem.FASTCHEM_UNKNOWN_SPECIES:
            fastchem_index = fastchem.getGasSpeciesIndex(strip_trailing_one(species))
        if fastchem_index == pyfastchem.FASTCHEM_UNKNOWN_SPECIES:
            raise NotImplementedError(
                f"FastChem-backed grid generation cannot align species '{species}' "
                f"for the current preset."
            )
        species_indices.append(fastchem_index)
    return species_indices


def _map_element_vector_to_fastchem_order(
    setup: ChemicalSetup,
    fastchem: "pyfastchem.FastChem",
    element_vector: Array,
) -> Sequence[float]:
    if setup.elements is None:
        raise ValueError("setup.elements is required for FastChem-backed grid generation.")

    setup_element_positions = {element: i for i, element in enumerate(setup.elements)}
    return [
        float(element_vector[setup_element_positions[fastchem.getElementSymbol(i)]])
        for i in range(fastchem.getElementNumber())
    ]


def _build_fastchem_outputs(
    setup: ChemicalSetup,
    temperature_axis: Array,
    pressure_axis: Array,
    log10_z_over_z_sun_axis: Array,
) -> EquilibriumGridOutputs:
    solve_point_fastchem, _ = _create_fastchem_point_solver(setup)
    return _build_grid_outputs(
        temperature_axis,
        pressure_axis,
        log10_z_over_z_sun_axis,
        lambda temperature, pressure, log10_z_over_z_sun: solve_point_fastchem(
            temperature,
            pressure,
            log10_z_over_z_sun,
        ),
    )


def _create_fastchem_point_solver(
    setup: ChemicalSetup,
) -> Tuple[
    Callable[[float, float, float], Tuple[Array, Array, Array, Array]],
    Sequence[int],
]:
    try:
        import pyfastchem
    except ImportError as exc:
        raise ImportError(
            "FastChem-backed grid generation requires the optional 'pyfastchem' package."
        ) from exc

    if setup.metadata is None or "fastchem" not in setup.metadata.get("source", "").lower():
        raise NotImplementedError(
            "FastChem-backed grid generation currently supports only the FastChem preset."
        )

    metadata = setup.metadata or {}
    fastchem_element_file = metadata.get(
        "fastchem_element_file",
        "fastchem/element_abundances/asplund_2020.dat",
    )
    fastchem_logk_file = metadata.get(
        "fastchem_logk_file",
        "fastchem/logK/logK.dat",
    )
    fastchem = pyfastchem.FastChem(
        str(get_data_filepath(fastchem_element_file)),
        str(get_data_filepath(fastchem_logk_file)),
        1,
    )
    fastchem.setVerboseLevel(0)
    species_indices = _build_fastchem_species_indices(setup, fastchem)

    def solve_point(
        temperature: float,
        pressure: float,
        log10_z_over_z_sun: float,
    ) -> Tuple[Array, Array, Array, Array]:
        element_vector = build_h_he_element_vector_from_log10_z_over_z_sun(
            setup,
            log10_z_over_z_sun,
        )
        fastchem.setElementAbundances(
            _map_element_vector_to_fastchem_order(setup, fastchem, element_vector)
        )
        input_data = pyfastchem.FastChemInput()
        output_data = pyfastchem.FastChemOutput()
        input_data.temperature = np.asarray([temperature], dtype=float)
        input_data.pressure = np.asarray([pressure], dtype=float)
        fastchem_flag = fastchem.calcDensities(input_data, output_data)
        if fastchem_flag != pyfastchem.FASTCHEM_SUCCESS:
            raise RuntimeError(
                "FastChem grid-point solve failed at "
                f"T={temperature}, P={pressure}, log10(Z/Zsun)={log10_z_over_z_sun}: "
                f"{pyfastchem.FASTCHEM_MSG[fastchem_flag]}"
            )

        n = jnp.asarray(np.asarray(output_data.number_densities, dtype=float)[0][species_indices])
        ntot = jnp.asarray(jnp.sum(n))
        x = n / jnp.clip(ntot, 1e-300)
        ln_n = jnp.log(jnp.clip(n, 1e-300))
        return ln_n, n, x, ntot

    return solve_point, species_indices


def _verify_exogibbs_grid_against_fastchem(
    setup: ChemicalSetup,
    temperature_axis: Array,
    pressure_axis: Array,
    log10_z_over_z_sun_axis: Array,
    exogibbs_outputs: EquilibriumGridOutputs,
    *,
    abundance_floor: float,
    tolerance_percent: float,
) -> Mapping[str, float]:
    if setup.metadata is None or "fastchem" not in setup.metadata.get("source", "").lower():
        raise NotImplementedError(
            "ExoGibbs-vs-FastChem verification currently supports only the FastChem preset."
        )

    solve_point_fastchem, _ = _create_fastchem_point_solver(setup)
    max_abs_percent_deviation = 0.0
    worst_temperature = None
    worst_pressure = None
    worst_log10_z_over_z_sun = None
    worst_species_index = None
    worst_species_name = None
    included_species_total = 0
    points_checked = 0

    for itemperature, temperature in enumerate(temperature_axis):
        for ipressure, pressure in enumerate(pressure_axis):
            for icomposition, log10_z_over_z_sun in enumerate(log10_z_over_z_sun_axis):
                _, _, fastchem_x, _ = solve_point_fastchem(
                    float(temperature),
                    float(pressure),
                    float(log10_z_over_z_sun),
                )
                exogibbs_x = exogibbs_outputs.x[itemperature, ipressure, icomposition]
                included_mask = jnp.maximum(fastchem_x, exogibbs_x) >= abundance_floor
                included_species = int(jnp.sum(included_mask))
                if included_species == 0:
                    points_checked += 1
                    continue

                relative_deviation = (
                    fastchem_x[included_mask]
                    / jnp.clip(exogibbs_x[included_mask], abundance_floor, None)
                    - 1.0
                )
                percent_deviation = 100.0 * relative_deviation
                abs_percent_deviation = jnp.abs(percent_deviation)
                point_max_index = int(jnp.argmax(abs_percent_deviation))
                point_max = float(abs_percent_deviation[point_max_index])
                if point_max > max_abs_percent_deviation:
                    max_abs_percent_deviation = point_max
                    worst_temperature = float(temperature)
                    worst_pressure = float(pressure)
                    worst_log10_z_over_z_sun = float(log10_z_over_z_sun)
                    included_species_indices = np.flatnonzero(np.asarray(included_mask))
                    worst_species_index = int(included_species_indices[point_max_index])
                    if setup.species is not None:
                        worst_species_name = str(setup.species[worst_species_index])
                included_species_total += included_species
                points_checked += 1

    verification_passed = max_abs_percent_deviation <= tolerance_percent
    return {
        "verification_abundance_floor": abundance_floor,
        "verification_tolerance_percent": tolerance_percent,
        "verification_points_checked": points_checked,
        "verification_species_compared": included_species_total,
        "verification_max_abs_percent_deviation": max_abs_percent_deviation,
        "verification_worst_temperature": worst_temperature,
        "verification_worst_pressure": worst_pressure,
        "verification_worst_log10_z_over_z_sun": worst_log10_z_over_z_sun,
        "verification_worst_species_index": worst_species_index,
        "verification_worst_species_name": worst_species_name,
        "verification_passed": verification_passed,
    }


def build_h_he_element_vector_from_log10_z_over_z_sun(
    setup: ChemicalSetup,
    log10_z_over_z_sun: float,
) -> Array:
    """Build the solver elemental abundance vector for an H/He atmosphere metallicity.

    The input ``log10_z_over_z_sun`` is interpreted as ``m = log10(Z/Zsun)``.
    The solver input vector is constructed from ``setup.element_vector_reference`` by
    solving for the uniform metal abundance scaling that yields the target physical
    metal mass fraction, while leaving H and He unchanged and forcing the electron
    abundance to zero when present.
    """
    b_ref = _require_h_he_reference_abundance_setup(setup)
    metallicity_scale = _h_he_metallicity_scale_from_log10_z_over_z_sun(
        setup,
        log10_z_over_z_sun,
    )
    metal_indices = jnp.asarray(
        [i for i, element in enumerate(setup.elements) if element not in {"H", "He", "e-"}],
        dtype=jnp.int32,
    )
    set_indices = None
    set_values = None
    if "e-" in setup.elements:
        set_indices = jnp.asarray([setup.elements.index("e-")], dtype=jnp.int32)
        set_values = jnp.asarray([0.0], dtype=b_ref.dtype)

    return update_element_vector(
        b_ref,
        scale_indices=metal_indices,
        scales=jnp.full(metal_indices.shape, metallicity_scale, dtype=b_ref.dtype),
        set_indices=set_indices,
        set_values=set_values,
    )


def build_equilibrium_grid(
    preset_name: str,
    temperature_axis: Array,
    pressure_axis: Array,
    log10_z_over_z_sun_axis: Array,
    *,
    source: EquilibriumGridSource = "exogibbs",
    options: Optional[EquilibriumOptions] = None,
    verify_exogibbs_against_fastchem: bool = True,
    verification_abundance_floor: float = _FASTCHEM_COMPARISON_ABUNDANCE_FLOOR,
    verification_tolerance_percent: float = _FASTCHEM_COMPARISON_TOLERANCE_PERCENT,
    setup_builder: Optional[Callable[[], ChemicalSetup]] = None,
) -> EquilibriumGrid:
    """Generate an in-memory equilibrium grid for a preset and source.

    The composition axis is explicitly ``log10(Z/Zsun)`` for an H/He atmosphere.
    When ``source="exogibbs"``, verification against FastChem at the same grid
    points is enabled by default for supported presets.
    """
    setup = setup_builder() if setup_builder is not None else _resolve_preset_builder(preset_name)()
    opts = options or EquilibriumOptions()
    temperature_axis = jnp.asarray(temperature_axis)
    pressure_axis = jnp.asarray(pressure_axis)
    log10_z_over_z_sun_axis = jnp.asarray(log10_z_over_z_sun_axis)
    verification_results = {}
    if source == "exogibbs":
        def solve_point(
            temperature: float,
            pressure: float,
            log10_z_over_z_sun: float,
        ) -> Tuple[Array, Array, Array, Array]:
            element_vector = build_h_he_element_vector_from_log10_z_over_z_sun(
                setup,
                log10_z_over_z_sun,
            )
            result = equilibrium(
                setup,
                temperature,
                pressure,
                element_vector,
                options=opts,
            )
            return result.ln_n, result.n, result.x, result.ntot

        outputs = _build_grid_outputs(
            temperature_axis,
            pressure_axis,
            log10_z_over_z_sun_axis,
            solve_point,
        )
        if verify_exogibbs_against_fastchem:
            verification_results = _verify_exogibbs_grid_against_fastchem(
                setup,
                temperature_axis,
                pressure_axis,
                log10_z_over_z_sun_axis,
                outputs,
                abundance_floor=verification_abundance_floor,
                tolerance_percent=verification_tolerance_percent,
            )
            if not verification_results["verification_passed"]:
                species_detail = ""
                if verification_results.get("verification_worst_species_name") is not None:
                    species_detail = (
                        f", species={verification_results['verification_worst_species_name']}"
                    )
                elif verification_results.get("verification_worst_species_index") is not None:
                    species_detail = (
                        f", species_index={verification_results['verification_worst_species_index']}"
                    )
                raise ValueError(
                    "ExoGibbs grid verification against FastChem failed: "
                    f"max abs percent deviation "
                    f"{verification_results['verification_max_abs_percent_deviation']:.6g}% "
                    f"exceeds tolerance {verification_tolerance_percent:.6g}% "
                    f"at T={verification_results['verification_worst_temperature']:.6g} K, "
                    f"P={verification_results['verification_worst_pressure']:.6g} bar, "
                    f"log10(Z/Zsun)="
                    f"{verification_results['verification_worst_log10_z_over_z_sun']:.6g}"
                    f"{species_detail}."
                    f"{_verification_dtype_warning()}"
                )
    elif source == "fastchem":
        outputs = _build_fastchem_outputs(
            setup,
            temperature_axis,
            pressure_axis,
            log10_z_over_z_sun_axis,
        )
        verify_exogibbs_against_fastchem = False
    else:
        raise ValueError(f"Unknown source '{source}'. Expected one of ('exogibbs', 'fastchem').")

    metadata = EquilibriumGridMetadata.from_setup(
        setup,
        preset_name=preset_name,
        source=source,
        exogibbs_epsilon_crit=opts.epsilon_crit,
        exogibbs_max_iter=opts.max_iter,
        verify_exogibbs_against_fastchem=verify_exogibbs_against_fastchem,
        verification_abundance_floor=verification_results.get("verification_abundance_floor"),
        verification_tolerance_percent=verification_results.get("verification_tolerance_percent"),
        verification_points_checked=verification_results.get("verification_points_checked"),
        verification_species_compared=verification_results.get("verification_species_compared"),
        verification_max_abs_percent_deviation=verification_results.get(
            "verification_max_abs_percent_deviation"
        ),
        verification_worst_temperature=verification_results.get("verification_worst_temperature"),
        verification_worst_pressure=verification_results.get("verification_worst_pressure"),
        verification_worst_log10_z_over_z_sun=verification_results.get(
            "verification_worst_log10_z_over_z_sun"
        ),
        verification_worst_species_index=verification_results.get(
            "verification_worst_species_index"
        ),
        verification_worst_species_name=verification_results.get(
            "verification_worst_species_name"
        ),
        verification_passed=verification_results.get("verification_passed"),
    )
    return EquilibriumGrid(
        temperature_axis=temperature_axis,
        pressure_axis=pressure_axis,
        log10_z_over_z_sun_axis=log10_z_over_z_sun_axis,
        outputs=outputs,
        metadata=metadata,
    )
