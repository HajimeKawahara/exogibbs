"""Dataset and NetCDF storage for gas-equilibrium grids."""

from __future__ import annotations

from dataclasses import fields
import json
from pathlib import Path
from typing import TYPE_CHECKING, Tuple

import jax.numpy as jnp
import numpy as np

from exogibbs.equilibrium.gas.grid.types import (
    _COMPOSITION_AXIS_NAME,
    _GRID_DIM_COMPOSITION,
    _GRID_DIM_PRESSURE,
    _GRID_DIM_SPECIES,
    _GRID_DIM_TEMPERATURE,
    _GRID_SCALAR_DIMS,
    _GRID_SPECIES_DIMS,
    _NONE_ATTR_SENTINEL,
    EquilibriumGrid,
    EquilibriumGridMetadata,
    EquilibriumGridOutputs,
)

if TYPE_CHECKING:
    import xarray as xr


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
