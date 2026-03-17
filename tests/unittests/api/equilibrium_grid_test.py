import jax.numpy as jnp
import xarray as xr

from exogibbs.api import (
    ChemicalSetup,
    EquilibriumGrid,
    EquilibriumGridMetadata,
    EquilibriumGridOutputs,
    build_equilibrium_grid,
    build_h_he_element_vector_from_log10_z_over_z_sun,
    equilibrium_grid_from_dataset,
    equilibrium_grid_to_dataset,
    load_equilibrium_grid_netcdf,
    save_equilibrium_grid_netcdf,
    validate_equilibrium_grid_compatibility,
)


def test_equilibrium_grid_metadata_from_setup_captures_preset_and_settings():
    setup = ChemicalSetup(
        formula_matrix=jnp.ones((2, 3)),
        hvector_func=lambda T: jnp.asarray([T, T, T]),
        elements=("H", "He"),
        species=("H2", "He", "H"),
        metadata={"source": "JANAF", "dataset": "gas"},
    )

    metadata = EquilibriumGridMetadata.from_setup(
        setup,
        preset_name="ykb4",
        source="exogibbs",
        exogibbs_epsilon_crit=1.0e-15,
        exogibbs_max_iter=1000,
    )

    assert metadata.preset_name == "ykb4"
    assert metadata.preset_setup_metadata == {"source": "JANAF", "dataset": "gas"}
    assert metadata.preset_elements == ("H", "He")
    assert metadata.preset_species == ("H2", "He", "H")
    assert metadata.source == "exogibbs"
    assert metadata.composition_axis_name == "log10(Z/Zsun)"
    assert "10**m" in metadata.composition_axis_definition
    assert metadata.exogibbs_epsilon_crit == 1.0e-15
    assert metadata.exogibbs_max_iter == 1000
    assert metadata.verify_exogibbs_against_fastchem is True
    assert metadata.verification_abundance_floor is None
    assert metadata.verification_tolerance_percent is None
    assert metadata.verification_passed is None


def test_equilibrium_grid_metadata_matches_setup_checks_preset_signature():
    setup = ChemicalSetup(
        formula_matrix=jnp.ones((2, 2)),
        hvector_func=lambda T: jnp.asarray([T, T]),
        elements=("H", "He"),
        species=("H2", "He"),
        metadata={"source": "fastchem v3.1.3", "dataset": "gas"},
    )
    mismatched_setup = ChemicalSetup(
        formula_matrix=jnp.ones((2, 2)),
        hvector_func=lambda T: jnp.asarray([T, T]),
        elements=("H", "He"),
        species=("H2", "He", "H"),
        metadata={"source": "fastchem v3.1.3", "dataset": "gas"},
    )

    metadata = EquilibriumGridMetadata.from_setup(
        setup,
        preset_name="fastchem",
        source="fastchem",
        verify_exogibbs_against_fastchem=False,
    )

    assert metadata.matches_setup(setup, "fastchem")
    assert not metadata.matches_setup(setup, "ykb4")
    assert not metadata.matches_setup(mismatched_setup, "fastchem")


def test_equilibrium_grid_container_holds_axes_outputs_and_metadata():
    outputs = EquilibriumGridOutputs(
        ln_n=jnp.zeros((2, 3, 4, 5)),
        n=jnp.ones((2, 3, 4, 5)),
        x=jnp.full((2, 3, 4, 5), 0.2),
        ntot=jnp.ones((2, 3, 4)),
    )
    metadata = EquilibriumGridMetadata(
        preset_name="ykb4",
        preset_setup_metadata={"source": "JANAF", "dataset": "gas"},
        preset_elements=("H", "He"),
        preset_species=("H2", "He"),
        source="fastchem",
        verify_exogibbs_against_fastchem=True,
    )

    grid = EquilibriumGrid(
        temperature_axis=jnp.asarray([300.0, 600.0]),
        pressure_axis=jnp.asarray([1.0e-6, 1.0, 1.0e2]),
        log10_z_over_z_sun_axis=jnp.asarray([-1.0, 0.0, 0.5, 1.0]),
        outputs=outputs,
        metadata=metadata,
    )

    assert grid.temperature_axis.shape == (2,)
    assert grid.pressure_axis.shape == (3,)
    assert grid.log10_z_over_z_sun_axis.shape == (4,)
    assert grid.outputs.ln_n.shape == (2, 3, 4, 5)
    assert grid.outputs.ntot.shape == (2, 3, 4)
    assert grid.metadata.source == "fastchem"


def test_build_h_he_element_vector_from_log10_z_over_z_sun_scales_metals_only():
    setup = ChemicalSetup(
        formula_matrix=jnp.ones((4, 2)),
        hvector_func=lambda T: jnp.asarray([T, T]),
        elements=("H", "He", "O", "e-"),
        element_vector_reference=jnp.asarray([0.9, 0.09, 0.01, 1.0e-8]),
    )

    b = build_h_he_element_vector_from_log10_z_over_z_sun(setup, 2.0)

    assert jnp.allclose(b, jnp.asarray([0.9, 0.09, 1.0, 0.0]))


def test_build_equilibrium_grid_exogibbs_path_returns_grid():
    grid = build_equilibrium_grid(
        "ykb4",
        temperature_axis=jnp.asarray([500.0]),
        pressure_axis=jnp.asarray([1.0]),
        log10_z_over_z_sun_axis=jnp.asarray([0.0]),
        source="exogibbs",
        verify_exogibbs_against_fastchem=False,
    )

    assert grid.temperature_axis.shape == (1,)
    assert grid.pressure_axis.shape == (1,)
    assert grid.log10_z_over_z_sun_axis.shape == (1,)
    assert grid.outputs.ln_n.ndim == 4
    assert grid.outputs.n.ndim == 4
    assert grid.outputs.x.ndim == 4
    assert grid.outputs.ntot.ndim == 3
    assert grid.metadata.preset_name == "ykb4"
    assert grid.metadata.source == "exogibbs"
    assert grid.metadata.verify_exogibbs_against_fastchem is False
    assert grid.metadata.verification_passed is None


def test_build_equilibrium_grid_exogibbs_fastchem_preset_verifies_by_default():
    grid = build_equilibrium_grid(
        "fastchem",
        temperature_axis=jnp.asarray([1000.0]),
        pressure_axis=jnp.asarray([1.0]),
        log10_z_over_z_sun_axis=jnp.asarray([0.0]),
        source="exogibbs",
    )

    assert grid.metadata.source == "exogibbs"
    assert grid.metadata.verify_exogibbs_against_fastchem is True
    assert grid.metadata.verification_abundance_floor == 1.0e-10
    assert grid.metadata.verification_tolerance_percent == 0.5
    assert grid.metadata.verification_points_checked == 1
    assert grid.metadata.verification_species_compared is not None
    assert grid.metadata.verification_species_compared > 0
    assert grid.metadata.verification_max_abs_percent_deviation is not None
    assert grid.metadata.verification_passed is True


def test_build_equilibrium_grid_fastchem_path_returns_grid():
    grid = build_equilibrium_grid(
        "fastchem",
        temperature_axis=jnp.asarray([1000.0]),
        pressure_axis=jnp.asarray([1.0]),
        log10_z_over_z_sun_axis=jnp.asarray([0.0]),
        source="fastchem",
    )

    assert grid.temperature_axis.shape == (1,)
    assert grid.pressure_axis.shape == (1,)
    assert grid.log10_z_over_z_sun_axis.shape == (1,)
    assert grid.outputs.ln_n.ndim == 4
    assert grid.outputs.n.ndim == 4
    assert grid.outputs.x.ndim == 4
    assert grid.outputs.ntot.ndim == 3
    assert grid.metadata.preset_name == "fastchem"
    assert grid.metadata.source == "fastchem"
    assert grid.metadata.verify_exogibbs_against_fastchem is False
    assert grid.metadata.verification_passed is None


def test_build_equilibrium_grid_fastchem_source_rejects_non_fastchem_preset():
    try:
        build_equilibrium_grid(
            "ykb4",
            temperature_axis=jnp.asarray([500.0]),
            pressure_axis=jnp.asarray([1.0]),
            log10_z_over_z_sun_axis=jnp.asarray([0.0]),
            source="fastchem",
        )
    except NotImplementedError as exc:
        assert "FastChem preset" in str(exc)
    else:
        raise AssertionError("Expected NotImplementedError for non-FastChem preset.")


def test_build_equilibrium_grid_exogibbs_verification_rejects_non_fastchem_preset():
    try:
        build_equilibrium_grid(
            "ykb4",
            temperature_axis=jnp.asarray([500.0]),
            pressure_axis=jnp.asarray([1.0]),
            log10_z_over_z_sun_axis=jnp.asarray([0.0]),
            source="exogibbs",
        )
    except NotImplementedError as exc:
        assert "FastChem preset" in str(exc)
    else:
        raise AssertionError("Expected NotImplementedError for unsupported verification preset.")


def test_equilibrium_grid_xarray_roundtrip_preserves_axes_species_and_metadata(tmp_path):
    grid = build_equilibrium_grid(
        "fastchem",
        temperature_axis=jnp.asarray([1000.0]),
        pressure_axis=jnp.asarray([1.0]),
        log10_z_over_z_sun_axis=jnp.asarray([0.0]),
        source="exogibbs",
    )

    dataset = equilibrium_grid_to_dataset(grid)

    assert tuple(dataset.coords) == ("temperature", "pressure", "log10_z_over_z_sun", "species")
    assert dataset["ln_n"].dims == ("temperature", "pressure", "log10_z_over_z_sun", "species")
    assert dataset["n"].dims == ("temperature", "pressure", "log10_z_over_z_sun", "species")
    assert dataset["x"].dims == ("temperature", "pressure", "log10_z_over_z_sun", "species")
    assert dataset["ntot"].dims == ("temperature", "pressure", "log10_z_over_z_sun")
    assert dataset.attrs["preset_name"] == "fastchem"
    assert dataset.attrs["source"] == "exogibbs"
    assert dataset.attrs["composition_axis_name"] == "log10(Z/Zsun)"

    path = tmp_path / "grid.nc"
    save_equilibrium_grid_netcdf(grid, str(path))
    loaded = load_equilibrium_grid_netcdf(str(path))

    assert jnp.allclose(loaded.temperature_axis, grid.temperature_axis)
    assert jnp.allclose(loaded.pressure_axis, grid.pressure_axis)
    assert jnp.allclose(loaded.log10_z_over_z_sun_axis, grid.log10_z_over_z_sun_axis)
    assert jnp.allclose(loaded.outputs.ln_n, grid.outputs.ln_n)
    assert jnp.allclose(loaded.outputs.n, grid.outputs.n)
    assert jnp.allclose(loaded.outputs.x, grid.outputs.x)
    assert jnp.allclose(loaded.outputs.ntot, grid.outputs.ntot)
    assert loaded.metadata == grid.metadata


def test_equilibrium_grid_from_dataset_requires_species_coord():
    dataset = xr.Dataset(
        data_vars={
            "ln_n": (("temperature", "pressure", "log10_z_over_z_sun", "species"), jnp.zeros((1, 1, 1, 1))),
            "n": (("temperature", "pressure", "log10_z_over_z_sun", "species"), jnp.zeros((1, 1, 1, 1))),
            "x": (("temperature", "pressure", "log10_z_over_z_sun", "species"), jnp.zeros((1, 1, 1, 1))),
            "ntot": (("temperature", "pressure", "log10_z_over_z_sun"), jnp.zeros((1, 1, 1))),
        },
        coords={
            "temperature": [1000.0],
            "pressure": [1.0],
            "log10_z_over_z_sun": [0.0],
        },
        attrs={},
    )

    try:
        equilibrium_grid_from_dataset(dataset)
    except ValueError as exc:
        assert "coordinate 'species'" in str(exc)
    else:
        raise AssertionError("Expected ValueError for missing species coordinate.")


def test_equilibrium_grid_from_dataset_requires_metadata_attrs():
    dataset = xr.Dataset(
        data_vars={
            "ln_n": (("temperature", "pressure", "log10_z_over_z_sun", "species"), jnp.zeros((1, 1, 1, 1))),
            "n": (("temperature", "pressure", "log10_z_over_z_sun", "species"), jnp.zeros((1, 1, 1, 1))),
            "x": (("temperature", "pressure", "log10_z_over_z_sun", "species"), jnp.zeros((1, 1, 1, 1))),
            "ntot": (("temperature", "pressure", "log10_z_over_z_sun"), jnp.zeros((1, 1, 1))),
        },
        coords={
            "temperature": [1000.0],
            "pressure": [1.0],
            "log10_z_over_z_sun": [0.0],
            "species": ["H2"],
        },
        attrs={"preset_name": "fastchem"},
    )

    try:
        equilibrium_grid_from_dataset(dataset)
    except ValueError as exc:
        assert "metadata field" in str(exc)
    else:
        raise AssertionError("Expected ValueError for incomplete metadata attrs.")


def test_validate_equilibrium_grid_compatibility_accepts_matching_setup():
    grid = build_equilibrium_grid(
        "fastchem",
        temperature_axis=jnp.asarray([1000.0]),
        pressure_axis=jnp.asarray([1.0]),
        log10_z_over_z_sun_axis=jnp.asarray([0.0]),
        source="fastchem",
    )
    setup = ChemicalSetup(
        formula_matrix=jnp.zeros(
            (len(grid.metadata.preset_elements), len(grid.metadata.preset_species))
        ),
        hvector_func=lambda T: jnp.asarray([T]),
        elements=grid.metadata.preset_elements,
        species=grid.metadata.preset_species,
        metadata=grid.metadata.preset_setup_metadata,
    )

    validate_equilibrium_grid_compatibility(grid, setup, "fastchem")


def test_validate_equilibrium_grid_compatibility_rejects_species_mismatch():
    grid = build_equilibrium_grid(
        "fastchem",
        temperature_axis=jnp.asarray([1000.0]),
        pressure_axis=jnp.asarray([1.0]),
        log10_z_over_z_sun_axis=jnp.asarray([0.0]),
        source="fastchem",
    )
    setup = ChemicalSetup(
        formula_matrix=jnp.zeros((1, 1)),
        hvector_func=lambda T: jnp.asarray([T]),
        elements=grid.metadata.preset_elements,
        species=grid.metadata.preset_species[:-1],
        metadata=grid.metadata.preset_setup_metadata,
    )

    try:
        validate_equilibrium_grid_compatibility(grid, setup, "fastchem")
    except ValueError as exc:
        assert "species mismatch" in str(exc)
    else:
        raise AssertionError("Expected ValueError for species mismatch.")


def test_validate_equilibrium_grid_compatibility_rejects_composition_axis_mismatch():
    metadata = EquilibriumGridMetadata(
        preset_name="fastchem",
        preset_setup_metadata={"source": "fastchem v3.1.3", "dataset": "gas"},
        preset_elements=("H", "He"),
        preset_species=("H2", "He"),
        source="fastchem",
        composition_axis_name="metallicity",
    )
    grid = EquilibriumGrid(
        temperature_axis=jnp.asarray([1000.0]),
        pressure_axis=jnp.asarray([1.0]),
        log10_z_over_z_sun_axis=jnp.asarray([0.0]),
        outputs=EquilibriumGridOutputs(
            ln_n=jnp.zeros((1, 1, 1, 2)),
            n=jnp.ones((1, 1, 1, 2)),
            x=jnp.full((1, 1, 1, 2), 0.5),
            ntot=jnp.ones((1, 1, 1)),
        ),
        metadata=metadata,
    )
    setup = ChemicalSetup(
        formula_matrix=jnp.zeros((2, 2)),
        hvector_func=lambda T: jnp.asarray([T, T]),
        elements=("H", "He"),
        species=("H2", "He"),
        metadata={"source": "fastchem v3.1.3", "dataset": "gas"},
    )

    try:
        validate_equilibrium_grid_compatibility(grid, setup, "fastchem")
    except ValueError as exc:
        assert "composition axis mismatch" in str(exc)
    else:
        raise AssertionError("Expected ValueError for composition-axis mismatch.")
