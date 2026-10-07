"""Construction and verification of gas-equilibrium grids."""

from __future__ import annotations

from typing import TYPE_CHECKING, Callable, Mapping, Optional, Sequence, Tuple

import jax.numpy as jnp
import numpy as np

from exogibbs.equilibrium.gas.grid.interpolation import (
    _h_he_metallicity_scale_from_log10_z_over_z_sun,
    _require_h_he_reference_abundance_setup,
    _verification_dtype_warning,
)
from exogibbs.equilibrium.gas.grid.types import (
    _FASTCHEM_COMPARISON_ABUNDANCE_FLOOR,
    _FASTCHEM_COMPARISON_TOLERANCE_PERCENT,
    Array,
    EquilibriumGrid,
    EquilibriumGridMetadata,
    EquilibriumGridOutputs,
    EquilibriumGridSource,
)
from exogibbs.equilibrium.gas.solve import equilibrium
from exogibbs.equilibrium.gas.types import EquilibriumOptions
from exogibbs.io.load_data import get_data_filepath
from exogibbs.thermo.composition import update_element_vector
from exogibbs.thermo.models import ChemicalSetup
from exogibbs.utils.nameparser import strip_trailing_one

if TYPE_CHECKING:
    import pyfastchem


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


def _normalize_number_densities_to_element_inventory_gauge(
    setup: ChemicalSetup,
    number_densities: Array,
    element_vector: Array,
) -> Tuple[Array, Array]:
    """Convert absolute number densities to the setup's amount gauge."""

    if setup.elements is None:
        raise ValueError("setup.elements is required for amount-gauge normalization.")
    formula_matrix = np.asarray(setup.formula_matrix, dtype=np.float64)
    densities = np.asarray(number_densities, dtype=np.float64)
    target = np.asarray(element_vector, dtype=np.float64)
    expected_density_shape = (formula_matrix.shape[1],)
    expected_target_shape = (formula_matrix.shape[0],)
    if densities.shape != expected_density_shape:
        raise ValueError(
            "number_densities must have one value per species: "
            f"expected {expected_density_shape}, got {densities.shape}."
        )
    if target.shape != expected_target_shape or len(setup.elements) != target.shape[0]:
        raise ValueError("element_vector must have one value per formula-matrix row.")
    if not np.all(np.isfinite(densities)) or np.any(densities < 0.0):
        raise ValueError("number_densities must be finite and non-negative.")
    if not np.all(np.isfinite(target)):
        raise ValueError("element_vector must contain only finite values.")

    physical_rows = np.asarray(
        [element not in {"e-", "electron"} for element in setup.elements],
        dtype=bool,
    )
    if np.any(target[physical_rows] < 0.0):
        raise ValueError("Non-charge element amounts must be non-negative.")
    positive_rows = physical_rows & (target > 0.0)
    if not np.any(positive_rows):
        raise ValueError("element_vector must contain a positive non-charge amount.")

    with np.errstate(invalid="ignore", over="ignore", divide="ignore"):
        source_inventory = formula_matrix @ densities
        source_total = float(np.sum(source_inventory[positive_rows]))
        target_total = float(np.sum(target[positive_rows]))
        gauge_scale = target_total / source_total
        amounts = densities * gauge_scale
    if (
        not np.all(np.isfinite(source_inventory))
        or not np.isfinite(source_total)
        or source_total <= 0.0
        or not np.isfinite(target_total)
        or target_total <= 0.0
        or not np.isfinite(gauge_scale)
        or gauge_scale <= 0.0
        or not np.all(np.isfinite(amounts))
    ):
        raise ValueError("Unable to convert number densities to a finite amount gauge.")

    log_floor = np.log(1.0e-300)
    with np.errstate(divide="ignore"):
        log_amounts = np.maximum(
            np.log(densities) + np.log(gauge_scale),
            log_floor,
        )
    output_dtype = jnp.result_type(jnp.asarray(setup.formula_matrix), jnp.float32)
    return (
        jnp.asarray(amounts, dtype=output_dtype),
        jnp.asarray(log_amounts, dtype=output_dtype),
    )


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
            "FastChem-backed grid generation and verification require the optional "
            "'pyfastchem' package. Install it with `pip install \"exogibbs[fastchem]\"`."
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

        physical_number_densities = np.asarray(
            output_data.number_densities,
            dtype=float,
        )[0][species_indices]
        n, ln_n = _normalize_number_densities_to_element_inventory_gauge(
            setup,
            physical_number_densities,
            element_vector,
        )
        ntot = jnp.asarray(jnp.sum(n))
        x = n / jnp.clip(ntot, 1e-300)
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
    verify_exogibbs_against_fastchem: bool = False,
    verification_abundance_floor: float = _FASTCHEM_COMPARISON_ABUNDANCE_FLOOR,
    verification_tolerance_percent: float = _FASTCHEM_COMPARISON_TOLERANCE_PERCENT,
    setup_builder: Optional[Callable[[], ChemicalSetup]] = None,
) -> EquilibriumGrid:
    """Generate an in-memory equilibrium grid for a preset and source.

    The composition axis is explicitly ``log10(Z/Zsun)`` for an H/He atmosphere.
    When ``source="exogibbs"``, verification against FastChem at the same grid
    points is available as an explicit opt-in for supported presets.
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
