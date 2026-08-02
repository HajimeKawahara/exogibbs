"""Run the FastChem 4 stand-alone executable as an independent comparator."""

from __future__ import annotations

from dataclasses import dataclass
import math
import os
from pathlib import Path
import shutil
import subprocess
import tempfile
from typing import Literal, Sequence, cast

import numpy as np


_CHEMISTRY_PREFIX = (
    "p(bar)",
    "T(K)",
    "n_<tot>(cm-3)",
    "n_g(cm-3)",
    "m(u)",
)
_MONITOR_PREFIX = (
    "grid_point",
    "iterations",
    "chem_iter",
    "cond_iter",
    "converged",
    "elem_conserved",
    "p(bar)",
    "T(K)",
    "n_<tot>(cm-3)",
    "n_g(cm-3)",
    "m(u)",
)
_PROFILE_MATCH_RTOL = 1.0e-9
FastChemChemistryMode = Literal["gas", "equilibrium_condensation"]
_CHEMISTRY_TYPES: dict[FastChemChemistryMode, str] = {
    "gas": "g",
    "equilibrium_condensation": "ce",
}


@dataclass(frozen=True)
class FastChemExecutableResult:
    """Structured output from one FastChem stand-alone profile calculation."""

    gas_names: tuple[str, ...]
    condensate_names: tuple[str, ...]
    gas_number_densities: np.ndarray
    condensate_number_densities: np.ndarray
    total_element_density: np.ndarray
    ideal_gas_density: np.ndarray
    mean_molecular_weight: np.ndarray
    convergence_status: np.ndarray
    element_conservation_status: np.ndarray
    total_iterations: np.ndarray
    chemistry_iterations: np.ndarray
    condensation_iterations: np.ndarray
    stdout: str
    chemistry_mode: FastChemChemistryMode = "equilibrium_condensation"

    @property
    def converged(self) -> np.ndarray:
        """Return the per-layer FastChem convergence flags."""

        return self.convergence_status == "ok"

    @property
    def elements_conserved(self) -> np.ndarray:
        """Return the per-layer aggregate element-conservation flags."""

        return self.element_conservation_status == "ok"

    @property
    def status(self) -> np.ndarray:
        """Alias for the raw per-layer convergence status strings."""

        return self.convergence_status

    @property
    def iterations(self) -> np.ndarray:
        """Alias for the per-layer total iteration counts."""

        return self.total_iterations


@dataclass(frozen=True)
class _ChemistryOutput:
    gas_names: tuple[str, ...]
    pressure: np.ndarray
    temperature: np.ndarray
    total_element_density: np.ndarray
    ideal_gas_density: np.ndarray
    mean_molecular_weight: np.ndarray
    gas_number_densities: np.ndarray


@dataclass(frozen=True)
class _MonitorOutput:
    element_names: tuple[str, ...]
    pressure: np.ndarray
    temperature: np.ndarray
    total_element_density: np.ndarray
    ideal_gas_density: np.ndarray
    mean_molecular_weight: np.ndarray
    convergence_status: tuple[str, ...]
    element_conservation_status: tuple[str, ...]
    total_iterations: np.ndarray
    chemistry_iterations: np.ndarray
    condensation_iterations: np.ndarray


@dataclass(frozen=True)
class _CondensateOutput:
    condensate_names: tuple[str, ...]
    pressure: np.ndarray
    temperature: np.ndarray
    condensate_number_densities: np.ndarray


def _parse_headered_table(
    text: str,
    *,
    source: str,
) -> tuple[tuple[str, ...], tuple[tuple[str, ...], ...]]:
    """Parse one FastChem whitespace-delimited output without merging columns."""

    lines = [
        (line_number, line.strip())
        for line_number, line in enumerate(text.splitlines(), 1)
    ]
    nonempty = [(line_number, line) for line_number, line in lines if line]
    if not nonempty:
        raise ValueError(f"{source} is empty.")

    header_line_number, header_line = nonempty[0]
    if not header_line.startswith("#"):
        raise ValueError(f"{source} line {header_line_number} is not a header.")

    header = tuple(token.lstrip("#") for token in header_line.split())
    if not header or any(not token for token in header):
        raise ValueError(f"{source} has an invalid header.")

    rows: list[tuple[str, ...]] = []
    for line_number, line in nonempty[1:]:
        if line.startswith("#"):
            raise ValueError(
                f"{source} has an unexpected header on line {line_number}."
            )
        row = tuple(line.split())
        if len(row) != len(header):
            raise ValueError(
                f"{source} line {line_number} has {len(row)} columns; "
                f"expected {len(header)}."
            )
        rows.append(row)

    if not rows:
        raise ValueError(f"{source} contains no data rows.")

    return header, tuple(rows)


def _numeric_columns(
    rows: tuple[tuple[str, ...], ...],
    *,
    source: str,
) -> np.ndarray:
    """Convert table rows to a finite floating-point matrix."""

    try:
        values = np.asarray(rows, dtype=float)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{source} contains a non-numeric value.") from exc

    if values.ndim != 2 or not np.all(np.isfinite(values)):
        raise ValueError(f"{source} contains a non-finite numeric value.")
    return values


def _parse_nonnegative_integer(value: str, *, source: str, field: str) -> int:
    try:
        parsed = int(value)
    except ValueError as exc:
        raise ValueError(
            f"{source} has a non-integer {field} value {value!r}."
        ) from exc
    if parsed < 0:
        raise ValueError(f"{source} has a negative {field} value {parsed}.")
    return parsed


def _validate_statuses(
    values: tuple[str, ...],
    *,
    source: str,
    field: str,
) -> None:
    unexpected = sorted(set(values) - {"ok", "fail"})
    if unexpected:
        raise ValueError(
            f"{source} has invalid {field} values: {', '.join(unexpected)}."
        )


def _parse_chemistry_output(
    text: str,
    *,
    source: str = "chemistry output",
) -> _ChemistryOutput:
    """Parse FastChem's ND chemistry output."""

    header, rows = _parse_headered_table(text, source=source)
    if header[: len(_CHEMISTRY_PREFIX)] != _CHEMISTRY_PREFIX:
        raise ValueError(
            f"{source} has an unexpected leading header: "
            f"{header[:len(_CHEMISTRY_PREFIX)]!r}."
        )

    gas_names = header[len(_CHEMISTRY_PREFIX) :]
    if not gas_names:
        raise ValueError(f"{source} contains no gas-species columns.")

    values = _numeric_columns(rows, source=source)
    if np.any(values[:, 0:2] <= 0.0) or np.any(values[:, 3] <= 0.0):
        raise ValueError(
            f"{source} has non-positive profile or ideal-gas values."
        )
    if np.any(values[:, (2, 4)] < 0.0):
        raise ValueError(f"{source} has a negative bulk-gas value.")
    gas_number_densities = values[:, len(_CHEMISTRY_PREFIX) :]
    if np.any(gas_number_densities < 0.0):
        raise ValueError(f"{source} has a negative gas-species number density.")

    return _ChemistryOutput(
        gas_names=gas_names,
        pressure=values[:, 0].copy(),
        temperature=values[:, 1].copy(),
        total_element_density=values[:, 2].copy(),
        ideal_gas_density=values[:, 3].copy(),
        mean_molecular_weight=values[:, 4].copy(),
        gas_number_densities=gas_number_densities.copy(),
    )


def _parse_monitor_output(
    text: str,
    *,
    source: str = "monitor output",
) -> _MonitorOutput:
    """Parse FastChem's per-layer convergence monitor."""

    header, rows = _parse_headered_table(text, source=source)
    if header[: len(_MONITOR_PREFIX)] != _MONITOR_PREFIX:
        raise ValueError(
            f"{source} has an unexpected leading header: "
            f"{header[:len(_MONITOR_PREFIX)]!r}."
        )

    element_names = header[len(_MONITOR_PREFIX) :]
    if not element_names:
        raise ValueError(f"{source} contains no element-status columns.")

    grid_points: list[int] = []
    total_iterations: list[int] = []
    chemistry_iterations: list[int] = []
    condensation_iterations: list[int] = []
    convergence_status: list[str] = []
    element_conservation_status: list[str] = []
    bulk_values: list[tuple[str, ...]] = []

    for row_number, row in enumerate(rows, 1):
        grid_points.append(
            _parse_nonnegative_integer(
                row[0],
                source=source,
                field=f"grid point on data row {row_number}",
            )
        )
        total_iterations.append(
            _parse_nonnegative_integer(
                row[1],
                source=source,
                field=f"iteration count on data row {row_number}",
            )
        )
        chemistry_iterations.append(
            _parse_nonnegative_integer(
                row[2],
                source=source,
                field=f"chemistry iteration count on data row {row_number}",
            )
        )
        condensation_iterations.append(
            _parse_nonnegative_integer(
                row[3],
                source=source,
                field=f"condensation iteration count on data row {row_number}",
            )
        )
        convergence_status.append(row[4])
        element_conservation_status.append(row[5])
        bulk_values.append(row[6:11])
        _validate_statuses(
            row[len(_MONITOR_PREFIX) :],
            source=source,
            field=f"per-element status on data row {row_number}",
        )

    expected_grid_points = list(range(len(rows)))
    if grid_points != expected_grid_points:
        raise ValueError(
            f"{source} grid points are {grid_points}; expected {expected_grid_points}."
        )

    convergence_status_tuple = tuple(convergence_status)
    element_conservation_status_tuple = tuple(element_conservation_status)
    _validate_statuses(
        convergence_status_tuple,
        source=source,
        field="convergence status",
    )
    _validate_statuses(
        element_conservation_status_tuple,
        source=source,
        field="element-conservation status",
    )

    bulk = _numeric_columns(tuple(bulk_values), source=source)
    if np.any(bulk[:, 0:2] <= 0.0) or np.any(bulk[:, 3] <= 0.0):
        raise ValueError(
            f"{source} has non-positive profile or ideal-gas values."
        )
    if np.any(bulk[:, (2, 4)] < 0.0):
        raise ValueError(f"{source} has a negative bulk-gas value.")

    return _MonitorOutput(
        element_names=element_names,
        pressure=bulk[:, 0].copy(),
        temperature=bulk[:, 1].copy(),
        total_element_density=bulk[:, 2].copy(),
        ideal_gas_density=bulk[:, 3].copy(),
        mean_molecular_weight=bulk[:, 4].copy(),
        convergence_status=convergence_status_tuple,
        element_conservation_status=element_conservation_status_tuple,
        total_iterations=np.asarray(total_iterations, dtype=np.int64),
        chemistry_iterations=np.asarray(chemistry_iterations, dtype=np.int64),
        condensation_iterations=np.asarray(
            condensation_iterations, dtype=np.int64
        ),
    )


def _parse_condensate_output(
    text: str,
    *,
    element_names: Sequence[str],
    source: str = "condensate output",
) -> _CondensateOutput:
    """Parse FastChem's condensate output while preserving duplicate names."""

    header, rows = _parse_headered_table(text, source=source)
    expected_prefix = ("p(bar)", "T(K)", *tuple(element_names))
    if header[: len(expected_prefix)] != expected_prefix:
        raise ValueError(
            f"{source} has an unexpected profile/element header: "
            f"{header[:len(expected_prefix)]!r}."
        )

    condensate_names = header[len(expected_prefix) :]
    if not condensate_names:
        raise ValueError(f"{source} contains no condensate-species columns.")

    values = _numeric_columns(rows, source=source)
    if np.any(values[:, 0:2] <= 0.0):
        raise ValueError(f"{source} has non-positive pressure or temperature.")
    condensate_number_densities = values[:, len(expected_prefix) :]
    if np.any(condensate_number_densities < 0.0):
        raise ValueError(f"{source} has a negative condensate number density.")

    return _CondensateOutput(
        condensate_names=condensate_names,
        pressure=values[:, 0].copy(),
        temperature=values[:, 1].copy(),
        condensate_number_densities=condensate_number_densities.copy(),
    )


def _validate_profile(
    temperatures: Sequence[float] | np.ndarray,
    pressures: Sequence[float] | np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Validate and copy a one-dimensional positive temperature-pressure profile."""

    try:
        temperature_array = np.asarray(temperatures, dtype=float)
        pressure_array = np.asarray(pressures, dtype=float)
    except (TypeError, ValueError) as exc:
        raise ValueError("temperatures and pressures must be numeric arrays.") from exc

    if temperature_array.ndim != 1 or pressure_array.ndim != 1:
        raise ValueError("temperatures and pressures must be one-dimensional.")
    if temperature_array.shape != pressure_array.shape:
        raise ValueError(
            "temperatures and pressures must have the same one-dimensional shape."
        )
    if temperature_array.size == 0:
        raise ValueError("temperatures and pressures must contain at least one layer.")
    if not np.all(np.isfinite(temperature_array)) or not np.all(
        np.isfinite(pressure_array)
    ):
        raise ValueError("temperatures and pressures must be finite.")
    if np.any(temperature_array <= 0.0) or np.any(pressure_array <= 0.0):
        raise ValueError("temperatures and pressures must be strictly positive.")

    return temperature_array.copy(), pressure_array.copy()


def _resolve_input_file(path: str | Path, *, name: str) -> Path:
    try:
        resolved = Path(path).resolve(strict=True)
    except (TypeError, OSError, RuntimeError) as exc:
        raise ValueError(f"{name} does not exist: {path}.") from exc
    if not resolved.is_file():
        raise ValueError(f"{name} is not a regular file: {resolved}.")
    return resolved


def _validate_positive_float(value: float, *, name: str) -> float:
    if isinstance(value, (bool, np.bool_)):
        raise ValueError(f"{name} must be a positive finite number.")
    try:
        parsed = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be a positive finite number.") from exc
    if not math.isfinite(parsed) or parsed <= 0.0:
        raise ValueError(f"{name} must be a positive finite number.")
    return parsed


def _validate_positive_integer(value: int, *, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, (int, np.integer)):
        raise ValueError(f"{name} must be a positive integer.")
    parsed = int(value)
    if parsed <= 0:
        raise ValueError(f"{name} must be a positive integer.")
    return parsed


def _validate_chemistry_mode(value: str) -> FastChemChemistryMode:
    if not isinstance(value, str) or value not in _CHEMISTRY_TYPES:
        valid = ", ".join(repr(mode) for mode in _CHEMISTRY_TYPES)
        raise ValueError(f"chemistry_mode must be one of: {valid}.")
    return cast(FastChemChemistryMode, value)


def _write_profile(
    path: Path,
    *,
    temperatures: np.ndarray,
    pressures: np.ndarray,
) -> None:
    lines = [
        f"{pressure:.17e}\t{temperature:.17e}"
        for temperature, pressure in zip(temperatures, pressures)
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _write_config(
    path: Path,
    *,
    chemistry_mode: FastChemChemistryMode,
    verbosity: int,
    chemistry_accuracy: float,
    element_conservation_accuracy: float,
    max_chemistry_iterations: int,
    max_internal_iterations: int,
) -> None:
    calculation_type = _CHEMISTRY_TYPES[chemistry_mode]
    chemistry_outputs = (
        "chemistry.dat condensates.dat"
        if chemistry_mode == "equilibrium_condensation"
        else "chemistry.dat"
    )
    species_inputs = (
        "gas_logk.dat condensate_logk.dat"
        if chemistry_mode == "equilibrium_condensation"
        else "gas_logk.dat"
    )
    config = f"""#Atmospheric profile input file
profile.dat

#Chemistry calculation type (g = gas, ce = equilibrium, cr = rainout)
{calculation_type}

#Chemistry output file
{chemistry_outputs}

#Monitor output file
monitor.dat

#FastChem console verbose level (1 - 4); 1 = almost silent, 4 = detailed console output
{verbosity}

#Output mixing ratios (MR) or particle number densities (ND, default)
ND

#Element abundance file
element_abundances.dat

#Species data files
{species_inputs}

#Accuracy of chemistry iteration
{chemistry_accuracy:.17e}

#Accuracy of element conservation
{element_conservation_accuracy:.17e}

#Max number of chemistry iterations
{max_chemistry_iterations}

#Max number internal solver iterations
{max_internal_iterations}
"""
    path.write_text(config, encoding="utf-8")


def _validate_layer_count(array: np.ndarray, *, layers: int, name: str) -> None:
    if array.shape[0] != layers:
        raise ValueError(
            f"{name} has {array.shape[0]} layers; expected exactly {layers}."
        )


def _validate_matching_values(
    actual: np.ndarray,
    expected: np.ndarray,
    *,
    name: str,
) -> None:
    if actual.shape != expected.shape or not np.allclose(
        actual,
        expected,
        rtol=_PROFILE_MATCH_RTOL,
        atol=0.0,
    ):
        raise ValueError(f"{name} does not match the requested profile.")


def _validate_consistent_outputs(
    chemistry: _ChemistryOutput,
    monitor: _MonitorOutput,
    condensates: _CondensateOutput,
    *,
    temperatures: np.ndarray,
    pressures: np.ndarray,
) -> None:
    layers = temperatures.size
    for name, array in (
        ("chemistry output", chemistry.pressure),
        ("monitor output", monitor.pressure),
        ("condensate output", condensates.pressure),
    ):
        _validate_layer_count(array, layers=layers, name=name)

    for label, actual in (
        ("chemistry pressure", chemistry.pressure),
        ("monitor pressure", monitor.pressure),
        ("condensate pressure", condensates.pressure),
    ):
        _validate_matching_values(actual, pressures, name=label)
    for label, actual in (
        ("chemistry temperature", chemistry.temperature),
        ("monitor temperature", monitor.temperature),
        ("condensate temperature", condensates.temperature),
    ):
        _validate_matching_values(actual, temperatures, name=label)

    for label, chemistry_values, monitor_values in (
        (
            "total element density",
            chemistry.total_element_density,
            monitor.total_element_density,
        ),
        (
            "ideal gas density",
            chemistry.ideal_gas_density,
            monitor.ideal_gas_density,
        ),
        (
            "mean molecular weight",
            chemistry.mean_molecular_weight,
            monitor.mean_molecular_weight,
        ),
    ):
        _validate_matching_values(
            monitor_values,
            chemistry_values,
            name=f"monitor {label}",
        )


def run_fastchem_executable(
    *,
    executable: str | Path,
    temperatures: Sequence[float] | np.ndarray,
    pressures: Sequence[float] | np.ndarray,
    element_abundance_file: str | Path,
    gas_logk_file: str | Path,
    condensate_logk_file: str | Path | None = None,
    chemistry_mode: FastChemChemistryMode = "equilibrium_condensation",
    verbosity: int = 1,
    chemistry_accuracy: float = 1.0e-5,
    element_conservation_accuracy: float = 1.0e-4,
    max_chemistry_iterations: int = 80_000,
    max_internal_iterations: int = 20_000,
) -> FastChemExecutableResult:
    """Run FastChem gas or equilibrium-condensation chemistry for a profile."""

    temperature_array, pressure_array = _validate_profile(
        temperatures, pressures
    )
    chemistry_mode_value = _validate_chemistry_mode(chemistry_mode)
    executable_path = _resolve_input_file(executable, name="executable")
    if not os.access(executable_path, os.X_OK):
        raise ValueError(f"executable is not executable: {executable_path}.")

    element_abundance_path = _resolve_input_file(
        element_abundance_file, name="element_abundance_file"
    )
    gas_logk_path = _resolve_input_file(gas_logk_file, name="gas_logk_file")
    condensate_logk_path: Path | None = None
    if chemistry_mode_value == "equilibrium_condensation":
        if condensate_logk_file is None:
            raise ValueError(
                "condensate_logk_file is required for "
                "equilibrium_condensation mode."
            )
        condensate_logk_path = _resolve_input_file(
            condensate_logk_file, name="condensate_logk_file"
        )

    if isinstance(verbosity, bool) or not isinstance(
        verbosity, (int, np.integer)
    ):
        raise ValueError("verbosity must be an integer from 1 through 4.")
    verbosity_value = int(verbosity)
    if verbosity_value not in range(1, 5):
        raise ValueError("verbosity must be an integer from 1 through 4.")

    chemistry_accuracy_value = _validate_positive_float(
        chemistry_accuracy, name="chemistry_accuracy"
    )
    element_conservation_accuracy_value = _validate_positive_float(
        element_conservation_accuracy,
        name="element_conservation_accuracy",
    )
    max_chemistry_iterations_value = _validate_positive_integer(
        max_chemistry_iterations,
        name="max_chemistry_iterations",
    )
    max_internal_iterations_value = _validate_positive_integer(
        max_internal_iterations,
        name="max_internal_iterations",
    )

    with tempfile.TemporaryDirectory(prefix="exogibbs_fastchem4_") as directory:
        work_directory = Path(directory)
        _write_profile(
            work_directory / "profile.dat",
            temperatures=temperature_array,
            pressures=pressure_array,
        )
        _write_config(
            work_directory / "config.input",
            chemistry_mode=chemistry_mode_value,
            verbosity=verbosity_value,
            chemistry_accuracy=chemistry_accuracy_value,
            element_conservation_accuracy=element_conservation_accuracy_value,
            max_chemistry_iterations=max_chemistry_iterations_value,
            max_internal_iterations=max_internal_iterations_value,
        )
        shutil.copyfile(
            element_abundance_path, work_directory / "element_abundances.dat"
        )
        shutil.copyfile(gas_logk_path, work_directory / "gas_logk.dat")
        if condensate_logk_path is not None:
            shutil.copyfile(
                condensate_logk_path, work_directory / "condensate_logk.dat"
            )

        try:
            completed = subprocess.run(
                [str(executable_path), "config.input"],
                cwd=work_directory,
                check=False,
                capture_output=True,
                text=True,
            )
        except OSError as exc:
            raise RuntimeError(
                f"Unable to execute FastChem at {executable_path}: {exc}"
            ) from exc

        if completed.returncode != 0:
            raise RuntimeError(
                "FastChem exited with return code "
                f"{completed.returncode}.\n"
                f"stdout:\n{completed.stdout}\n"
                f"stderr:\n{completed.stderr}"
            )

        output_paths = {
            "chemistry output": work_directory / "chemistry.dat",
            "monitor output": work_directory / "monitor.dat",
        }
        if chemistry_mode_value == "equilibrium_condensation":
            output_paths["condensate output"] = (
                work_directory / "condensates.dat"
            )
        missing = [
            name
            for name, output_path in output_paths.items()
            if not output_path.is_file()
        ]
        if missing:
            raise RuntimeError(
                "FastChem returned success without creating: "
                + ", ".join(missing)
                + "."
            )

        chemistry = _parse_chemistry_output(
            output_paths["chemistry output"].read_text(encoding="utf-8"),
            source="chemistry output",
        )
        monitor = _parse_monitor_output(
            output_paths["monitor output"].read_text(encoding="utf-8"),
            source="monitor output",
        )
        if chemistry_mode_value == "equilibrium_condensation":
            condensates = _parse_condensate_output(
                output_paths["condensate output"].read_text(
                    encoding="utf-8"
                ),
                element_names=monitor.element_names,
                source="condensate output",
            )
        else:
            condensates = _CondensateOutput(
                condensate_names=(),
                pressure=pressure_array.copy(),
                temperature=temperature_array.copy(),
                condensate_number_densities=np.empty(
                    (temperature_array.size, 0),
                    dtype=np.float64,
                ),
            )

        _validate_consistent_outputs(
            chemistry,
            monitor,
            condensates,
            temperatures=temperature_array,
            pressures=pressure_array,
        )

        return FastChemExecutableResult(
            gas_names=chemistry.gas_names,
            condensate_names=condensates.condensate_names,
            gas_number_densities=chemistry.gas_number_densities,
            condensate_number_densities=condensates.condensate_number_densities,
            total_element_density=chemistry.total_element_density,
            ideal_gas_density=chemistry.ideal_gas_density,
            mean_molecular_weight=chemistry.mean_molecular_weight,
            convergence_status=np.asarray(
                monitor.convergence_status, dtype=np.str_
            ),
            element_conservation_status=np.asarray(
                monitor.element_conservation_status, dtype=np.str_
            ),
            total_iterations=monitor.total_iterations,
            chemistry_iterations=monitor.chemistry_iterations,
            condensation_iterations=monitor.condensation_iterations,
            stdout=completed.stdout,
            chemistry_mode=chemistry_mode_value,
        )
