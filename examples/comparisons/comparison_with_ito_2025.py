"""Compare the Ito et al. (2025) rainout profile with two equilibrium solvers.

The Ito workbook contains total-pressure gas mole fractions ordered from the
ground upward.  For target Layer ``i >= 2``, this example reconstructs the
H/O/Si inventory from the gas in the one-grid-higher-pressure Layer ``i - 1``
and independently solves local equilibrium with condensates at ``(T_i, P_i)``.
This is an Ito-anchored one-step rainout comparison: neither solver's output is
propagated into the next layer.

The ExoGibbs chemistry contains exactly the five Ito gas molecules H2, H2O,
O2, SiO, and SiH4 and the two condensates SiO and SiO2.  FastChem necessarily
adds the elemental reference gases H, O, and Si; this unavoidable catalog
difference is retained and reported rather than changing the Ito network on
the ExoGibbs side.  Helium is not a conserved chemistry element.  Instead,
its fixed Ito EOS ratio is enforced through a scalar reactive-pressure fixed
point.

Only the H/O/Si ratios carry physical information in the solver input.  The
ExoGibbs call therefore applies a documented uniform abundance gauge so that
trace depleted elements remain above the production solver's absolute
numerical floor; condensate amounts are converted back to the original gauge.
The preferred gauge is retried with a more conservative scale if needed.

The XLSX reader uses only the Python standard library and reads cached cell
values.  This keeps the example offline and avoids an optional spreadsheet
dependency.
"""

from __future__ import annotations

import argparse
from collections import Counter
import csv
from dataclasses import dataclass
import hashlib
import json
import math
import os
from pathlib import Path, PurePosixPath
import re
import sys
import tempfile
from typing import Any, Iterable, Mapping, Optional, Sequence
from xml.etree import ElementTree
from zipfile import ZipFile


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPOSITORY_ROOT))
sys.path.insert(0, str(REPOSITORY_ROOT / "src"))

os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.environ.setdefault("JAX_PLATFORM_NAME", "cpu")
os.environ.setdefault("JAX_ENABLE_X64", "1")
os.environ.setdefault("MPLCONFIGDIR", "/tmp/exogibbs_matplotlib")

import jax
from jax import config
import jax.numpy as jnp
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np

from benchmarks.fastchem4.comparison import align_species_values
from benchmarks.fastchem4.fastchem_executable import run_fastchem_executable
from exogibbs.api.condensate import (
    CondensateEquilibriumOptions,
    solve as solve_condensate_equilibrium,
)
from exogibbs.equilibrium.condensate.setup import (
    build_condensate_chemical_setup,
)
from exogibbs.presets.fastchem4_cond import condensate_chemical_setup
from exogibbs.thermo.models import ChemicalSetup
from exogibbs.utils.fastchem_parity import normalize_species_name


config.update("jax_enable_x64", True)

DATA_ROOT = REPOSITORY_ROOT / "src" / "exogibbs" / "data" / "FastChem4"
GAS_LOGK_FILE = DATA_ROOT / "logK" / "logK_wo_ions.dat"
CONDENSATE_LOGK_FILE = DATA_ROOT / "logK" / "logK_condensates.dat"
DEFAULT_INPUT = REPOSITORY_ROOT / "external_data" / "Ito_2025.xlsx"
DEFAULT_OUTPUT_ROOT = REPOSITORY_ROOT / "results" / "ito_2025"
DEFAULT_FIGURE = DEFAULT_OUTPUT_ROOT / "ito_2025_exogibbs_fastchem4.png"
DEFAULT_TABLE = DEFAULT_OUTPUT_ROOT / "ito_2025_exogibbs_fastchem4.csv"
DEFAULT_SUMMARY = DEFAULT_OUTPUT_ROOT / "ito_2025_exogibbs_fastchem4.json"
DEFAULT_CHECKPOINT = DEFAULT_OUTPUT_ROOT / "ito_2025_exogibbs_fastchem4.npz"

ELEMENTS = ("H", "O", "Si")
ITO_SPECIES = ("H2", "He", "H2O", "O2", "SiO", "SiH4")
ITO_REACTIVE_SPECIES = ("H2", "H2O", "O2", "SiO", "SiH4")
COMMON_GAS_SPECIES = (
    "H1",
    "O1",
    "Si1",
    "H2",
    "H2O1",
    "O2",
    "O1Si1",
    "H4Si1",
)
FASTCHEM_MOLECULES = ("H2", "H2O1", "H4Si1", "O1Si1", "O2")
CONDENSATE_SPECIES = ("SiO(s)", "SiO2(s,l)")
ATOMIC_GAS_SPECIES = ("H1", "O1", "Si1")
ITO_TO_COMMON = {
    "H2": "H2",
    "H2O": "H2O1",
    "O2": "O2",
    "SiO": "O1Si1",
    "SiH4": "H4Si1",
}
EXOGIBBS_GAS_SPECIES = tuple(
    ITO_TO_COMMON[name] for name in ITO_REACTIVE_SPECIES
)

HELIUM_MASS_FRACTION = 0.275
HE_TO_H2_RATIO = HELIUM_MASS_FRACTION / (
    2.0 * (1.0 - HELIUM_MASS_FRACTION)
)
FIXED_POINT_RTOL = 1.0e-9
FIXED_POINT_MAX_ITERATIONS = 6
EXOGIBBS_ELEMENT_GAUGES = (1.0e-3, 1.0e-4, 1.0e-5, 1.0e-6, 1.0e-7)
CHECKPOINT_SCHEMA = "exogibbs_ito_2025_comparison_v2"
PLOT_FLOOR = 1.0e-45

_SPREADSHEET_NS = "http://schemas.openxmlformats.org/spreadsheetml/2006/main"
_RELATIONSHIP_NS = "http://schemas.openxmlformats.org/package/2006/relationships"
_OFFICE_RELATIONSHIP_NS = (
    "http://schemas.openxmlformats.org/officeDocument/2006/relationships"
)
_CELL_REFERENCE = re.compile(r"^([A-Z]+)[0-9]+$")
_FASTCHEM_RECORD = re.compile(r"^\s*([^\s:#]+)\s+[^:]*:\s*")


@dataclass(frozen=True)
class ItoProfile:
    """Validated Ito workbook data in ground-to-upper-atmosphere order."""

    layer: np.ndarray
    pressure_bar: np.ndarray
    temperature_k: np.ndarray
    gas_fractions: np.ndarray
    atomic_fractions: np.ndarray


@dataclass(frozen=True)
class SolverLayerResult:
    """One solver state converted to the Ito total-pressure convention."""

    gas_fractions: np.ndarray
    atomic_gas_fractions: np.ndarray
    condensate_amounts: np.ndarray
    abundance_scale: float
    reactive_pressure_bar: float
    fixed_point_iterations: int
    converged: bool
    status: str
    solver_iterations: int


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Compare the Ito et al. (2025) one-grid rainout profile with "
            "ExoGibbs and the standalone FastChem 4 executable."
        )
    )
    parser.add_argument(
        "--fastchem-executable",
        required=True,
        type=Path,
        help="Path to the audited FastChem 4 standalone executable.",
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=DEFAULT_INPUT,
        help="Ito workbook path.",
    )
    parser.add_argument(
        "--figure",
        type=Path,
        default=DEFAULT_FIGURE,
        help="Output comparison PNG.",
    )
    parser.add_argument(
        "--table",
        type=Path,
        default=DEFAULT_TABLE,
        help="Output layer-by-layer CSV.",
    )
    parser.add_argument(
        "--summary",
        type=Path,
        default=DEFAULT_SUMMARY,
        help="Output machine-readable JSON summary.",
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=DEFAULT_CHECKPOINT,
        help="Restartable NPZ calculation checkpoint.",
    )
    parser.add_argument(
        "--no-resume",
        action="store_true",
        help="Ignore and overwrite an existing checkpoint.",
    )
    parser.add_argument(
        "--retry-failed",
        action="store_true",
        help="Retry attempted solver layers whose checkpoint status did not converge.",
    )
    parser.add_argument(
        "--solver",
        choices=("both", "exogibbs", "fastchem"),
        default="both",
        help="Run both solvers or fill only one side of the checkpoint.",
    )
    parser.add_argument(
        "--layer-step",
        type=int,
        default=1,
        help="Use every Nth target layer; the scientific default is 1.",
    )
    parser.add_argument(
        "--max-layers",
        type=int,
        default=None,
        help="Optional leading-layer limit for smoke tests.",
    )
    parser.add_argument(
        "--checkpoint-every",
        type=int,
        default=10,
        help="Save after this many newly completed solver-layer calculations.",
    )
    parser.add_argument(
        "--show",
        action="store_true",
        help="Display the figure after saving it.",
    )
    args = parser.parse_args()
    for name in ("layer_step", "checkpoint_every"):
        if getattr(args, name) <= 0:
            parser.error(f"--{name.replace('_', '-')} must be positive")
    if args.max_layers is not None and args.max_layers <= 0:
        parser.error("--max-layers must be positive")
    return args


def _column_index(cell_reference: str) -> int:
    match = _CELL_REFERENCE.match(cell_reference)
    if match is None:
        raise ValueError(f"Invalid XLSX cell reference: {cell_reference!r}")
    value = 0
    for character in match.group(1):
        value = value * 26 + ord(character) - ord("A") + 1
    return value - 1


def _shared_strings(archive: ZipFile) -> tuple[str, ...]:
    try:
        payload = archive.read("xl/sharedStrings.xml")
    except KeyError:
        return ()
    root = ElementTree.fromstring(payload)
    strings = []
    text_tag = f"{{{_SPREADSHEET_NS}}}t"
    for item in root.findall(f"{{{_SPREADSHEET_NS}}}si"):
        strings.append("".join(node.text or "" for node in item.iter(text_tag)))
    return tuple(strings)


def _first_worksheet_path(archive: ZipFile) -> str:
    workbook = ElementTree.fromstring(archive.read("xl/workbook.xml"))
    sheets = workbook.find(f"{{{_SPREADSHEET_NS}}}sheets")
    if sheets is None or len(sheets) == 0:
        raise ValueError("The Ito workbook contains no worksheets.")
    relationship_id = sheets[0].attrib.get(
        f"{{{_OFFICE_RELATIONSHIP_NS}}}id"
    )
    if not relationship_id:
        raise ValueError("The first worksheet has no relationship identifier.")
    relationships = ElementTree.fromstring(
        archive.read("xl/_rels/workbook.xml.rels")
    )
    targets = {
        item.attrib["Id"]: item.attrib["Target"]
        for item in relationships.findall(f"{{{_RELATIONSHIP_NS}}}Relationship")
    }
    try:
        target = targets[relationship_id]
    except KeyError as exc:
        raise ValueError("Unable to resolve the first worksheet path.") from exc
    if target.startswith("/"):
        return target.lstrip("/")
    return str(PurePosixPath("xl") / target)


def _cell_value(cell: ElementTree.Element, shared: Sequence[str]) -> Any:
    cell_type = cell.attrib.get("t")
    if cell_type == "inlineStr":
        inline = cell.find(f"{{{_SPREADSHEET_NS}}}is")
        if inline is None:
            return ""
        text_tag = f"{{{_SPREADSHEET_NS}}}t"
        return "".join(node.text or "" for node in inline.iter(text_tag))
    value_node = cell.find(f"{{{_SPREADSHEET_NS}}}v")
    if value_node is None or value_node.text is None:
        return None
    raw = value_node.text
    if cell_type == "s":
        return shared[int(raw)]
    if cell_type in {"str", "e"}:
        return raw
    if cell_type == "b":
        return raw == "1"
    try:
        return float(raw)
    except ValueError:
        return raw


def _read_first_worksheet(path: Path) -> list[list[Any]]:
    with ZipFile(path) as archive:
        shared = _shared_strings(archive)
        worksheet = ElementTree.fromstring(
            archive.read(_first_worksheet_path(archive))
        )
    sheet_data = worksheet.find(f"{{{_SPREADSHEET_NS}}}sheetData")
    if sheet_data is None:
        raise ValueError("The first worksheet has no sheet data.")
    rows = []
    for row in sheet_data.findall(f"{{{_SPREADSHEET_NS}}}row"):
        values: dict[int, Any] = {}
        for cell in row.findall(f"{{{_SPREADSHEET_NS}}}c"):
            reference = cell.attrib.get("r")
            if reference is None:
                continue
            values[_column_index(reference)] = _cell_value(cell, shared)
        if values:
            width = max(values) + 1
            rows.append([values.get(index) for index in range(width)])
    if not rows:
        raise ValueError("The first worksheet contains no non-empty rows.")
    return rows


def _numeric_column(
    rows: Sequence[Sequence[Any]],
    header_index: Mapping[str, int],
    name: str,
) -> np.ndarray:
    try:
        column = header_index[name]
    except KeyError as exc:
        raise ValueError(f"The Ito workbook is missing column {name!r}.") from exc
    values = []
    for row_number, row in enumerate(rows, 2):
        value = row[column] if column < len(row) else None
        try:
            values.append(float(value))
        except (TypeError, ValueError) as exc:
            raise ValueError(
                f"The Ito workbook has a non-numeric {name!r} value "
                f"on worksheet row {row_number}."
            ) from exc
    return np.asarray(values, dtype=np.float64)


def reconstruct_atomic_inventory(gas_fractions: np.ndarray) -> np.ndarray:
    """Return unnormalized H/O/Si/He atom counts from Ito gas fractions."""

    fractions = np.asarray(gas_fractions, dtype=np.float64)
    if fractions.shape[-1] != len(ITO_SPECIES):
        raise ValueError(
            f"gas_fractions must end with {len(ITO_SPECIES)} Ito species."
        )
    h2, helium, h2o, o2, sio, sih4 = np.moveaxis(fractions, -1, 0)
    return np.stack(
        (
            2.0 * h2 + 2.0 * h2o + 4.0 * sih4,
            h2o + 2.0 * o2 + sio,
            sio + sih4,
            helium,
        ),
        axis=-1,
    )


def reactive_element_abundances(gas_fractions: np.ndarray) -> np.ndarray:
    """Return sum-normalized H/O/Si abundances, excluding inert helium."""

    inventory = reconstruct_atomic_inventory(gas_fractions)[..., :3]
    totals = np.sum(inventory, axis=-1, keepdims=True)
    if np.any(~np.isfinite(inventory)) or np.any(inventory <= 0.0):
        raise ValueError("Ito-derived H/O/Si inventories must be finite and positive.")
    return inventory / totals


def load_ito_profile(path: Path) -> ItoProfile:
    """Load and validate the supplied Ito workbook."""

    rows = _read_first_worksheet(path)
    header_index = {
        str(value).strip(): index
        for index, value in enumerate(rows[0])
        if value is not None
    }
    data_rows = rows[1:]
    layer_values = _numeric_column(
        data_rows, header_index, "Layer number"
    )
    if np.any(layer_values != np.round(layer_values)):
        raise ValueError("Ito layer numbers must be integers.")
    layer = layer_values.astype(np.int64)
    pressure = _numeric_column(data_rows, header_index, "P [bar]")
    temperature = _numeric_column(data_rows, header_index, "T[K]")
    gas = np.column_stack(
        [
            _numeric_column(data_rows, header_index, name)
            for name in (
                "Fraction of H2",
                "Fraction of He",
                "Fraction of H2O",
                "Fracton of O2",
                "Fraction of SiO",
                "Fraction of SiH4",
            )
        ]
    )
    if layer.size < 2:
        raise ValueError("The Ito profile must contain the ground and an upper layer.")
    if not np.array_equal(layer, np.arange(1, layer.size + 1)):
        raise ValueError("Ito layer numbers must be contiguous and start at 1.")
    if np.any(~np.isfinite(pressure)) or np.any(pressure <= 0.0):
        raise ValueError("Ito pressures must be finite and positive.")
    if np.any(~np.isfinite(temperature)) or np.any(temperature <= 0.0):
        raise ValueError("Ito temperatures must be finite and positive.")
    if np.any(np.diff(pressure) >= 0.0):
        raise ValueError("Ito pressures must decrease from the ground upward.")
    if np.any(np.diff(temperature) >= 0.0):
        raise ValueError("Ito temperatures must decrease from the ground upward.")
    if np.any(~np.isfinite(gas)) or np.any(gas < 0.0):
        raise ValueError("Ito gas fractions must be finite and non-negative.")
    gas_sums = np.sum(gas, axis=1)
    if not np.allclose(gas_sums, 1.0, rtol=0.0, atol=1.0e-5):
        raise ValueError("Ito gas fractions do not sum to one within file precision.")
    helium_ratio = gas[:, ITO_SPECIES.index("He")] / gas[
        :, ITO_SPECIES.index("H2")
    ]
    if not np.allclose(
        helium_ratio,
        HE_TO_H2_RATIO,
        rtol=1.0e-12,
        atol=1.0e-14,
    ):
        raise ValueError("Ito He/H2 is inconsistent with the fixed EOS ratio.")
    atomic_inventory = reconstruct_atomic_inventory(gas)
    atomic_fractions = atomic_inventory / np.sum(
        atomic_inventory, axis=1, keepdims=True
    )
    return ItoProfile(
        layer=layer,
        pressure_bar=pressure,
        temperature_k=temperature,
        gas_fractions=gas,
        atomic_fractions=atomic_fractions,
    )


def filter_fastchem_records(text: str, selected_names: Iterable[str]) -> str:
    """Keep exact named FastChem records and the original leading comments."""

    selected = set(selected_names)
    lines = text.splitlines()
    starts = []
    for index, line in enumerate(lines):
        match = _FASTCHEM_RECORD.match(line)
        if match is not None and not line.lstrip().startswith("#"):
            starts.append((index, match.group(1)))
    if not starts:
        raise ValueError("The FastChem table contains no species records.")
    found = [name for _, name in starts if name in selected]
    missing = sorted(selected - set(found))
    duplicates = sorted(name for name in selected if found.count(name) != 1)
    if missing or duplicates:
        raise ValueError(
            "Unable to select unique FastChem records: "
            f"missing={missing}, non-unique={duplicates}."
        )
    output = lines[: starts[0][0]]
    for position, (start, name) in enumerate(starts):
        if name not in selected:
            continue
        end = starts[position + 1][0] if position + 1 < len(starts) else len(lines)
        output.extend(lines[start:end])
    return "\n".join(output).rstrip() + "\n"


def _write_filtered_fastchem_inputs(directory: Path) -> tuple[Path, Path]:
    gas_path = directory / "ito_gas_logk.dat"
    condensate_path = directory / "ito_condensate_logk.dat"
    gas_path.write_text(
        filter_fastchem_records(
            GAS_LOGK_FILE.read_text(encoding="utf-8"),
            FASTCHEM_MOLECULES,
        ),
        encoding="utf-8",
    )
    condensate_path.write_text(
        filter_fastchem_records(
            CONDENSATE_LOGK_FILE.read_text(encoding="utf-8"),
            CONDENSATE_SPECIES,
        ),
        encoding="utf-8",
    )
    return gas_path, condensate_path


def _subset_hvector(function, indices: tuple[int, ...]):
    index_array = jnp.asarray(indices, dtype=jnp.int32)

    def hvector(temperature):
        return function(temperature)[..., index_array]

    return jax.jit(hvector)


def build_ito_exogibbs_setup():
    """Build Ito's H/O/Si, five-gas, two-condensate reaction network."""

    full = condensate_chemical_setup(silent=True)
    element_indices = tuple(full.elements.index(name) for name in ELEMENTS)
    gas_indices = tuple(
        full.gas_species.index(name) for name in EXOGIBBS_GAS_SPECIES
    )
    condensate_indices = tuple(
        full.condensate_species.index(name) for name in CONDENSATE_SPECIES
    )
    gas_matrix = np.asarray(full.formula_matrix)[
        np.ix_(element_indices, gas_indices)
    ]
    condensate_matrix = np.asarray(full.formula_matrix_cond)[
        np.ix_(element_indices, condensate_indices)
    ]
    gas_setup = ChemicalSetup(
        formula_matrix=jnp.asarray(gas_matrix, dtype=jnp.float64),
        hvector_func=_subset_hvector(full.gas_setup.hvector_func, gas_indices),
        elements=ELEMENTS,
        species=EXOGIBBS_GAS_SPECIES,
        metadata={
            "source": "FastChem4 filtered for Ito et al. (2025)",
            "reaction_network": "Ito et al. (2025) molecules only",
        },
    )
    validity = full.condensate_setup.temperature_validity_upper
    if validity is None:
        raise ValueError("FastChem4 condensates have no validity bounds.")
    condensate_setup = ChemicalSetup(
        formula_matrix=jnp.asarray(condensate_matrix, dtype=jnp.float64),
        hvector_func=_subset_hvector(
            full.condensate_setup.hvector_func, condensate_indices
        ),
        elements=ELEMENTS,
        species=CONDENSATE_SPECIES,
        metadata={
            "source": "FastChem4 filtered for Ito et al. (2025)",
        },
        temperature_validity_upper=tuple(
            float(validity[index]) for index in condensate_indices
        ),
    )
    return build_condensate_chemical_setup(
        gas_setup=gas_setup,
        condensate_setup=condensate_setup,
    )


def _initial_reactive_pressure(
    total_pressure_bar: float,
    previous_ito_gas: np.ndarray,
) -> float:
    reactive_fraction = 1.0 - float(
        previous_ito_gas[ITO_SPECIES.index("He")]
    )
    previous_reactive_h2 = float(
        previous_ito_gas[ITO_SPECIES.index("H2")]
    ) / reactive_fraction
    return total_pressure_bar / (1.0 + HE_TO_H2_RATIO * previous_reactive_h2)


def _convert_reactive_gas(
    common_gas_fractions: np.ndarray,
    gas_species: Sequence[str] = COMMON_GAS_SPECIES,
) -> tuple[np.ndarray, np.ndarray, float]:
    fractions = np.asarray(common_gas_fractions, dtype=np.float64)
    names = tuple(gas_species)
    if fractions.shape != (len(names),):
        raise ValueError("Unexpected common gas fraction shape.")
    if np.any(~np.isfinite(fractions)) or np.any(fractions < 0.0):
        raise ValueError("Reactive gas fractions must be finite and non-negative.")
    fractions = fractions / np.sum(fractions)
    reactive_h2 = fractions[names.index("H2")]
    reactive_pressure_fraction = 1.0 / (
        1.0 + HE_TO_H2_RATIO * reactive_h2
    )
    total_fractions = np.zeros(len(ITO_SPECIES), dtype=np.float64)
    for ito_name in ITO_REACTIVE_SPECIES:
        common_name = ITO_TO_COMMON[ito_name]
        total_fractions[ITO_SPECIES.index(ito_name)] = (
            reactive_pressure_fraction
            * fractions[names.index(common_name)]
        )
    total_fractions[ITO_SPECIES.index("He")] = (
        HE_TO_H2_RATIO * total_fractions[ITO_SPECIES.index("H2")]
    )
    atomic = np.asarray(
        [
            (
                reactive_pressure_fraction * fractions[names.index(name)]
                if name in names
                else np.nan
            )
            for name in ATOMIC_GAS_SPECIES
        ]
    )
    return total_fractions, atomic, reactive_pressure_fraction


def _fixed_point_converged(current: float, updated: float) -> bool:
    return abs(math.log(updated / current)) <= FIXED_POINT_RTOL


def _exogibbs_abundance_scales(abundance: np.ndarray) -> tuple[float, ...]:
    """Return preferred and conservative gauges without changing element ratios."""

    values = np.asarray(abundance, dtype=np.float64)
    if values.shape != (len(ELEMENTS),):
        raise ValueError("Unexpected ExoGibbs element-abundance shape.")
    if np.any(~np.isfinite(values)) or np.any(values <= 0.0):
        raise ValueError("ExoGibbs element abundances must be finite and positive.")
    minimum = float(np.min(values))
    scales = tuple(
        max(1.0, gauge / minimum) for gauge in EXOGIBBS_ELEMENT_GAUGES
    )
    return tuple(dict.fromkeys(scales))


def solve_exogibbs_layer(
    setup,
    *,
    temperature_k: float,
    total_pressure_bar: float,
    abundance: np.ndarray,
    initial_reactive_pressure_bar: float,
) -> SolverLayerResult:
    """Solve one ExoGibbs layer with the fixed-He pressure convention."""

    last_status = "not_attempted"
    last_pressure = float(initial_reactive_pressure_bar)
    last_fixed_point_iteration = 0
    last_scale = 1.0
    for abundance_scale in _exogibbs_abundance_scales(abundance):
        pressure = float(initial_reactive_pressure_bar)
        scaled_abundance = (
            np.asarray(abundance, dtype=np.float64) * abundance_scale
        )
        last_scale = abundance_scale
        result = None
        total_fractions = None
        atomic = None
        for fixed_point_iteration in range(1, FIXED_POINT_MAX_ITERATIONS + 1):
            last_fixed_point_iteration = fixed_point_iteration
            try:
                result = solve_condensate_equilibrium(
                    setup,
                    T=float(temperature_k),
                    P=pressure,
                    b=jnp.asarray(scaled_abundance, dtype=jnp.float64),
                    options=CondensateEquilibriumOptions(
                        return_diagnostics=False
                    ),
                )
                jax.block_until_ready(
                    (result.gas_x, result.condensate_amounts)
                )
            except (FloatingPointError, ValueError) as error:
                last_status = f"numerical_error:{type(error).__name__}"
                break
            last_status = str(result.status)
            if not result.converged:
                break
            total_fractions, atomic, pressure_fraction = _convert_reactive_gas(
                np.asarray(result.gas_x, dtype=np.float64),
                EXOGIBBS_GAS_SPECIES,
            )
            updated_pressure = total_pressure_bar * pressure_fraction
            last_pressure = pressure
            if _fixed_point_converged(pressure, updated_pressure):
                diagnostics = result.diagnostics or {}
                iteration_value = diagnostics.get("iterations", -1)
                try:
                    solver_iterations = int(np.asarray(iteration_value))
                except (TypeError, ValueError):
                    solver_iterations = -1
                assert total_fractions is not None
                assert atomic is not None
                return SolverLayerResult(
                    gas_fractions=total_fractions,
                    atomic_gas_fractions=atomic,
                    condensate_amounts=np.asarray(
                        result.condensate_amounts, dtype=np.float64
                    )
                    / abundance_scale,
                    abundance_scale=abundance_scale,
                    reactive_pressure_bar=float(pressure),
                    fixed_point_iterations=fixed_point_iteration,
                    converged=True,
                    status=str(result.status),
                    solver_iterations=solver_iterations,
                )
            pressure = updated_pressure
        else:
            last_status = "helium_fixed_point_not_converged"
        last_pressure = pressure
    return SolverLayerResult(
        gas_fractions=np.full(len(ITO_SPECIES), np.nan),
        atomic_gas_fractions=np.full(len(ATOMIC_GAS_SPECIES), np.nan),
        condensate_amounts=np.full(len(CONDENSATE_SPECIES), np.nan),
        abundance_scale=last_scale,
        reactive_pressure_bar=last_pressure,
        fixed_point_iterations=last_fixed_point_iteration,
        converged=False,
        status=last_status,
        solver_iterations=-1,
    )


def _write_fastchem_abundance(path: Path, abundance: np.ndarray) -> None:
    values = np.asarray(abundance, dtype=np.float64)
    if values.shape != (len(ELEMENTS),) or np.any(values <= 0.0):
        raise ValueError("FastChem H/O/Si abundances must be positive.")
    hydrogen = values[ELEMENTS.index("H")]
    lines = [
        "# Ito gas-derived H/O/Si abundance",
        "e-  0.0",
        "H   12.0",
    ]
    for element in ("O", "Si"):
        value = 12.0 + math.log10(values[ELEMENTS.index(element)] / hydrogen)
        lines.append(f"{element:<2s}  {value:.17e}")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _validate_fastchem_catalog(
    gas_names: Sequence[str],
    condensate_names: Sequence[str],
) -> None:
    expected_gases = Counter(
        normalize_species_name(name) for name in (*COMMON_GAS_SPECIES, "e-")
    )
    actual_gases = Counter(normalize_species_name(name) for name in gas_names)
    expected_condensates = Counter(
        normalize_species_name(name) for name in CONDENSATE_SPECIES
    )
    actual_condensates = Counter(
        normalize_species_name(name) for name in condensate_names
    )
    if (
        actual_gases != expected_gases
        or actual_condensates != expected_condensates
    ):
        raise RuntimeError(
            "Filtered FastChem catalog differs from the shared catalog: "
            f"expected_gases={expected_gases}, actual_gases={actual_gases}, "
            f"expected_condensates={expected_condensates}, "
            f"actual_condensates={actual_condensates}."
        )


def solve_fastchem_layer(
    executable: Path,
    *,
    gas_logk_file: Path,
    condensate_logk_file: Path,
    abundance_file: Path,
    temperature_k: float,
    total_pressure_bar: float,
    initial_reactive_pressure_bar: float,
) -> SolverLayerResult:
    """Solve one standalone FastChem layer with the fixed-He convention."""

    pressure = float(initial_reactive_pressure_bar)
    result = None
    total_fractions = None
    atomic = None
    aligned_condensates = None
    for fixed_point_iteration in range(1, FIXED_POINT_MAX_ITERATIONS + 1):
        result = run_fastchem_executable(
            executable=executable,
            temperatures=[temperature_k],
            pressures=[pressure],
            element_abundance_file=abundance_file,
            gas_logk_file=gas_logk_file,
            condensate_logk_file=condensate_logk_file,
            chemistry_mode="equilibrium_condensation",
        )
        _validate_fastchem_catalog(result.gas_names, result.condensate_names)
        if not bool(result.converged[0]) or not bool(result.elements_conserved[0]):
            raise RuntimeError(
                "FastChem failed at "
                f"T={temperature_k:g} K, P_reactive={pressure:g} bar: "
                f"status={result.status[0]}, "
                f"elements={result.element_conservation_status[0]}."
            )
        gas_density = np.asarray(result.gas_number_densities[0], dtype=np.float64)
        total_gas_density = float(np.sum(gas_density))
        aligned_density = align_species_values(
            COMMON_GAS_SPECIES,
            result.gas_names,
            gas_density,
        )
        total_fractions, atomic, pressure_fraction = _convert_reactive_gas(
            aligned_density / total_gas_density
        )
        aligned_condensates = align_species_values(
            CONDENSATE_SPECIES,
            result.condensate_names,
            result.condensate_number_densities[0],
        ) / float(result.total_element_density[0])
        updated_pressure = total_pressure_bar * pressure_fraction
        if _fixed_point_converged(pressure, updated_pressure):
            break
        pressure = updated_pressure
    else:
        raise RuntimeError("FastChem helium-pressure fixed point did not converge.")
    assert result is not None
    assert total_fractions is not None
    assert atomic is not None
    assert aligned_condensates is not None
    return SolverLayerResult(
        gas_fractions=total_fractions,
        atomic_gas_fractions=atomic,
        condensate_amounts=aligned_condensates,
        abundance_scale=1.0,
        reactive_pressure_bar=float(pressure),
        fixed_point_iterations=fixed_point_iteration,
        converged=True,
        status=str(result.status[0]),
        solver_iterations=int(result.iterations[0]),
    )


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _target_indices(
    profile: ItoProfile,
    *,
    step: int,
    max_layers: Optional[int],
) -> np.ndarray:
    indices = np.arange(1, profile.layer.size, step, dtype=np.int64)
    if max_layers is not None:
        indices = indices[:max_layers]
    if indices.size == 0:
        raise ValueError("No Layer 2-or-higher targets were selected.")
    return indices


def _new_checkpoint(
    profile: ItoProfile,
    target_indices: np.ndarray,
    *,
    input_sha256: str,
    executable_sha256: str,
    script_sha256: str = "",
    gas_logk_sha256: str = "",
    condensate_logk_sha256: str = "",
) -> dict[str, np.ndarray]:
    count = target_indices.size
    source_indices = target_indices - 1
    return {
        "schema": np.asarray(CHECKPOINT_SCHEMA),
        "input_sha256": np.asarray(input_sha256),
        "fastchem_executable_sha256": np.asarray(executable_sha256),
        "script_sha256": np.asarray(script_sha256),
        "gas_logk_sha256": np.asarray(gas_logk_sha256),
        "condensate_logk_sha256": np.asarray(condensate_logk_sha256),
        "target_indices": target_indices,
        "source_layers": profile.layer[source_indices],
        "layers": profile.layer[target_indices],
        "pressure_bar": profile.pressure_bar[target_indices],
        "temperature_k": profile.temperature_k[target_indices],
        "ito_gas_fractions": profile.gas_fractions[target_indices],
        "input_abundances": reactive_element_abundances(
            profile.gas_fractions[source_indices]
        ),
        "exogibbs_done": np.zeros(count, dtype=bool),
        "exogibbs_converged": np.zeros(count, dtype=bool),
        "exogibbs_status": np.full(count, "", dtype="<U80"),
        "exogibbs_gas_fractions": np.full(
            (count, len(ITO_SPECIES)), np.nan, dtype=np.float64
        ),
        "exogibbs_atomic_gas_fractions": np.full(
            (count, len(ATOMIC_GAS_SPECIES)), np.nan, dtype=np.float64
        ),
        "exogibbs_condensate_amounts": np.full(
            (count, len(CONDENSATE_SPECIES)), np.nan, dtype=np.float64
        ),
        "exogibbs_abundance_scale": np.full(count, np.nan),
        "exogibbs_reactive_pressure_bar": np.full(count, np.nan),
        "exogibbs_fixed_point_iterations": np.full(count, -1, dtype=np.int64),
        "exogibbs_solver_iterations": np.full(count, -1, dtype=np.int64),
        "fastchem_done": np.zeros(count, dtype=bool),
        "fastchem_converged": np.zeros(count, dtype=bool),
        "fastchem_status": np.full(count, "", dtype="<U80"),
        "fastchem_gas_fractions": np.full(
            (count, len(ITO_SPECIES)), np.nan, dtype=np.float64
        ),
        "fastchem_atomic_gas_fractions": np.full(
            (count, len(ATOMIC_GAS_SPECIES)), np.nan, dtype=np.float64
        ),
        "fastchem_condensate_amounts": np.full(
            (count, len(CONDENSATE_SPECIES)), np.nan, dtype=np.float64
        ),
        "fastchem_abundance_scale": np.full(count, np.nan),
        "fastchem_reactive_pressure_bar": np.full(count, np.nan),
        "fastchem_fixed_point_iterations": np.full(count, -1, dtype=np.int64),
        "fastchem_solver_iterations": np.full(count, -1, dtype=np.int64),
    }


def _load_checkpoint(
    path: Path,
    expected: Mapping[str, np.ndarray],
) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as archive:
        state = {name: archive[name].copy() for name in archive.files}
    required = set(expected)
    if set(state) != required:
        raise ValueError("The checkpoint schema fields do not match this script.")
    for name in (
        "schema",
        "input_sha256",
        "fastchem_executable_sha256",
        "script_sha256",
        "gas_logk_sha256",
        "condensate_logk_sha256",
        "target_indices",
        "source_layers",
        "layers",
        "pressure_bar",
        "temperature_k",
        "ito_gas_fractions",
        "input_abundances",
    ):
        if not np.array_equal(state[name], expected[name]):
            raise ValueError(f"Checkpoint input mismatch for {name!r}.")
    return state


def _save_checkpoint(path: Path, state: Mapping[str, np.ndarray]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = tempfile.NamedTemporaryFile(
        mode="wb",
        prefix=f".{path.name}.",
        suffix=".tmp",
        dir=path.parent,
        delete=False,
    )
    temporary_path = Path(temporary.name)
    try:
        with temporary:
            np.savez_compressed(temporary, **state)
        os.replace(temporary_path, path)
    finally:
        if temporary_path.exists():
            temporary_path.unlink()


def _store_layer_result(
    state: dict[str, np.ndarray],
    solver: str,
    index: int,
    result: SolverLayerResult,
) -> None:
    state[f"{solver}_done"][index] = True
    state[f"{solver}_converged"][index] = result.converged
    state[f"{solver}_status"][index] = result.status
    state[f"{solver}_gas_fractions"][index] = result.gas_fractions
    state[f"{solver}_atomic_gas_fractions"][index] = (
        result.atomic_gas_fractions
    )
    state[f"{solver}_condensate_amounts"][index] = result.condensate_amounts
    state[f"{solver}_abundance_scale"][index] = result.abundance_scale
    state[f"{solver}_reactive_pressure_bar"][index] = (
        result.reactive_pressure_bar
    )
    state[f"{solver}_fixed_point_iterations"][index] = (
        result.fixed_point_iterations
    )
    state[f"{solver}_solver_iterations"][index] = result.solver_iterations


def _run_calculations(
    args: argparse.Namespace,
    profile: ItoProfile,
    state: dict[str, np.ndarray],
) -> None:
    run_exogibbs = args.solver in {"both", "exogibbs"}
    run_fastchem = args.solver in {"both", "fastchem"}
    setup = build_ito_exogibbs_setup() if run_exogibbs else None
    newly_completed = 0
    with tempfile.TemporaryDirectory(prefix="exogibbs_ito_2025_") as directory:
        temporary_root = Path(directory)
        gas_logk, condensate_logk = _write_filtered_fastchem_inputs(
            temporary_root
        )
        abundance_file = temporary_root / "element_abundances.dat"
        for output_index, target_index in enumerate(state["target_indices"]):
            target = int(target_index)
            previous = profile.gas_fractions[target - 1]
            initial_pressure = _initial_reactive_pressure(
                float(profile.pressure_bar[target]), previous
            )
            abundance = state["input_abundances"][output_index]
            layer = int(profile.layer[target])
            retry_exogibbs = (
                args.retry_failed
                and state["exogibbs_done"][output_index]
                and not state["exogibbs_converged"][output_index]
            )
            if run_exogibbs and (
                not state["exogibbs_done"][output_index] or retry_exogibbs
            ):
                assert setup is not None
                result = solve_exogibbs_layer(
                    setup,
                    temperature_k=float(profile.temperature_k[target]),
                    total_pressure_bar=float(profile.pressure_bar[target]),
                    abundance=abundance,
                    initial_reactive_pressure_bar=initial_pressure,
                )
                _store_layer_result(state, "exogibbs", output_index, result)
                newly_completed += 1
                print(
                    f"Layer {layer}: ExoGibbs {result.status}; "
                    f"He fixed point {result.fixed_point_iterations}",
                    flush=True,
                )
            retry_fastchem = (
                args.retry_failed
                and state["fastchem_done"][output_index]
                and not state["fastchem_converged"][output_index]
            )
            if run_fastchem and (
                not state["fastchem_done"][output_index] or retry_fastchem
            ):
                _write_fastchem_abundance(abundance_file, abundance)
                result = solve_fastchem_layer(
                    args.fastchem_executable,
                    gas_logk_file=gas_logk,
                    condensate_logk_file=condensate_logk,
                    abundance_file=abundance_file,
                    temperature_k=float(profile.temperature_k[target]),
                    total_pressure_bar=float(profile.pressure_bar[target]),
                    initial_reactive_pressure_bar=initial_pressure,
                )
                _store_layer_result(state, "fastchem", output_index, result)
                newly_completed += 1
                print(
                    f"Layer {layer}: FastChem {result.status}; "
                    f"He fixed point {result.fixed_point_iterations}",
                    flush=True,
                )
            if newly_completed >= args.checkpoint_every:
                _save_checkpoint(args.checkpoint, state)
                newly_completed = 0
    _save_checkpoint(args.checkpoint, state)


def _write_table(path: Path, state: Mapping[str, np.ndarray]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    header = [
        "source_layer",
        "target_layer",
        "pressure_bar",
        "temperature_k",
        *[f"input_b_{element}" for element in ELEMENTS],
    ]
    for source in ("ito", "exogibbs", "fastchem"):
        header.extend(f"{source}_x_{species}" for species in ITO_SPECIES)
    for source in ("exogibbs", "fastchem"):
        header.extend(
            f"{source}_x_{normalize_species_name(species)}"
            for species in ATOMIC_GAS_SPECIES
        )
        header.extend(
            f"{source}_amount_{species}" for species in CONDENSATE_SPECIES
        )
        header.extend(
            (
                f"{source}_input_abundance_scale",
                f"{source}_reactive_pressure_bar",
                f"{source}_fixed_point_iterations",
                f"{source}_solver_iterations",
                f"{source}_converged",
                f"{source}_status",
            )
        )
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(header)
        for index in range(state["layers"].size):
            row: list[Any] = [
                int(state["source_layers"][index]),
                int(state["layers"][index]),
                f"{state['pressure_bar'][index]:.17e}",
                f"{state['temperature_k'][index]:.17e}",
                *[
                    f"{value:.17e}"
                    for value in state["input_abundances"][index]
                ],
            ]
            for source in ("ito", "exogibbs", "fastchem"):
                key = (
                    "ito_gas_fractions"
                    if source == "ito"
                    else f"{source}_gas_fractions"
                )
                row.extend(f"{value:.17e}" for value in state[key][index])
            for source in ("exogibbs", "fastchem"):
                row.extend(
                    f"{value:.17e}"
                    for value in state[f"{source}_atomic_gas_fractions"][index]
                )
                row.extend(
                    f"{value:.17e}"
                    for value in state[f"{source}_condensate_amounts"][index]
                )
                row.extend(
                    (
                        f"{state[f'{source}_abundance_scale'][index]:.17e}",
                        f"{state[f'{source}_reactive_pressure_bar'][index]:.17e}",
                        int(state[f"{source}_fixed_point_iterations"][index]),
                        (
                            int(state[f"{source}_solver_iterations"][index])
                            if state[f"{source}_solver_iterations"][index] >= 0
                            else ""
                        ),
                        bool(state[f"{source}_converged"][index]),
                        str(state[f"{source}_status"][index]),
                    )
                )
            writer.writerow(row)


def _plot_values(values: np.ndarray) -> np.ndarray:
    """Clip finite non-negative values only for readable logarithmic plots."""

    array = np.asarray(values, dtype=np.float64)
    output = np.full(array.shape, np.nan, dtype=np.float64)
    valid = np.isfinite(array) & (array >= 0.0)
    output[valid] = np.maximum(array[valid], PLOT_FLOOR)
    return output


def make_comparison_figure(state: Mapping[str, np.ndarray]):
    """Make one species-resolved profile panel for each Ito gas."""

    fig, axes = plt.subplots(3, 2, figsize=(12.0, 12.5), sharey=True)
    pressure = state["pressure_bar"]
    method_styles = {
        "Ito et al. (2025)": (
            "ito_gas_fractions",
            "black",
            "-",
            2.2,
        ),
        "ExoGibbs": ("exogibbs_gas_fractions", "tab:blue", "--", 1.8),
        "FastChem 4": ("fastchem_gas_fractions", "tab:orange", ":", 2.0),
    }
    active_methods: set[str] = set()
    for species_index, (axis, species) in enumerate(
        zip(axes.flat, ITO_SPECIES)
    ):
        clipped_count = 0
        for label, (key, color, linestyle, linewidth) in method_styles.items():
            values = state[key][:, species_index]
            if not np.any(np.isfinite(values)):
                continue
            active_methods.add(label)
            clipped_count += int(
                np.count_nonzero(
                    np.isfinite(values) & (values >= 0.0) & (values < PLOT_FLOOR)
                )
            )
            axis.plot(
                _plot_values(values),
                pressure,
                color=color,
                linestyle=linestyle,
                linewidth=linewidth,
            )
        axis.set_xscale("log")
        axis.set_yscale("log")
        axis.set_title(species)
        axis.set_xlabel("Total-pressure gas mole fraction")
        axis.grid(alpha=0.25)
        if species_index % 2 == 0:
            axis.set_ylabel("Pressure [bar]")
        if clipped_count:
            axis.text(
                0.98,
                0.03,
                f"{clipped_count} values below plot floor",
                transform=axis.transAxes,
                ha="right",
                va="bottom",
                fontsize=7.5,
                color="0.4",
            )
    axes.flat[0].set_ylim(
        float(np.max(pressure)) * 1.02,
        float(np.min(pressure)) / 1.02,
    )
    method_handles = [
        Line2D(
            (0,),
            (0,),
            color=color,
            linestyle=linestyle,
            linewidth=linewidth,
            label=label,
        )
        for label, (_key, color, linestyle, linewidth) in method_styles.items()
        if label in active_methods
    ]
    fig.legend(
        handles=method_handles,
        title="Calculation",
        loc="upper center",
        bbox_to_anchor=(0.5, 0.953),
        ncol=len(method_handles),
    )
    first_layer = int(state["layers"][0])
    last_layer = int(state["layers"][-1])
    title_details = [
        f"Layers {first_layer}--{last_layer}",
        "Layer 1 excluded",
    ]
    if "ExoGibbs" in active_methods:
        converged = int(np.count_nonzero(state["exogibbs_converged"]))
        title_details.append(
            f"ExoGibbs converged {converged}/{state['layers'].size}"
        )
    fig.suptitle(
        "Ito et al. (2025) one-grid rainout comparison\n"
        + "; ".join(title_details),
        fontsize=14,
    )
    fig.text(
        0.5,
        0.018,
        "Each target uses the Ito gas inventory one grid below. "
        "H/O/Si chemistry includes SiO and SiO2 condensates; "
        r"$x_{\rm He}/x_{\rm H2}=0.189655$ is fixed outside chemistry."
        "\nExoGibbs uses Ito's exact five-molecule network; FastChem necessarily "
        f"adds H/O/Si atomic gases. Plot floor: {PLOT_FLOOR:.0e}.",
        ha="center",
        fontsize=8.5,
    )
    fig.tight_layout(rect=(0.02, 0.06, 0.98, 0.91), h_pad=2.0, w_pad=1.4)
    return fig


def _finite_log_differences(
    model: np.ndarray,
    reference: np.ndarray,
) -> np.ndarray:
    model_values = np.asarray(model, dtype=np.float64)
    reference_values = np.asarray(reference, dtype=np.float64)
    if model_values.shape != reference_values.shape:
        raise ValueError("Model and reference arrays must have the same shape.")
    valid = (
        np.isfinite(model_values)
        & (model_values >= 0.0)
        & np.isfinite(reference_values)
        & (reference_values > 0.0)
    )
    output = np.full(model_values.shape, np.nan)
    compared_model = np.maximum(model_values[valid], PLOT_FLOOR)
    output[valid] = np.log10(compared_model / reference_values[valid])
    return output


def _trailing_plateau_start_index(values: np.ndarray) -> Optional[int]:
    """Return the first row of an exact repeated trailing value, if present."""

    array = np.asarray(values, dtype=np.float64)
    if array.ndim != 1 or array.size < 2 or not np.isfinite(array[-1]):
        return None
    start = array.size - 1
    while start > 0 and array[start - 1] == array[-1]:
        start -= 1
    return start if start < array.size - 1 else None


def _ito_plateau_stops(reference: np.ndarray) -> tuple[int, ...]:
    stops = []
    for index in range(len(ITO_SPECIES)):
        start = _trailing_plateau_start_index(reference[:, index])
        stops.append(reference.shape[0] if start is None else start)
    return tuple(stops)


def _difference_summary(
    model: np.ndarray,
    reference: np.ndarray,
    *,
    stop_indices: Optional[Sequence[int]] = None,
) -> dict[str, Any]:
    differences = _finite_log_differences(model, reference)
    rows = {}
    for index, species in enumerate(ITO_SPECIES):
        stop = differences.shape[0] if stop_indices is None else stop_indices[index]
        model_values = np.asarray(model[:stop, index], dtype=np.float64)
        reference_values = np.asarray(reference[:stop, index], dtype=np.float64)
        reference_valid = np.isfinite(reference_values) & (reference_values > 0.0)
        compared = (
            reference_valid
            & np.isfinite(model_values)
            & (model_values >= 0.0)
        )
        values = np.abs(differences[:stop, index])
        finite = values[np.isfinite(values)]
        rows[species] = {
            "reference_count": int(np.count_nonzero(reference_valid)),
            "compared_count": int(np.count_nonzero(compared)),
            "model_missing_or_invalid_count": int(
                np.count_nonzero(reference_valid & ~compared)
            ),
            "model_zero_count": int(np.count_nonzero(compared & (model_values == 0.0))),
            "model_zero_or_below_plot_floor_count": int(
                np.count_nonzero(compared & (model_values <= PLOT_FLOOR))
            ),
            "maximum_absolute_dex": (
                float(np.max(finite)) if finite.size else None
            ),
            "median_absolute_dex": (
                float(np.median(finite)) if finite.size else None
            ),
        }
    return rows


def _finite_maximum_or_none(values: np.ndarray) -> Optional[float]:
    finite = np.asarray(values, dtype=np.float64)
    finite = finite[np.isfinite(finite)]
    return float(np.max(finite)) if finite.size else None


def _plateau_summary(
    reference: np.ndarray,
    layers: np.ndarray,
) -> dict[str, Any]:
    output = {}
    for index, species in enumerate(ITO_SPECIES):
        start = _trailing_plateau_start_index(reference[:, index])
        output[species] = {
            "start_layer": None if start is None else int(layers[start]),
            "row_count": 0 if start is None else int(reference.shape[0] - start),
        }
    return output


def _write_summary(
    path: Path,
    state: Mapping[str, np.ndarray],
    *,
    input_path: Path,
    input_sha256: str,
    executable_path: Path,
    executable_sha256: str,
    script_sha256: str,
    gas_logk_sha256: str,
    condensate_logk_sha256: str,
) -> None:
    reference = state["ito_gas_fractions"]
    plateau_stops = _ito_plateau_stops(reference)
    payload = {
        "schema": CHECKPOINT_SCHEMA,
        "input": {
            "workbook": str(input_path.resolve()),
            "sha256": input_sha256,
            "source_paper": "Ito et al. 2025, ApJ 987, 174, Figure 2(a)",
            "source_layer_for_target": "target layer minus one",
            "excluded_layers": [1],
        },
        "reproducibility": {
            "script_sha256": script_sha256,
            "gas_logk_sha256": gas_logk_sha256,
            "condensate_logk_sha256": condensate_logk_sha256,
        },
        "chemistry": {
            "elements": list(ELEMENTS),
            "ito_gases": list(ITO_SPECIES),
            "exogibbs_reactive_gases": list(EXOGIBBS_GAS_SPECIES),
            "fastchem_reactive_gases": list(COMMON_GAS_SPECIES),
            "condensates": list(CONDENSATE_SPECIES),
            "helium_treatment": "fixed outside equilibrium chemistry",
            "helium_to_h2_number_ratio": HE_TO_H2_RATIO,
            "reactive_pressure_fixed_point_rtol": FIXED_POINT_RTOL,
            "exogibbs_element_abundance_gauge_sequence": list(
                EXOGIBBS_ELEMENT_GAUGES
            ),
            "exogibbs_gauge_treatment": (
                "uniformly scale all H/O/Si input amounts for numerical "
                "conditioning, then scale condensate amounts back"
            ),
            "thermochemistry": "packaged FastChem 4.0.3 logK tables",
        },
        "fastchem": {
            "executable": str(executable_path.resolve()),
            "executable_sha256": executable_sha256,
        },
        "profile": {
            "layer_count": int(state["layers"].size),
            "first_layer": int(state["layers"][0]),
            "last_layer": int(state["layers"][-1]),
            "pressure_bar_min": float(np.min(state["pressure_bar"])),
            "pressure_bar_max": float(np.max(state["pressure_bar"])),
            "temperature_k_min": float(np.min(state["temperature_k"])),
            "temperature_k_max": float(np.max(state["temperature_k"])),
            "ito_exact_trailing_plateaus": _plateau_summary(
                reference, state["layers"]
            ),
        },
        "exogibbs": {
            "attempted_layers": int(np.count_nonzero(state["exogibbs_done"])),
            "converged_layers": int(
                np.count_nonzero(state["exogibbs_converged"])
            ),
            "maximum_atomic_gas_fraction": _finite_maximum_or_none(
                state["exogibbs_atomic_gas_fractions"]
            ),
            "solver_iterations_recorded": False,
            "difference_from_ito": {
                "full_profile": _difference_summary(
                    state["exogibbs_gas_fractions"], reference
                ),
                "before_ito_trailing_plateau": _difference_summary(
                    state["exogibbs_gas_fractions"],
                    reference,
                    stop_indices=plateau_stops,
                ),
            },
        },
        "fastchem4": {
            "attempted_layers": int(np.count_nonzero(state["fastchem_done"])),
            "converged_layers": int(
                np.count_nonzero(state["fastchem_converged"])
            ),
            "maximum_atomic_gas_fraction": _finite_maximum_or_none(
                state["fastchem_atomic_gas_fractions"]
            ),
            "difference_from_ito": {
                "full_profile": _difference_summary(
                    state["fastchem_gas_fractions"], reference
                ),
                "before_ito_trailing_plateau": _difference_summary(
                    state["fastchem_gas_fractions"],
                    reference,
                    stop_indices=plateau_stops,
                ),
            },
        },
        "exogibbs_vs_fastchem4": {
            "full_profile": _difference_summary(
                state["exogibbs_gas_fractions"],
                state["fastchem_gas_fractions"],
            ),
            "before_ito_trailing_plateau": _difference_summary(
                state["exogibbs_gas_fractions"],
                state["fastchem_gas_fractions"],
                stop_indices=plateau_stops,
            ),
        },
        "plot": {
            "mole_fraction_floor": PLOT_FLOOR,
            "raw_values_are_preserved_in_csv_and_checkpoint": True,
        },
        "interpretation_limits": [
            "Ito condensate amounts are unavailable and are not plotted.",
            (
                "Ito and the shared FastChem4 table use different SiO2 "
                "thermochemical sources."
            ),
            (
                "FastChem necessarily adds elemental reference gases that are "
                "absent from the exact Ito/ExoGibbs reaction network."
            ),
            (
                "Upper-profile Ito fractions contain finite-output plateaus "
                "and should not be treated as unlimited numerical precision."
            ),
        ],
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def _print_summary(state: Mapping[str, np.ndarray]) -> None:
    print("Ito et al. (2025) one-grid rainout comparison")
    print(
        f"  layers: {int(state['layers'][0])}--{int(state['layers'][-1])} "
        f"({state['layers'].size} targets; Layer 1 excluded)"
    )
    for solver in ("exogibbs", "fastchem"):
        print(
            f"  {solver}: completed "
            f"{np.count_nonzero(state[f'{solver}_done'])}/"
            f"{state['layers'].size}; converged "
            f"{np.count_nonzero(state[f'{solver}_converged'])}/"
            f"{state['layers'].size}"
        )
    reference = state["ito_gas_fractions"]
    stops = _ito_plateau_stops(reference)
    for label, model, comparison in (
        ("ExoGibbs/Ito", state["exogibbs_gas_fractions"], reference),
        ("FastChem/Ito", state["fastchem_gas_fractions"], reference),
        (
            "ExoGibbs/FastChem",
            state["exogibbs_gas_fractions"],
            state["fastchem_gas_fractions"],
        ),
    ):
        rows = _difference_summary(
            model,
            comparison,
            stop_indices=stops,
        )
        maxima = [
            row["maximum_absolute_dex"]
            for row in rows.values()
            if row["maximum_absolute_dex"] is not None
        ]
        if maxima:
            print(
                f"  maximum {label} difference before Ito plateaus: "
                f"{max(maxima):.4g} dex"
            )


def _validate_requested_solvers(
    state: Mapping[str, np.ndarray],
    solver: str,
) -> None:
    """Fail when any requested layer is incomplete or unconverged."""

    layers = np.asarray(state["layers"])
    if layers.ndim != 1 or layers.size == 0:
        raise RuntimeError(
            "Ito comparison release criteria failed: invalid layer array"
        )
    requested = ("exogibbs", "fastchem") if solver == "both" else (solver,)
    failures = []
    for name in requested:
        done = np.asarray(state[f"{name}_done"])
        converged = np.asarray(state[f"{name}_converged"])
        if (
            done.dtype.kind != "b"
            or converged.dtype.kind != "b"
            or done.shape != layers.shape
            or converged.shape != layers.shape
        ):
            failures.append(f"{name}: invalid completion arrays")
            continue
        incomplete = np.flatnonzero(~done)
        unconverged = np.flatnonzero(done & ~converged)
        if incomplete.size or unconverged.size:
            failures.append(
                f"{name}: incomplete rows={incomplete.tolist()}, "
                f"unconverged rows={unconverged.tolist()}"
            )
    if failures:
        raise RuntimeError(
            "Ito comparison release criteria failed: " + "; ".join(failures)
        )


def main() -> None:
    args = _parse_args()
    input_path = args.input.resolve(strict=True)
    executable_path = args.fastchem_executable.resolve(strict=True)
    if not executable_path.is_file() or not os.access(executable_path, os.X_OK):
        raise ValueError(
            f"FastChem executable is not an executable file: {executable_path}."
        )
    profile = load_ito_profile(input_path)
    target_indices = _target_indices(
        profile,
        step=args.layer_step,
        max_layers=args.max_layers,
    )
    input_sha256 = _sha256(input_path)
    executable_sha256 = _sha256(executable_path)
    script_sha256 = _sha256(Path(__file__).resolve())
    gas_logk_sha256 = _sha256(GAS_LOGK_FILE)
    condensate_logk_sha256 = _sha256(CONDENSATE_LOGK_FILE)
    expected = _new_checkpoint(
        profile,
        target_indices,
        input_sha256=input_sha256,
        executable_sha256=executable_sha256,
        script_sha256=script_sha256,
        gas_logk_sha256=gas_logk_sha256,
        condensate_logk_sha256=condensate_logk_sha256,
    )
    if args.checkpoint.exists() and not args.no_resume:
        state = _load_checkpoint(args.checkpoint, expected)
        print(f"resuming checkpoint: {args.checkpoint}", flush=True)
    else:
        state = expected
    args.figure.unlink(missing_ok=True)
    _run_calculations(args, profile, state)
    _write_table(args.table, state)
    _write_summary(
        args.summary,
        state,
        input_path=input_path,
        input_sha256=input_sha256,
        executable_path=executable_path,
        executable_sha256=executable_sha256,
        script_sha256=script_sha256,
        gas_logk_sha256=gas_logk_sha256,
        condensate_logk_sha256=condensate_logk_sha256,
    )
    _validate_requested_solvers(state, args.solver)
    figure = make_comparison_figure(state)
    args.figure.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(args.figure, dpi=220, bbox_inches="tight")
    if args.show:
        plt.show()
    plt.close(figure)
    _print_summary(state)
    print(
        "  release gate: requested solver layers complete and converged "
        f"({args.solver})"
    )
    print(f"figure: {args.figure}")
    print(f"table: {args.table}")
    print(f"summary: {args.summary}")
    print(f"checkpoint: {args.checkpoint}")


if __name__ == "__main__":
    main()
