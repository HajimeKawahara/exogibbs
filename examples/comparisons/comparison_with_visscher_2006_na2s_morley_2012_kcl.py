"""Compare low-temperature KCl and Na2S condensation benchmarks.

Two independent reduced local-equilibrium systems use Lodders (2003) solar
abundances.  H/He/K/Cl permits only KCl to condense, while H/He/Na/S permits
only Na2S to condense.  This isolates each cloud reaction from unrelated
refractory condensates and from the other benchmark.

ExoGibbs always runs.  If a standalone FastChem 4 executable is supplied,
FastChem independently solves the same temperatures, pressure, abundances,
gas catalog, condensate catalog, and JANAF-derived thermochemistry.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import os
from pathlib import Path
import re
import sys
import tempfile
from typing import Callable, Iterable, Sequence


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
import numpy as np

from benchmarks.fastchem4.comparison import align_species_values
from benchmarks.fastchem4.fastchem_executable import run_fastchem_executable
from exogibbs.api.condensate import (
    CondensateEquilibriumOptions,
    solve_profile as solve_condensate_profile,
)
from exogibbs.equilibrium.condensate.setup import (
    CondensateChemicalSetup,
    build_condensate_chemical_setup,
)
from exogibbs.presets.fastchem4_cond import condensate_chemical_setup
from exogibbs.thermo.models import ChemicalSetup
from exogibbs.utils.fastchem_parity import normalize_species_name


config.update("jax_enable_x64", True)

DATA_ROOT = REPOSITORY_ROOT / "src" / "exogibbs" / "data" / "FastChem4"
GAS_LOGK_FILE = DATA_ROOT / "logK" / "logK_wo_ions.dat"
CONDENSATE_LOGK_FILE = DATA_ROOT / "logK" / "logK_condensates.dat"
ELEMENT_ABUNDANCE_FILE = (
    DATA_ROOT / "element_abundances" / "lodders_2003.dat"
)
DEFAULT_FIGURE = (
    REPOSITORY_ROOT
    / "results"
    / "visscher_2006_na2s_morley_2012_kcl"
    / "condensation_comparison.png"
)

PRESSURE_BAR = 1.0
FASTCHEM_GAS_AGREEMENT_TOLERANCE_DEX = 1.0e-3
KCL_VAPOR_PRESSURE_RELATIVE_TOLERANCE = 3.0e-2
NA2S_SULFUR_FRACTION_RELATIVE_TOLERANCE = 2.0e-2
DEFAULT_TEMPERATURES_K = np.unique(
    np.concatenate(
        (
            np.arange(700.0, 1101.0, 10.0),
            np.arange(785.0, 811.0, 1.0),
            np.arange(985.0, 1011.0, 1.0),
        )
    )
)

_FASTCHEM_RECORD = re.compile(r"^\s*([^\s:#]+)\s+[^:]*:\s*")


@dataclass(frozen=True)
class DemoSolution:
    """One solver's reduced-system equilibrium profile."""

    temperatures_k: np.ndarray
    gas_x: np.ndarray
    condensate_amounts: np.ndarray
    converged: np.ndarray
    status: tuple[str, ...]


@dataclass(frozen=True)
class BenchmarkCase:
    """One deliberately reduced literature-condensation system."""

    label: str
    elements: tuple[str, ...]
    gas_species: tuple[str, ...]
    molecule_species: tuple[str, ...]
    condensate_species: str


KCL_CASE = BenchmarkCase(
    label="KCl",
    elements=("H", "He", "K", "Cl"),
    gas_species=("H1", "He1", "K1", "Cl1", "H2", "Cl1K1"),
    molecule_species=("H2", "Cl1K1"),
    condensate_species="KCl(s,l)",
)
NA2S_CASE = BenchmarkCase(
    label="Na2S",
    elements=("H", "He", "Na", "S"),
    gas_species=("H1", "He1", "Na1", "S1", "H2", "H2S1"),
    molecule_species=("H2", "H2S1"),
    condensate_species="Na2S(s,l)",
)
BENCHMARK_CASES = (KCL_CASE, NA2S_CASE)


def kcl_condensation_temperature(
    pressure_bar: float | np.ndarray,
    metallicity_dex: float = 0.0,
) -> np.ndarray:
    """Return the Morley et al. (2012) KCl condensation fit in kelvin."""

    pressure = np.asarray(pressure_bar, dtype=np.float64)
    denominator = (
        12.479
        - 0.879 * np.log10(pressure)
        - 0.879 * float(metallicity_dex)
    )
    return 1.0e4 / denominator


def na2s_condensation_temperature(
    pressure_bar: float | np.ndarray,
    metallicity_dex: float = 0.0,
) -> np.ndarray:
    """Return the Visscher et al. (2006) Na2S condensation fit in kelvin."""

    pressure = np.asarray(pressure_bar, dtype=np.float64)
    denominator = (
        10.05
        - 0.72 * np.log10(pressure)
        - 1.08 * float(metallicity_dex)
    )
    return 1.0e4 / denominator


def kcl_saturation_pressure(temperature_k: Sequence[float] | np.ndarray) -> np.ndarray:
    """Return the Morley et al. (2012) KCl vapor pressure in bar."""

    temperature = np.asarray(temperature_k, dtype=np.float64)
    return 10.0 ** (7.611 - 11382.0 / temperature)


def sodium_saturation_pressure(
    temperature_k: Sequence[float] | np.ndarray,
    metallicity_dex: float = 0.0,
) -> np.ndarray:
    """Return the Morley et al. (2012) Na pressure over Na2S in bar."""

    temperature = np.asarray(temperature_k, dtype=np.float64)
    return 10.0 ** (
        8.550 - 13889.0 / temperature - 0.5 * float(metallicity_dex)
    )


def _subset_hvector(
    function: Callable[[jnp.ndarray], jnp.ndarray],
    indices: tuple[int, ...],
) -> Callable[[jnp.ndarray], jnp.ndarray]:
    index_array = jnp.asarray(indices, dtype=jnp.int32)

    @jax.jit
    def hvector(temperature: jnp.ndarray) -> jnp.ndarray:
        return function(temperature)[..., index_array]

    return hvector


def build_reduced_setup(case: BenchmarkCase) -> CondensateChemicalSetup:
    """Build one minimal gas and single-condensate benchmark catalog."""

    full = condensate_chemical_setup(
        species_default_elements=False,
        element_file="FastChem4/element_abundances/lodders_2003.dat",
        silent=True,
    )
    element_indices = tuple(
        full.elements.index(name) for name in case.elements
    )
    gas_indices = tuple(
        full.gas_species.index(name) for name in case.gas_species
    )
    condensate_indices = (
        full.condensate_species.index(case.condensate_species),
    )

    gas_matrix = np.asarray(full.formula_matrix)[
        np.ix_(element_indices, gas_indices)
    ]
    condensate_matrix = np.asarray(full.formula_matrix_cond)[
        np.ix_(element_indices, condensate_indices)
    ]
    reference = np.asarray(full.gas_setup.element_vector_reference)[
        np.asarray(element_indices)
    ]
    reference = reference / np.sum(reference)

    gas_setup = ChemicalSetup(
        formula_matrix=jnp.asarray(gas_matrix, dtype=jnp.float64),
        hvector_func=_subset_hvector(
            full.gas_setup.hvector_func,
            gas_indices,
        ),
        elements=case.elements,
        species=case.gas_species,
        element_vector_reference=jnp.asarray(reference, dtype=jnp.float64),
        metadata={
            "source": f"FastChem4/JANAF reduced {case.label} benchmark",
            "element_abundances": "Lodders (2003)",
        },
    )

    validity = full.condensate_setup.temperature_validity_upper
    if validity is None:
        raise ValueError("FastChem4 condensates have no validity bounds.")
    condensate_setup = ChemicalSetup(
        formula_matrix=jnp.asarray(condensate_matrix, dtype=jnp.float64),
        hvector_func=_subset_hvector(
            full.condensate_setup.hvector_func,
            condensate_indices,
        ),
        elements=case.elements,
        species=(case.condensate_species,),
        temperature_validity_upper=tuple(
            float(validity[index]) for index in condensate_indices
        ),
        metadata={
            "source": f"FastChem4/JANAF reduced {case.label} benchmark",
            "condensate_catalog_mode": "reduced_explicit",
        },
    )
    return build_condensate_chemical_setup(
        gas_setup=gas_setup,
        condensate_setup=condensate_setup,
    )


def solar_element_budget(setup: CondensateChemicalSetup) -> jnp.ndarray:
    """Return the normalized Lodders (2003) benchmark inventory."""

    reference = setup.gas_setup.element_vector_reference
    if reference is None:
        raise ValueError("The reduced setup has no reference element budget.")
    budget = jnp.asarray(reference, dtype=jnp.float64)
    return budget / jnp.sum(budget)


def solve_exogibbs(
    setup: CondensateChemicalSetup,
    temperatures_k: Sequence[float] | np.ndarray,
) -> DemoSolution:
    """Solve the reduced profile with the production ExoGibbs API."""

    temperatures = np.asarray(temperatures_k, dtype=np.float64)
    pressures = np.full_like(temperatures, PRESSURE_BAR)
    profile = solve_condensate_profile(
        setup,
        T=jnp.asarray(temperatures, dtype=jnp.float64),
        P=jnp.asarray(pressures, dtype=jnp.float64),
        b=solar_element_budget(setup),
        options=CondensateEquilibriumOptions(),
    )
    jax.block_until_ready(profile.batched_arrays)
    solution = DemoSolution(
        temperatures_k=temperatures,
        gas_x=np.stack(
            [np.asarray(layer.gas_x, dtype=np.float64) for layer in profile.layers]
        ),
        condensate_amounts=np.stack(
            [
                np.asarray(layer.condensate_amounts, dtype=np.float64)
                for layer in profile.layers
            ]
        ),
        converged=np.asarray(
            [layer.converged for layer in profile.layers],
            dtype=bool,
        ),
        status=tuple(layer.status for layer in profile.layers),
    )
    if not np.all(solution.converged):
        failed = np.flatnonzero(~solution.converged).tolist()
        raise RuntimeError(f"ExoGibbs did not converge at profile rows {failed}.")
    return solution


def _filter_fastchem_records(text: str, selected_names: Iterable[str]) -> str:
    """Keep exact named FastChem records and the original leading comments."""

    selected = set(selected_names)
    lines = text.splitlines()
    starts: list[tuple[int, str]] = []
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
        end = (
            starts[position + 1][0]
            if position + 1 < len(starts)
            else len(lines)
        )
        output.extend(lines[start:end])
    return "\n".join(output).rstrip() + "\n"


def _reduced_element_abundance_text(case: BenchmarkCase) -> str:
    """Filter the packaged Lodders (2003) table to the benchmark elements."""

    values: dict[str, str] = {}
    for line in ELEMENT_ABUNDANCE_FILE.read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        fields = stripped.split()
        if len(fields) >= 2 and fields[0] in case.elements:
            values[fields[0]] = fields[1]
    missing = sorted(set(case.elements) - set(values))
    if missing:
        raise ValueError(f"Missing Lodders (2003) abundances: {missing}.")
    rows = [f"# Reduced Lodders (2003) abundances for the {case.label} demo"]
    rows.extend(f"{element}  {values[element]}" for element in case.elements)
    return "\n".join(rows) + "\n"


def _write_reduced_fastchem_inputs(
    directory: Path,
    case: BenchmarkCase,
) -> tuple[Path, Path, Path]:
    abundance_path = directory / "element_abundances.dat"
    gas_path = directory / "gas_logk.dat"
    condensate_path = directory / "condensate_logk.dat"
    abundance_path.write_text(
        _reduced_element_abundance_text(case),
        encoding="utf-8",
    )
    gas_path.write_text(
        _filter_fastchem_records(
            GAS_LOGK_FILE.read_text(encoding="utf-8"),
            case.molecule_species,
        ),
        encoding="utf-8",
    )
    condensate_path.write_text(
        _filter_fastchem_records(
            CONDENSATE_LOGK_FILE.read_text(encoding="utf-8"),
            (case.condensate_species,),
        ),
        encoding="utf-8",
    )
    return abundance_path, gas_path, condensate_path


def _normalized_catalog(names: Sequence[str]) -> set[str]:
    return {normalize_species_name(str(name)) for name in names}


def solve_fastchem4(
    executable: Path,
    case: BenchmarkCase,
    setup: CondensateChemicalSetup,
    temperatures_k: Sequence[float] | np.ndarray,
) -> DemoSolution:
    """Independently solve the reduced profile with standalone FastChem 4."""

    temperatures = np.asarray(temperatures_k, dtype=np.float64)
    pressures = np.full_like(temperatures, PRESSURE_BAR)
    with tempfile.TemporaryDirectory(
        prefix=f"exogibbs_{case.label.lower()}_fastchem4_"
    ) as directory:
        abundance_path, gas_path, condensate_path = (
            _write_reduced_fastchem_inputs(Path(directory), case)
        )
        result = run_fastchem_executable(
            executable=executable,
            temperatures=temperatures,
            pressures=pressures,
            element_abundance_file=abundance_path,
            gas_logk_file=gas_path,
            condensate_logk_file=condensate_path,
            chemistry_mode="equilibrium_condensation",
        )

    if _normalized_catalog(result.gas_names) != _normalized_catalog(
        setup.gas_species
    ):
        raise RuntimeError("FastChem and ExoGibbs gas catalogs do not match.")
    if _normalized_catalog(result.condensate_names) != _normalized_catalog(
        setup.condensate_species
    ):
        raise RuntimeError(
            "FastChem and ExoGibbs condensate catalogs do not match."
        )
    gas_density = align_species_values(
        setup.gas_species,
        result.gas_names,
        result.gas_number_densities,
    )
    condensate_density = align_species_values(
        setup.condensate_species,
        result.condensate_names,
        result.condensate_number_densities,
    )
    solution = DemoSolution(
        temperatures_k=temperatures,
        gas_x=gas_density / np.sum(gas_density, axis=1, keepdims=True),
        condensate_amounts=(
            condensate_density / result.total_element_density[:, None]
        ),
        converged=np.asarray(
            result.converged & result.elements_conserved,
            dtype=bool,
        ),
        status=tuple(
            f"converged={converged}; elements_conserved={conserved}"
            for converged, conserved in zip(
                result.convergence_status,
                result.element_conservation_status,
            )
        ),
    )
    if not np.all(solution.converged):
        failed = np.flatnonzero(~solution.converged).tolist()
        raise RuntimeError(f"FastChem 4 failed at profile rows {failed}.")
    return solution


def condensation_bracket(
    temperatures_k: Sequence[float] | np.ndarray,
    amounts: Sequence[float] | np.ndarray,
) -> tuple[float, float]:
    """Bracket the transition using a scale-relative condensate floor."""

    temperatures = np.asarray(temperatures_k, dtype=np.float64)
    values = np.asarray(amounts, dtype=np.float64)
    if temperatures.ndim != 1 or values.shape != temperatures.shape:
        raise ValueError("temperatures and amounts must be matching 1D arrays.")
    if np.any(~np.isfinite(values)) or np.any(values < 0.0):
        raise ValueError("amounts must be finite and non-negative.")
    threshold = float(np.max(values)) * 1.0e-12
    positive = values > threshold
    if not np.any(positive) or np.all(positive):
        raise ValueError("The grid must cross the condensation boundary.")
    cold_edge = float(np.max(temperatures[positive]))
    warmer = temperatures[(temperatures > cold_edge) & ~positive]
    if warmer.size == 0:
        raise ValueError("No warmer uncondensed grid point brackets the boundary.")
    return cold_edge, float(np.min(warmer))


def condensed_element_fractions(
    solution: DemoSolution,
    setup: CondensateChemicalSetup,
) -> dict[str, np.ndarray]:
    """Return K, Na, and S inventory fractions held in target condensates."""

    budget = np.asarray(solar_element_budget(setup), dtype=np.float64)
    amounts = solution.condensate_amounts[:, 0]
    if setup.condensate_species == ("KCl(s,l)",):
        return {
            "K in KCl": amounts / budget[setup.elements.index("K")],
        }
    if setup.condensate_species == ("Na2S(s,l)",):
        return {
            "Na in Na2S": (
                2.0 * amounts / budget[setup.elements.index("Na")]
            ),
            "S in Na2S": amounts / budget[setup.elements.index("S")],
        }
    raise ValueError("Unexpected condensate catalog for the benchmark.")


def make_figure(
    *,
    setups: dict[str, CondensateChemicalSetup],
    exogibbs: dict[str, DemoSolution],
    fastchem4: dict[str, DemoSolution] | None = None,
) -> plt.Figure:
    """Create the gas and condensed-inventory comparison figure."""

    figure, axes = plt.subplots(
        2,
        2,
        figsize=(10.0, 7.2),
        sharex=True,
        constrained_layout=True,
    )
    ax_kcl_gas, ax_na_gas, ax_k_cond, ax_na_cond = axes.flat
    temperatures = exogibbs[KCL_CASE.label].temperatures_k
    for solution in exogibbs.values():
        if not np.array_equal(temperatures, solution.temperatures_k):
            raise ValueError("ExoGibbs benchmark temperature grids differ.")
    if fastchem4 is not None:
        for solution in fastchem4.values():
            if not np.array_equal(temperatures, solution.temperatures_k):
                raise ValueError(
                    "ExoGibbs and FastChem temperature grids differ."
                )

    def plot_gas(
        axis: plt.Axes,
        case: BenchmarkCase,
        species: str,
        label: str,
        color: str,
    ) -> None:
        setup = setups[case.label]
        exogibbs_solution = exogibbs[case.label]
        index = setup.gas_species.index(species)
        axis.plot(
            temperatures,
            exogibbs_solution.gas_x[:, index],
            color=color,
            linewidth=2.0,
            label=f"{label}: ExoGibbs",
        )
        if fastchem4 is not None:
            fastchem_solution = fastchem4[case.label]
            axis.plot(
                temperatures,
                fastchem_solution.gas_x[:, index],
                color=color,
                linestyle="none",
                marker="o",
                markerfacecolor="none",
                markersize=3.5,
                markevery=max(1, temperatures.size // 20),
                label=f"{label}: FastChem 4",
            )

    plot_gas(
        ax_kcl_gas,
        KCL_CASE,
        "Cl1K1",
        "KCl(g)",
        "tab:blue",
    )
    plot_gas(ax_na_gas, NA2S_CASE, "Na1", "Na(g)", "tab:orange")
    plot_gas(ax_na_gas, NA2S_CASE, "H2S1", "H2S(g)", "tab:green")

    kcl_reference = float(kcl_condensation_temperature(PRESSURE_BAR))
    na2s_reference = float(na2s_condensation_temperature(PRESSURE_BAR))
    kcl_cold = temperatures <= kcl_reference
    na2s_cold = temperatures <= na2s_reference
    ax_kcl_gas.plot(
        temperatures[kcl_cold],
        kcl_saturation_pressure(temperatures[kcl_cold]) / PRESSURE_BAR,
        color="black",
        linestyle="--",
        linewidth=1.4,
        label="Morley vapor-pressure fit",
    )
    ax_na_gas.plot(
        temperatures[na2s_cold],
        sodium_saturation_pressure(temperatures[na2s_cold]) / PRESSURE_BAR,
        color="black",
        linestyle="--",
        linewidth=1.4,
        label="Morley Na-pressure fit",
    )

    exogibbs_fractions = {
        case.label: condensed_element_fractions(
            exogibbs[case.label],
            setups[case.label],
        )
        for case in BENCHMARK_CASES
    }
    fastchem_fractions = None
    if fastchem4 is not None:
        fastchem_fractions = {
            case.label: condensed_element_fractions(
                fastchem4[case.label],
                setups[case.label],
            )
            for case in BENCHMARK_CASES
        }
    fraction_styles = (
        (KCL_CASE, "K in KCl", "tab:blue", ax_k_cond),
        (NA2S_CASE, "Na in Na2S", "tab:orange", ax_na_cond),
        (NA2S_CASE, "S in Na2S", "tab:red", ax_na_cond),
    )
    for case, label, color, axis in fraction_styles:
        axis.plot(
            temperatures,
            exogibbs_fractions[case.label][label],
            color=color,
            linewidth=2.0,
            label=f"{label}: ExoGibbs",
        )
        if fastchem_fractions is not None:
            axis.plot(
                temperatures,
                fastchem_fractions[case.label][label],
                color=color,
                linestyle="none",
                marker="o",
                markerfacecolor="none",
                markersize=3.5,
                markevery=max(1, temperatures.size // 20),
                label=f"{label}: FastChem 4",
            )

    for axis in (ax_kcl_gas, ax_k_cond):
        axis.axvline(
            kcl_reference,
            color="0.35",
            linestyle=":",
            linewidth=1.5,
            label=f"Morley Tcond = {kcl_reference:.1f} K",
        )
    for axis in (ax_na_gas, ax_na_cond):
        axis.axvline(
            na2s_reference,
            color="0.35",
            linestyle=":",
            linewidth=1.5,
            label=f"Visscher Tcond = {na2s_reference:.1f} K",
        )

    for axis in (ax_kcl_gas, ax_na_gas):
        axis.set_yscale("log")
        axis.set_ylabel("Gas mole fraction")
    for axis in (ax_k_cond, ax_na_cond):
        axis.set_ylim(-0.025, 1.025)
        axis.set_ylabel("Fraction of elemental inventory condensed")
        axis.set_xlabel("Temperature (K)")
    for axis in axes.flat:
        axis.grid(alpha=0.25)
        axis.legend(fontsize=7, loc="best")

    ax_kcl_gas.set_title("KCl gas and saturation vapor pressure")
    ax_na_gas.set_title("Na and H2S gas across Na2S condensation")
    ax_k_cond.set_title("K sequestration by KCl")
    ax_na_cond.set_title("Na-limited Na2S and residual sulfur")
    figure.suptitle(
        "KCl and Na2S local-equilibrium condensation\n"
        "1 bar, Lodders (2003) solar abundances, independent reduced systems"
    )
    return figure


def _maximum_key_gas_difference(
    left: DemoSolution,
    right: DemoSolution,
    setup: CondensateChemicalSetup,
) -> float:
    key_species = (
        ("Cl1K1", "H2")
        if setup.condensate_species == ("KCl(s,l)",)
        else ("Na1", "H2S1", "H2")
    )
    indices = [setup.gas_species.index(name) for name in key_species]
    left_values = np.maximum(left.gas_x[:, indices], 1.0e-300)
    right_values = np.maximum(right.gas_x[:, indices], 1.0e-300)
    return float(np.max(np.abs(np.log10(left_values / right_values))))


def _temperature_index(solution: DemoSolution, temperature_k: float) -> int:
    matches = np.flatnonzero(solution.temperatures_k == temperature_k)
    if matches.size != 1:
        raise RuntimeError(
            f"The release profile must contain {temperature_k:g} K exactly."
        )
    return int(matches[0])


def _validate_release_criteria(
    *,
    setups: dict[str, CondensateChemicalSetup],
    exogibbs: dict[str, DemoSolution],
    fastchem4: dict[str, DemoSolution] | None,
) -> None:
    """Require the literature and optional cross-code checks to pass."""

    kcl_setup = setups[KCL_CASE.label]
    kcl_solution = exogibbs[KCL_CASE.label]
    kcl_index = _temperature_index(kcl_solution, 750.0)
    kcl_gas_index = kcl_setup.gas_species.index("Cl1K1")
    calculated_kcl_pressure = (
        kcl_solution.gas_x[kcl_index, kcl_gas_index] * PRESSURE_BAR
    )
    reference_kcl_pressure = float(kcl_saturation_pressure([750.0])[0])
    if not np.isclose(
        calculated_kcl_pressure,
        reference_kcl_pressure,
        rtol=KCL_VAPOR_PRESSURE_RELATIVE_TOLERANCE,
        atol=0.0,
    ):
        raise RuntimeError(
            "The cold KCl gas does not follow the Morley vapor-pressure fit: "
            f"calculated={calculated_kcl_pressure:.6g} bar, "
            f"reference={reference_kcl_pressure:.6g} bar."
        )

    na2s_setup = setups[NA2S_CASE.label]
    na2s_solution = exogibbs[NA2S_CASE.label]
    na2s_index = _temperature_index(na2s_solution, 700.0)
    fractions = condensed_element_fractions(na2s_solution, na2s_setup)
    sodium_fraction = float(fractions["Na in Na2S"][na2s_index])
    sulfur_fraction = float(fractions["S in Na2S"][na2s_index])
    expected_sulfur_fraction = 10.0 ** (6.37 - 7.26) / 2.0
    if sodium_fraction <= 0.99 or sulfur_fraction >= 0.07 or not np.isclose(
        sulfur_fraction,
        expected_sulfur_fraction,
        rtol=NA2S_SULFUR_FRACTION_RELATIVE_TOLERANCE,
        atol=0.0,
    ):
        raise RuntimeError(
            "The cold Na2S solution is not sodium-limited as expected: "
            f"Na fraction={sodium_fraction:.6g}, "
            f"S fraction={sulfur_fraction:.6g}."
        )

    if fastchem4 is None:
        return
    for case in BENCHMARK_CASES:
        exogibbs_solution = exogibbs[case.label]
        fastchem_solution = fastchem4[case.label]
        exogibbs_bracket = condensation_bracket(
            exogibbs_solution.temperatures_k,
            exogibbs_solution.condensate_amounts[:, 0],
        )
        fastchem_bracket = condensation_bracket(
            fastchem_solution.temperatures_k,
            fastchem_solution.condensate_amounts[:, 0],
        )
        gas_difference = _maximum_key_gas_difference(
            exogibbs_solution,
            fastchem_solution,
            setups[case.label],
        )
        if exogibbs_bracket != fastchem_bracket:
            raise RuntimeError(
                f"{case.label} ExoGibbs/FastChem phase brackets differ: "
                f"{exogibbs_bracket} versus {fastchem_bracket}."
            )
        if (
            not np.isfinite(gas_difference)
            or gas_difference > FASTCHEM_GAS_AGREEMENT_TOLERANCE_DEX
        ):
            raise RuntimeError(
                f"{case.label} ExoGibbs/FastChem gas difference is "
                f"{gas_difference:.6g} dex; limit "
                f"{FASTCHEM_GAS_AGREEMENT_TOLERANCE_DEX:.1e} dex."
            )


def print_summary(
    *,
    setups: dict[str, CondensateChemicalSetup],
    exogibbs: dict[str, DemoSolution],
    fastchem4: dict[str, DemoSolution] | None,
) -> None:
    """Print condensation brackets and optional cross-code differences."""

    references = {
        KCL_CASE.label: float(kcl_condensation_temperature(PRESSURE_BAR)),
        NA2S_CASE.label: float(na2s_condensation_temperature(PRESSURE_BAR)),
    }
    print("One-bar solar condensation benchmark")
    for case in BENCHMARK_CASES:
        solution = exogibbs[case.label]
        cold, warm = condensation_bracket(
            solution.temperatures_k,
            solution.condensate_amounts[:, 0],
        )
        print(
            f"{case.condensate_species}: ExoGibbs boundary in "
            f"({cold:.1f}, {warm:.1f}] K; "
            f"literature fit {references[case.label]:.2f} K"
        )
        print(
            f"{case.label} ExoGibbs converged layers: "
            f"{np.count_nonzero(solution.converged)}/"
            f"{solution.converged.size}"
        )
    if fastchem4 is not None:
        for case in BENCHMARK_CASES:
            setup = setups[case.label]
            exogibbs_solution = exogibbs[case.label]
            fastchem_solution = fastchem4[case.label]
            gas_dex = _maximum_key_gas_difference(
                exogibbs_solution,
                fastchem_solution,
                setup,
            )
            condensate_difference = float(
                np.max(
                    np.abs(
                        exogibbs_solution.condensate_amounts
                        - fastchem_solution.condensate_amounts
                    )
                )
            )
            print(
                f"{case.label} ExoGibbs-FastChem 4 maximum key-gas "
                f"difference: {gas_dex:.3e} dex"
            )
            print(
                f"{case.label} ExoGibbs-FastChem 4 maximum condensate-"
                f"amount difference: {condensate_difference:.3e}"
            )
    cross_code = (
        ""
        if fastchem4 is None
        else (
            "; matching FastChem phase brackets and gas <= "
            f"{FASTCHEM_GAS_AGREEMENT_TOLERANCE_DEX:.1e} dex"
        )
    )
    print(
        "Release gate: passed "
        f"(KCl vapor pressure <= "
        f"{KCL_VAPOR_PRESSURE_RELATIVE_TOLERANCE:.0%}; "
        f"Na2S sulfur-fraction error <= "
        f"{NA2S_SULFUR_FRACTION_RELATIVE_TOLERANCE:.0%}{cross_code})"
    )


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Compare the Visscher Na2S and Morley KCl condensation fits with "
            "ExoGibbs and, optionally, standalone FastChem 4."
        )
    )
    parser.add_argument(
        "--fastchem-executable",
        type=Path,
        default=None,
        help="Optional path to a standalone FastChem 4 executable.",
    )
    parser.add_argument(
        "--figure",
        type=Path,
        default=DEFAULT_FIGURE,
        help="Output PNG path.",
    )
    parser.add_argument(
        "--show",
        action="store_true",
        help="Display the figure after writing it.",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    fastchem_executable = None
    if args.fastchem_executable is not None:
        fastchem_executable = args.fastchem_executable.resolve(strict=True)
        if not fastchem_executable.is_file() or not os.access(
            fastchem_executable, os.X_OK
        ):
            raise ValueError(
                "FastChem executable is not an executable file: "
                f"{fastchem_executable}."
            )
    args.figure.unlink(missing_ok=True)
    setups = {
        case.label: build_reduced_setup(case) for case in BENCHMARK_CASES
    }
    exogibbs = {
        case.label: solve_exogibbs(
            setups[case.label],
            DEFAULT_TEMPERATURES_K,
        )
        for case in BENCHMARK_CASES
    }
    fastchem4 = None
    if fastchem_executable is not None:
        fastchem4 = {
            case.label: solve_fastchem4(
                fastchem_executable,
                case,
                setups[case.label],
                DEFAULT_TEMPERATURES_K,
            )
            for case in BENCHMARK_CASES
        }
    _validate_release_criteria(
        setups=setups,
        exogibbs=exogibbs,
        fastchem4=fastchem4,
    )
    print_summary(setups=setups, exogibbs=exogibbs, fastchem4=fastchem4)

    figure = make_figure(
        setups=setups,
        exogibbs=exogibbs,
        fastchem4=fastchem4,
    )
    args.figure.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(args.figure, dpi=200, bbox_inches="tight")
    print(f"figure: {args.figure}")
    if args.show:
        plt.show()
    plt.close(figure)


if __name__ == "__main__":
    main()
