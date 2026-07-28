#!/usr/bin/env python
"""Compare the ExoGibbs production solver with an independent FastChem4 run."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import time
from typing import Any, Mapping, Sequence

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
SOURCE_ROOT = ROOT / "src"
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(SOURCE_ROOT))

DATA_ROOT = SOURCE_ROOT / "exogibbs" / "data" / "FastChem4"
ELEMENT_ABUNDANCE_FILE = DATA_ROOT / "element_abundances" / "asplund_2021.dat"
GAS_LOGK_FILE = DATA_ROOT / "logK" / "logK_wo_ions.dat"
CONDENSATE_LOGK_FILE = DATA_ROOT / "logK" / "logK_condensates.dat"
DEFAULT_OUTPUT = (
    ROOT / "results" / "fastchem4_production_comparison" / "summary.json"
)
DEFAULT_POINT = (1400.0, 0.1)
CONDENSATE_THRESHOLDS = (1.0e-20, 1.0e-12, 1.0e-8)
BOLTZMANN_CGS = 1.380649e-16
SCHEMA = "exogibbs_fastchem4_production_comparison_v1"
FASTCHEM_REFERENCE_TAG = "v4.0.3"
FASTCHEM_REFERENCE_COMMIT = "ae67cbd559bc64a3233a1cee6030b8e6b50520de"
FASTCHEM_FIXED_SETTINGS = {
    "chemistry_accuracy": 1.0e-5,
    "element_conservation_accuracy": 1.0e-4,
    "max_chemistry_iterations": 80_000,
    "max_internal_iterations": 20_000,
}


def _parse_point(text: str) -> tuple[float, float]:
    """Parse a temperature,pressure CLI point."""

    parts = text.split(",")
    if len(parts) != 2:
        raise argparse.ArgumentTypeError(
            "points must use the form TEMPERATURE_K,PRESSURE_BAR"
        )
    try:
        temperature, pressure = (float(part) for part in parts)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(
            "point temperature and pressure must be numeric"
        ) from exc
    if not math.isfinite(temperature) or temperature <= 0.0:
        raise argparse.ArgumentTypeError("point temperature must be positive")
    if not math.isfinite(pressure) or pressure <= 0.0:
        raise argparse.ArgumentTypeError("point pressure must be positive")
    return temperature, pressure


def build_parser() -> argparse.ArgumentParser:
    """Build the command-line parser without importing JAX."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--fastchem-executable",
        type=Path,
        required=True,
        help="Path to an independently built FastChem4 standalone executable.",
    )
    parser.add_argument(
        "--fastchem-version-label",
        default="unknown",
        help="Audited FastChem source version, for example '4.0.3 (ae67cbd)'.",
    )
    parser.add_argument(
        "--fastchem-source-root",
        type=Path,
        default=ROOT / "FastChem",
        help="Optional FastChem source checkout used for provenance and data parity.",
    )
    parser.add_argument(
        "--point",
        action="append",
        type=_parse_point,
        default=None,
        metavar="TEMPERATURE_K,PRESSURE_BAR",
        help=(
            "Comparison point. Repeat for a profile; defaults to 1400,0.1 "
            "when omitted."
        ),
    )
    parser.add_argument(
        "--jax-platform",
        choices=("cpu", "gpu"),
        default="cpu",
        help="JAX platform for the ExoGibbs production solve.",
    )
    parser.add_argument(
        "--major-threshold",
        type=float,
        default=1.0e-8,
        help="Union mixing-ratio threshold for major gas species.",
    )
    parser.add_argument(
        "--ratio-floor",
        type=float,
        default=1.0e-300,
        help="Positive floor used only when forming log10 ratios.",
    )
    parser.add_argument(
        "--budget-relative-floor",
        type=float,
        default=1.0e-12,
        help="Element-budget denominator floor.",
    )
    parser.add_argument(
        "--top-n",
        type=int,
        default=12,
        help="Number of largest per-species differences to retain.",
    )
    parser.add_argument(
        "--fastchem-verbosity",
        type=int,
        default=1,
        choices=range(1, 5),
        metavar="{1,2,3,4}",
        help="FastChem console verbosity.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT,
        help="JSON summary path. A Markdown report is written beside it.",
    )
    parser.add_argument(
        "--preflight-only",
        action="store_true",
        help="Validate provenance and input files without running either solver.",
    )
    return parser


def _utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace(
        "+00:00", "Z"
    )


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _run_text(command: Sequence[str], *, cwd: Path) -> str | None:
    completed = subprocess.run(
        list(command),
        cwd=cwd,
        check=False,
        capture_output=True,
        text=True,
    )
    if completed.returncode != 0:
        return None
    return completed.stdout.strip()


def _git_metadata(root: Path) -> dict[str, Any]:
    if not (root / ".git").exists():
        return {
            "available": False,
            "root": str(root),
        }
    status = _run_text(("git", "status", "--short"), cwd=root)
    return {
        "available": True,
        "root": str(root.resolve()),
        "commit": _run_text(("git", "rev-parse", "HEAD"), cwd=root),
        "branch": _run_text(("git", "branch", "--show-current"), cwd=root),
        "describe": _run_text(
            ("git", "describe", "--tags", "--always", "--dirty"), cwd=root
        ),
        "worktree_clean": status == "",
        "worktree_status": status,
    }


def _data_record(path: Path) -> dict[str, Any]:
    return {
        "path": str(path.resolve()),
        "size_bytes": path.stat().st_size,
        "sha256": _sha256(path),
    }


def _source_data_parity(source_root: Path) -> dict[str, Any]:
    pairs = {
        "element_abundances": (
            ELEMENT_ABUNDANCE_FILE,
            source_root / "input" / "element_abundances" / "asplund_2021.dat",
        ),
        "gas_logk": (
            GAS_LOGK_FILE,
            source_root / "input" / "logK" / "logK_wo_ions.dat",
        ),
        "condensate_logk": (
            CONDENSATE_LOGK_FILE,
            source_root / "input" / "logK" / "logK_condensates.dat",
        ),
    }
    rows: dict[str, Any] = {}
    for name, (packaged, source) in pairs.items():
        packaged_exists = packaged.is_file()
        source_exists = source.is_file()
        packaged_hash = _sha256(packaged) if packaged_exists else None
        source_hash = _sha256(source) if source_exists else None
        rows[name] = {
            "packaged_path": str(packaged.resolve()),
            "source_path": str(source.resolve()),
            "packaged_exists": packaged_exists,
            "source_exists": source_exists,
            "packaged_sha256": packaged_hash,
            "source_sha256": source_hash,
            "byte_identical": bool(
                packaged_exists
                and source_exists
                and packaged_hash == source_hash
            ),
        }
    return {
        "files": rows,
        "all_packaged_files_present": all(
            row["packaged_exists"] for row in rows.values()
        ),
        "all_source_files_present": all(
            row["source_exists"] for row in rows.values()
        ),
        "all_byte_identical": all(
            row["byte_identical"] for row in rows.values()
        ),
    }


def _preflight(args: argparse.Namespace) -> dict[str, Any]:
    executable = args.fastchem_executable.resolve()
    data_files = {
        "element_abundances": ELEMENT_ABUNDANCE_FILE,
        "gas_logk": GAS_LOGK_FILE,
        "condensate_logk": CONDENSATE_LOGK_FILE,
    }
    data_present = {
        name: path.is_file() for name, path in data_files.items()
    }
    source_root = args.fastchem_source_root.resolve()
    source_parity = (
        _source_data_parity(source_root)
        if source_root.is_dir()
        else {
            "files": {},
            "all_source_files_present": False,
            "all_byte_identical": False,
        }
    )
    fastchem_git = _git_metadata(source_root)
    version_label = args.fastchem_version_label.strip()
    git_describe = str(fastchem_git.get("describe") or "")
    checks = {
        "fastchem_executable_exists": executable.is_file(),
        "fastchem_executable_is_executable": os.access(executable, os.X_OK),
        "all_packaged_data_present": all(data_present.values()),
        "fastchem_version_label_identifies_reference": (
            bool(version_label)
            and "4.0.3" in version_label.lower()
            and FASTCHEM_REFERENCE_COMMIT[:7] in version_label.lower()
        ),
        "fastchem_source_checkout_present": source_root.is_dir(),
        "fastchem_source_git_available": bool(
            fastchem_git.get("available")
        ),
        "fastchem_source_commit_is_reference": (
            fastchem_git.get("commit") == FASTCHEM_REFERENCE_COMMIT
        ),
        "fastchem_source_worktree_clean": bool(
            fastchem_git.get("worktree_clean")
        ),
        "fastchem_source_describe_is_reference": (
            git_describe == FASTCHEM_REFERENCE_TAG
        ),
        "packaged_and_source_data_byte_identical": source_parity[
            "all_byte_identical"
        ],
    }
    return {
        "schema": "exogibbs_fastchem4_production_comparison_preflight_v1",
        "generated_at_utc": _utc_now(),
        "checks": checks,
        "passed": all(checks.values()),
        "packaged_data_present": data_present,
        "packaged_data": {
            name: _data_record(path)
            for name, path in data_files.items()
            if path.is_file()
        },
        "source_data_parity": source_parity,
        "fastchem_executable": (
            _data_record(executable) if executable.is_file() else None
        ),
        "fastchem_version_label": args.fastchem_version_label,
        "exogibbs_git": _git_metadata(ROOT),
        "fastchem_git": fastchem_git,
        "binary_source_correspondence": (
            "operator-asserted by --fastchem-version-label; executable SHA256 "
            "and source Git commit are recorded independently"
        ),
    }


def _write_json(path: Path, payload: Any) -> None:
    from benchmarks.fastchem4.comparison import to_json_safe

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(to_json_safe(payload), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _preflight_path(output_path: Path) -> Path:
    """Return an output-specific preflight path without cross-run collisions."""

    return output_path.with_name(f"{output_path.stem}.preflight.json")


def _failure_payload(
    *,
    preflight: Mapping[str, Any],
    stage: str,
    error: Exception,
) -> dict[str, Any]:
    """Build a fail-closed report that cannot be mistaken for an old result."""

    return {
        "schema": SCHEMA,
        "generated_at_utc": _utc_now(),
        "preflight": preflight,
        "failure": {
            "stage": stage,
            "error_type": type(error).__name__,
            "message": str(error),
        },
        "summary": {
            "comparison_completed": False,
            "scientific_acceptance_thresholds_applied": False,
            "status": "failed",
        },
    }


def _write_failure_report(
    *,
    output_path: Path,
    preflight: Mapping[str, Any],
    stage: str,
    error: Exception,
) -> None:
    payload = _failure_payload(
        preflight=preflight,
        stage=stage,
        error=error,
    )
    _write_json(output_path, payload)
    output_path.with_suffix(".md").write_text(
        "\n".join(
            [
                "# ExoGibbs production vs FastChem4",
                "",
                f"Status: `failed` during `{stage}`.",
                "",
                f"Error: `{type(error).__name__}: {error}`",
                "",
                "No scientific comparison result was produced.",
                "",
            ]
        ),
        encoding="utf-8",
    )


def _configure_jax(platform: str) -> None:
    os.environ["JAX_ENABLE_X64"] = "1"
    os.environ["JAX_PLATFORMS"] = platform
    os.environ["JAX_PLATFORM_NAME"] = platform


def _block_until_ready(tree: Any) -> None:
    import jax

    for leaf in jax.tree_util.tree_leaves(tree):
        block = getattr(leaf, "block_until_ready", None)
        if block is not None:
            block()


def _run_exogibbs(
    temperatures: np.ndarray,
    pressures: np.ndarray,
) -> tuple[Any, np.ndarray, Any, tuple[Mapping[str, Any], ...], dict[str, Any]]:
    import jax
    from jax import config as jax_config
    import jax.numpy as jnp

    jax_config.update("jax_enable_x64", True)

    import exogibbs
    from exogibbs.api.condensate import (
        CondensateEquilibriumOptions,
        solve_profile as solve_condensate_profile,
    )
    from exogibbs.api.gas import solve as solve_gas
    from exogibbs.presets.fastchem4_cond import condensate_chemical_setup
    from exogibbs.utils.fastchem_parity import (
        build_aligned_abundance_vector,
    )

    expected_root = (SOURCE_ROOT / "exogibbs").resolve()
    imported_root = Path(exogibbs.__file__).resolve().parent
    if imported_root != expected_root:
        raise RuntimeError(
            "Imported exogibbs from outside this repository: "
            f"{imported_root} != {expected_root}"
        )

    setup = condensate_chemical_setup(
        gas_path="FastChem4/logK/logK_wo_ions.dat",
        condensate_path="FastChem4/logK/logK_condensates.dat",
        species_default_elements=False,
        element_file="FastChem4/element_abundances/asplund_2021.dat",
        silent=True,
    )
    aligned = build_aligned_abundance_vector(
        setup.elements,
        source="fastchem_file",
        normalize=True,
        element_file=ELEMENT_ABUNDANCE_FILE,
    )
    budget = jnp.asarray(aligned.vector, dtype=jnp.float64)

    started = time.perf_counter()
    result = solve_condensate_profile(
        setup,
        T=jnp.asarray(temperatures, dtype=jnp.float64),
        P=jnp.asarray(pressures, dtype=jnp.float64),
        b=budget,
        options=CondensateEquilibriumOptions(return_diagnostics=True),
        return_diagnostics=True,
    )
    _block_until_ready(result.batched_arrays)
    wall_seconds = time.perf_counter() - started

    gas_diagnostics: list[Mapping[str, Any]] = []
    for temperature, pressure in zip(temperatures, pressures):
        gas_result, diagnostics = solve_gas(
            setup.gas_setup,
            T=float(temperature),
            P=float(pressure),
            b=budget,
            return_diagnostics=True,
        )
        _block_until_ready((gas_result, diagnostics))
        gas_diagnostics.append(diagnostics)

    metadata = {
        "wall_seconds": wall_seconds,
        "jax_backend": jax.default_backend(),
        "jax_devices": [str(device) for device in jax.devices()],
        "jax_enable_x64": bool(jax_config.x64_enabled),
        "runtime_package_version": exogibbs.__version__,
        "version_provenance_note": (
            "Runtime package metadata can lag an editable checkout; "
            "provenance.exogibbs_git.commit identifies the source tree."
        ),
        "route": "head_v2",
        "fixed_support_v2_preset": "validated_2026_07",
        "profile_method": result.method,
        "abundance": {
            "source": aligned.source,
            "source_path": aligned.source_path,
            "normalized": aligned.normalized,
            "metadata": aligned.metadata,
        },
    }
    return setup, np.asarray(budget), result, tuple(gas_diagnostics), metadata


def _common_gibbs_over_rt(
    *,
    setup: Any,
    temperature: float,
    pressure: float,
    gas_amounts: np.ndarray,
    condensate_amounts: np.ndarray,
    reference_pressure: float = 1.0,
) -> float:
    """Evaluate both solver states with one ExoGibbs thermodynamic objective."""

    gas = np.asarray(gas_amounts, dtype=np.float64)
    condensates = np.asarray(condensate_amounts, dtype=np.float64)
    total_gas = float(np.sum(gas))
    if (
        gas.ndim != 1
        or condensates.ndim != 1
        or not math.isfinite(total_gas)
        or total_gas <= 0.0
    ):
        return math.nan
    h_gas = np.asarray(setup.gas_setup.hvector_func(temperature), dtype=float)
    h_cond = np.asarray(
        setup.condensate_setup.hvector_func(temperature), dtype=float
    )
    positive_gas = np.isfinite(gas) & (gas > 0.0)
    positive_cond = np.isfinite(condensates) & (condensates > 0.0)
    gas_source = h_gas[positive_gas] + math.log(
        pressure / reference_pressure
    )
    gas_term = np.sum(
        gas[positive_gas]
        * (
            gas_source
            + np.log(gas[positive_gas])
            - math.log(total_gas)
        )
    )
    cond_term = np.sum(condensates[positive_cond] * h_cond[positive_cond])
    return float(gas_term + cond_term)


def _host(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _host(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_host(item) for item in value]
    if value is None or isinstance(value, (str, bool, int, float)):
        return value
    try:
        import jax

        value = jax.device_get(value)
    except (ImportError, TypeError):
        pass
    array = np.asarray(value)
    if array.shape == ():
        return array.item()
    return array.tolist()


def _exogibbs_status(
    layer: Any,
    gas_diagnostics: Mapping[str, Any],
) -> dict[str, Any]:
    diagnostics = layer.diagnostics or {}
    lifecycle = diagnostics.get("fixed_support_v2", {})
    return {
        "status": layer.status,
        "converged": bool(layer.converged),
        "acceptance_tier": layer.acceptance_tier,
        "selected_route": layer.selected_route,
        "head_route_name": layer.head_route_name,
        "head_route_version": layer.head_route_version,
        "support_count": int(np.asarray(layer.condensate_support_indices).size),
        "support_names": list(layer.condensate_support_names),
        "lifecycle_outcome": lifecycle.get("outcome"),
        "fixed_support_converged": lifecycle.get(
            "fixed_support_converged"
        ),
        "support_closed": lifecycle.get("support_closed"),
        "independent_kkt_passed": lifecycle.get(
            "independent_kkt_passed"
        ),
        "independent_kkt": _host(lifecycle.get("independent_kkt")),
        "independent_kkt_state_note": (
            "The lifecycle KKT diagnostics describe the accepted "
            "fixed-support state before any optional full-budget gas polish; "
            "species comparisons use the final public state."
        ),
        "terminal_status_name": lifecycle.get("terminal_status_name"),
        "final_state_values_finite": lifecycle.get(
            "final_state_values_finite"
        ),
        "full_budget_gate": _host(
            diagnostics.get("full_condensate_budget_residual_gate")
        ),
        "full_budget_gas_log_amount_polish": _host(
            diagnostics.get("full_condensate_budget_gas_log_amount_polish")
        ),
        "gas_only_diagnostics": _host(gas_diagnostics),
    }


def _fastchem_layer_status(result: Any, index: int) -> dict[str, Any]:
    return {
        "status": str(result.status[index]),
        "converged": bool(result.converged[index]),
        "elements_conserved": bool(result.elements_conserved[index]),
        "iterations": int(result.iterations[index]),
        "chemistry_iterations": int(result.chemistry_iterations[index]),
        "condensation_iterations": int(
            result.condensation_iterations[index]
        ),
    }


def _fastchem_common_basis_state(
    *,
    gas_number_densities: np.ndarray,
    condensate_number_densities: np.ndarray,
    total_element_density: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Convert FastChem number densities to the shared elemental-budget gauge."""

    gas_density = np.asarray(gas_number_densities, dtype=np.float64)
    condensate_density = np.asarray(
        condensate_number_densities, dtype=np.float64
    )
    element_density = np.asarray(total_element_density, dtype=np.float64)
    if gas_density.ndim != 2 or condensate_density.ndim != 2:
        raise ValueError("FastChem species number densities must be 2-D.")
    if element_density.ndim != 1:
        raise ValueError("FastChem total element density must be 1-D.")
    if (
        gas_density.shape[0] != condensate_density.shape[0]
        or gas_density.shape[0] != element_density.size
    ):
        raise ValueError("FastChem common-basis arrays have incompatible layers.")
    if (
        not np.all(np.isfinite(gas_density))
        or not np.all(np.isfinite(condensate_density))
        or np.any(gas_density < 0.0)
        or np.any(condensate_density < 0.0)
    ):
        raise ValueError(
            "FastChem species number densities must be finite and nonnegative."
        )

    invalid_element_density = (
        ~np.isfinite(element_density) | (element_density <= 0.0)
    )
    if np.any(invalid_element_density):
        indices = np.flatnonzero(invalid_element_density).tolist()
        raise RuntimeError(
            "FastChem total element density must be finite and positive "
            f"before common-basis normalization; invalid layers: {indices}."
        )
    gas_density_sum = np.sum(gas_density, axis=1)
    invalid_gas_density = (
        ~np.isfinite(gas_density_sum) | (gas_density_sum <= 0.0)
    )
    if np.any(invalid_gas_density):
        indices = np.flatnonzero(invalid_gas_density).tolist()
        raise RuntimeError(
            "FastChem total gas density must be finite and positive before "
            f"mixing-ratio normalization; invalid layers: {indices}."
        )

    gas_amounts = gas_density / element_density[:, None]
    condensate_amounts = condensate_density / element_density[:, None]
    gas_mixing_ratios = gas_density / gas_density_sum[:, None]
    return gas_amounts, condensate_amounts, gas_mixing_ratios


def _make_layer_reports(
    *,
    setup: Any,
    budget: np.ndarray,
    exogibbs_result: Any,
    gas_diagnostics: Sequence[Mapping[str, Any]],
    fastchem_result: Any,
    temperatures: np.ndarray,
    pressures: np.ndarray,
    major_threshold: float,
    ratio_floor: float,
    budget_relative_floor: float,
    top_n: int,
) -> tuple[list[dict[str, Any]], np.ndarray, np.ndarray]:
    from benchmarks.fastchem4.comparison import (
        align_species_values,
        condensate_comparison_metrics,
        element_budget_metrics,
        gas_major_species_metrics,
    )

    fastchem_gas_density = align_species_values(
        setup.gas_species,
        fastchem_result.gas_names,
        fastchem_result.gas_number_densities,
    )
    fastchem_cond_density = align_species_values(
        setup.condensate_species,
        fastchem_result.condensate_names,
        fastchem_result.condensate_number_densities,
    )
    (
        fastchem_gas_amounts,
        fastchem_cond_amounts,
        fastchem_gas_x,
    ) = _fastchem_common_basis_state(
        gas_number_densities=fastchem_gas_density,
        condensate_number_densities=fastchem_cond_density,
        total_element_density=fastchem_result.total_element_density,
    )
    exogibbs_gas = np.asarray(
        exogibbs_result.batched_arrays["gas_n"], dtype=np.float64
    )
    exogibbs_cond = np.asarray(
        exogibbs_result.batched_arrays["condensate_amounts"],
        dtype=np.float64,
    )
    exogibbs_gas_x = np.asarray(
        exogibbs_result.batched_arrays["gas_x"], dtype=np.float64
    )
    gas_formula = np.asarray(setup.formula_matrix, dtype=np.float64)
    cond_formula = np.asarray(setup.formula_matrix_cond, dtype=np.float64)

    reports: list[dict[str, Any]] = []
    for index, (temperature, pressure) in enumerate(
        zip(temperatures, pressures)
    ):
        ideal_density = (
            float(pressure)
            * 1.0e6
            / (BOLTZMANN_CGS * float(temperature))
        )
        exo_total_gas = float(np.sum(exogibbs_gas[index]))
        fc_total_gas = float(np.sum(fastchem_gas_amounts[index]))
        exo_physical_scale = ideal_density / exo_total_gas
        exo_gibbs = _common_gibbs_over_rt(
            setup=setup,
            temperature=float(temperature),
            pressure=float(pressure),
            gas_amounts=exogibbs_gas[index],
            condensate_amounts=exogibbs_cond[index],
        )
        fc_gibbs = _common_gibbs_over_rt(
            setup=setup,
            temperature=float(temperature),
            pressure=float(pressure),
            gas_amounts=fastchem_gas_amounts[index],
            condensate_amounts=fastchem_cond_amounts[index],
        )
        condensate_metrics = {
            f"{threshold:.0e}": condensate_comparison_metrics(
                names=setup.condensate_species,
                left_values=exogibbs_cond[index],
                right_values=fastchem_cond_amounts[index],
                active_floor=threshold,
                ratio_floor=max(ratio_floor, threshold),
                top_n=top_n,
            )
            for threshold in CONDENSATE_THRESHOLDS
        }
        reports.append(
            {
                "index": index,
                "temperature_K": float(temperature),
                "pressure_bar": float(pressure),
                "status": {
                    "exogibbs": _exogibbs_status(
                        exogibbs_result.layers[index],
                        gas_diagnostics[index],
                    ),
                    "fastchem": _fastchem_layer_status(
                        fastchem_result, index
                    ),
                },
                "element_budget": {
                    "exogibbs": element_budget_metrics(
                        gas_formula_matrix=gas_formula,
                        condensate_formula_matrix=cond_formula,
                        gas_amounts=exogibbs_gas[index],
                        condensate_amounts=exogibbs_cond[index],
                        target=budget,
                        element_names=setup.elements,
                        relative_floor=budget_relative_floor,
                    ),
                    "fastchem": element_budget_metrics(
                        gas_formula_matrix=gas_formula,
                        condensate_formula_matrix=cond_formula,
                        gas_amounts=fastchem_gas_amounts[index],
                        condensate_amounts=fastchem_cond_amounts[index],
                        target=budget,
                        element_names=setup.elements,
                        relative_floor=budget_relative_floor,
                    ),
                },
                "total_gas": {
                    "ideal_gas_number_density_cm-3": ideal_density,
                    "fastchem_sum_gas_number_density_cm-3": float(
                        np.sum(fastchem_gas_density[index])
                    ),
                    "fastchem_reported_ideal_gas_density_cm-3": float(
                        fastchem_result.ideal_gas_density[index]
                    ),
                    "fastchem_total_element_density_cm-3": float(
                        fastchem_result.total_element_density[index]
                    ),
                    "exogibbs_normalized_amount": exo_total_gas,
                    "fastchem_normalized_amount": fc_total_gas,
                    "normalized_amount_relative_difference": (
                        exo_total_gas / fc_total_gas - 1.0
                    ),
                    "exogibbs_implied_total_element_density_cm-3": (
                        exo_physical_scale
                    ),
                    "implied_total_element_density_relative_difference": (
                        exo_physical_scale
                        / float(fastchem_result.total_element_density[index])
                        - 1.0
                    ),
                },
                "gas_major_species": gas_major_species_metrics(
                    names=setup.gas_species,
                    left_values=exogibbs_gas_x[index],
                    right_values=fastchem_gas_x[index],
                    threshold=major_threshold,
                    ratio_floor=ratio_floor,
                    excluded_names=("e-",),
                    top_n=top_n,
                ),
                "condensates": condensate_metrics,
                "gibbs_over_rt": {
                    "evaluator": (
                        "common ExoGibbs thermodynamic objective; "
                        "not FastChem-reported Gibbs energy"
                    ),
                    "exogibbs": exo_gibbs,
                    "fastchem_state": fc_gibbs,
                    "exogibbs_minus_fastchem": exo_gibbs - fc_gibbs,
                    "interpretable_only_when_budgets_close": True,
                },
            }
        )
    return reports, exogibbs_cond, fastchem_cond_amounts


def _phase_transition_reports(
    *,
    names: Sequence[str],
    exogibbs_amounts: np.ndarray,
    fastchem_amounts: np.ndarray,
    temperatures: np.ndarray,
    pressures: np.ndarray,
) -> dict[str, Any]:
    from benchmarks.fastchem4.comparison import profile_phase_transitions

    conditions = [
        {
            "index": index,
            "temperature_K": float(temperature),
            "pressure_bar": float(pressure),
        }
        for index, (temperature, pressure) in enumerate(
            zip(temperatures, pressures)
        )
    ]
    return {
        f"{threshold:.0e}": {
            "threshold": threshold,
            "conditions": conditions,
            "exogibbs": profile_phase_transitions(
                names=names,
                amounts=exogibbs_amounts,
                threshold=threshold,
            ),
            "fastchem": profile_phase_transitions(
                names=names,
                amounts=fastchem_amounts,
                threshold=threshold,
            ),
        }
        for threshold in CONDENSATE_THRESHOLDS
    }


def _summary(
    *,
    preflight: Mapping[str, Any],
    layers: Sequence[Mapping[str, Any]],
    gas_catalog_match: bool,
    condensate_catalog_match: bool,
) -> dict[str, Any]:
    exo_converged = [
        bool(layer["status"]["exogibbs"]["converged"]) for layer in layers
    ]
    fastchem_converged = [
        bool(layer["status"]["fastchem"]["converged"]) for layer in layers
    ]
    fastchem_conserved = [
        bool(layer["status"]["fastchem"]["elements_conserved"])
        for layer in layers
    ]
    def finite_number(value: Any) -> bool:
        if value is None:
            return False
        try:
            return math.isfinite(float(value))
        except (TypeError, ValueError):
            return False

    def finite_values(values: Sequence[Any]) -> list[float]:
        output = []
        for value in values:
            if finite_number(value):
                output.append(float(value))
        return output

    exo_budget = finite_values(
        [
            layer["element_budget"]["exogibbs"].get(
                "max_absolute_relative_residual"
            )
            for layer in layers
        ]
    )
    fc_budget = finite_values(
        [
            layer["element_budget"]["fastchem"].get(
                "max_absolute_relative_residual"
            )
            for layer in layers
        ]
    )
    gas_dex = finite_values(
        [
            layer["gas_major_species"].get(
                "max_absolute_log10_ratio"
            )
            for layer in layers
        ]
    )
    gibbs_delta = finite_values(
        [
            abs(float(layer["gibbs_over_rt"]["exogibbs_minus_fastchem"]))
            if layer["gibbs_over_rt"]["exogibbs_minus_fastchem"] is not None
            else None
            for layer in layers
        ]
    )
    expected_condensate_keys = {
        f"{threshold:.0e}" for threshold in CONDENSATE_THRESHOLDS
    }
    condensate_metrics_finite = bool(
        all(
            set(layer["condensates"]) == expected_condensate_keys
            and all(
                bool(metrics.get("finite"))
                for metrics in layer["condensates"].values()
            )
            for layer in layers
        )
    )
    total_gas_metrics_finite = bool(
        all(
            all(finite_number(value) for value in layer["total_gas"].values())
            for layer in layers
        )
    )
    metrics_finite = bool(
        all(
            layer["element_budget"]["exogibbs"]["finite"]
            and layer["element_budget"]["fastchem"]["finite"]
            and layer["gas_major_species"]["finite"]
            and all(
                finite_number(layer["gibbs_over_rt"].get(key))
                for key in (
                    "exogibbs",
                    "fastchem_state",
                    "exogibbs_minus_fastchem",
                )
            )
            for layer in layers
        )
        and condensate_metrics_finite
        and total_gas_metrics_finite
    )
    complete = bool(
        preflight["passed"]
        and gas_catalog_match
        and condensate_catalog_match
        and all(exo_converged)
        and all(fastchem_converged)
        and all(fastchem_conserved)
        and metrics_finite
    )
    return {
        "comparison_completed": complete,
        "scientific_acceptance_thresholds_applied": False,
        "all_exogibbs_layers_converged": all(exo_converged),
        "all_fastchem_layers_converged": all(fastchem_converged),
        "all_fastchem_element_checks_passed": all(fastchem_conserved),
        "gas_catalog_match": gas_catalog_match,
        "condensate_catalog_match": condensate_catalog_match,
        "all_comparison_metrics_finite": metrics_finite,
        "all_condensate_metrics_finite": condensate_metrics_finite,
        "all_total_gas_metrics_finite": total_gas_metrics_finite,
        "max_exogibbs_budget_relative_residual": max(
            exo_budget, default=None
        ),
        "max_fastchem_budget_relative_residual": max(fc_budget, default=None),
        "max_major_gas_abs_log10_ratio": max(
            gas_dex, default=None
        ),
        "max_abs_common_gibbs_over_rt_difference": max(
            gibbs_delta, default=None
        ),
        "status": "complete" if complete else "incomplete",
    }


def _format_number(value: Any, digits: int = 3) -> str:
    if value is None:
        return "n/a"
    number = float(value)
    if not math.isfinite(number):
        return "n/a"
    return f"{number:.{digits}g}"


def _markdown_report(payload: Mapping[str, Any]) -> str:
    summary = payload["summary"]
    lines = [
        "# ExoGibbs production vs FastChem4",
        "",
        f"Generated: `{payload['generated_at_utc']}`",
        "",
        (
            "FastChem is used only as an independent post-solve comparison "
            "target. No FastChem runtime value is supplied to an ExoGibbs "
            "solver constructor or initializer."
        ),
        "",
        "## Input contract",
        "",
        (
            f"- FastChem version label: "
            f"`{payload['provenance']['fastchem_version_label']}`"
        ),
        "- Gas data: `FastChem4/logK/logK_wo_ions.dat`",
        "- Condensate data: `FastChem4/logK/logK_condensates.dat`",
        "- Abundances: `FastChem4/element_abundances/asplund_2021.dat`",
        "- Element vector: file-backed, reordered, and sum-normalized",
        (
            "- Packaged/source data byte-identical: "
            f"`{payload['preflight']['source_data_parity']['all_byte_identical']}`"
        ),
        "",
        "## Layer summary",
        "",
        "| # | T [K] | P [bar] | Exo status | FC status | "
        "max budget (Exo/FC) | major gas max [dex] | G/RT Exo-FC |",
        "|---:|---:|---:|---|---|---:|---:|---:|",
    ]
    for layer in payload["layers"]:
        lines.append(
            "| {index} | {temperature} | {pressure} | {exo} | {fc} | "
            "{exo_budget} / {fc_budget} | {gas_dex} | {gibbs} |".format(
                index=layer["index"],
                temperature=_format_number(layer["temperature_K"], 6),
                pressure=_format_number(layer["pressure_bar"], 6),
                exo=layer["status"]["exogibbs"]["status"],
                fc=layer["status"]["fastchem"]["status"],
                exo_budget=_format_number(
                    layer["element_budget"]["exogibbs"][
                        "max_absolute_relative_residual"
                    ]
                ),
                fc_budget=_format_number(
                    layer["element_budget"]["fastchem"][
                        "max_absolute_relative_residual"
                    ]
                ),
                gas_dex=_format_number(
                    layer["gas_major_species"].get(
                        "max_absolute_log10_ratio"
                    )
                ),
                gibbs=_format_number(
                    layer["gibbs_over_rt"][
                        "exogibbs_minus_fastchem"
                    ]
                ),
            )
        )
    lines.extend(
        [
            "",
            "## Condensate active-set summary",
            "",
            "| # | amount floor | ratio clip floor | Exo active | FC active | "
            "common / union | Jaccard | max floor-clipped [dex] |",
            "|---:|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for layer in payload["layers"]:
        for threshold in CONDENSATE_THRESHOLDS:
            key = f"{threshold:.0e}"
            metrics = layer["condensates"][key]
            lines.append(
                "| {index} | {threshold} | {ratio_floor} | {exo} | {fc} | "
                "{intersection} / {union} | {jaccard} | {max_dex} |".format(
                    index=layer["index"],
                    threshold=key,
                    ratio_floor=_format_number(metrics["ratio_floor"]),
                    exo=metrics["left_active_count"],
                    fc=metrics["right_active_count"],
                    intersection=metrics["intersection_active_count"],
                    union=metrics["union_active_count"],
                    jaccard=_format_number(
                        metrics["active_set_jaccard"]
                    ),
                    max_dex=_format_number(
                        metrics["max_absolute_log10_ratio"]
                    ),
                )
            )
    lines.extend(
        [
            "",
            (
                "Condensate dex values are clipped at the recorded ratio "
                "floor. Active counts, overlap, and absolute amounts are the "
                "primary evidence when a phase is absent from one solver."
            ),
            "",
            "## Phase-boundary coverage",
            "",
        ]
    )
    if len(payload["layers"]) < 2:
        lines.append(
            "Not evaluated: this run contains one profile point and therefore "
            "has no adjacent interval in which to observe a phase entry or exit."
        )
    else:
        lines.extend(
            [
                "| amount floor | Exo transitions | FC transitions |",
                "|---:|---:|---:|",
            ]
        )
        for threshold in CONDENSATE_THRESHOLDS:
            key = f"{threshold:.0e}"
            transitions = payload["profile_phase_transitions"][key]
            lines.append(
                "| {threshold} | {exo} | {fc} |".format(
                    threshold=key,
                    exo=transitions["exogibbs"]["transition_count"],
                    fc=transitions["fastchem"]["transition_count"],
                )
            )
    lines.extend(
        [
            "",
            "## Run summary",
            "",
            f"- Status: `{summary['status']}`",
            (
                "- All ExoGibbs layers converged: "
                f"`{summary['all_exogibbs_layers_converged']}`"
            ),
            (
                "- All FastChem layers converged: "
                f"`{summary['all_fastchem_layers_converged']}`"
            ),
            (
                "- Gas/condensate catalog match: "
                f"`{summary['gas_catalog_match']}` / "
                f"`{summary['condensate_catalog_match']}`"
            ),
            (
                "- Scientific acceptance thresholds applied: "
                "`False` (this is an observational comparison report)"
            ),
            "",
            (
                "The reported FastChem-state `G/RT` is evaluated afterward "
                "with the common ExoGibbs objective. FastChem does not expose "
                "a public Gibbs-energy diagnostic here."
            ),
            "",
        ]
    )
    return "\n".join(lines)


def _run(args: argparse.Namespace, preflight: Mapping[str, Any]) -> dict[str, Any]:
    from benchmarks.fastchem4.comparison import (
        occurrence_keys,
    )
    from benchmarks.fastchem4.fastchem_executable import (
        run_fastchem_executable,
    )

    points = args.point or [DEFAULT_POINT]
    temperatures = np.asarray([point[0] for point in points], dtype=np.float64)
    pressures = np.asarray([point[1] for point in points], dtype=np.float64)

    fastchem_started = time.perf_counter()
    fastchem_result = run_fastchem_executable(
        executable=args.fastchem_executable,
        temperatures=temperatures,
        pressures=pressures,
        element_abundance_file=ELEMENT_ABUNDANCE_FILE,
        gas_logk_file=GAS_LOGK_FILE,
        condensate_logk_file=CONDENSATE_LOGK_FILE,
        verbosity=args.fastchem_verbosity,
        **FASTCHEM_FIXED_SETTINGS,
    )
    fastchem_wall_seconds = time.perf_counter() - fastchem_started

    setup, budget, exogibbs_result, gas_diagnostics, exogibbs_metadata = (
        _run_exogibbs(temperatures, pressures)
    )
    gas_catalog_match = sorted(occurrence_keys(setup.gas_species)) == sorted(
        occurrence_keys(fastchem_result.gas_names)
    )
    condensate_catalog_match = (
        occurrence_keys(setup.condensate_species)
        == occurrence_keys(fastchem_result.condensate_names)
    )
    if not gas_catalog_match or not condensate_catalog_match:
        raise RuntimeError(
            "FastChem and ExoGibbs species catalogs do not align "
            "occurrence-by-occurrence."
        )

    layers, exogibbs_cond, fastchem_cond = _make_layer_reports(
        setup=setup,
        budget=budget,
        exogibbs_result=exogibbs_result,
        gas_diagnostics=gas_diagnostics,
        fastchem_result=fastchem_result,
        temperatures=temperatures,
        pressures=pressures,
        major_threshold=args.major_threshold,
        ratio_floor=args.ratio_floor,
        budget_relative_floor=args.budget_relative_floor,
        top_n=args.top_n,
    )
    transitions = _phase_transition_reports(
        names=setup.condensate_species,
        exogibbs_amounts=exogibbs_cond,
        fastchem_amounts=fastchem_cond,
        temperatures=temperatures,
        pressures=pressures,
    )
    summary = _summary(
        preflight=preflight,
        layers=layers,
        gas_catalog_match=gas_catalog_match,
        condensate_catalog_match=condensate_catalog_match,
    )
    return {
        "schema": SCHEMA,
        "generated_at_utc": _utc_now(),
        "preflight": preflight,
        "provenance": {
            "exogibbs_git": preflight["exogibbs_git"],
            "fastchem_git": preflight["fastchem_git"],
            "fastchem_version_label": args.fastchem_version_label,
            "fastchem_executable": preflight["fastchem_executable"],
            "fastchem_stdout": fastchem_result.stdout,
        },
        "input_contract": {
            "temperature_K": temperatures,
            "pressure_bar": pressures,
            "reference_pressure_bar": 1.0,
            "element_names": setup.elements,
            "element_abundance_vector": budget,
            "element_abundance_normalization": "sum of non-electron values is 1",
            "gas_species_count": len(setup.gas_species),
            "condensate_species_count": len(setup.condensate_species),
            "gas_catalog_occurrence_match": gas_catalog_match,
            "condensate_catalog_occurrence_match": condensate_catalog_match,
            "fastchem_values_used_as_exogibbs_constructor_inputs": False,
            "shared_inputs_only": (
                "thermochemical files, elemental abundance file, "
                "temperature, pressure"
            ),
        },
        "fastchem": {
            "backend": "standalone_executable",
            "wall_seconds": fastchem_wall_seconds,
            "equilibrium_condensation": True,
            "rainout_condensation": False,
            "number_density_unit": "cm^-3",
            "settings": {
                "verbosity": args.fastchem_verbosity,
                **FASTCHEM_FIXED_SETTINGS,
            },
        },
        "exogibbs": exogibbs_metadata,
        "layers": layers,
        "profile_phase_transitions": transitions,
        "summary": summary,
    }


def main() -> None:
    args = build_parser().parse_args()
    if args.major_threshold <= 0.0 or not math.isfinite(
        args.major_threshold
    ):
        raise ValueError("--major-threshold must be finite and positive.")
    if args.ratio_floor <= 0.0 or not math.isfinite(args.ratio_floor):
        raise ValueError("--ratio-floor must be finite and positive.")
    if args.budget_relative_floor <= 0.0 or not math.isfinite(
        args.budget_relative_floor
    ):
        raise ValueError(
            "--budget-relative-floor must be finite and positive."
        )
    if args.top_n < 1:
        raise ValueError("--top-n must be at least 1.")

    preflight = _preflight(args)
    preflight_path = _preflight_path(args.output)
    _write_json(preflight_path, preflight)
    if not preflight["passed"]:
        error = RuntimeError(
            f"FastChem4 comparison preflight failed; see {preflight_path}."
        )
        _write_failure_report(
            output_path=args.output,
            preflight=preflight,
            stage="preflight",
            error=error,
        )
        raise error
    if args.preflight_only:
        print(preflight_path)
        return

    _configure_jax(args.jax_platform)
    try:
        payload = _run(args, preflight)
    except Exception as error:
        _write_failure_report(
            output_path=args.output,
            preflight=preflight,
            stage="comparison",
            error=error,
        )
        raise
    _write_json(args.output, payload)
    markdown_path = args.output.with_suffix(".md")
    markdown_path.write_text(
        _markdown_report(payload),
        encoding="utf-8",
    )
    print(args.output)
    print(markdown_path)


if __name__ == "__main__":
    main()
