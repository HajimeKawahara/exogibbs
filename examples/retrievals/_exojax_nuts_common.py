"""Shared ExoJAX spectrum and NumPyro helpers for the VJP demos.

The module keeps ExoJAX and NumPyro imports lazy so repository tests and the
chemistry-only condensate preflight do not require either optional package.
No database download is attempted: callers must supply a complete local CO
ExoMol directory.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import itertools
import json
from pathlib import Path
import time
from typing import Any, Callable, Mapping, Optional, Sequence, Union

import jax
import jax.numpy as jnp
import numpy as np


jax.config.update("jax_enable_x64", True)

TRUTH_T0_K = 1160.0
TRUTH_ALPHA = 0.03
TRUTH_LOG_CO_SCALE = 0.0

T0_PRIOR_BOUNDS_K = (1120.0, 1200.0)
ALPHA_PRIOR_BOUNDS = (0.025, 0.035)
LOG_CO_SCALE_PRIOR_BOUNDS = (-0.15, 0.15)

PRESSURE_TOP_BAR = 1.0e-3
PRESSURE_BOTTOM_BAR = 10.0
REFERENCE_PRESSURE_BAR = 1.0
TEMPERATURE_RANGE_K = (700.0, 1500.0)
DEFAULT_RELATIVE_NOISE = 2.0e-3
DEFAULT_QUICK_NUM_WARMUP = 5
DEFAULT_QUICK_NUM_SAMPLES = 10
DEFAULT_QUICK_MAX_TREE_DEPTH = 4
GAS_QUICK_NUM_WARMUP = 100
GAS_QUICK_NUM_SAMPLES = 100
GAS_QUICK_MAX_TREE_DEPTH = 8
RUN_STATUS_FILENAME = "run_status.json"
_SHARED_OUTPUT_FILENAMES = (
    "mock_spectrum.npz",
    "posterior_samples.npz",
    "run_summary.json",
)


@dataclass(frozen=True)
class SpectralContext:
    """Static ExoJAX objects used by a differentiable CO spectrum."""

    nu_grid: Any
    art: Any
    opa: Any
    molmass: float
    gravity: float
    mean_molecular_weight: float


@dataclass(frozen=True)
class MockObservation:
    """Normalized deterministic mock spectrum and its known noise."""

    observed: Any
    truth: Any
    flux_scale: Any
    noise_std: Any


@dataclass(frozen=True)
class RunSettings:
    """NUTS settings after applying the optional quick profile."""

    num_warmup: int
    num_samples: int
    seed: int
    progress_bar: bool
    max_tree_depth: int
    quick: bool


def require_local_co_database(
    path: Optional[Union[str, Path]],
) -> Path:
    """Validate a local Li2015-like ExoMol directory without downloading."""

    if path is None:
        raise ValueError(
            "A local CO ExoMol directory is required. Pass --co-database "
            "/path/to/CO/12C-16O/Li2015."
        )
    database = Path(path).expanduser().resolve()
    if not database.is_dir():
        raise FileNotFoundError(f"CO database directory does not exist: {database}")

    exact_name = database.parent.name
    database_name = database.name
    prefix = f"{exact_name}__{database_name}"
    required = (database / f"{prefix}.def", database / f"{prefix}.pf")
    missing = [item.name for item in required if not item.is_file()]
    state_files = tuple(database.glob(f"{prefix}.states*"))
    transition_files = tuple(database.glob(f"{prefix}.trans*"))
    if not state_files:
        missing.append(f"{prefix}.states[.bz2/.hdf5/.feather]")
    if not transition_files:
        missing.append(f"{prefix}.trans[.bz2/.hdf5/.feather]")
    if missing:
        raise FileNotFoundError(
            "The CO database is incomplete and this offline demo will not "
            f"download missing files. Missing: {', '.join(missing)}"
        )
    return database


def build_spectral_context(
    co_database_path: Union[str, Path],
    *,
    nlayer: int,
    nu_points: int,
) -> SpectralContext:
    """Build the CO opacity and pure-absorption emission calculation."""

    if nlayer < 2:
        raise ValueError("nlayer must be at least two.")
    if nu_points < 128:
        raise ValueError("nu_points must be at least 128 for PreMODIT.")
    database = require_local_co_database(co_database_path)

    from exojax.database.exomol.api import MdbExomol
    from exojax.database.molinfo.mass import isotope_molmass
    from exojax.opacity import OpaPremodit
    from exojax.rt import ArtEmisPure
    from exojax.utils.grids import wavenumber_grid

    nu_grid, _wavelength, _resolution = wavenumber_grid(
        22920.0,
        23000.0,
        nu_points,
        unit="AA",
        xsmode="premodit",
    )
    # broadf=False and broadf_download=False make the offline contract explicit.
    # The default .def broadening parameters are sufficient for this VJP demo.
    mdb = MdbExomol(
        str(database),
        nurange=nu_grid,
        broadf=False,
        broadf_download=False,
        gpu_transfer=False,
    )
    opa = OpaPremodit(
        mdb,
        nu_grid,
        auto_trange=list(TEMPERATURE_RANGE_K),
        dit_grid_resolution=1.0,
    )
    art = ArtEmisPure(
        nu_grid=nu_grid,
        pressure_btm=PRESSURE_BOTTOM_BAR,
        pressure_top=PRESSURE_TOP_BAR,
        nlayer=nlayer,
        rtsolver="ibased",
        nstream=4,
    )
    art.change_temperature_range(*TEMPERATURE_RANGE_K)
    return SpectralContext(
        nu_grid=nu_grid,
        art=art,
        opa=opa,
        molmass=float(isotope_molmass("12C-16O")),
        gravity=1.0e5,
        mean_molecular_weight=2.33,
    )


def co_emission_flux(
    context: SpectralContext,
    temperature: Any,
    co_vmr: Any,
) -> jax.Array:
    """Return an ExoJAX CO-only emission spectrum."""

    from exojax.atm.atmconvert import vmr_to_mmr

    temperature_array = jnp.asarray(temperature)
    co_vmr_array = jnp.asarray(co_vmr)
    co_mmr = vmr_to_mmr(
        co_vmr_array,
        context.molmass,
        context.mean_molecular_weight,
    )
    cross_section = context.opa.xsmatrix(
        temperature_array,
        context.art.pressure,
    )
    optical_depth = context.art.opacity_profile_xs(
        cross_section,
        co_mmr,
        context.molmass,
        context.gravity,
    )
    return context.art.run(optical_depth, temperature_array)


def make_mock_observation(
    raw_truth_flux: Any,
    *,
    seed: int,
    relative_noise: float,
) -> MockObservation:
    """Normalize a truth spectrum and add deterministic Gaussian noise."""

    if relative_noise <= 0.0:
        raise ValueError("relative_noise must be positive.")
    raw = np.asarray(jax.device_get(raw_truth_flux), dtype=np.float64)
    if raw.ndim != 1 or not np.all(np.isfinite(raw)):
        raise ValueError("raw_truth_flux must be a finite one-dimensional array.")
    scale = float(np.median(np.abs(raw)))
    if not np.isfinite(scale) or scale <= 0.0:
        raise ValueError("The truth spectrum does not have a positive flux scale.")
    truth = raw / scale
    noise = np.random.default_rng(seed).normal(
        loc=0.0,
        scale=relative_noise,
        size=truth.shape,
    )
    dtype = jnp.asarray(raw_truth_flux).dtype
    return MockObservation(
        observed=jnp.asarray(truth + noise, dtype=dtype),
        truth=jnp.asarray(truth, dtype=dtype),
        flux_scale=jnp.asarray(scale, dtype=dtype),
        noise_std=jnp.asarray(relative_noise, dtype=dtype),
    )


def add_common_cli_arguments(
    parser: argparse.ArgumentParser,
    *,
    default_output_dir: Union[str, Path],
) -> None:
    """Add consistent data, profile, sampling, and output arguments."""

    parser.add_argument(
        "--co-database",
        type=Path,
        default=None,
        help="Exact local CO/12C-16O/Li2015 directory (never downloaded).",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(default_output_dir),
        help="Directory for JSON and NumPy output artifacts.",
    )
    parser.add_argument("--nlayer", type=int, default=24)
    parser.add_argument("--nu-points", type=int, default=1024)
    parser.add_argument("--num-warmup", type=int, default=500)
    parser.add_argument("--num-samples", type=int, default=1000)
    parser.add_argument("--max-tree-depth", type=int, default=10)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--quick",
        action="store_true",
        help="Use the demo's bounded quick profile with 8 layers and 256 points.",
    )
    parser.add_argument(
        "--preflight-only",
        action="store_true",
        help="Run primal and reverse-mode checks without NUTS.",
    )
    parser.add_argument(
        "--no-progress-bar",
        action="store_true",
        help="Disable the NumPyro progress bar.",
    )


def resolve_run_settings(
    args: argparse.Namespace,
    *,
    quick_num_warmup: int = DEFAULT_QUICK_NUM_WARMUP,
    quick_num_samples: int = DEFAULT_QUICK_NUM_SAMPLES,
    quick_max_tree_depth: int = DEFAULT_QUICK_MAX_TREE_DEPTH,
) -> RunSettings:
    """Validate sampling arguments and apply the quick upper bounds."""

    for name in ("num_warmup", "num_samples", "max_tree_depth"):
        if int(getattr(args, name)) <= 0:
            raise ValueError(f"--{name.replace('_', '-')} must be positive.")
    if int(args.num_samples) < 2:
        raise ValueError("--num-samples must be at least two.")
    if min(quick_num_warmup, quick_num_samples, quick_max_tree_depth) <= 0:
        raise ValueError("Quick-profile limits must be positive.")
    if quick_num_samples < 2:
        raise ValueError("The quick-profile sample limit must be at least two.")
    if args.quick:
        num_warmup = min(int(args.num_warmup), quick_num_warmup)
        num_samples = min(int(args.num_samples), quick_num_samples)
        max_tree_depth = min(int(args.max_tree_depth), quick_max_tree_depth)
    else:
        num_warmup = int(args.num_warmup)
        num_samples = int(args.num_samples)
        max_tree_depth = int(args.max_tree_depth)
    return RunSettings(
        num_warmup=num_warmup,
        num_samples=num_samples,
        seed=int(args.seed),
        progress_bar=not bool(args.no_progress_bar),
        max_tree_depth=max_tree_depth,
        quick=bool(args.quick),
    )


def resolve_demo_shape(args: argparse.Namespace) -> tuple[int, int]:
    """Validate the requested shape and apply quick-mode upper bounds."""

    nlayer = int(args.nlayer)
    nu_points = int(args.nu_points)
    if args.quick:
        nlayer = min(nlayer, 8)
        nu_points = min(nu_points, 256)
    if nlayer < 2:
        raise ValueError("--nlayer must be at least two.")
    if nu_points < 128:
        raise ValueError("--nu-points must be at least 128.")
    return nlayer, nu_points


def run_reverse_mode_nuts(
    model: Callable[[], None],
    observation: MockObservation,
    settings: RunSettings,
):
    """Run NumPyro NUTS with reverse-mode differentiation explicitly selected."""

    del observation  # The closed-over model already contains the observation.
    from numpyro.infer import MCMC, NUTS

    kernel = NUTS(
        model,
        forward_mode_differentiation=False,
        max_tree_depth=settings.max_tree_depth,
    )
    mcmc = MCMC(
        kernel,
        num_warmup=settings.num_warmup,
        num_samples=settings.num_samples,
        num_chains=1,
        progress_bar=settings.progress_bar,
    )
    started = time.perf_counter()
    mcmc.run(
        jax.random.PRNGKey(settings.seed),
        extra_fields=(
            "accept_prob",
            "num_steps",
            "adapt_state.step_size",
        ),
    )
    jax.block_until_ready(mcmc.get_samples())
    setattr(mcmc, "exogibbs_elapsed_seconds", time.perf_counter() - started)
    return mcmc


def _json_ready(value: Any) -> Any:
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, Mapping):
        return {str(key): _json_ready(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_json_ready(item) for item in value]
    array = np.asarray(jax.device_get(value))
    if array.ndim == 0:
        return array.item()
    return array.tolist()


def _preflight_filename(case_name: str) -> str:
    """Return the exact case-local preflight filename."""

    if not case_name or Path(case_name).name != case_name:
        raise ValueError("case_name must be a non-empty filename component.")
    return f"{case_name}_preflight.json"


def _write_strict_json(path: Path, value: Mapping[str, Any]) -> None:
    """Write standard JSON without NaN or Infinity tokens."""

    text = json.dumps(
        value,
        indent=2,
        sort_keys=True,
        allow_nan=False,
    )
    path.write_text(text + "\n", encoding="utf-8")


def write_run_status(
    output_dir: Union[str, Path],
    *,
    case_name: str,
    preflight_only: bool,
    state: str,
) -> None:
    """Write a compact manifest for the current retrieval invocation."""

    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    artifact_names = (*_SHARED_OUTPUT_FILENAMES, _preflight_filename(case_name))
    _write_strict_json(
        output / RUN_STATUS_FILENAME,
        {
            "schema": "exogibbs_retrieval_run_status_v1",
            "case_name": case_name,
            "mode": "preflight" if preflight_only else "sampling",
            "state": state,
            "artifacts": {
                name: (output / name).is_file() for name in artifact_names
            },
        },
    )


def initialize_run_output(
    output_dir: Union[str, Path],
    *,
    case_name: str,
    preflight_only: bool,
) -> Path:
    """Remove only known stale artifacts and mark a new invocation started."""

    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    artifact_names = (*_SHARED_OUTPUT_FILENAMES, _preflight_filename(case_name))
    for name in artifact_names:
        try:
            (output / name).unlink()
        except FileNotFoundError:
            pass
    write_run_status(
        output,
        case_name=case_name,
        preflight_only=preflight_only,
        state="started",
    )
    return output


def _run_config(settings: RunSettings) -> dict[str, Any]:
    """Return the effective single-chain NUTS configuration."""

    return {
        "num_warmup": settings.num_warmup,
        "num_samples": settings.num_samples,
        "num_chains": 1,
        "max_tree_depth": settings.max_tree_depth,
        "seed": settings.seed,
        "quick": settings.quick,
    }


def _finite_float_or_none(value: Any) -> Optional[float]:
    """Return a finite JSON-safe float, or None otherwise."""

    result = float(value)
    return result if np.isfinite(result) else None


def _posterior_statistics(values: np.ndarray) -> dict[str, Optional[float]]:
    """Return finite posterior statistics without non-standard JSON values."""

    if not values.size or not np.all(np.isfinite(values)):
        return {
            "mean": None,
            "standard_deviation": None,
            "q05": None,
            "q50": None,
            "q95": None,
        }
    return {
        "mean": float(np.mean(values)),
        "standard_deviation": float(np.std(values)),
        "q05": float(np.quantile(values, 0.05)),
        "q50": float(np.quantile(values, 0.50)),
        "q95": float(np.quantile(values, 0.95)),
    }


def _sampling_diagnostics(
    mcmc: Any,
    samples: Mapping[str, np.ndarray],
    settings: RunSettings,
) -> dict[str, Any]:
    """Summarize diagnostics needed to reject an unusable demo chain."""

    sample_counts = {values.shape[0] for values in samples.values()}
    sample_count = sample_counts.pop() if len(sample_counts) == 1 else 0
    all_samples_finite = bool(
        samples and all(np.all(np.isfinite(values)) for values in samples.values())
    )
    parameters_with_variation = {
        name: bool(values.shape[0] > 1 and np.any(values[1:] != values[0]))
        for name, values in samples.items()
    }
    all_parameters_vary = bool(
        parameters_with_variation and all(parameters_with_variation.values())
    )
    extra_fields = {
        name: np.asarray(jax.device_get(values))
        for name, values in mcmc.get_extra_fields().items()
    }
    sampler_field_names = (
        "accept_prob",
        "num_steps",
        "adapt_state.step_size",
    )
    sampler_statistics_available = all(
        name in extra_fields and extra_fields[name].shape == (sample_count,)
        for name in sampler_field_names
    )
    sampler_statistics = (
        [extra_fields[name] for name in sampler_field_names]
        if sampler_statistics_available
        else []
    )
    sampler_statistics_finite = bool(
        sampler_statistics
        and all(np.all(np.isfinite(values)) for values in sampler_statistics)
    )
    divergent = extra_fields.get("diverging")
    divergence_diagnostics_available = bool(
        divergent is not None
        and divergent.dtype == np.bool_
        and divergent.shape == (sample_count,)
    )
    divergence_count = (
        int(np.sum(divergent)) if divergence_diagnostics_available else None
    )
    diagnostics: dict[str, Any] = {
        "sample_count": sample_count,
        "sample_count_matches_config": sample_count == settings.num_samples,
        "all_samples_finite": all_samples_finite,
        "parameters_with_variation": parameters_with_variation,
        "all_parameters_vary": all_parameters_vary,
        "sampler_statistics_available": sampler_statistics_available,
        "sampler_statistics_finite": sampler_statistics_finite,
        "divergence_diagnostics_available": divergence_diagnostics_available,
        "divergences": divergence_count,
        "divergence_fraction": (
            float(divergence_count / sample_count)
            if divergence_count is not None and sample_count
            else None
        ),
        "effective_sample_size": None,
        "r_hat": None,
        "convergence_diagnostics_note": (
            "ESS and R-hat are not reported for this single-chain demo."
        ),
    }
    accept_probability = extra_fields.get("accept_prob")
    if accept_probability is not None and accept_probability.size:
        diagnostics.update(
            {
                "mean_accept_probability": _finite_float_or_none(
                    np.mean(accept_probability)
                ),
                "minimum_accept_probability": _finite_float_or_none(
                    np.min(accept_probability)
                ),
            }
        )
    num_steps = extra_fields.get("num_steps")
    if num_steps is not None and num_steps.size:
        diagnostics.update(
            {
                "mean_num_steps": _finite_float_or_none(np.mean(num_steps)),
                "maximum_num_steps": (
                    int(np.max(num_steps))
                    if np.all(np.isfinite(num_steps))
                    else None
                ),
            }
        )
    step_size = extra_fields.get("adapt_state.step_size")
    if step_size is not None and step_size.size:
        diagnostics["step_size"] = _finite_float_or_none(
            np.ravel(step_size)[-1]
        )
    failure_reasons = []
    if not diagnostics["sample_count_matches_config"]:
        failure_reasons.append("saved sample count does not match the run config")
    if not all_samples_finite:
        failure_reasons.append("posterior samples are not finite")
    if not all_parameters_vary:
        failure_reasons.append("one or more posterior parameters did not vary")
    if not sampler_statistics_finite:
        failure_reasons.append("sampler statistics are missing or non-finite")
    if not divergence_diagnostics_available:
        failure_reasons.append("divergence diagnostics are unavailable")
    elif divergence_count:
        failure_reasons.append(f"{divergence_count} divergent transition(s)")
    diagnostics["failure_reasons"] = failure_reasons
    diagnostics["passed"] = not failure_reasons
    return diagnostics


def write_run_outputs(
    output_dir: Union[str, Path],
    *,
    case_name: str,
    context: SpectralContext,
    observation: MockObservation,
    mcmc: Any = None,
    metadata: Optional[Mapping[str, Any]] = None,
    settings: Optional[RunSettings] = None,
) -> None:
    """Write deterministic inputs, posterior samples, and a compact summary."""

    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output / "mock_spectrum.npz",
        wavenumber=np.asarray(jax.device_get(context.nu_grid)),
        observed=np.asarray(jax.device_get(observation.observed)),
        truth=np.asarray(jax.device_get(observation.truth)),
        noise_std=float(observation.noise_std),
        flux_scale=float(observation.flux_scale),
    )
    summary: dict[str, Any] = {
        "schema": "exogibbs_exojax_nuts_vjp_demo_v1",
        "case_name": case_name,
        "jax_backend": jax.default_backend(),
        "jax_devices": [str(device) for device in jax.devices()],
        "nlayer": int(context.art.pressure.shape[0]),
        "nu_points": int(np.asarray(context.nu_grid).shape[0]),
        "metadata": _json_ready(metadata or {}),
    }
    if settings is not None:
        summary["run_config"] = _run_config(settings)
    if mcmc is not None:
        if settings is None:
            raise ValueError("settings are required when writing MCMC outputs.")
        samples = {
            name: np.asarray(jax.device_get(values))
            for name, values in mcmc.get_samples().items()
        }
        np.savez_compressed(output / "posterior_samples.npz", **samples)
        sampling_diagnostics = _sampling_diagnostics(mcmc, samples, settings)
        summary.update(
            {
                "elapsed_seconds": float(
                    getattr(mcmc, "exogibbs_elapsed_seconds", np.nan)
                ),
                "divergences": sampling_diagnostics["divergences"],
                "sampling_diagnostics": sampling_diagnostics,
                "posterior": {
                    name: _posterior_statistics(values)
                    for name, values in samples.items()
                },
            }
        )
        summary["elapsed_seconds"] = _finite_float_or_none(
            summary["elapsed_seconds"]
        )
    try:
        _write_strict_json(output / "run_summary.json", summary)
    except (TypeError, ValueError):
        write_run_status(
            output,
            case_name=case_name,
            preflight_only=mcmc is None,
            state="failed",
        )
        raise
    sampling_failed = bool(
        mcmc is not None and not summary["sampling_diagnostics"]["passed"]
    )
    write_run_status(
        output,
        case_name=case_name,
        preflight_only=mcmc is None,
        state=(
            "failed"
            if sampling_failed
            else "preflight_complete" if mcmc is None else "complete"
        ),
    )
    if sampling_failed:
        diagnostics = summary["sampling_diagnostics"]
        raise RuntimeError(
            "NUTS sampling diagnostics failed: "
            f"{'; '.join(diagnostics['failure_reasons'])}. Outputs were "
            f"written to {output}."
        )


def _gas_prior_corners() -> tuple[tuple[float, float, float], ...]:
    return tuple(
        itertools.product(
            T0_PRIOR_BOUNDS_K,
            ALPHA_PRIOR_BOUNDS,
            LOG_CO_SCALE_PRIOR_BOUNDS,
        )
    )


def _scale_carbon_and_oxygen(
    reference: Any,
    carbon_index: int,
    oxygen_index: int,
    log_co_scale: Any,
) -> jax.Array:
    values = jnp.asarray(reference)
    indices = jnp.asarray([carbon_index, oxygen_index], dtype=jnp.int32)
    scale = jnp.power(jnp.asarray(10.0, dtype=values.dtype), log_co_scale)
    return values.at[indices].set(values[indices] * scale)


def run_gas_demo(
    *,
    use_grid_initializer: bool,
    case_name: str,
    argv: Optional[Sequence[str]] = None,
) -> int:
    """Run one gas-only retrieval; the two cases differ only by initializer."""

    parser = argparse.ArgumentParser(description=__doc__)
    add_common_cli_arguments(
        parser,
        default_output_dir=Path("results") / "vjp_retrieval" / case_name,
    )
    parser.add_argument(
        "--relative-noise",
        type=float,
        default=DEFAULT_RELATIVE_NOISE,
    )
    args = parser.parse_args(argv)
    settings = resolve_run_settings(
        args,
        quick_num_warmup=GAS_QUICK_NUM_WARMUP,
        quick_num_samples=GAS_QUICK_NUM_SAMPLES,
        quick_max_tree_depth=GAS_QUICK_MAX_TREE_DEPTH,
    )
    nlayer, nu_points = resolve_demo_shape(args)
    co_database = require_local_co_database(args.co_database)
    if not np.isfinite(args.relative_noise) or args.relative_noise <= 0.0:
        raise ValueError("--relative-noise must be positive.")
    initialize_run_output(
        args.output_dir,
        case_name=case_name,
        preflight_only=args.preflight_only,
    )
    context = build_spectral_context(
        co_database,
        nlayer=nlayer,
        nu_points=nu_points,
    )

    from exogibbs.api.gas import EquilibriumOptions, GridEquilibriumInitializer
    from exogibbs.api.gas import solve_profile as solve_gas_profile
    from exogibbs.presets.fastchem import chemsetup

    chemistry = chemsetup(silent=True)
    if chemistry.element_vector_reference is None:
        raise ValueError("The FastChem preset has no reference element vector.")
    reference = jnp.asarray(chemistry.element_vector_reference, dtype=jnp.float64)
    carbon_index = chemistry.elements.index("C")
    oxygen_index = chemistry.elements.index("O")
    co_species_index = chemistry.species.index("C1O1")

    initializer = None
    grid = None
    if use_grid_initializer:
        from exogibbs.api import (
            get_default_equilibrium_grid_path,
            load_equilibrium_grid_netcdf,
        )

        grid = load_equilibrium_grid_netcdf(
            str(get_default_equilibrium_grid_path("fastchem"))
        )
        initializer = GridEquilibriumInitializer(
            grid=grid,
            preset_name="fastchem",
        )
    options = EquilibriumOptions(
        epsilon_crit=1.0e-10,
        max_iter=1000,
        method="vmap_cold",
    )
    pressure = jnp.asarray(context.art.pressure, dtype=jnp.float64)

    def raw_flux(t0_kelvin, alpha, log_co_scale):
        temperature = context.art.powerlaw_temperature(t0_kelvin, alpha)
        inventory = _scale_carbon_and_oxygen(
            reference,
            carbon_index,
            oxygen_index,
            log_co_scale,
        )
        result = solve_gas_profile(
            chemistry,
            temperature,
            pressure,
            inventory,
            Pref=REFERENCE_PRESSURE_BAR,
            initializer=initializer,
            options=options,
        )
        return co_emission_flux(
            context,
            temperature,
            result.x[:, co_species_index],
        )

    truth_flux = raw_flux(TRUTH_T0_K, TRUTH_ALPHA, TRUTH_LOG_CO_SCALE)
    observation = make_mock_observation(
        truth_flux,
        seed=settings.seed,
        relative_noise=args.relative_noise,
    )

    corner_reports = []
    for t0_kelvin, alpha, log_co_scale in _gas_prior_corners():
        temperature = context.art.powerlaw_temperature(t0_kelvin, alpha)
        inventory = _scale_carbon_and_oxygen(
            reference,
            carbon_index,
            oxygen_index,
            log_co_scale,
        )
        _result, diagnostics = solve_gas_profile(
            chemistry,
            temperature,
            pressure,
            inventory,
            Pref=REFERENCE_PRESSURE_BAR,
            initializer=initializer,
            options=options,
            return_diagnostics=True,
        )
        converged = bool(jnp.all(diagnostics["converged"]))
        finite = bool(jnp.all(jnp.isfinite(diagnostics["final_residual"])))
        corner_reports.append(
            {
                "t0_kelvin": t0_kelvin,
                "alpha": alpha,
                "log_co_scale": log_co_scale,
                "temperature_min_kelvin": float(jnp.min(temperature)),
                "temperature_max_kelvin": float(jnp.max(temperature)),
                "converged": converged,
                "finite_residual": finite,
                "maximum_iterations": int(jnp.max(diagnostics["n_iter"])),
            }
        )
    if not all(
        item["converged"] and item["finite_residual"]
        for item in corner_reports
    ):
        raise RuntimeError(f"Gas solver prior-corner preflight failed: {corner_reports}")

    def spectral_loss(t0_kelvin, alpha, log_co_scale):
        prediction = raw_flux(t0_kelvin, alpha, log_co_scale)
        prediction = prediction / observation.flux_scale
        return jnp.mean(jnp.square(prediction - observation.observed))

    loss, gradient = jax.value_and_grad(
        spectral_loss,
        argnums=(0, 1, 2),
    )(TRUTH_T0_K, TRUTH_ALPHA, TRUTH_LOG_CO_SCALE)
    gradient_values = tuple(float(value) for value in gradient)
    if not np.isfinite(float(loss)) or not np.all(np.isfinite(gradient_values)):
        raise RuntimeError(
            f"Reverse-mode spectrum preflight is non-finite: {loss}, {gradient_values}"
        )
    preflight = {
        "case_name": case_name,
        "initializer": "packaged_fastchem_grid" if initializer else "uniform_cold",
        "reverse_mode_gradient": {
            "T0": gradient_values[0],
            "alpha": gradient_values[1],
            "log_co_scale": gradient_values[2],
        },
        "spectral_loss": float(loss),
        "prior_corners": corner_reports,
        "passed": True,
    }
    if grid is not None:
        preflight["grid_bounds"] = {
            "temperature": [
                float(jnp.min(grid.temperature_axis)),
                float(jnp.max(grid.temperature_axis)),
            ],
            "pressure": [
                float(jnp.min(grid.pressure_axis)),
                float(jnp.max(grid.pressure_axis)),
            ],
            "log10_z_over_z_sun": [
                float(jnp.min(grid.log10_z_over_z_sun_axis)),
                float(jnp.max(grid.log10_z_over_z_sun_axis)),
            ],
        }

    if args.preflight_only:
        write_run_outputs(
            args.output_dir,
            case_name=case_name,
            context=context,
            observation=observation,
            metadata={"preflight": preflight},
            settings=settings,
        )
        print(f"{case_name}: preflight passed; outputs: {args.output_dir}")
        return 0

    import numpyro
    import numpyro.distributions as dist

    def model():
        t0_kelvin = numpyro.sample("T0", dist.Uniform(*T0_PRIOR_BOUNDS_K))
        alpha = numpyro.sample("alpha", dist.Uniform(*ALPHA_PRIOR_BOUNDS))
        log_co_scale = numpyro.sample(
            "log_co_scale",
            dist.Uniform(*LOG_CO_SCALE_PRIOR_BOUNDS),
        )
        prediction = raw_flux(t0_kelvin, alpha, log_co_scale)
        prediction = prediction / observation.flux_scale
        numpyro.sample(
            "spectrum",
            dist.Normal(prediction, observation.noise_std),
            obs=observation.observed,
        )

    mcmc = run_reverse_mode_nuts(model, observation, settings)
    write_run_outputs(
        args.output_dir,
        case_name=case_name,
        context=context,
        observation=observation,
        mcmc=mcmc,
        metadata={"preflight": preflight},
        settings=settings,
    )
    print(f"{case_name}: completed; outputs: {args.output_dir}")
    return 0


__all__ = (
    "MockObservation",
    "RunSettings",
    "SpectralContext",
    "TRUTH_ALPHA",
    "TRUTH_LOG_CO_SCALE",
    "TRUTH_T0_K",
    "add_common_cli_arguments",
    "build_spectral_context",
    "co_emission_flux",
    "initialize_run_output",
    "make_mock_observation",
    "require_local_co_database",
    "resolve_demo_shape",
    "resolve_run_settings",
    "run_gas_demo",
    "run_reverse_mode_nuts",
    "write_run_outputs",
    "write_run_status",
)
