"""ExoGibbs portions of the examples listed in the documentation."""

from __future__ import annotations

from collections import Counter
from typing import Any, Callable, Sequence

import numpy as np

from benchmarks.documented_examples.instrumentation import TimingCollector
from benchmarks.documented_examples.manifest import DocumentedExampleCase


ACCEPTED_STATUSES = frozenset({"converged", "converged_with_caveat"})


def _limited(values: Sequence[float], smoke_layers: int | None) -> np.ndarray:
    array = np.asarray(values, dtype=np.float64)
    if smoke_layers is None:
        return array.copy()
    if isinstance(smoke_layers, bool) or smoke_layers <= 0:
        raise ValueError("smoke_layers must be a positive integer.")
    return array[: min(int(smoke_layers), array.size)].copy()


def _status_summary(statuses: Sequence[str]) -> dict[str, int]:
    return dict(sorted(Counter(str(status) for status in statuses).items()))


def _solution_summary(solution: Any) -> dict[str, Any]:
    converged = np.asarray(solution.converged, dtype=bool)
    return {
        "layer_count": int(converged.size),
        "converged_layer_count": int(np.count_nonzero(converged)),
        "all_layers_converged": bool(np.all(converged)),
        "statuses": _status_summary(solution.status),
    }


def _profile_summary(profile: Any) -> dict[str, Any]:
    """Summarize convergence from a public profile result."""

    converged = np.asarray(
        [layer.converged for layer in profile.layers],
        dtype=bool,
    )
    statuses = tuple(layer.status for layer in profile.layers)
    return {
        "layer_count": int(converged.size),
        "converged_layer_count": int(np.count_nonzero(converged)),
        "all_layers_converged": bool(np.all(converged)),
        "statuses": _status_summary(statuses),
    }


def _profile_physical_audit(profile: Any) -> dict[str, Any]:
    """Summarize the production zero-barrier acceptance gate per layer."""

    positive_condensate_layers = 0
    audited_positive_layers = 0
    failed_layer_indices = []
    acceptance_tiers = Counter()
    for index, layer in enumerate(profile.layers):
        acceptance_tiers[str(layer.acceptance_tier)] += 1
        amounts = np.asarray(layer.condensate_amounts, dtype=np.float64)
        physically_valid = bool(
            np.all(np.isfinite(np.asarray(layer.gas_n, dtype=np.float64)))
            and np.all(np.isfinite(amounts))
            and np.all(np.asarray(layer.gas_n, dtype=np.float64) >= 0.0)
            and np.all(amounts >= 0.0)
        )
        lifecycle = (
            (layer.diagnostics or {}).get("fixed_support_v2", {})
            if layer.diagnostics is not None
            else {}
        )
        positive_condensate = bool(np.any(amounts > 0.0))
        zero_barrier_passed = True
        if positive_condensate:
            positive_condensate_layers += 1
            caller_audit = lifecycle.get("caller_gauge_zero_barrier_kkt", {})
            zero_barrier_passed = caller_audit.get("accepted") is True
            if zero_barrier_passed:
                audited_positive_layers += 1
        if (
            not bool(layer.converged)
            or str(layer.status) not in ACCEPTED_STATUSES
            or not physically_valid
            or not zero_barrier_passed
        ):
            failed_layer_indices.append(index)
    return {
        "layer_count": len(profile.layers),
        "positive_condensate_layer_count": positive_condensate_layers,
        "audited_positive_condensate_layer_count": audited_positive_layers,
        "all_layers_finite_and_physically_accepted": not failed_layer_indices,
        "failed_layer_indices": tuple(failed_layer_indices),
        "acceptance_tiers": dict(sorted(acceptance_tiers.items())),
    }


def _gas_profile_physical_audit(
    gas_x: Any,
    converged: Any,
    species_count: int,
) -> dict[str, Any]:
    """Check the gas-only profile contract used by the source example."""

    fractions = np.asarray(gas_x, dtype=np.float64)
    converged_array = np.asarray(converged, dtype=bool)
    expected_shape = (converged_array.size, species_count)
    shape_valid = fractions.shape == expected_shape
    if shape_valid:
        invalid_layers = np.flatnonzero(
            np.any(~np.isfinite(fractions) | (fractions < 0.0), axis=1)
            | (np.sum(fractions, axis=1) <= 0.0)
        )
    else:
        invalid_layers = np.arange(converged_array.size)
    return {
        "layer_count": int(converged_array.size),
        "expected_shape": expected_shape,
        "actual_shape": fractions.shape,
        "invalid_layer_indices": tuple(int(item) for item in invalid_layers),
        "all_layers_finite_nonnegative_and_nonempty": bool(
            shape_valid and invalid_layers.size == 0
        ),
    }


def run_visscher_2006(
    collector: TimingCollector,
    smoke_layers: int | None,
) -> dict[str, Any]:
    """Run the documented reduced KCl and Na2S ExoGibbs scans."""

    from examples.comparisons import (
        comparison_with_visscher_2006_na2s_morley_2012_kcl as example,
    )

    temperatures = _limited(example.DEFAULT_TEMPERATURES_K, smoke_layers)
    with collector.phase("build_reduced_setups", category="setup"):
        setups = {
            case.label: example.build_reduced_setup(case)
            for case in example.BENCHMARK_CASES
        }

    summaries = {}
    audits = {}
    for case in example.BENCHMARK_CASES:
        phase = f"solve_{case.label.lower()}"
        with collector.phase(phase, category="solver"):
            profile = example.solve_condensate_profile(
                setups[case.label],
                T=example.jnp.asarray(temperatures, dtype=example.jnp.float64),
                P=example.jnp.full(
                    temperatures.shape,
                    example.PRESSURE_BAR,
                    dtype=example.jnp.float64,
                ),
                b=example.solar_element_budget(setups[case.label]),
                options=example.CondensateEquilibriumOptions(),
            )
            example.jax.block_until_ready(profile.batched_arrays)
        summaries[case.label] = _profile_summary(profile)
        audits[case.label] = _profile_physical_audit(profile)
    return {
        "output_layer_count": sum(
            summary["layer_count"] for summary in summaries.values()
        ),
        "profiles": summaries,
        "physical_audits": audits,
        "all_layers_converged": all(
            summary["all_layers_converged"] for summary in summaries.values()
        ) and all(
            audit["all_layers_finite_and_physically_accepted"]
            for audit in audits.values()
        ),
    }


def run_visscher_2010(
    collector: TimingCollector,
    smoke_layers: int | None,
) -> dict[str, Any]:
    """Run the documented silicate competition ExoGibbs scans."""

    from examples.comparisons import (
        comparison_with_visscher_2010_forsterite_enstatite as example,
    )

    temperatures = _limited(example.DEFAULT_TEMPERATURES_K, smoke_layers)
    with collector.phase("build_reduced_setups", category="setup"):
        setups = {
            run.key: example.build_reduced_setup(run)
            for run in example.COMPETITION_RUNS
        }

    summaries = {}
    audits = {}
    for run in example.COMPETITION_RUNS:
        phase = f"solve_{run.key}"
        with collector.phase(phase, category="solver"):
            profile = example.solve_condensate_profile(
                setups[run.key],
                T=example.jnp.asarray(temperatures, dtype=example.jnp.float64),
                P=example.jnp.full(
                    temperatures.shape,
                    example.PRESSURE_BAR,
                    dtype=example.jnp.float64,
                ),
                b=example.solar_element_budget(setups[run.key]),
                options=example.CondensateEquilibriumOptions(),
            )
            example.jax.block_until_ready(profile.batched_arrays)
        summaries[run.key] = _profile_summary(profile)
        audits[run.key] = _profile_physical_audit(profile)
    return {
        "output_layer_count": sum(
            summary["layer_count"] for summary in summaries.values()
        ),
        "profiles": summaries,
        "physical_audits": audits,
        "all_layers_converged": all(
            summary["all_layers_converged"] for summary in summaries.values()
        ) and all(
            audit["all_layers_finite_and_physically_accepted"]
            for audit in audits.values()
        ),
    }


def run_ito_2025_rainout(
    collector: TimingCollector,
    smoke_layers: int | None,
) -> dict[str, Any]:
    """Run the ExoGibbs half of the propagated Ito rainout example."""

    from examples.comparisons import comparison_with_ito_2025 as ito_base
    from examples.comparisons import comparison_with_ito_2025_rainout as example

    with collector.phase("load_ito_profile", category="setup"):
        profile = ito_base.load_ito_profile(example.DEFAULT_INPUT)
        target = example._target_profile(profile, smoke_layers)
        layer1_abundance = ito_base.reactive_element_abundances(
            profile.gas_fractions[0]
        )
    with collector.phase("solve_propagated_rainout", category="solver"):
        solution = example.solve_exogibbs_rainout(
            target,
            layer1_abundance=layer1_abundance,
        )
    summary = _solution_summary(solution)
    physically_valid = bool(
        np.all(np.isfinite(solution.gas_fractions))
        and np.all(np.isfinite(solution.condensate_amounts))
        and np.all(np.asarray(solution.gas_fractions) >= 0.0)
        and np.all(np.asarray(solution.condensate_amounts) >= 0.0)
        and np.all(np.asarray(solution.converged, dtype=bool))
        and all(str(status) in ACCEPTED_STATUSES for status in solution.status)
    )
    return {
        "output_layer_count": summary["layer_count"],
        "profiles": {"propagated_rainout": summary},
        "fixed_point_iterations": int(solution.fixed_point_iterations),
        "all_layers_finite_and_accepted": physically_valid,
        "all_layers_converged": (
            summary["all_layers_converged"] and physically_valid
        ),
    }


def run_fe_fes_rainout(
    collector: TimingCollector,
    smoke_layers: int | None,
) -> dict[str, Any]:
    """Run the documented Fe-FeS local and rainout profiles."""

    from examples.comparisons import demo_fe_fes_rainout as example

    temperatures = _limited(example.DEFAULT_TEMPERATURES_K, smoke_layers)
    with collector.phase("build_reduced_setup", category="setup"):
        setup = example.build_reduced_setup()
    summaries = {}
    audits = {}
    for name, rainout in (("local", False), ("rainout", True)):
        with collector.phase(f"solve_{name}", category="solver"):
            options = example.CondensateEquilibriumOptions(
                rainout=rainout,
                profile_method="scan_hot_from_bottom" if rainout else None,
            )
            profile = example.solve_condensate_profile(
                setup,
                T=example.jnp.asarray(temperatures, dtype=example.jnp.float64),
                P=example.jnp.full(
                    temperatures.shape,
                    example.PRESSURE_BAR,
                    dtype=example.jnp.float64,
                ),
                b=example.solar_element_budget(setup),
                options=options,
            )
            example.jax.block_until_ready(profile.batched_arrays)
        summaries[name] = _profile_summary(profile)
        audits[name] = _profile_physical_audit(profile)
    return {
        "output_layer_count": sum(
            summary["layer_count"] for summary in summaries.values()
        ),
        "profiles": summaries,
        "physical_audits": audits,
        "all_layers_converged": all(
            summary["all_layers_converged"] for summary in summaries.values()
        ) and all(
            audit["all_layers_finite_and_physically_accepted"]
            for audit in audits.values()
        ),
    }


def run_fastchem4_l_dwarf(
    collector: TimingCollector,
    smoke_layers: int | None,
) -> dict[str, Any]:
    """Run the ExoGibbs profiles used by the documented L-dwarf figure."""

    from examples.comparisons import (
        comparison_with_fastchem4_condensates as example,
    )
    from exogibbs.api.condensate import CondensateEquilibriumOptions
    from exogibbs.api.gas import EquilibriumOptions as GasEquilibriumOptions

    temperatures, pressures = example._profile_conditions("l-dwarf")
    temperatures = _limited(temperatures, smoke_layers)
    pressures = _limited(pressures, smoke_layers)
    with collector.phase("build_full_catalog_setup", category="setup"):
        setup = example.condensate_chemical_setup(
            gas_path="FastChem4/logK/logK_wo_ions.dat",
            condensate_path="FastChem4/logK/logK_condensates.dat",
            species_default_elements=False,
            element_file="FastChem4/element_abundances/asplund_2021.dat",
            silent=True,
        )
        aligned_abundance = example.build_aligned_abundance_vector(
            setup.elements,
            source="fastchem_file",
            normalize=True,
            element_file=example.ELEMENT_FILE,
        )
        budget = example.jnp.asarray(
            aligned_abundance.vector,
            dtype=example.jnp.float64,
        )

    with collector.phase("solve_condensate_profile", category="solver"):
        condensate_profile = example.solve_condensate_profile(
            setup,
            T=example.jnp.asarray(temperatures, dtype=example.jnp.float64),
            P=example.jnp.asarray(pressures, dtype=example.jnp.float64),
            b=budget,
            options=CondensateEquilibriumOptions(return_diagnostics=True),
            return_diagnostics=True,
        )
        example.jax.block_until_ready(condensate_profile.batched_arrays)
    condensate_converged = np.asarray(
        [layer.converged for layer in condensate_profile.layers],
        dtype=bool,
    )
    condensate_statuses = tuple(
        layer.status for layer in condensate_profile.layers
    )

    with collector.phase("solve_gas_only_profile", category="solver"):
        gas_profile, gas_diagnostics = example.solve_gas_profile(
            setup.gas_setup,
            T=example.jnp.asarray(temperatures, dtype=example.jnp.float64),
            P=example.jnp.asarray(pressures, dtype=example.jnp.float64),
            b=budget,
            options=GasEquilibriumOptions(),
            return_diagnostics=True,
        )
        example.jax.block_until_ready(
            (gas_profile.x, gas_diagnostics["converged"])
        )
    gas_converged = np.asarray(gas_diagnostics["converged"], dtype=bool)
    gas_x = np.asarray(gas_profile.x, dtype=np.float64)
    gas_audit = _gas_profile_physical_audit(
        gas_x,
        gas_converged,
        len(setup.gas_species),
    )
    gas_physically_valid = gas_audit[
        "all_layers_finite_nonnegative_and_nonempty"
    ]

    profiles = {
        "condensate": {
            "layer_count": int(condensate_converged.size),
            "converged_layer_count": int(
                np.count_nonzero(condensate_converged)
            ),
            "all_layers_converged": bool(np.all(condensate_converged)),
            "statuses": _status_summary(condensate_statuses),
        },
        "gas_only": {
            "layer_count": int(gas_converged.size),
            "converged_layer_count": int(np.count_nonzero(gas_converged)),
            "all_layers_converged": bool(np.all(gas_converged)),
        },
    }
    condensate_audit = _profile_physical_audit(condensate_profile)
    return {
        "output_layer_count": sum(
            profile["layer_count"] for profile in profiles.values()
        ),
        "profiles": profiles,
        "physical_audits": {
            "condensate": condensate_audit,
            "gas_only": gas_audit,
        },
        "all_layers_converged": all(
            profile["all_layers_converged"] for profile in profiles.values()
        )
        and condensate_audit["all_layers_finite_and_physically_accepted"]
        and gas_physically_valid,
        "catalog": {
            "element_count": len(setup.elements),
            "gas_species_count": len(setup.gas_species),
            "condensate_species_count": len(setup.condensate_species),
        },
    }


WORKLOADS: dict[
    str,
    Callable[[TimingCollector, int | None], dict[str, Any]],
] = {
    "run_visscher_2006": run_visscher_2006,
    "run_visscher_2010": run_visscher_2010,
    "run_ito_2025_rainout": run_ito_2025_rainout,
    "run_fe_fes_rainout": run_fe_fes_rainout,
    "run_fastchem4_l_dwarf": run_fastchem4_l_dwarf,
}


def run_case(
    case: DocumentedExampleCase,
    collector: TimingCollector,
    smoke_layers: int | None,
) -> dict[str, Any]:
    """Execute one manifest workload and enforce its convergence contract."""

    try:
        workload = WORKLOADS[case.workload]
    except KeyError as error:
        raise ValueError(
            f"No workload is registered for {case.workload!r}."
        ) from error
    result = workload(collector, smoke_layers)
    if not result.get("all_layers_converged", False):
        raise RuntimeError(f"Not every output layer converged: {result!r}")
    actual = int(result.get("output_layer_count", 0))
    if actual <= 0:
        raise RuntimeError("The workload returned no output layers.")
    expected = case.expected_output_layer_count(smoke_layers)
    if actual != expected:
        scope = "Full" if smoke_layers is None else "Smoke"
        raise RuntimeError(
            f"{scope} workload returned {actual} output layers; expected "
            f"{expected}."
        )
    return result


__all__ = ("WORKLOADS", "run_case")
