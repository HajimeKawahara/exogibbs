"""Benchmark a fixed-support JAX PD-IPM reduced-step core.

This scratch benchmark isolates the most GPU-friendly part of the condensate
solver: a fixed-shape active support, no route retries, no diagnostics payload,
and no SciPy/NumPy work inside the timed JIT core.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass, fields, is_dataclass
import json
import os
import statistics
import sys
import time
from pathlib import Path
from typing import Any, Mapping

import numpy as np


def _bootstrap_jax_platform() -> str:
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument(
        "--jax-platform",
        choices=("cpu", "cuda", "gpu", "default"),
        default="cpu",
    )
    args, _ = parser.parse_known_args()
    platform = str(args.jax_platform)
    if platform != "default":
        os.environ["JAX_PLATFORMS"] = platform
        os.environ["JAX_PLATFORM_NAME"] = platform
    return platform


REQUESTED_JAX_PLATFORM = _bootstrap_jax_platform()
os.environ.setdefault("JAX_ENABLE_X64", "1")

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import jax  # noqa: E402
import jax.numpy as jnp  # noqa: E402

from exogibbs.api.chemistry import ThermoState  # noqa: E402
from exogibbs.api.condensate_equilibrium import _ln_normalized_pressure  # noqa: E402
from exogibbs.condensates.curated_profiles import (  # noqa: E402
    FRESH_CURATED_PROFILES,
    case_id_for_profile,
    element_budget_for_profile,
    support_payload_for_profile,
)
from exogibbs.optimize.minimize_cond import (  # noqa: E402
    CondensateEquilibriumInit,
    CondensateRGIEReducedCouplingConfig,
    _solve_pdipm_rgie_v11_activity_correction_layer,
    solve_gas_equilibrium_with_duals,
    solve_restricted_support_condensate_layer,
)
from exogibbs.presets.fastchem4_cond import condensate_chemical_setup  # noqa: E402


DEFAULT_TARGETS = (
    "solar_metal_sulfide_or_Fe_Ni_S_region:8",
    "solar_water_condensation:4",
    "lowT_strong_condensation_budget_stress:4",
)


@dataclass(frozen=True)
class FixedSupportInputs:
    family: str
    layer_index: int
    case_id: str
    temperature: float
    pressure: float
    support_indices: np.ndarray
    support_amounts_init: np.ndarray
    formula_matrix: np.ndarray
    formula_matrix_cond_active: np.ndarray
    element_inventory_target: np.ndarray
    hvector: np.ndarray
    hvector_cond_active: np.ndarray
    ln_normalized_pressure: float
    q0: np.ndarray
    r0: np.ndarray
    qtot0: float
    lambda0: np.ndarray
    rho0: np.ndarray
    epsilon0: np.ndarray
    max_iter: int


def _block_tree(value: Any) -> None:
    if hasattr(value, "block_until_ready"):
        value.block_until_ready()
        return
    if is_dataclass(value) and not isinstance(value, type):
        for field in fields(value):
            _block_tree(getattr(value, field.name))
        return
    if isinstance(value, Mapping):
        for item in value.values():
            _block_tree(item)
        return
    if isinstance(value, (list, tuple)):
        for item in value:
            _block_tree(item)


def _json_safe(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, (np.bool_, bool)):
        return bool(value)
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, (np.floating, float)):
        return float(value)
    if hasattr(value, "tolist"):
        return _json_safe(value.tolist())
    return value


def _jax_runtime_report(requested_platform: str) -> dict[str, Any]:
    devices = jax.devices()
    return {
        "requested_platform": requested_platform,
        "default_backend": jax.default_backend(),
        "default_device": str(devices[0]) if devices else None,
        "devices": [
            {
                "id": int(getattr(device, "id", index)),
                "platform": str(device.platform),
                "device_kind": str(getattr(device, "device_kind", "")),
                "process_index": int(getattr(device, "process_index", 0)),
            }
            for index, device in enumerate(devices)
        ],
        "jax_enable_x64": os.environ.get("JAX_ENABLE_X64"),
        "jax_platforms_env": os.environ.get("JAX_PLATFORMS"),
        "jax_platform_name_env": os.environ.get("JAX_PLATFORM_NAME"),
    }


def _parse_target(raw: str) -> tuple[str, int]:
    family, _, layer = raw.partition(":")
    if not family or not layer:
        raise ValueError(f"target must be FAMILY:LAYER, got {raw!r}")
    if family not in FRESH_CURATED_PROFILES:
        raise ValueError(f"unknown curated family: {family!r}")
    return family, int(layer)


def _build_fixed_support_inputs(
    *,
    setup: Any,
    family: str,
    layer_index: int,
    support_mode: str,
) -> FixedSupportInputs:
    definition = FRESH_CURATED_PROFILES[family]
    temperatures = tuple(definition.temperatures)
    pressures = tuple(definition.pressures)
    if layer_index < 0 or layer_index >= len(temperatures):
        raise ValueError(f"layer {layer_index} out of range for {family}")
    temperature = float(temperatures[layer_index])
    pressure = float(pressures[layer_index])
    budget = element_budget_for_profile(setup, definition)
    support_indices, support_amounts_init = support_payload_for_profile(
        setup,
        definition,
        budget,
    )
    support_indices = np.asarray(support_indices, dtype=np.int32)
    support_amounts_init = np.asarray(support_amounts_init, dtype=np.float64)
    if support_mode != "explicit_payload":
        raise ValueError("only support_mode='explicit_payload' is implemented")
    if support_indices.size == 0:
        raise ValueError("fixed-support core requires a non-empty support")

    state = ThermoState(
        temperature=temperature,
        ln_normalized_pressure=_ln_normalized_pressure(pressure, 1.0),
        element_vector=jnp.asarray(budget, dtype=jnp.float64),
    )
    gas = solve_gas_equilibrium_with_duals(
        state,
        setup.formula_matrix,
        setup.gas_setup.hvector_func,
        gas_epsilon_crit=1.0e-12,
    )
    hvector = np.asarray(setup.gas_setup.hvector_func(temperature), dtype=np.float64)
    hcond_full = np.asarray(
        setup.condensate_setup.hvector_func(temperature),
        dtype=np.float64,
    )
    formula_matrix = np.asarray(setup.formula_matrix, dtype=np.float64)
    formula_matrix_cond = np.asarray(setup.formula_matrix_cond, dtype=np.float64)
    formula_matrix_cond_active = formula_matrix_cond[:, support_indices]
    hvector_cond_active = hcond_full[support_indices]
    q0 = np.asarray(gas["ln_nk"], dtype=np.float64)
    r0 = np.log(np.maximum(support_amounts_init, 1.0e-300))
    qtot0 = float(np.asarray(gas["ln_ntot"], dtype=np.float64))
    gas_stationarity_source = hvector + float(state.ln_normalized_pressure) - qtot0
    lambda0 = np.linalg.lstsq(
        formula_matrix.T,
        q0 + gas_stationarity_source,
        rcond=None,
    )[0]
    positive_stoich = formula_matrix_cond_active > 0.0
    capacity = np.full_like(formula_matrix_cond_active, np.inf, dtype=np.float64)
    np.divide(
        np.asarray(budget, dtype=np.float64)[:, np.newaxis],
        formula_matrix_cond_active,
        out=capacity,
        where=positive_stoich,
    )
    reference_element_indices = np.argmin(capacity, axis=0)
    reference_element_budget = np.asarray(budget, dtype=np.float64)[
        reference_element_indices
    ]
    epsilon0 = np.log(np.maximum(1.0e-15 * reference_element_budget, 1.0e-300))
    return FixedSupportInputs(
        family=family,
        layer_index=layer_index,
        case_id=case_id_for_profile(definition, temperature, pressure),
        temperature=temperature,
        pressure=pressure,
        support_indices=support_indices,
        support_amounts_init=support_amounts_init,
        formula_matrix=formula_matrix,
        formula_matrix_cond_active=formula_matrix_cond_active,
        element_inventory_target=np.asarray(budget, dtype=np.float64),
        hvector=hvector,
        hvector_cond_active=hvector_cond_active,
        ln_normalized_pressure=float(state.ln_normalized_pressure),
        q0=q0,
        r0=r0,
        qtot0=qtot0,
        lambda0=lambda0,
        rho0=np.zeros_like(r0),
        epsilon0=epsilon0,
        max_iter=int(definition.max_inner_iterations),
    )


def _gk(q: jax.Array, qtot: jax.Array, hvector: jax.Array, ln_pressure: jax.Array) -> jax.Array:
    return hvector + q - qtot + ln_pressure


def _l2(values: jax.Array) -> jax.Array:
    scale = jnp.max(jnp.abs(values), initial=jnp.asarray(0.0, dtype=values.dtype))
    return jnp.where(scale == 0.0, 0.0, scale * jnp.linalg.norm(values / scale))


def _jax_reduced_step(
    q: jax.Array,
    r: jax.Array,
    lam: jax.Array,
    rho: jax.Array,
    qtot: jax.Array,
    ag: jax.Array,
    ac: jax.Array,
    target: jax.Array,
    hvector: jax.Array,
    hcond: jax.Array,
    ln_pressure: jax.Array,
    epsilon: jax.Array,
    r_cap: jax.Array,
    alpha_grid: jax.Array,
) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array, jax.Array, jax.Array, jax.Array]:
    gas_stationarity_source = _gk(q, qtot, hvector, ln_pressure) - q
    log_activity_proxy = ac.T @ lam - hcond
    jac_mask = log_activity_proxy > -0.1
    jac_mask = jnp.where(jnp.any(jac_mask), jac_mask, jnp.arange(r.shape[0]) == jnp.argmax(log_activity_proxy))

    n = jnp.exp(q)
    m = jnp.exp(r)
    eta = jnp.exp(rho)
    j_vec = m / jnp.maximum(eta, 1.0e-300)
    t_vec = r + rho - epsilon
    geff = q + gas_stationarity_source
    gas_inventory = ag @ n
    delta_bhat = target - gas_inventory - ac @ m
    delta_ntot = jnp.sum(n) - jnp.exp(qtot)
    qhat = ag @ (n[:, None] * ag.T) + ac @ (j_vec[:, None] * ac.T)
    qhat = qhat + 1.0e-14 * jnp.eye(qhat.shape[0], dtype=qhat.dtype)
    rhs_top = ag @ (n * geff) + ac @ (j_vec * hcond + m * t_vec - m) + delta_bhat
    rhs_bottom = jnp.dot(n, geff) - delta_ntot
    matrix = jnp.block(
        [
            [qhat, gas_inventory[:, None]],
            [gas_inventory[None, :], jnp.asarray([[delta_ntot]], dtype=qhat.dtype)],
        ]
    )
    rhs = jnp.concatenate([rhs_top, jnp.asarray([rhs_bottom], dtype=qhat.dtype)])
    solution = jnp.linalg.lstsq(matrix, rhs, rcond=None)[0]
    solution = jnp.nan_to_num(solution, nan=0.0, posinf=0.0, neginf=0.0)
    pi = solution[:-1]
    delta_qtot = solution[-1]
    raw_delta_q = ag.T @ pi + delta_qtot - geff
    raw_delta_rho = (hcond - ac.T @ pi) / jnp.maximum(eta, 1.0e-300) - 1.0
    raw_delta_r = -raw_delta_rho - t_vec
    delta_q = jnp.clip(raw_delta_q, -2.0, 2.0)
    delta_r = jnp.clip(raw_delta_r, -5.0, 5.0)
    delta_rho = jnp.clip(raw_delta_rho, -5.0, 5.0)
    delta_lam = jnp.clip(pi - lam, -100.0, 100.0)
    alpha_values = alpha_grid

    def residual_norm(qi: jax.Array, ri: jax.Array, lami: jax.Array, rhoi: jax.Array, qtoti: jax.Array) -> jax.Array:
        ni = jnp.exp(qi)
        mi = jnp.exp(ri)
        etai = jnp.exp(rhoi)
        gas = qi + gas_stationarity_source + qtot - qtoti - ag.T @ lami
        cond = hcond - ac.T @ lami - etai
        cond_masked = jnp.where(jac_mask, cond, 0.0)
        budget = ag @ ni + ac @ mi - target
        comp = ri + rhoi - epsilon
        total_density = jnp.asarray([jnp.sum(ni) - jnp.exp(qtoti)], dtype=qi.dtype)
        return _l2(jnp.concatenate([gas, cond_masked, budget, comp, total_density]))

    initial_norm = residual_norm(q, r, lam, rho, qtot)

    def trial(alpha: jax.Array) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array, jax.Array, jax.Array]:
        tq = q + alpha * delta_q
        tr = r + alpha * delta_r
        tlam = lam + alpha * delta_lam
        trho = rho + alpha * delta_rho
        tqtot = qtot + alpha * delta_qtot
        tr = jnp.minimum(tr, r_cap)
        return tq, tr, tlam, trho, tqtot, residual_norm(tq, tr, tlam, trho, tqtot)

    tq, tr, tlam, trho, tqtot, norms = jax.vmap(trial)(alpha_values)
    finite = jnp.isfinite(norms)
    accepted_mask = finite & (norms < initial_norm)
    any_accepted = jnp.any(accepted_mask)
    first_index = jnp.argmax(accepted_mask)
    best_index = jnp.argmin(jnp.where(finite, norms, jnp.inf))
    selected = jnp.where(any_accepted, first_index, best_index)
    accepted = any_accepted
    next_q = jnp.where(accepted, tq[selected], q)
    next_r = jnp.where(accepted, tr[selected], r)
    next_lam = jnp.where(accepted, tlam[selected], lam)
    next_rho = jnp.where(accepted, trho[selected], rho)
    next_qtot = jnp.where(accepted, tqtot[selected], qtot)
    next_norm = jnp.where(accepted, norms[selected], initial_norm)
    return (
        next_q,
        next_r,
        next_lam,
        next_rho,
        next_qtot,
        next_norm,
        accepted,
    )


def _make_scan_fn(iterations: int):
    alpha_grid = jnp.asarray(
        (1.0, 0.5, 0.25, 0.125, 0.0625, 0.03125, 0.015625, 0.01, 0.003, 0.001, 0.0003, 0.0001, 1.0e-5),
        dtype=jnp.float64,
    )

    @jax.jit
    def run(
        q0: jax.Array,
        r0: jax.Array,
        lam0: jax.Array,
        rho0: jax.Array,
        qtot0: jax.Array,
        r_cap: jax.Array,
        ag: jax.Array,
        ac: jax.Array,
        target: jax.Array,
        hvector: jax.Array,
        hcond: jax.Array,
        ln_pressure: jax.Array,
        epsilon: jax.Array,
    ) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array, jax.Array, jax.Array, jax.Array]:
        def body(carry, _):
            q, r, lam, rho, qtot, residual, still_running = carry
            stepped = _jax_reduced_step(
                q,
                r,
                lam,
                rho,
                qtot,
                ag,
                ac,
                target,
                hvector,
                hcond,
                ln_pressure,
                epsilon,
                r_cap,
                alpha_grid,
            )
            next_q, next_r, next_lam, next_rho, next_qtot, next_residual, accepted = stepped
            apply_step = still_running & accepted
            next_carry = (
                jnp.where(apply_step, next_q, q),
                jnp.where(apply_step, next_r, r),
                jnp.where(apply_step, next_lam, lam),
                jnp.where(apply_step, next_rho, rho),
                jnp.where(apply_step, next_qtot, qtot),
                jnp.where(still_running, next_residual, residual),
                apply_step,
            )
            return next_carry, next_carry[-2:]

        initial = (
            q0,
            r0,
            lam0,
            rho0,
            qtot0,
            jnp.asarray(jnp.inf, dtype=jnp.float64),
            jnp.asarray(True),
        )
        final, history = jax.lax.scan(body, initial, xs=None, length=iterations)
        return (*final, history[0], history[1])

    return run


def _time_call(fn: Any, args: tuple[Any, ...], *, warmup: int, repeat: int) -> dict[str, Any]:
    t0 = time.perf_counter()
    first = fn(*args)
    _block_tree(first)
    first_elapsed = time.perf_counter() - t0
    for _ in range(warmup):
        _block_tree(fn(*args))
    times = []
    last = first
    for _ in range(repeat):
        start = time.perf_counter()
        last = fn(*args)
        _block_tree(last)
        times.append(time.perf_counter() - start)
    return {
        "first_call_seconds": first_elapsed,
        "warmup": warmup,
        "repeat": repeat,
        "warm_call_seconds": times,
        "warm_median_seconds": statistics.median(times),
        "warm_min_seconds": min(times),
        "warm_max_seconds": max(times),
        "last_result": last,
    }


def _benchmark_target(
    *,
    setup: Any,
    target: str,
    support_mode: str,
    iterations: int,
    warmup: int,
    repeat: int,
    include_python_restricted: bool,
) -> dict[str, Any]:
    family, layer = _parse_target(target)
    inputs = _build_fixed_support_inputs(
        setup=setup,
        family=family,
        layer_index=layer,
        support_mode=support_mode,
    )
    scan_fn = _make_scan_fn(iterations)
    jax_args = (
        jnp.asarray(inputs.q0, dtype=jnp.float64),
        jnp.asarray(inputs.r0, dtype=jnp.float64),
        jnp.asarray(inputs.lambda0, dtype=jnp.float64),
        jnp.asarray(inputs.rho0, dtype=jnp.float64),
        jnp.asarray(inputs.qtot0, dtype=jnp.float64),
        jnp.asarray(inputs.r0, dtype=jnp.float64),
        jnp.asarray(inputs.formula_matrix, dtype=jnp.float64),
        jnp.asarray(inputs.formula_matrix_cond_active, dtype=jnp.float64),
        jnp.asarray(inputs.element_inventory_target, dtype=jnp.float64),
        jnp.asarray(inputs.hvector, dtype=jnp.float64),
        jnp.asarray(inputs.hvector_cond_active, dtype=jnp.float64),
        jnp.asarray(inputs.ln_normalized_pressure, dtype=jnp.float64),
        jnp.asarray(inputs.epsilon0, dtype=jnp.float64),
    )
    jax_timing = _time_call(scan_fn, jax_args, warmup=warmup, repeat=repeat)
    final = jax_timing.pop("last_result")
    final_norm = float(np.asarray(jax.device_get(final[5])))
    accepted_count = int(np.sum(np.asarray(jax.device_get(final[-1]), dtype=bool)))
    result = {
        "target": target,
        "family": inputs.family,
        "layer_index": inputs.layer_index,
        "case_id": inputs.case_id,
        "temperature": inputs.temperature,
        "pressure": inputs.pressure,
        "support_mode": support_mode,
        "support_count": int(inputs.support_indices.size),
        "gas_species_count": int(inputs.formula_matrix.shape[1]),
        "element_count": int(inputs.formula_matrix.shape[0]),
        "iterations": int(iterations),
        "jax_fixed_core": {
            **jax_timing,
            "final_residual_norm": final_norm,
            "accepted_iteration_count": accepted_count,
        },
    }
    if include_python_restricted:
        state = ThermoState(
            temperature=inputs.temperature,
            ln_normalized_pressure=inputs.ln_normalized_pressure,
            element_vector=jnp.asarray(inputs.element_inventory_target),
        )
        restricted_args = (
            state,
            setup.formula_matrix,
            setup.formula_matrix_cond,
            setup.gas_setup.hvector_func,
            setup.condensate_setup.hvector_func,
        )
        restricted_kwargs = {
            "support_indices": inputs.support_indices,
            "condensate_species": setup.condensate_species,
            "element_names": setup.elements,
            "support_amounts_init": jnp.asarray(inputs.support_amounts_init),
            "reduced_coupling_config": CondensateRGIEReducedCouplingConfig(
                reduced_coupling_mode="pdipm_rgie_v11_activity_correction",
                alpha_s=1.0,
            ),
            "max_iter": inputs.max_iter,
        }

        def run_restricted(*args: Any) -> Any:
            return solve_restricted_support_condensate_layer(
                *args,
                **restricted_kwargs,
            )

        result["python_restricted_solver"] = _time_call(
            run_restricted,
            restricted_args,
            warmup=0,
            repeat=max(1, min(repeat, 3)),
        )
        py_last = result["python_restricted_solver"].pop("last_result")
        result["python_restricted_solver"]["solver_success"] = bool(
            py_last.get("solver_success")
        )
        result["python_restricted_solver"]["n_iter"] = _json_safe(
            py_last.get("diagnostics", {}).get("n_iter")
        )
    return _json_safe(result)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--jax-platform",
        choices=("cpu", "cuda", "gpu", "default"),
        default=REQUESTED_JAX_PLATFORM,
    )
    parser.add_argument("--targets", nargs="*", default=list(DEFAULT_TARGETS))
    parser.add_argument("--iterations", type=int, default=8)
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--repeat", type=int, default=20)
    parser.add_argument(
        "--support-mode",
        choices=("explicit_payload",),
        default="explicit_payload",
    )
    parser.add_argument("--include-python-restricted", action="store_true")
    parser.add_argument("--print-jax-devices", action="store_true")
    parser.add_argument(
        "--output",
        default="volatiles_artifacts/pdipm_fixed_support_core_benchmark.json",
    )
    args = parser.parse_args()
    if args.iterations < 1:
        raise ValueError("--iterations must be at least 1")
    if args.repeat < 1:
        raise ValueError("--repeat must be at least 1")
    if args.warmup < 0:
        raise ValueError("--warmup must be non-negative")
    runtime = _jax_runtime_report(str(args.jax_platform))
    if args.print_jax_devices:
        print(json.dumps(runtime, sort_keys=True), flush=True)
    setup = condensate_chemical_setup(silent=True)
    rows = []
    for target in args.targets:
        row = _benchmark_target(
            setup=setup,
            target=str(target),
            support_mode=str(args.support_mode),
            iterations=int(args.iterations),
            warmup=int(args.warmup),
            repeat=int(args.repeat),
            include_python_restricted=bool(args.include_python_restricted),
        )
        rows.append(row)
        print(
            json.dumps(
                {
                    "target": target,
                    "support_count": row["support_count"],
                    "jax_warm_median_seconds": row["jax_fixed_core"][
                        "warm_median_seconds"
                    ],
                    "jax_first_call_seconds": row["jax_fixed_core"][
                        "first_call_seconds"
                    ],
                    "final_residual_norm": row["jax_fixed_core"][
                        "final_residual_norm"
                    ],
                },
                sort_keys=True,
            ),
            flush=True,
        )
    payload = {
        "schema": "exogibbs_pdipm_fixed_support_core_benchmark_v1",
        "jax_runtime": runtime,
        "iterations": int(args.iterations),
        "warmup": int(args.warmup),
        "repeat": int(args.repeat),
        "rows": rows,
        "notes": (
            "The JAX fixed core is a scratch reduced-step scan. It is not yet "
            "the production route and intentionally omits route retries, "
            "diagnostics payload construction, BVLS, and support discovery."
        ),
    }
    output = ROOT / str(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(_json_safe(payload), indent=2, sort_keys=True) + "\n")
    print(f"wrote {output}")


if __name__ == "__main__":
    main()
