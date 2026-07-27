import math
import time

import jax
import jax.numpy as jnp
import numpy as np
from jax import custom_vjp
from jax import jacrev
from jax.lax import while_loop, stop_gradient
from jax.scipy.linalg import cho_factor
from jax.scipy.linalg import cho_solve
from functools import partial
from typing import Any, Tuple, Callable, Dict

from exogibbs.equilibrium.gas.types import ThermoState
from exogibbs.equilibrium.gas.kernel.equations import _A_diagn_At
from exogibbs.equilibrium.gas.kernel.equations import _compute_gk
from exogibbs.equilibrium.gas.kernel.autodiff import vjp_elements
from exogibbs.equilibrium.gas.kernel.autodiff import vjp_pressure
from exogibbs.equilibrium.gas.kernel.autodiff import vjp_temperature

_CHO_EPS = 1.0e-18


def build_minimize_gibbs_core_lnnk_output_source_trace(
    ln_nk_output: Any,
    ln_ntot_output: Any,
    n_iter: Any,
    final_residual: Any,
    *,
    case_key: str = "diagnostic",
    newton_iter: int = 0,
) -> dict[str, Any]:
    """Describe the gas-only Gibbs core ln_nk output without changing solver inputs."""

    raw = np.asarray(jax.device_get(ln_nk_output))
    finite = np.isfinite(raw.astype(np.float64, copy=False))
    double_min_log = math.log(float.fromhex("0x1p-1022"))
    return {
        "diagnostic_only": True,
        "default_off": True,
        "constructor_input": False,
        "reference_trace_input": False,
        "FastChem_trace_values_used_as_inputs": False,
        "FastChem_trace_values_used_as_KL_constructor_inputs": False,
        "used_as_KL_constructor_input": False,
        "available": True,
        "case_key": str(case_key),
        "newton_iter": int(newton_iter),
        "source_stage": "minimize_gibbs_core ln_nk output source",
        "producer_function": "src/exogibbs/optimize/minimize.py::minimize_gibbs_core",
        "raw_input_type": type(ln_nk_output).__name__,
        "raw_input_dtype": str(raw.dtype),
        "shape": [int(dim) for dim in raw.shape],
        "finite_count": int(finite.sum()),
        "below_double_normal_log_count": int(np.sum(raw < double_min_log)),
        "native_longdouble_provenance_available": False,
        "preserves_native_longdouble_bits": False,
        "reconstructed_from_float64": True,
        "ln_ntot_output": float(np.asarray(jax.device_get(ln_ntot_output))),
        "n_iter": int(np.asarray(jax.device_get(n_iter))),
        "final_residual": float(np.asarray(jax.device_get(final_residual))),
        "floor_policy": "gas-only PIPM minimizer core output; no native long-double floor policy",
        "next_required_field": (
            "minimize_gibbs_core while_loop final carry before JAX float64 "
            "ln_nk output materialization"
        ),
    }


def build_minimize_gibbs_core_final_carry_source_trace(
    ln_nk_output: Any,
    ln_ntot_output: Any,
    n_iter: Any,
    final_residual: Any,
    *,
    case_key: str = "diagnostic",
    newton_iter: int = 0,
) -> dict[str, Any]:
    """Describe the final carry boundary for diagnostic ln_nk provenance."""

    trace = build_minimize_gibbs_core_lnnk_output_source_trace(
        ln_nk_output,
        ln_ntot_output,
        n_iter,
        final_residual,
        case_key=case_key,
        newton_iter=newton_iter,
    )
    trace.update(
        {
            "source_stage": "minimize_gibbs_core while_loop final carry ln_nk source",
            "producer_function": (
                "src/exogibbs/optimize/minimize.py::"
                "minimize_gibbs_core_with_source_trace"
            ),
            "trace_boundary": "lax.while_loop final carry",
            "final_carry_tuple_index": 0,
            "final_carry_tuple_field": "ln_nk",
            "final_carry_tuple_size": 13,
            "observed_after_while_loop_return": True,
            "next_required_field": (
                "minimize_gibbs_core body/update_all ln_nk_new source before "
                "JAX float64 carry materialization"
            ),
        }
    )
    return trace


def _sample_array(value: Any, limit: int = 5) -> list[float]:
    raw = np.ravel(np.asarray(jax.device_get(value), dtype=np.float64))
    return [float(item) for item in raw[:limit]]


def compare_solve_iteration_system_longdouble(
    bmatrix: Any,
    rhs: Any,
    An: Any,
    resn: Any,
    *,
    reference_binv_rhs: Any,
    reference_binv_an: Any,
    reference_schur_safe: Any,
) -> dict[str, Any]:
    """Attempt a host long-double replay of the iteration linear solve."""

    result: dict[str, Any] = {
        "diagnostic_only": True,
        "default_off": True,
        "constructor_input": False,
        "reference_trace_input": False,
        "FastChem_trace_values_used_as_KL_constructor_inputs": False,
        "source_stage": "long-double linear solve comparator",
        "producer_function": (
            "src/exogibbs/optimize/minimize.py::"
            "compare_solve_iteration_system_longdouble"
        ),
        "attempted": True,
        "available": False,
        "solver": "numpy.linalg.solve on np.longdouble host arrays",
        "input_dtype": "np.longdouble",
    }
    def _compare_solution(
        binv_rhs_cmp: Any,
        binv_an_cmp: Any,
        schur_safe_cmp: Any,
        solver: str,
    ) -> None:
        ref_rhs = np.asarray(jax.device_get(reference_binv_rhs), dtype=np.float64)
        ref_an = np.asarray(jax.device_get(reference_binv_an), dtype=np.float64)
        ref_schur = float(np.asarray(jax.device_get(reference_schur_safe)))
        binv_rhs_arr = np.asarray(binv_rhs_cmp, dtype=np.float64)
        binv_an_arr = np.asarray(binv_an_cmp, dtype=np.float64)
        schur_safe_float = float(schur_safe_cmp)
        result.update(
            {
                "available": True,
                "solver": solver,
                "binv_rhs_max_abs_delta_vs_jax": float(
                    np.max(np.abs(binv_rhs_arr - ref_rhs))
                ),
                "binv_an_max_abs_delta_vs_jax": float(
                    np.max(np.abs(binv_an_arr - ref_an))
                ),
                "schur_safe_abs_delta_vs_jax": float(
                    abs(schur_safe_float - ref_schur)
                ),
                "binv_rhs_sample": [float(v) for v in binv_rhs_arr[:5]],
                "binv_an_sample": [float(v) for v in binv_an_arr[:5]],
                "schur_safe": schur_safe_float,
            }
        )

    def _solve_with_numpy_longdouble() -> None:
        bmatrix_ld = np.asarray(jax.device_get(bmatrix), dtype=np.longdouble)
        rhs_ld = np.asarray(jax.device_get(rhs), dtype=np.longdouble)
        an_ld = np.asarray(jax.device_get(An), dtype=np.longdouble)
        resn_ld = np.asarray(jax.device_get(resn), dtype=np.longdouble)
        jitter = np.asarray(_CHO_EPS, dtype=np.longdouble)
        eye = np.eye(bmatrix_ld.shape[0], dtype=np.longdouble)
        rhs_pair = np.stack((rhs_ld, an_ld), axis=1)
        solved_pair = np.linalg.solve(bmatrix_ld + jitter * eye, rhs_pair)
        binv_rhs_ld = solved_pair[:, 0]
        binv_an_ld = solved_pair[:, 1]
        schur_ld = resn_ld - np.vdot(an_ld, binv_an_ld)
        schur_safe_ld = np.where(
            np.abs(schur_ld) < jitter,
            np.where(schur_ld < 0.0, -jitter, jitter),
            schur_ld,
        )
        _compare_solution(
            binv_rhs_ld,
            binv_an_ld,
            schur_safe_ld,
            "numpy.linalg.solve on np.longdouble host arrays",
        )

    def _solve_with_scipy_longdouble() -> None:
        import scipy.linalg

        bmatrix_ld = np.asarray(jax.device_get(bmatrix), dtype=np.longdouble)
        rhs_ld = np.asarray(jax.device_get(rhs), dtype=np.longdouble)
        an_ld = np.asarray(jax.device_get(An), dtype=np.longdouble)
        resn_ld = np.asarray(jax.device_get(resn), dtype=np.longdouble)
        jitter = np.asarray(_CHO_EPS, dtype=np.longdouble)
        eye = np.eye(bmatrix_ld.shape[0], dtype=np.longdouble)
        rhs_pair = np.stack((rhs_ld, an_ld), axis=1)
        solved_pair = scipy.linalg.solve(
            bmatrix_ld + jitter * eye,
            rhs_pair,
            assume_a="gen",
            check_finite=False,
        )
        binv_rhs_ld = solved_pair[:, 0]
        binv_an_ld = solved_pair[:, 1]
        schur_ld = resn_ld - np.vdot(an_ld, binv_an_ld)
        schur_safe_ld = np.where(
            np.abs(schur_ld) < jitter,
            np.where(schur_ld < 0.0, -jitter, jitter),
            schur_ld,
        )
        _compare_solution(
            binv_rhs_ld,
            binv_an_ld,
            schur_safe_ld,
            "scipy.linalg.solve on np.longdouble host arrays",
        )

    def _solve_with_mpmath() -> None:
        import mpmath as mp

        mp.mp.dps = 80
        bmatrix_f64 = np.asarray(jax.device_get(bmatrix), dtype=np.float64)
        rhs_f64 = np.asarray(jax.device_get(rhs), dtype=np.float64)
        an_f64 = np.asarray(jax.device_get(An), dtype=np.float64)
        resn_mp = mp.mpf(str(float(np.asarray(jax.device_get(resn)))))
        jitter_mp = mp.mpf(str(_CHO_EPS))
        matrix = mp.matrix(
            [
                [
                    mp.mpf(str(float(value + (_CHO_EPS if i == j else 0.0))))
                    for j, value in enumerate(row)
                ]
                for i, row in enumerate(bmatrix_f64)
            ]
        )
        rhs_pair = mp.matrix(
            [
                [mp.mpf(str(float(rhs_f64[i]))), mp.mpf(str(float(an_f64[i])))]
                for i in range(bmatrix_f64.shape[0])
            ]
        )
        solved_pair = mp.lu_solve(matrix, rhs_pair)
        binv_rhs_mp = [solved_pair[i, 0] for i in range(bmatrix_f64.shape[0])]
        binv_an_mp = [solved_pair[i, 1] for i in range(bmatrix_f64.shape[0])]
        schur_mp = resn_mp - mp.fsum(
            mp.mpf(str(float(an_f64[i]))) * binv_an_mp[i]
            for i in range(bmatrix_f64.shape[0])
        )
        if abs(schur_mp) < jitter_mp:
            schur_safe_mp = -jitter_mp if schur_mp < 0 else jitter_mp
        else:
            schur_safe_mp = schur_mp
        _compare_solution(
            [float(value) for value in binv_rhs_mp],
            [float(value) for value in binv_an_mp],
            float(schur_safe_mp),
            "mpmath.lu_solve high-precision comparator from float64 materialized inputs",
        )
        result["mpmath_decimal_precision"] = int(mp.mp.dps)
        result["input_dtype"] = "float64 materialized values promoted to mpmath"

    failures: list[dict[str, str]] = []
    for solver in (
        _solve_with_numpy_longdouble,
        _solve_with_scipy_longdouble,
        _solve_with_mpmath,
    ):
        try:
            solver()
            break
        except Exception as exc:  # pragma: no cover - backend availability varies
            failures.append(
                {
                    "solver": solver.__name__,
                    "failure_type": type(exc).__name__,
                    "failure_message": str(exc),
                }
            )
    if not result["available"]:
        result.update(
            {
                "available": False,
                "failures": failures,
                "next_required_field": (
                    "platform-supported long-double linear algebra backend or "
                    "external comparator"
                ),
            }
        )
    return result


def build_hvector_provider_source_trace(
    hvector_func: Any,
    hvector: Any,
    temperature: Any,
    *,
    case_key: str = "diagnostic",
    newton_iter: int = 0,
) -> dict[str, Any]:
    """Describe the concrete hvector provider boundary observed by diagnostics."""

    raw = np.asarray(jax.device_get(hvector))
    return {
        "diagnostic_only": True,
        "default_off": True,
        "constructor_input": False,
        "reference_trace_input": False,
        "FastChem_trace_values_used_as_KL_constructor_inputs": False,
        "case_key": str(case_key),
        "newton_iter": int(newton_iter),
        "source_stage": "concrete hvector_func thermochemical provider boundary",
        "producer_function": "hvector_func",
        "hvector_func_type": type(hvector_func).__name__,
        "hvector_func_name": getattr(hvector_func, "__name__", None),
        "hvector_func_module": getattr(hvector_func, "__module__", None),
        "temperature": float(np.asarray(jax.device_get(temperature))),
        "hvector_output_dtype": str(raw.dtype),
        "hvector_shape": [int(dim) for dim in raw.shape],
        "hvector_sample": [float(v) for v in np.ravel(raw.astype(np.float64))[:5]],
        "provider_metadata_available": False,
        "native_longdouble_provenance_available": False,
        "preserves_native_longdouble_bits": False,
        "next_required_field": (
            "preset/provider-specific hvector construction trace before JAX jit "
            "and float64 materialization"
        ),
    }


def build_keyed_final_iteration_provider_linear_trace(
    hvector_func: Any,
    components: dict[str, Any],
    *,
    case_key: str = "diagnostic",
    newton_iter: int = 0,
) -> dict[str, Any]:
    """Persist provider source and final linear-system inputs under one join key."""

    temperature = components["temperature"]
    provider_trace_func = getattr(hvector_func, "fastchem_hvector_logk_source_trace", None)
    provider_trace: dict[str, Any] | None = None
    provider_trace_error: str | None = None
    if callable(provider_trace_func):
        try:
            provider_trace = provider_trace_func(temperature, limit=16)
        except Exception as exc:  # pragma: no cover - provider-specific diagnostics
            provider_trace_error = f"{type(exc).__name__}: {exc}"

    bmatrix = np.asarray(jax.device_get(components["bmatrix"]), dtype=np.float64)
    rhs = np.asarray(jax.device_get(components["rhs"]), dtype=np.float64)
    an = np.asarray(jax.device_get(components["An"]), dtype=np.float64)
    hvector = np.asarray(jax.device_get(components["hvector"]), dtype=np.float64)
    gk = np.asarray(jax.device_get(components["gk_before_update"]), dtype=np.float64)
    trace = {
        "diagnostic_only": True,
        "default_off": True,
        "constructor_input": False,
        "reference_trace_input": False,
        "FastChem_trace_values_used_as_KL_constructor_inputs": False,
        "case_key": str(case_key),
        "newton_iter": int(newton_iter),
        "iteration": int(components["iteration"]),
        "join_key": {
            "case_key": str(case_key),
            "newton_iter": int(newton_iter),
            "iteration": int(components["iteration"]),
            "source_stage": "final gas-only minimizer iteration",
        },
        "source_stage": "keyed final iteration provider source plus linear inputs",
        "producer_function": (
            "src/exogibbs/optimize/minimize.py::"
            "build_keyed_final_iteration_provider_linear_trace"
        ),
        "provider_function_identity": getattr(hvector_func, "__name__", None),
        "provider_function_type": type(hvector_func).__name__,
        "provider_trace_available": provider_trace is not None,
        "provider_trace_error": provider_trace_error,
        "provider_source_trace": provider_trace,
        "temperature": float(np.asarray(jax.device_get(temperature))),
        "temperature_dtype": str(np.asarray(jax.device_get(temperature)).dtype),
        "linear_system_inputs": {
            "bmatrix_shape": [int(dim) for dim in bmatrix.shape],
            "bmatrix_dtype": str(bmatrix.dtype),
            "bmatrix": bmatrix.tolist(),
            "rhs_shape": [int(dim) for dim in rhs.shape],
            "rhs_dtype": str(rhs.dtype),
            "rhs": rhs.tolist(),
            "An_shape": [int(dim) for dim in an.shape],
            "An_dtype": str(an.dtype),
            "An": an.tolist(),
            "resn": float(np.asarray(jax.device_get(components["resn"]))),
            "scalar_rhs": float(np.asarray(jax.device_get(components["scalar_rhs"]))),
            "schur_safe": float(np.asarray(jax.device_get(components["schur_safe"]))),
            "ln_ntot": float(np.asarray(jax.device_get(components["ln_ntot_before"]))),
            "ln_normalized_pressure": float(
                np.asarray(jax.device_get(components["ln_normalized_pressure"]))
            ),
        },
        "thermochemical_inputs": {
            "hvector_shape": [int(dim) for dim in hvector.shape],
            "hvector_dtype": str(hvector.dtype),
            "hvector": hvector.tolist(),
            "gk_shape": [int(dim) for dim in gk.shape],
            "gk_dtype": str(gk.dtype),
            "gk": gk.tolist(),
            "ln_nk_sample": _sample_array(components["ln_nk_before"], limit=12),
        },
        "join_readiness": {
            "provider_source_keyed_with_linear_inputs": provider_trace is not None,
            "final_bmatrix_rhs_An_resn_persisted": True,
            "A1_donor_rows_still_require_species_donor_mapping": True,
        },
        "next_required_field": (
            "species/donor mapping from v54 A1 donor rows to final iteration "
            "provider source and linear-system rows"
        ),
    }
    return trace


def trace_minimize_gibbs_core_update_all_lnnk_new_source_components(
    state: ThermoState,
    ln_nk_init: jnp.ndarray,
    ln_ntot_init: float,
    formula_matrix: jnp.ndarray,
    hvector_func,
    epsilon_crit: float = 1.0e-11,
    max_iter: int = 1000,
    *,
    case_key: str = "diagnostic",
    newton_iter: int = 0,
) -> dict[str, Any]:
    """Replay the core loop in Python and trace final ln_nk_new source terms."""

    hvector = hvector_func(state.temperature)
    gk = _compute_gk(
        state.temperature,
        ln_nk_init,
        ln_ntot_init,
        hvector,
        state.ln_normalized_pressure,
    )
    An = formula_matrix @ jnp.exp(ln_nk_init)
    ln_nk = ln_nk_init
    ln_ntot = ln_ntot_init
    epsilon = jnp.asarray(jnp.inf, dtype=jnp.asarray(ln_nk).dtype)
    counter = 0
    last_components: dict[str, Any] | None = None
    while (float(jax.device_get(epsilon)) > float(epsilon_crit)) and counter < max_iter:
        _, _, resn, bmatrix, rhs, scalar_rhs = _prepare_iteration_system(
            ln_nk, ln_ntot, formula_matrix, state.element_vector, gk, An
        )
        binv_rhs, binv_an, schur_safe = _solve_iteration_system(bmatrix, rhs, An, resn)
        pi_vector, delta_ln_ntot = _finish_iteration_solve(
            binv_rhs, binv_an, An, scalar_rhs, schur_safe
        )
        at_pi = formula_matrix.T @ pi_vector
        delta_ln_nk = at_pi + delta_ln_ntot - gk
        lam = _cea_lambda(delta_ln_nk, delta_ln_ntot, ln_nk, ln_ntot)
        ln_nk_new = ln_nk + lam * delta_ln_nk
        ln_ntot_new = ln_ntot + lam * delta_ln_ntot
        gk_before_update = gk
        gk, An, epsilon = _evaluate_iteration_state(
            ln_nk_new,
            ln_ntot_new,
            formula_matrix,
            state.element_vector,
            state.temperature,
            state.ln_normalized_pressure,
            hvector,
            gk,
            lam,
            at_pi,
            pi_vector,
        )
        last_components = {
            "iteration": counter,
            "ln_nk_before": ln_nk,
            "delta_ln_nk": delta_ln_nk,
            "lambda": lam,
            "lambda_times_delta_ln_nk": lam * delta_ln_nk,
            "ln_nk_new": ln_nk_new,
            "delta_ln_ntot": delta_ln_ntot,
            "pi_vector": pi_vector,
            "at_pi": at_pi,
            "gk_before_update": gk_before_update,
            "binv_rhs": binv_rhs,
            "binv_an": binv_an,
            "schur_safe": schur_safe,
            "scalar_rhs": scalar_rhs,
            "rhs": rhs,
            "bmatrix": bmatrix,
            "An": An,
            "resn": resn,
            "hvector": hvector,
            "ln_ntot_before": ln_ntot,
            "ln_normalized_pressure": state.ln_normalized_pressure,
            "temperature": state.temperature,
        }
        ln_nk = ln_nk_new
        ln_ntot = ln_ntot_new
        counter += 1

    trace = build_minimize_gibbs_core_final_carry_source_trace(
        ln_nk,
        ln_ntot,
        counter,
        epsilon,
        case_key=case_key,
        newton_iter=newton_iter,
    )
    trace.update(
        {
            "source_stage": "update_all/_apply_iteration_step ln_nk_new source",
            "producer_function": (
                "src/exogibbs/optimize/minimize.py::"
                "trace_minimize_gibbs_core_update_all_lnnk_new_source_components"
            ),
            "trace_boundary": "diagnostic Python replay of final update_all step",
            "update_all_source_components_available": last_components is not None,
            "observed_after_while_loop_return": False,
            "next_required_field": (
                "_finish_iteration_solve pi_vector/delta_ln_ntot source and "
                "_compute_gk gk source before JAX float64 step materialization"
            ),
        }
    )
    if last_components is not None:
        trace["update_all_source_components"] = {
            "iteration": int(last_components["iteration"]),
            "formula": "ln_nk_new = ln_nk + lambda * delta_ln_nk",
            "ln_nk_before_sample": _sample_array(last_components["ln_nk_before"]),
            "delta_ln_nk_sample": _sample_array(last_components["delta_ln_nk"]),
            "lambda": float(np.asarray(jax.device_get(last_components["lambda"]))),
            "lambda_times_delta_ln_nk_sample": _sample_array(
                last_components["lambda_times_delta_ln_nk"]
            ),
            "ln_nk_new_sample": _sample_array(last_components["ln_nk_new"]),
            "delta_ln_ntot": float(
                np.asarray(jax.device_get(last_components["delta_ln_ntot"]))
            ),
            "at_pi_sample": _sample_array(last_components["at_pi"]),
            "gk_before_update_sample": _sample_array(
                last_components["gk_before_update"]
            ),
            "pi_vector_sample": _sample_array(last_components["pi_vector"]),
            "binv_rhs_sample": _sample_array(last_components["binv_rhs"]),
            "binv_an_sample": _sample_array(last_components["binv_an"]),
            "rhs_sample": _sample_array(last_components["rhs"]),
            "schur_safe": float(np.asarray(jax.device_get(last_components["schur_safe"]))),
            "scalar_rhs": float(np.asarray(jax.device_get(last_components["scalar_rhs"]))),
            "source_component_dtype": str(
                np.asarray(jax.device_get(last_components["ln_nk_new"])).dtype
            ),
            "delta_ln_nk_formula": "delta_ln_nk = at_pi + delta_ln_ntot - gk",
            "finish_iteration_solve_formula": (
                "pi_vector = binv_rhs - binv_an * delta_ln_ntot"
            ),
        }
        trace["delta_ln_nk_source_components_trace"] = {
            "diagnostic_only": True,
            "default_off": True,
            "constructor_input": False,
            "reference_trace_input": False,
            "FastChem_trace_values_used_as_KL_constructor_inputs": False,
            "source_stage": "_finish_iteration_solve/_compute_gk delta_ln_nk source",
            "producer_function": (
                "src/exogibbs/optimize/minimize.py::"
                "trace_minimize_gibbs_core_update_all_lnnk_new_source_components"
            ),
            "source_formula": "delta_ln_nk = at_pi + delta_ln_ntot - gk",
            "at_pi_source_formula": "at_pi = formula_matrix.T @ pi_vector",
            "pi_vector_source_formula": (
                "pi_vector = binv_rhs - binv_an * delta_ln_ntot"
            ),
            "delta_ln_ntot_source_formula": (
                "delta_ln_ntot = (scalar_rhs - vdot(An, binv_rhs)) / schur_safe"
            ),
            "gk_source_formula": "_compute_gk(T, ln_nk, ln_ntot, hvector, ln_normalized_pressure)",
            "component_fields": [
                "pi_vector_sample",
                "delta_ln_ntot",
                "at_pi_sample",
                "gk_before_update_sample",
                "binv_rhs_sample",
                "binv_an_sample",
                "schur_safe",
                "scalar_rhs",
            ],
            "native_longdouble_provenance_available": False,
            "preserves_native_longdouble_bits": False,
            "next_required_field": (
                "_solve_iteration_system binv_rhs/binv_an/schur_safe and "
                "_compute_gk thermochemical source before JAX float64 materialization"
            ),
        }
        trace["linear_solve_and_gk_source_trace"] = {
            "diagnostic_only": True,
            "default_off": True,
            "constructor_input": False,
            "reference_trace_input": False,
            "FastChem_trace_values_used_as_KL_constructor_inputs": False,
            "source_stage": "_solve_iteration_system/_compute_gk source inputs",
            "producer_function": (
                "src/exogibbs/optimize/minimize.py::"
                "trace_minimize_gibbs_core_update_all_lnnk_new_source_components"
            ),
            "solve_iteration_system_formula": (
                "binv_rhs, binv_an = cho_solve(cho_factor(bmatrix + jitter * eye), "
                "stack(rhs, An)); schur_safe = safe(resn - vdot(An, binv_an))"
            ),
            "bmatrix_source_formula": "bmatrix = _A_diagn_At(exp(ln_nk), formula_matrix)",
            "rhs_source_formula": "rhs = formula_matrix @ (gk * exp(ln_nk)) + b - An",
            "An_source_formula": "An = formula_matrix @ exp(ln_nk)",
            "gk_source_formula": (
                "gk = hvector + ln_nk - ln_ntot + ln_normalized_pressure"
            ),
            "bmatrix_sample": _sample_array(last_components["bmatrix"]),
            "rhs_sample": _sample_array(last_components["rhs"]),
            "An_sample": _sample_array(last_components["An"]),
            "resn": float(np.asarray(jax.device_get(last_components["resn"]))),
            "hvector_sample": _sample_array(last_components["hvector"]),
            "ln_nk_sample": _sample_array(last_components["ln_nk_before"]),
            "ln_ntot": float(
                np.asarray(jax.device_get(last_components["ln_ntot_before"]))
            ),
            "ln_normalized_pressure": float(
                np.asarray(jax.device_get(last_components["ln_normalized_pressure"]))
            ),
            "temperature": float(
                np.asarray(jax.device_get(last_components["temperature"]))
            ),
            "source_component_dtype": str(
                np.asarray(jax.device_get(last_components["bmatrix"])).dtype
            ),
            "native_longdouble_provenance_available": False,
            "preserves_native_longdouble_bits": False,
            "next_required_field": (
                "_A_diagn_At/cho_solve numerical precision and hvector_func thermochemical source before JAX float64 materialization"
            ),
        }
        trace["hvector_and_linear_precision_source_trace"] = {
            "diagnostic_only": True,
            "default_off": True,
            "constructor_input": False,
            "reference_trace_input": False,
            "FastChem_trace_values_used_as_KL_constructor_inputs": False,
            "source_stage": "hvector_func thermochemical source and cho_solve precision",
            "producer_function": (
                "src/exogibbs/optimize/minimize.py::"
                "trace_minimize_gibbs_core_update_all_lnnk_new_source_components"
            ),
            "hvector_func_type": type(hvector_func).__name__,
            "hvector_func_name": getattr(hvector_func, "__name__", None),
            "hvector_output_dtype": str(np.asarray(jax.device_get(hvector)).dtype),
            "hvector_sample": _sample_array(hvector),
            "hvector_native_longdouble_provenance_available": False,
            "linear_solver_factorization": "jax.scipy.linalg.cho_factor",
            "linear_solver_solve": "jax.scipy.linalg.cho_solve",
            "linear_solver_jitter": float(_CHO_EPS),
            "linear_solver_matrix_dtype": str(
                np.asarray(jax.device_get(last_components["bmatrix"])).dtype
            ),
            "linear_solver_rhs_dtype": str(
                np.asarray(jax.device_get(last_components["rhs"])).dtype
            ),
            "linear_solver_native_longdouble_provenance_available": False,
            "native_longdouble_provenance_available": False,
            "preserves_native_longdouble_bits": False,
            "next_required_field": (
                "thermochemical provider hvector construction and optional "
                "long-double linear algebra comparator before JAX float64 "
                "materialization"
            ),
        }
        trace["hvector_provider_source_trace"] = build_hvector_provider_source_trace(
            hvector_func,
            hvector,
            last_components["temperature"],
            case_key=case_key,
            newton_iter=newton_iter,
        )
        trace["longdouble_linear_solve_comparator_trace"] = (
            compare_solve_iteration_system_longdouble(
                last_components["bmatrix"],
                last_components["rhs"],
                last_components["An"],
                last_components["resn"],
                reference_binv_rhs=last_components["binv_rhs"],
                reference_binv_an=last_components["binv_an"],
                reference_schur_safe=last_components["schur_safe"],
            )
        )
        trace["keyed_final_iteration_provider_linear_trace"] = (
            build_keyed_final_iteration_provider_linear_trace(
                hvector_func,
                last_components,
                case_key=case_key,
                newton_iter=newton_iter,
            )
        )
    return trace


def _minimize_gibbs_cond_fun(carry):
    (
        _ln_nk,
        _ln_ntot,
        _gk,
        _an,
        epsilon,
        counter,
        _formula_matrix,
        _element_vector,
        _temperature,
        _ln_normalized_pressure,
        _hvector,
        epsilon_crit,
        max_iter,
    ) = carry
    return (epsilon > epsilon_crit) & (counter < max_iter)


def _minimize_gibbs_body_fun(carry):
    (
        ln_nk,
        ln_ntot,
        gk,
        An,
        _epsilon,
        counter,
        formula_matrix,
        element_vector,
        temperature,
        ln_normalized_pressure,
        hvector,
        epsilon_crit,
        max_iter,
    ) = carry
    ln_nk_new, ln_ntot_new, epsilon, gk, An = update_all(
        ln_nk,
        ln_ntot,
        formula_matrix,
        element_vector,
        temperature,
        ln_normalized_pressure,
        hvector,
        gk,
        An,
    )
    # Keep cond/body at module scope and thread solver context through the
    # carry so repeated calls reuse the same while_loop callable identities.
    return (
        ln_nk_new,
        ln_ntot_new,
        gk,
        An,
        epsilon,
        counter + 1,
        formula_matrix,
        element_vector,
        temperature,
        ln_normalized_pressure,
        hvector,
        epsilon_crit,
        max_iter,
    )

def solve_gibbs_iteration_equations(
    nk: jnp.ndarray,
    ntotk: float,
    formula_matrix: jnp.ndarray,
    b: jnp.ndarray,
    gk: jnp.ndarray,
    An: jnp.ndarray,
) -> Tuple[jnp.ndarray, float]:
    """
    Solve the Gibbs iteration equations using the Lagrange multipliers.
    This function computes the matrix and vector to solve the system of equations
    that arises from the Gibbs energy minimization problem.

    Args:
        nk: number of species vector (n_species,) for k-th iteration.
        ntotk: Total number of species for k-th iteration.
        formula_matrix: Formula matrix for stoichiometric constraints (n_elements, n_species).
        b: Element abundance vector (n_elements, ).
        gk: gk vector (n_species,) for k-th iteration.
        An: formula_matrix @ nk vector (n_elements, ).

    Returns:
        Tuple containing:
            - The pi vector (nspecies, ).
            - The update of the  log total number of species (delta_ln_ntot).
    """
    resn = jnp.sum(nk) - ntotk
    bmatrix = _A_diagn_At(nk, formula_matrix)
    gk_nk = gk * nk
    Angk = formula_matrix @ gk_nk
    ngk = jnp.dot(nk, gk)
    rhs = Angk + b - An
    scalar_rhs = ngk - resn

    # Solve the bordered system through its Schur complement on
    # B = A diag(n) A^T to avoid assembling the dense (E+1)x(E+1) matrix.
    jitter = jnp.asarray(_CHO_EPS, dtype=bmatrix.dtype)
    eye = jnp.eye(bmatrix.shape[0], dtype=bmatrix.dtype)
    c_factor, lower = cho_factor(bmatrix + jitter * eye)

    rhs_pair = jnp.stack((rhs, An), axis=1)
    solved_pair = cho_solve((c_factor, lower), rhs_pair)
    binv_rhs = solved_pair[:, 0]
    binv_an = solved_pair[:, 1]

    schur = resn - jnp.vdot(An, binv_an)
    schur_safe = jnp.where(
        jnp.abs(schur) < jitter,
        jnp.where(schur < 0.0, -jitter, jitter),
        schur,
    )
    delta_ln_ntot = (scalar_rhs - jnp.vdot(An, binv_rhs)) / schur_safe
    pi_vector = binv_rhs - binv_an * delta_ln_ntot
    return pi_vector, delta_ln_ntot


def compute_residuals_with_at_pi(
    nk: jnp.ndarray,
    ntotk: float,
    b: jnp.ndarray,
    gk: jnp.ndarray,
    An: jnp.ndarray,
    at_pi: jnp.ndarray,
) -> float:
    ress = nk * (at_pi - gk)
    ress_squared = jnp.dot(ress, ress)

    An_b = An - b
    resj_squared = jnp.dot(An_b, An_b)

    resn = jnp.sum(nk) - ntotk
    resn_squared = jnp.dot(resn, resn)
    return jnp.sqrt(ress_squared + resj_squared + resn_squared)

CEA_SIZE = 18.420681        # = -ln(1e-8)
LN_X_CAP = 9.2103404        # = -ln(1e-4)

def _cea_lambda(delta_ln_nk, delta_ln_ntot, ln_nk, ln_ntot, size=CEA_SIZE):
    # λ1: ensure |Δln n|<=0.4, |Δln n_k|<=2
    cap_ntot = 5.0 * jnp.abs(delta_ln_ntot)           # 1/0.4
    cap_sp   = jnp.max(jnp.abs(delta_ln_nk))
    denom1   = jnp.maximum(jnp.maximum(cap_ntot, cap_sp), 1e-300)
    lam1     = 2.0 / denom1

    # maintain x_k<=1e-4 if increasing when x_k<=1e-8
    ln_xk  = ln_nk - ln_ntot
    small  = (ln_xk <= -size) & (delta_ln_nk >= 0.0)
    denom2 = delta_ln_nk - delta_ln_ntot
    safe   = small & (denom2 > 0.0)
    cand   = ( -LN_X_CAP - ln_xk ) / denom2           # (-ln 1e-4 - ln xk)/(Δln nk - Δln n)
    lam2   = jnp.where(jnp.any(safe), jnp.min(jnp.where(safe, cand, jnp.inf)), jnp.inf)

    lam = jnp.minimum(1.0, jnp.minimum(lam1, lam2))
    # safe guard
    lam = jnp.clip(lam, 1e-6, 1.0)
    return lam


def _prepare_iteration_system(
    ln_nk,
    ln_ntot,
    formula_matrix,
    b,
    gk,
    An,
):
    nk = jnp.exp(ln_nk)
    ntot = jnp.exp(ln_ntot)
    resn = jnp.sum(nk) - ntot
    bmatrix = _A_diagn_At(nk, formula_matrix)
    gk_nk = gk * nk
    Angk = formula_matrix @ gk_nk
    ngk = jnp.dot(nk, gk)
    rhs = Angk + b - An
    scalar_rhs = ngk - resn
    return nk, ntot, resn, bmatrix, rhs, scalar_rhs


def _solve_iteration_system(bmatrix, rhs, An, resn):
    jitter = jnp.asarray(_CHO_EPS, dtype=bmatrix.dtype)
    eye = jnp.eye(bmatrix.shape[0], dtype=bmatrix.dtype)
    c_factor, lower = cho_factor(bmatrix + jitter * eye)

    rhs_pair = jnp.stack((rhs, An), axis=1)
    solved_pair = cho_solve((c_factor, lower), rhs_pair)
    binv_rhs = solved_pair[:, 0]
    binv_an = solved_pair[:, 1]

    schur = resn - jnp.vdot(An, binv_an)
    schur_safe = jnp.where(
        jnp.abs(schur) < jitter,
        jnp.where(schur < 0.0, -jitter, jitter),
        schur,
    )
    return binv_rhs, binv_an, schur_safe


def _finish_iteration_solve(binv_rhs, binv_an, An, scalar_rhs, schur_safe):
    delta_ln_ntot = (scalar_rhs - jnp.vdot(An, binv_rhs)) / schur_safe
    pi_vector = binv_rhs - binv_an * delta_ln_ntot
    return pi_vector, delta_ln_ntot


def _apply_iteration_step(
    ln_nk,
    ln_ntot,
    formula_matrix,
    gk,
    pi_vector,
    delta_ln_ntot,
):
    at_pi = formula_matrix.T @ pi_vector
    delta_ln_nk = at_pi + delta_ln_ntot - gk
    lam = _cea_lambda(delta_ln_nk, delta_ln_ntot, ln_nk, ln_ntot)
    ln_ntot_new = ln_ntot + lam * delta_ln_ntot
    ln_nk_new = ln_nk + lam * delta_ln_nk
    return ln_nk_new, ln_ntot_new, lam, delta_ln_nk, at_pi


def _evaluate_iteration_state(
    ln_nk,
    ln_ntot,
    formula_matrix,
    b,
    T,
    ln_normalized_pressure,
    hvector,
    gk_prev,
    lam,
    at_pi,
    pi_vector,
):
    nk = jnp.exp(ln_nk)
    ntot = jnp.exp(ln_ntot)
    del T, ln_normalized_pressure, hvector, pi_vector
    gk = gk_prev + lam * (at_pi - gk_prev)
    An = formula_matrix @ nk
    epsilon = compute_residuals_with_at_pi(nk, ntot, b, gk, An, at_pi)
    return gk, An, epsilon

def update_all(
    ln_nk,
    ln_ntot,
    formula_matrix,
    b,
    T,
    ln_normalized_pressure,
    hvector,
    gk,
    An,
):
    _, _, resn, bmatrix, rhs, scalar_rhs = _prepare_iteration_system(
        ln_nk, ln_ntot, formula_matrix, b, gk, An
    )
    binv_rhs, binv_an, schur_safe = _solve_iteration_system(bmatrix, rhs, An, resn)
    pi_vector, delta_ln_ntot = _finish_iteration_solve(
        binv_rhs, binv_an, An, scalar_rhs, schur_safe
    )
    ln_nk, ln_ntot, lam, _, at_pi = _apply_iteration_step(
        ln_nk, ln_ntot, formula_matrix, gk, pi_vector, delta_ln_ntot
    )
    gk, An, epsilon = _evaluate_iteration_state(
        ln_nk,
        ln_ntot,
        formula_matrix,
        b,
        T,
        ln_normalized_pressure,
        hvector,
        gk,
        lam,
        at_pi,
        pi_vector,
    )
    return ln_nk, ln_ntot, epsilon, gk, An


def profile_minimize_gibbs_iterations(
    state: ThermoState,
    ln_nk_init: jnp.ndarray,
    ln_ntot_init: float,
    formula_matrix: jnp.ndarray,
    hvector_func: Callable[[float], jnp.ndarray],
    epsilon_crit: float = 1.0e-11,
    max_iter: int = 1000,
) -> Dict[str, object]:
    """Run the same Newton iterations in Python and time major sub-steps.

    This is a profiling helper, not the production solve path.
    """

    def _block(x):
        return jax.tree_util.tree_map(
            lambda y: y.block_until_ready() if hasattr(y, "block_until_ready") else y,
            x,
        )

    prepare_system = jax.jit(_prepare_iteration_system)
    solve_system = jax.jit(_solve_iteration_system)
    finish_solve = jax.jit(_finish_iteration_solve)
    apply_step = jax.jit(_apply_iteration_step)
    eval_state = jax.jit(_evaluate_iteration_state)

    hvector = hvector_func(state.temperature)
    _block(hvector)

    gk = _compute_gk(
        state.temperature,
        ln_nk_init,
        ln_ntot_init,
        hvector,
        state.ln_normalized_pressure,
    )
    An = formula_matrix @ jnp.exp(ln_nk_init)
    _block((gk, An))

    ln_nk = ln_nk_init
    ln_ntot = ln_ntot_init
    epsilon = jnp.asarray(jnp.inf, dtype=ln_nk.dtype)
    epsilon_host = float("inf")
    counter = 0

    # Compile the per-part kernels once outside the timed loop.
    _, _, resn0, bmatrix0, rhs0, scalar_rhs0 = prepare_system(
        ln_nk, ln_ntot, formula_matrix, state.element_vector, gk, An
    )
    _block((resn0, bmatrix0, rhs0, scalar_rhs0))
    binv_rhs0, binv_an0, schur_safe0 = solve_system(bmatrix0, rhs0, An, resn0)
    _block((binv_rhs0, binv_an0, schur_safe0))
    pi_vector0, delta_ln_ntot0 = finish_solve(
        binv_rhs0, binv_an0, An, scalar_rhs0, schur_safe0
    )
    _block((pi_vector0, delta_ln_ntot0))
    ln_nk1, ln_ntot1, lam1, _, at_pi1 = apply_step(
        ln_nk, ln_ntot, formula_matrix, gk, pi_vector0, delta_ln_ntot0
    )
    _block((ln_nk1, ln_ntot1))
    gk1, An1, epsilon1 = eval_state(
        ln_nk1,
        ln_ntot1,
        formula_matrix,
        state.element_vector,
        state.temperature,
        state.ln_normalized_pressure,
        hvector,
        gk,
        lam1,
        at_pi1,
        pi_vector0,
    )
    _block((gk1, An1, epsilon1))

    timings_s = {
        "prepare_system": 0.0,
        "linear_solve": 0.0,
        "finish_solve": 0.0,
        "step_update_damping": 0.0,
        "residual_evaluation": 0.0,
        "convergence_check": 0.0,
    }

    while True:
        t0 = time.perf_counter()
        keep_going = (epsilon_host > epsilon_crit) and (counter < max_iter)
        timings_s["convergence_check"] += time.perf_counter() - t0
        if not keep_going:
            break

        t0 = time.perf_counter()
        _, _, resn, bmatrix, rhs, scalar_rhs = prepare_system(
            ln_nk, ln_ntot, formula_matrix, state.element_vector, gk, An
        )
        _block((resn, bmatrix, rhs, scalar_rhs))
        timings_s["prepare_system"] += time.perf_counter() - t0

        t0 = time.perf_counter()
        binv_rhs, binv_an, schur_safe = solve_system(bmatrix, rhs, An, resn)
        _block((binv_rhs, binv_an, schur_safe))
        timings_s["linear_solve"] += time.perf_counter() - t0

        t0 = time.perf_counter()
        pi_vector, delta_ln_ntot = finish_solve(
            binv_rhs, binv_an, An, scalar_rhs, schur_safe
        )
        _block((pi_vector, delta_ln_ntot))
        timings_s["finish_solve"] += time.perf_counter() - t0

        t0 = time.perf_counter()
        ln_nk, ln_ntot, lam, _, at_pi = apply_step(
            ln_nk, ln_ntot, formula_matrix, gk, pi_vector, delta_ln_ntot
        )
        _block((ln_nk, ln_ntot))
        timings_s["step_update_damping"] += time.perf_counter() - t0

        t0 = time.perf_counter()
        gk, An, epsilon = eval_state(
            ln_nk,
            ln_ntot,
            formula_matrix,
            state.element_vector,
            state.temperature,
            state.ln_normalized_pressure,
            hvector,
            gk,
            lam,
            at_pi,
            pi_vector,
        )
        _block((gk, An, epsilon))
        epsilon_host = float(jax.device_get(epsilon))
        timings_s["residual_evaluation"] += time.perf_counter() - t0

        counter += 1

    total_profiled_s = sum(timings_s.values())
    average_iteration_s = total_profiled_s / counter if counter else 0.0
    average_breakdown_s = {
        key: value / counter if counter else 0.0 for key, value in timings_s.items()
    }

    return {
        "ln_nk": ln_nk,
        "ln_ntot": ln_ntot,
        "n_iter": counter,
        "final_residual": epsilon,
        "timings_s": timings_s,
        "average_iteration_s": average_iteration_s,
        "average_breakdown_s": average_breakdown_s,
        "total_profiled_s": total_profiled_s,
    }

def minimize_gibbs_core(
    state: ThermoState,
    ln_nk_init: jnp.ndarray,
    ln_ntot_init: float,
    formula_matrix: jnp.ndarray,
    hvector_func,
    epsilon_crit: float = 1.0e-11,
    max_iter: int = 1000,
) -> Tuple[jnp.ndarray, float, int, jnp.ndarray]:
    """Compute log(number of species) by minimizing the Gibbs energy using the Lagrange multipliers method.

    Args:
        state: Thermodynamic state containing temperature, pressure, and element abundances.
        ln_nk_init: Initial log number of species vector (n_species,).
        ln_ntot_init: Initial log total number of species.
        formula_matrix: Stoichiometric formula matrix (n_elements, n_species).
        hvector: Chemical potential over RT vector (n_species,).
        epsilon_crit: Convergence tolerance for residual norm.
        max_iter: Maximum number of iterations allowed.

    Returns:
        Tuple containing:
            - Final log number of species vector (n_species,).
            - Final log total number of species.
            - Number of iterations performed.
            - Final residual norm used in convergence checks.
    """

    hvector = hvector_func(state.temperature)

    gk = _compute_gk(
        state.temperature,
        ln_nk_init,
        ln_ntot_init,
        hvector,
        state.ln_normalized_pressure,
    )
    An = formula_matrix @ jnp.exp(ln_nk_init)

    init_carry = (
        ln_nk_init,
        ln_ntot_init,
        gk,
        An,
        jnp.inf,
        0,
        formula_matrix,
        state.element_vector,
        state.temperature,
        state.ln_normalized_pressure,
        hvector,
        epsilon_crit,
        max_iter,
    )
    ln_nk, ln_tot, _, _, epsilon, counter, _, _, _, _, _, _, _ = while_loop(
        _minimize_gibbs_cond_fun,
        _minimize_gibbs_body_fun,
        init_carry,
    )
    return ln_nk, ln_tot, counter, epsilon


def minimize_gibbs_core_with_source_trace(
    state: ThermoState,
    ln_nk_init: jnp.ndarray,
    ln_ntot_init: float,
    formula_matrix: jnp.ndarray,
    hvector_func,
    epsilon_crit: float = 1.0e-11,
    max_iter: int = 1000,
    *,
    source_trace_case_key: str = "diagnostic",
    source_trace_newton_iter: int = 0,
) -> Tuple[jnp.ndarray, float, int, jnp.ndarray, dict[str, Any]]:
    """Run the core solver and attach a default-off final-carry source trace."""

    ln_nk, ln_tot, counter, epsilon = minimize_gibbs_core(
        state,
        ln_nk_init,
        ln_ntot_init,
        formula_matrix,
        hvector_func,
        epsilon_crit,
        max_iter,
    )
    source_trace = build_minimize_gibbs_core_final_carry_source_trace(
        ln_nk,
        ln_tot,
        counter,
        epsilon,
        case_key=source_trace_case_key,
        newton_iter=source_trace_newton_iter,
    )
    source_trace["update_all_lnnk_new_source_trace"] = (
        trace_minimize_gibbs_core_update_all_lnnk_new_source_components(
            state,
            ln_nk_init,
            ln_ntot_init,
            formula_matrix,
            hvector_func,
            epsilon_crit,
            max_iter,
            case_key=source_trace_case_key,
            newton_iter=source_trace_newton_iter,
        )
    )
    return ln_nk, ln_tot, counter, epsilon, source_trace


def _minimize_gibbs_solve_impl(
    state: ThermoState,
    ln_nk0: jnp.ndarray,
    ln_ntot0: float,
    formula_matrix: jnp.ndarray,
    hvector_func: Callable[[float], jnp.ndarray],
    epsilon_crit: float,
    max_iter: int,
) -> jnp.ndarray:
    ln_nk, _, _, _ = minimize_gibbs_core(
        state,
        ln_nk0,
        ln_ntot0,
        formula_matrix,
        hvector_func,
        epsilon_crit,
        max_iter,
    )
    return ln_nk


# Keep the transformed solver at module scope so repeated calls reuse the same
# Python callable identity instead of rebuilding a new custom_vjp closure.
@partial(custom_vjp, nondiff_argnums=(3, 4, 5, 6))
def _minimize_gibbs_solve(
    state: ThermoState,
    ln_nk0: jnp.ndarray,
    ln_ntot0: float,
    formula_matrix: jnp.ndarray,
    hvector_func: Callable[[float], jnp.ndarray],
    epsilon_crit: float,
    max_iter: int,
) -> jnp.ndarray:
    return _minimize_gibbs_solve_impl(
        state,
        ln_nk0,
        ln_ntot0,
        formula_matrix,
        hvector_func,
        epsilon_crit,
        max_iter,
    )


def _minimize_gibbs_solve_fwd(
    state: ThermoState,
    ln_nk0: jnp.ndarray,
    ln_ntot0: float,
    formula_matrix: jnp.ndarray,
    hvector_func: Callable[[float], jnp.ndarray],
    epsilon_crit: float,
    max_iter: int,
):
    ln_nk, ln_ntot, _, _ = minimize_gibbs_core(
        state,
        ln_nk0,
        ln_ntot0,
        formula_matrix,
        hvector_func,
        epsilon_crit,
        max_iter,
    )
    dfunc = jacrev(hvector_func)
    hdot = dfunc(state.temperature)
    residuals = (ln_nk, hdot, state.element_vector, ln_ntot)
    return ln_nk, residuals


def _minimize_gibbs_solve_bwd(
    formula_matrix: jnp.ndarray,
    hvector_func: Callable[[float], jnp.ndarray],
    epsilon_crit: float,
    max_iter: int,
    res,
    g,
):
    del hvector_func, epsilon_crit, max_iter
    ln_nk, hdot, element_vector, ln_ntot = res

    nk = jnp.exp(ln_nk)
    ntot_result = jnp.exp(ln_ntot)

    Bmatrix = _A_diagn_At(nk, formula_matrix)
    c, lower = cho_factor(Bmatrix)
    alpha = cho_solve((c, lower), formula_matrix @ g)
    beta = cho_solve((c, lower), element_vector)
    beta_dot_b_element = jnp.vdot(beta, element_vector)

    cot_T = vjp_temperature(
        g,
        nk,
        formula_matrix,
        hdot,
        alpha,
        beta,
        element_vector,
        beta_dot_b_element,
    )
    cot_P = vjp_pressure(g, ntot_result, alpha, element_vector, beta_dot_b_element)
    cot_b = vjp_elements(g, alpha, beta, element_vector, beta_dot_b_element)
    # No gradients for initialization arguments.
    return (ThermoState(jnp.asarray(cot_T), jnp.asarray(cot_P), cot_b), None, None)


_minimize_gibbs_solve.defvjp(_minimize_gibbs_solve_fwd, _minimize_gibbs_solve_bwd)


def minimize_gibbs(
    state: ThermoState,
    ln_nk_init: jnp.ndarray,
    ln_ntot_init: float,
    formula_matrix: jnp.ndarray,
    hvector_func: Callable[[float], jnp.ndarray],
    epsilon_crit: float = 1.0e-11,
    max_iter: int = 1000,
) -> jnp.ndarray:
    """Compute log(number of species) by minimizing the Gibbs energy using the Lagrange multipliers method.

    Args:
        state: Thermodynamic state containing temperature, pressure, and element abundances.
        ln_nk_init: Initial natural log number of species vector (n_species,).
        ln_ntot_init: Initial natural log total number of species.
        formula_matrix: Stoichiometric formula matrix (n_elements, n_species).
        hvector_func: Function that returns chemical potential over RT vector (n_species,).
        epsilon_crit: Convergence tolerance for residual norm.
        max_iter: Maximum number of iterations allowed.

    Returns:
        Final log number of species vector (n_species,).
    """
    # Treat initial guesses as non-differentiable inputs
    ln_nk0 = stop_gradient(ln_nk_init)
    ln_ntot0 = stop_gradient(ln_ntot_init)
    return _minimize_gibbs_solve(
        state,
        ln_nk0,
        ln_ntot0,
        formula_matrix,
        hvector_func,
        epsilon_crit,
        max_iter,
    )


def minimize_gibbs_with_diagnostics(
    state: ThermoState,
    ln_nk_init: jnp.ndarray,
    ln_ntot_init: float,
    formula_matrix: jnp.ndarray,
    hvector_func: Callable[[float], jnp.ndarray],
    epsilon_crit: float = 1.0e-11,
    max_iter: int = 1000,
) -> Tuple[jnp.ndarray, Dict[str, jnp.ndarray]]:
    """Run Gibbs minimization and return lightweight convergence diagnostics."""
    ln_nk, _, n_iter, final_residual = minimize_gibbs_core(
        state,
        ln_nk_init,
        ln_ntot_init,
        formula_matrix,
        hvector_func,
        epsilon_crit,
        max_iter,
    )
    epsilon_crit_used = jnp.asarray(epsilon_crit, dtype=final_residual.dtype)
    max_iter_used = jnp.asarray(max_iter, dtype=n_iter.dtype)
    converged = final_residual <= epsilon_crit_used
    hit_max_iter = (n_iter >= max_iter_used) & (~converged)

    diagnostics = {
        "n_iter": n_iter,
        "converged": converged,
        "hit_max_iter": hit_max_iter,
        "final_residual": final_residual,
        "epsilon_crit": epsilon_crit_used,
        "max_iter": max_iter_used,
    }
    return ln_nk, diagnostics
