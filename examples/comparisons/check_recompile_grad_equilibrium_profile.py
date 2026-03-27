import io
import logging
import re
import time
from contextlib import contextmanager

import jax
import numpy as np
from jax import config

from exogibbs.api.equilibrium import EquilibriumOptions, equilibrium_profile
from exogibbs.presets.ykb4 import chemsetup


config.update("jax_enable_x64", True)
config.update("jax_log_compiles", True)


@contextmanager
def capture_compile_log():
    stream = io.StringIO()
    handler = logging.StreamHandler(stream)
    logger = logging.getLogger("jax._src.interpreters.pxla")

    old_level = logger.level
    old_handlers = list(logger.handlers)
    old_propagate = logger.propagate

    logger.setLevel(logging.WARNING)
    logger.propagate = False
    logger.handlers = [handler]
    try:
        yield stream
    finally:
        logger.setLevel(old_level)
        logger.propagate = old_propagate
        logger.handlers = old_handlers


def block_tree(tree):
    jax.tree_util.tree_map(jax.block_until_ready, tree)


def summarize_compile_log(text):
    compile_lines = [line.strip() for line in text.splitlines() if "Compiling " in line]
    return {
        "total_compile_count": len(compile_lines),
        "scan_compile_count": len(re.findall(r"Compiling scan\\b", text)),
        "while_compile_count": len(re.findall(r"Compiling while\\b", text)),
        "jit_compile_count": len(re.findall(r"Compiling jit\\b", text)),
        "compile_lines": compile_lines,
    }


def main():
    setup = chemsetup()
    base_b = np.asarray(setup.element_vector_reference)

    nlayer = 5
    base_temperature = np.linspace(900.0, 1200.0, nlayer, dtype=np.float64)
    base_pressure = np.logspace(-3.0, 1.0, num=nlayer, dtype=np.float64)

    opts = EquilibriumOptions(
        method="scan_hot_from_top",
        epsilon_crit=1.0e-11,
        max_iter=200,
    )

    def objective(T_in, P_in, b_in):
        result = equilibrium_profile(
            setup,
            T_in,
            P_in,
            b_in,
            Pref=1.0,
            options=opts,
            return_diagnostics=False,
        )
        return result.ln_n.sum()

    grad_fn = jax.jit(jax.value_and_grad(objective, argnums=(0, 1, 2)))

    print("Differentiated function:")
    print("  grad_fn = jax.jit(jax.value_and_grad(objective, argnums=(0, 1, 2)))")
    print("  objective(T, P, b) = sum(equilibrium_profile(setup, T, P, b, options=opts).ln_n)")
    print("  This exercises reverse-mode AD through equilibrium_profile(...) and")
    print("  requests gradients with respect to temperature, pressure, and element abundances.")
    print("")

    niter = 3
    times = []
    per_iter_summaries = []

    for i in range(niter):
        T_in = base_temperature + 0.5 * i
        P_in = base_pressure * np.exp(0.01 * i)
        b_in = base_b.copy()
        b_in[0] = b_in[0] * np.exp(0.02 * i)

        with capture_compile_log() as log:
            t0 = time.time()
            value, grads = grad_fn(T_in, P_in, b_in)
            block_tree((value, grads))
            elapsed = time.time() - t0

        summary = summarize_compile_log(log.getvalue())
        per_iter_summaries.append(summary)
        times.append(elapsed)

        grad_T, grad_P, grad_b = grads
        print(f"=== iteration {i} ===")
        print(f"value = {float(value):.16e}")
        print(f"time_seconds = {elapsed:.6f}")
        print(
            "grad_norms = "
            f"T:{np.linalg.norm(np.asarray(grad_T)):.6e} "
            f"P:{np.linalg.norm(np.asarray(grad_P)):.6e} "
            f"b:{np.linalg.norm(np.asarray(grad_b)):.6e}"
        )
        print(
            "compile_counts = "
            f"total:{summary['total_compile_count']} "
            f"jit:{summary['jit_compile_count']} "
            f"scan:{summary['scan_compile_count']} "
            f"while:{summary['while_compile_count']}"
        )
        if summary["compile_lines"]:
            print("compile_lines:")
            for line in summary["compile_lines"]:
                print(f"  {line}")
        else:
            print("compile_lines: none")
        print("")

    repeated_recompile = any(
        summary["total_compile_count"] > 0 for summary in per_iter_summaries[1:]
    )
    print("=== summary ===")
    print(f"times = {times}")
    print(f"mean_time_excluding_first = {float(np.mean(times[1:])):.6f}")
    print(f"repeated_iterations_show_compile_logs = {repeated_recompile}")
    print("Interpretation:")
    print("  - iteration 0 compile activity is expected for the first trace/compile")
    print("  - iterations 1+ should normally show zero compile lines if recompilation is stable")
    print("  - compile lines on iterations 1+ indicate possible recompilation in the differentiated path")
    print("  - because gradients are taken w.r.t. T, P, and b, the custom_vjp backward path is exercised")


if __name__ == "__main__":
    main()
