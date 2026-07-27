"""Diagnostic and source-trace entry points for the gas kernel."""

from exogibbs.equilibrium.gas.kernel.solver import (
    build_hvector_provider_source_trace,
    build_keyed_final_iteration_provider_linear_trace,
    build_minimize_gibbs_core_final_carry_source_trace,
    build_minimize_gibbs_core_lnnk_output_source_trace,
    compare_solve_iteration_system_longdouble,
    minimize_gibbs_core_with_source_trace,
    minimize_gibbs_with_diagnostics,
    profile_minimize_gibbs_iterations,
    trace_minimize_gibbs_core_update_all_lnnk_new_source_components,
)


__all__ = (
    "build_hvector_provider_source_trace",
    "build_keyed_final_iteration_provider_linear_trace",
    "build_minimize_gibbs_core_final_carry_source_trace",
    "build_minimize_gibbs_core_lnnk_output_source_trace",
    "compare_solve_iteration_system_longdouble",
    "minimize_gibbs_core_with_source_trace",
    "minimize_gibbs_with_diagnostics",
    "profile_minimize_gibbs_iterations",
    "trace_minimize_gibbs_core_update_all_lnnk_new_source_components",
)
