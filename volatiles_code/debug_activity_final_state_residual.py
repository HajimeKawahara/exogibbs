from __future__ import annotations

import json
import os
import sys

import numpy as np

os.environ.setdefault("JAX_ENABLE_X64", "1")
os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.environ.setdefault("JAX_PLATFORM_NAME", "cpu")

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if os.path.join(ROOT, "src") not in sys.path:
    sys.path.insert(0, os.path.join(ROOT, "src"))

import jax.numpy as jnp  # noqa: E402

from exogibbs.api.condensate_equilibrium import (  # noqa: E402
    CondensateEquilibriumOptions,
    condensate_equilibrium,
)
from exogibbs.condensates.curated_profiles import FRESH_CURATED_PROFILES  # noqa: E402
from exogibbs.optimize.pdipm_rgie_cond import _algorithm_v11_residuals  # noqa: E402
from exogibbs.presets.fastchem4_cond import condensate_chemical_setup  # noqa: E402


def _l2(values):
    values = np.asarray(values, dtype=np.float64)
    scale = np.max(np.abs(values)) if values.size else 0.0
    return 0.0 if scale == 0.0 else float(scale * np.linalg.norm(values / scale))


def main() -> None:
    family = sys.argv[1] if len(sys.argv) > 1 else "solar_water_condensation"
    layer = int(sys.argv[2]) if len(sys.argv) > 2 else 0
    setup = condensate_chemical_setup(silent=True)
    definition = FRESH_CURATED_PROFILES[family]
    b = jnp.asarray(setup.gas_setup.element_vector_reference, dtype=jnp.float64)
    result = condensate_equilibrium(
        setup,
        float(definition.temperatures[layer]),
        float(definition.pressures[layer]),
        b,
        options=CondensateEquilibriumOptions(
            max_inner_iterations=16,
            return_diagnostics=True,
            enable_full_condensate_budget_residual_gate=False,
        ),
    )
    lifecycle = result.diagnostics["head_route_lifecycle"]
    primary = lifecycle["primary_execution_report"]
    continuation = primary["continuation_report"]
    final_state = continuation["final_state"]
    support = tuple(int(v) for v in primary["filter_report"]["valid_support_indices"])
    q = np.asarray(final_state["ln_nk"], dtype=np.float64)
    r = np.asarray(final_state["ln_mk"], dtype=np.float64)
    lam = np.asarray(final_state["element_potential"], dtype=np.float64)
    rho = np.asarray(final_state["rho"], dtype=np.float64)
    qtot = float(final_state["ln_ntot"])
    temperature = float(definition.temperatures[layer])
    pressure = float(definition.pressures[layer])
    hgas = np.asarray(setup.gas_setup.hvector_func(temperature), dtype=np.float64)
    hcond_full = np.asarray(
        setup.condensate_setup.hvector_func(temperature),
        dtype=np.float64,
    )
    ac = np.asarray(setup.formula_matrix_cond, dtype=np.float64)[:, support]
    continuation_input = lifecycle.get("continuation_input", {})
    gas_source_payload = continuation_input.get("gas_stationarity_source")
    gas_source_guess = (
        np.asarray(gas_source_payload, dtype=np.float64)
        if gas_source_payload is not None
        else hgas + np.log(pressure) - qtot
    )
    gas_source_formula = hgas + np.log(pressure) - qtot
    final_epsilon = float(continuation.get("final_epsilon", -10.0))
    residual = _algorithm_v11_residuals(
        formula_matrix=np.asarray(setup.formula_matrix, dtype=np.float64),
        formula_matrix_cond_active=ac,
        element_inventory_target=np.asarray(b, dtype=np.float64),
        gas_stationarity_source=gas_source_guess,
        condensate_standard_source=hcond_full[list(support)],
        q=q,
        r=r,
        lam=lam,
        rho=rho,
        qtot=qtot,
        epsilon=final_epsilon,
        qtot_reference=qtot,
    )
    print(
        json.dumps(
            {
                "family": family,
                "layer": layer,
                "result_converged": bool(result.converged),
                "result_route": result.selected_route,
                "result_support_count": int(len(result.condensate_support_indices)),
                "final_state_support_count": len(support),
                "continuation_final_residual_l2": continuation.get("final_residual_l2"),
                "continuation_final_epsilon": final_epsilon,
                "computed_combined_l2": _l2(residual["combined"]),
                "gas_source_formula_delta_l2": _l2(gas_source_guess - gas_source_formula),
                "components": {
                    key: _l2(value)
                    for key, value in residual.items()
                    if key != "combined"
                },
                "final_state_keys": sorted(final_state.keys()),
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
