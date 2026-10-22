"""A solution phase in equilibrium with a pure solid
=================================================

This synthetic example uses the public fixed-support multiphase API. Both
phases carry the fictitious element X; only the ideal binary liquid carries Y. With solid
standard Gibbs energy -RT*log(2), coexistence requires a liquid X fraction of
one half. A budget of 3 mol X and 1 mol Y therefore gives 1 mol of each liquid
component and 2 mol of solid X.

Run ``python examples/multiphase_equilibrium.py`` from the repository root.
No ExoEOS, ExoInventory or external data are needed. The caller selects both
phases and all three positive components. Phase selection and insertion
stability are not audited by a fixed-support solve.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np

from exogibbs.api import multiphase


GAS_CONSTANT_J_MOL_K = 8.31446261815324


def liquid_gibbs_j(
    temperature_k: float, pressure_pa: float,
    amounts_mol: jax.Array, parameters: dict[str, jax.Array],
) -> jax.Array:
    """Return the full extensive ideal-solution Gibbs energy in joules."""
    total = jnp.sum(amounts_mol)
    fractions = amounts_mol / jnp.where(total > 0, total, 1.0)
    log_fractions = jnp.log(jnp.where(amounts_mol > 0, fractions, 1.0))
    return GAS_CONSTANT_J_MOL_K * temperature_k * (amounts_mol @ log_fractions)


def solid_gibbs_j(
    temperature_k: float, pressure_pa: float,
    amounts_mol: jax.Array, parameters: dict[str, jax.Array],
) -> jax.Array:
    """Return n*g for the pure solid, using the same energy reference."""
    return (GAS_CONSTANT_J_MOL_K * temperature_k
            * parameters["solid_standard_rt"] * amounts_mol[0])


def run_example() -> multiphase.Result:
    """Solve a declared support and check the two-phase elemental ledger."""
    jax.config.update("jax_enable_x64", True)
    problem = multiphase.Problem(
        elements=("X", "Y"),
        phases=(
            multiphase.Phase(
                name="liquid", components=("X(l)", "Y(l)"),
                atom_matrix=((1.0, 0.0), (0.0, 1.0)), gibbs_j=liquid_gibbs_j,
            ),
            multiphase.Phase(
                name="solid", components=("X(s)",),
                atom_matrix=((1.0,), (0.0,)), gibbs_j=solid_gibbs_j,
            ),
        ),
    )
    budget_mol = jnp.array([3.0, 1.0])
    result = multiphase.solve_fixed_support(
        problem, temperature_k=1200.0, pressure_pa=1.0e5,
        element_amounts_mol=budget_mol, init=jnp.array([1.5, 1.0, 1.5]),
        parameters={"solid_standard_rt": -jnp.log(2.0)},
        support=multiphase.Support(components=(0, 1, 2)),
    )
    if not all(bool(value) for value in (
        result.converged, result.local_kkt_valid,
        result.minimum_valid, result.support_valid,
    )):
        raise RuntimeError("The multiphase result failed its numerical or local-minimum checks.")
    np.testing.assert_allclose(
        problem.atom_matrix @ np.asarray(result.amounts_mol), budget_mol,
        rtol=1.0e-8, atol=1.0e-10,
    )
    return result


if __name__ == "__main__":
    result = run_example()
    print("Component amounts [X(l), Y(l), X(s)] (mol):", np.asarray(result.amounts_mol))
    print("Phase amounts [liquid, solid] (mol):", np.asarray(result.phase_amounts_mol))
    print("Phase stability (fixed-support solve):", result.stability_status)
