import jax
import jax.numpy as jnp
import pytest

from exogibbs.optimize.fixed_support_soc import (
    fixed_support_soc_correction_direction,
)

jax.config.update("jax_enable_x64", True)


def test_soc_direction_satisfies_full_kkt_linearization_for_primal_defect():
    ag = jnp.asarray([[1.0, 0.0], [0.0, 1.0]])
    ac = jnp.asarray([[1.0], [1.0]])
    n = jnp.asarray([0.31, 0.42])
    m = jnp.asarray([0.025])
    eta = jnp.asarray([0.004])
    gas_inventory = ag @ n
    total_residual = jnp.sum(n) - 0.74
    budget_defect = jnp.asarray([0.003, -0.002])
    total_defect = jnp.asarray(0.001)
    dq, dr, dlam, drho, dqtot = fixed_support_soc_correction_direction(
        formula_matrix=ag,
        formula_matrix_cond_active=ac,
        gas_amounts=n,
        condensate_amounts=m,
        condensate_duals=eta,
        gas_inventory=gas_inventory,
        total_density_residual=total_residual,
        budget_defect=budget_defect,
        total_density_defect=total_defect,
        max_abs_primal_step=1.0e6,
    )

    assert dq - ag.T @ dlam - dqtot == pytest.approx(jnp.zeros_like(dq), abs=1e-12)
    assert -ac.T @ dlam - eta * drho == pytest.approx(
        jnp.zeros_like(dr), abs=1e-12
    )
    assert dr + drho == pytest.approx(jnp.zeros_like(dr), abs=1e-12)
    assert ag @ (n * dq) + ac @ (m * dr) == pytest.approx(
        -budget_defect, abs=1e-11
    )
    assert jnp.dot(n, dq) - 0.74 * dqtot == pytest.approx(
        -total_defect, abs=1e-11
    )
