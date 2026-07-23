import jax
import jax.numpy as jnp
import pytest

from exogibbs.optimize.fixed_support_ipopt_soc import (
    fixed_support_linearized_rhs_residual_blocks,
    fixed_support_reduced_direction_from_rhs,
    fixed_support_reduced_direction_from_rhs_with_diagnostics,
    fixed_support_soc_constraint_rhs,
    fixed_support_soc_trial_from_current,
)

jax.config.update("jax_enable_x64", True)


def _problem():
    return {
        "ag": jnp.asarray([[1.0, 0.2, 0.4], [0.1, 1.0, 0.3]]),
        "ac": jnp.asarray([[0.7, 0.1], [0.2, 0.8]]),
        "n": jnp.asarray([0.31, 0.42, 0.27]),
        "m": jnp.asarray([0.025, 0.04]),
        "eta": jnp.asarray([0.004, 0.006]),
        "ntot": jnp.asarray(1.15),
        "rg": jnp.asarray([0.3, -0.2, 0.1]),
        "rc": jnp.asarray([0.05, -0.08]),
        "rb": jnp.asarray([0.003, -0.002]),
        "rt_comp": jnp.asarray([0.02, -0.03]),
        "rt_total": jnp.asarray(0.001),
    }


def _reduced_direction(problem):
    return fixed_support_reduced_direction_from_rhs(
        formula_matrix=problem["ag"],
        formula_matrix_cond_active=problem["ac"],
        gas_amounts=problem["n"],
        condensate_amounts=problem["m"],
        condensate_duals=problem["eta"],
        total_gas_amount=problem["ntot"],
        gas_rhs=problem["rg"],
        condensate_rhs=problem["rc"],
        budget_rhs=problem["rb"],
        complementarity_rhs=problem["rt_comp"],
        total_density_rhs=problem["rt_total"],
    )


def test_generic_reduced_direction_matches_dense_full_kkt_solve():
    p = _problem()
    ng, nc = p["n"].shape[0], p["m"].shape[0]
    ne = p["ag"].shape[0]
    zeros = jnp.zeros
    full_matrix = jnp.block(
        [
            [
                jnp.eye(ng),
                zeros((ng, nc)),
                -p["ag"].T,
                zeros((ng, nc)),
                -jnp.ones((ng, 1)),
            ],
            [
                zeros((nc, ng)),
                zeros((nc, nc)),
                -p["ac"].T,
                -jnp.diag(p["eta"]),
                zeros((nc, 1)),
            ],
            [
                p["ag"] * p["n"][None, :],
                p["ac"] * p["m"][None, :],
                zeros((ne, ne)),
                zeros((ne, nc)),
                zeros((ne, 1)),
            ],
            [
                zeros((nc, ng)),
                jnp.eye(nc),
                zeros((nc, ne)),
                jnp.eye(nc),
                zeros((nc, 1)),
            ],
            [
                p["n"][None, :],
                zeros((1, nc)),
                zeros((1, ne)),
                zeros((1, nc)),
                -p["ntot"].reshape((1, 1)),
            ],
        ]
    )
    rhs = jnp.concatenate(
        [p["rg"], p["rc"], p["rb"], p["rt_comp"], p["rt_total"].reshape((1,))]
    )
    dense = jnp.linalg.solve(full_matrix, -rhs)
    reduced = jnp.concatenate(
        [
            *_reduced_direction(p)[:-1],
            _reduced_direction(p)[-1].reshape((1,)),
        ]
    )

    assert reduced == pytest.approx(dense, abs=2.0e-12)


def test_generic_reduced_direction_satisfies_all_five_rhs_blocks():
    p = _problem()
    direction = _reduced_direction(p)
    blocks = fixed_support_linearized_rhs_residual_blocks(
        formula_matrix=p["ag"],
        formula_matrix_cond_active=p["ac"],
        gas_amounts=p["n"],
        condensate_amounts=p["m"],
        condensate_duals=p["eta"],
        total_gas_amount=p["ntot"],
        gas_rhs=p["rg"],
        condensate_rhs=p["rc"],
        budget_rhs=p["rb"],
        complementarity_rhs=p["rt_comp"],
        total_density_rhs=p["rt_total"],
        delta_q=direction[0],
        delta_r=direction[1],
        delta_element_potential=direction[2],
        delta_rho=direction[3],
        delta_qtot=direction[4],
    )

    for block in blocks:
        assert block == pytest.approx(jnp.zeros_like(block), abs=2.0e-12)


def test_reduced_direction_reports_raw_solve_quality():
    p = _problem()
    result = fixed_support_reduced_direction_from_rhs_with_diagnostics(
        formula_matrix=p["ag"],
        formula_matrix_cond_active=p["ac"],
        gas_amounts=p["n"],
        condensate_amounts=p["m"],
        condensate_duals=p["eta"],
        total_gas_amount=p["ntot"],
        gas_rhs=p["rg"],
        condensate_rhs=p["rc"],
        budget_rhs=p["rb"],
        complementarity_rhs=p["rt_comp"],
        total_density_rhs=p["rt_total"],
    )

    assert bool(result[5])
    assert float(result[6]) < 1.0e-12
    assert jnp.isfinite(result[7])
    assert float(result[8]) > 0.0
    assert float(result[9]) >= float(result[8])
    assert float(jnp.linalg.norm(result[13])) == pytest.approx(1.0, abs=1.0e-12)


def test_charge_schur_direction_matches_coupled_direction():
    p = _problem()
    common = {
        "formula_matrix": p["ag"],
        "formula_matrix_cond_active": p["ac"],
        "gas_amounts": p["n"],
        "condensate_amounts": p["m"],
        "condensate_duals": p["eta"],
        "total_gas_amount": p["ntot"],
        "gas_rhs": p["rg"],
        "condensate_rhs": p["rc"],
        "budget_rhs": p["rb"],
        "complementarity_rhs": p["rt_comp"],
        "total_density_rhs": p["rt_total"],
    }
    coupled = fixed_support_reduced_direction_from_rhs(**common)
    eliminated = fixed_support_reduced_direction_from_rhs(
        **common,
        charge_solve_policy="charge_schur",
        charge_row_index=1,
    )

    for coupled_block, eliminated_block in zip(coupled, eliminated):
        assert eliminated_block == pytest.approx(coupled_block, abs=2.0e-11)


def test_charge_schur_solve_retains_weak_charge_equation():
    p = _problem()
    p["ag"] = p["ag"].at[1].multiply(1.0e-8)
    p["ac"] = p["ac"].at[1].set(0.0)
    result = fixed_support_reduced_direction_from_rhs_with_diagnostics(
        formula_matrix=p["ag"],
        formula_matrix_cond_active=p["ac"],
        gas_amounts=p["n"],
        condensate_amounts=p["m"],
        condensate_duals=p["eta"],
        total_gas_amount=p["ntot"],
        gas_rhs=p["rg"],
        condensate_rhs=p["rc"],
        budget_rhs=p["rb"].at[1].set(1.0e-10),
        complementarity_rhs=p["rt_comp"],
        total_density_rhs=p["rt_total"],
        charge_solve_policy="charge_schur",
        charge_row_index=1,
    )

    assert bool(result[5])
    assert float(result[12]) < 1.0e-12


def test_reduced_direction_can_remove_smallest_singular_mode():
    p = _problem()
    result = fixed_support_reduced_direction_from_rhs_with_diagnostics(
        formula_matrix=p["ag"],
        formula_matrix_cond_active=p["ac"],
        gas_amounts=p["n"],
        condensate_amounts=p["m"],
        condensate_duals=p["eta"],
        total_gas_amount=p["ntot"],
        gas_rhs=p["rg"],
        condensate_rhs=p["rc"],
        budget_rhs=p["rb"],
        complementarity_rhs=p["rt_comp"],
        total_density_rhs=p["rt_total"],
        reduced_mode_policy="remove_smallest_mode",
    )
    reduced_solution = jnp.concatenate([result[2], result[4].reshape((1,))])

    assert jnp.dot(reduced_solution, result[13][: reduced_solution.shape[0]]) == pytest.approx(
        0.0, abs=2.0e-12
    )


def test_reduced_direction_can_report_dominant_svd_solution_component():
    p = _problem()
    result = fixed_support_reduced_direction_from_rhs_with_diagnostics(
        formula_matrix=p["ag"],
        formula_matrix_cond_active=p["ac"],
        gas_amounts=p["n"],
        condensate_amounts=p["m"],
        condensate_duals=p["eta"],
        total_gas_amount=p["ntot"],
        gas_rhs=p["rg"],
        condensate_rhs=p["rc"],
        budget_rhs=p["rb"],
        complementarity_rhs=p["rt_comp"],
        total_density_rhs=p["rt_total"],
        diagnostic_mode_vector_policy="dominant_solution_component",
    )
    solution = jnp.concatenate([result[2], result[4].reshape((1,))])
    component = result[13][: solution.shape[0]]

    assert jnp.linalg.norm(component) <= jnp.linalg.norm(solution) + 1.0e-12
    assert jnp.dot(solution - component, component) == pytest.approx(
        0.0, abs=2.0e-12
    )
    metrics = result[13][solution.shape[0] :]
    rhs_projection_sum = jnp.sum(metrics[jnp.asarray([10, 11, 12, 15, 16])])
    assert rhs_projection_sum == pytest.approx(
        metrics[1] * metrics[2], abs=2.0e-11
    )


def test_symmetric_equilibration_stabilizes_ill_scaled_reduced_system():
    p = _problem()
    p.update(
        {
            "n": jnp.asarray([1.0e-12, 0.42, 0.27]),
            "m": jnp.asarray([1.0e-14, 0.04]),
            "eta": jnp.asarray([1.0e-18, 0.006]),
            "ntot": jnp.asarray(0.690000000001),
        }
    )
    result = fixed_support_reduced_direction_from_rhs_with_diagnostics(
        formula_matrix=p["ag"],
        formula_matrix_cond_active=p["ac"],
        gas_amounts=p["n"],
        condensate_amounts=p["m"],
        condensate_duals=p["eta"],
        total_gas_amount=p["ntot"],
        gas_rhs=p["rg"],
        condensate_rhs=p["rc"],
        budget_rhs=p["rb"],
        complementarity_rhs=p["rt_comp"],
        total_density_rhs=p["rt_total"],
    )
    unscaled_condition = result[9] / result[8]
    scaled_condition = result[11] / result[10]

    assert bool(result[5])
    assert float(scaled_condition) < 0.01 * float(unscaled_condition)
    assert float(result[12]) < 1.0e-12
    assert float(jnp.linalg.norm(result[13])) == pytest.approx(1.0, abs=1.0e-12)


def test_generic_reduced_direction_reproduces_absolute_pi_ordinary_system():
    p = _problem()
    lam = jnp.asarray([0.12, -0.07])
    q_plus_source = jnp.asarray([0.4, -0.1, 0.2])
    hcond = jnp.asarray([0.03, -0.04])
    comp = p["rt_comp"]
    budget = p["rb"]
    total = p["rt_total"]
    p = {
        **p,
        "rg": q_plus_source - p["ag"].T @ lam,
        "rc": hcond - p["ac"].T @ lam - p["eta"],
        "rb": budget,
        "rt_comp": comp,
        "rt_total": total,
    }
    direction = _reduced_direction(p)
    j_vec = p["m"] / p["eta"]
    gas_inventory = p["ag"] @ p["n"]
    qhat = (
        p["ag"] @ (p["n"][:, None] * p["ag"].T)
        + p["ac"] @ (j_vec[:, None] * p["ac"].T)
    )
    matrix = jnp.block(
        [
            [qhat, gas_inventory[:, None]],
            [
                gas_inventory[None, :],
                (jnp.sum(p["n"]) - p["ntot"]).reshape((1, 1)),
            ],
        ]
    )
    rhs_top = (
        p["ag"] @ (p["n"] * q_plus_source)
        + p["ac"] @ (j_vec * hcond + p["m"] * comp - p["m"])
        - budget
    )
    rhs_bottom = jnp.dot(p["n"], q_plus_source) - total
    absolute = jnp.linalg.solve(
        matrix, jnp.concatenate([rhs_top, rhs_bottom.reshape((1,))])
    )

    assert lam + direction[2] == pytest.approx(absolute[:-1], abs=2.0e-12)
    assert direction[4] == pytest.approx(absolute[-1], abs=2.0e-12)


def test_soc_constraint_rhs_uses_ipopt_recurrence():
    current_budget = jnp.asarray([0.2, -0.1])
    current_total = jnp.asarray(0.05)
    first_trial_budget = jnp.asarray([0.03, 0.04])
    first_trial_total = jnp.asarray(-0.02)
    first_budget, first_total = fixed_support_soc_constraint_rhs(
        trial_budget_residual=first_trial_budget,
        trial_total_density_residual=first_trial_total,
        previous_soc_budget_rhs=current_budget,
        previous_soc_total_density_rhs=current_total,
        alpha_soc=0.5,
    )
    second_budget, second_total = fixed_support_soc_constraint_rhs(
        trial_budget_residual=jnp.asarray([-0.01, 0.02]),
        trial_total_density_residual=jnp.asarray(0.03),
        previous_soc_budget_rhs=first_budget,
        previous_soc_total_density_rhs=first_total,
        alpha_soc=0.25,
    )

    assert first_budget == pytest.approx(first_trial_budget + 0.5 * current_budget)
    assert first_total == pytest.approx(first_trial_total + 0.5 * current_total)
    assert second_budget == pytest.approx(jnp.asarray([-0.01, 0.02]) + 0.25 * first_budget)
    assert second_total == pytest.approx(0.03 + 0.25 * first_total)


def test_soc_rhs_direction_uses_soc_constraints_in_primal_blocks():
    p = _problem()
    soc_budget, soc_total = fixed_support_soc_constraint_rhs(
        trial_budget_residual=jnp.asarray([0.04, -0.03]),
        trial_total_density_residual=jnp.asarray(0.02),
        previous_soc_budget_rhs=p["rb"],
        previous_soc_total_density_rhs=p["rt_total"],
        alpha_soc=0.5,
    )
    p = {**p, "rb": soc_budget, "rt_total": soc_total}
    direction = _reduced_direction(p)
    blocks = fixed_support_linearized_rhs_residual_blocks(
        formula_matrix=p["ag"],
        formula_matrix_cond_active=p["ac"],
        gas_amounts=p["n"],
        condensate_amounts=p["m"],
        condensate_duals=p["eta"],
        total_gas_amount=p["ntot"],
        gas_rhs=p["rg"],
        condensate_rhs=p["rc"],
        budget_rhs=soc_budget,
        complementarity_rhs=p["rt_comp"],
        total_density_rhs=soc_total,
        delta_q=direction[0],
        delta_r=direction[1],
        delta_element_potential=direction[2],
        delta_rho=direction[3],
        delta_qtot=direction[4],
    )

    for block in blocks:
        assert block == pytest.approx(jnp.zeros_like(block), abs=2.0e-12)


def test_soc_trial_starts_from_current_and_separates_step_sizes():
    trial = fixed_support_soc_trial_from_current(
        q=jnp.asarray([1.0]),
        r=jnp.asarray([2.0]),
        element_potential=jnp.asarray([3.0]),
        rho=jnp.asarray([4.0]),
        qtot=jnp.asarray(5.0),
        delta_q=jnp.asarray([10.0]),
        delta_r=jnp.asarray([20.0]),
        delta_element_potential=jnp.asarray([30.0]),
        delta_rho=jnp.asarray([40.0]),
        delta_qtot=jnp.asarray(50.0),
        alpha_test=0.125,
        alpha_soc=0.5,
        alpha_y=0.25,
        alpha_dual=0.75,
    )

    assert trial[0] == pytest.approx(jnp.asarray([6.0]))
    assert trial[1] == pytest.approx(jnp.asarray([12.0]))
    assert trial[2] == pytest.approx(jnp.asarray([10.5]))
    assert trial[3] == pytest.approx(jnp.asarray([34.0]))
    assert trial[4] == pytest.approx(jnp.asarray(30.0))
    assert trial[5] == pytest.approx(jnp.asarray(0.125))
