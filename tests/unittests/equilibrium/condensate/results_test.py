import jax.numpy as jnp

from exogibbs.equilibrium.condensate.results import (
    build_condensate_equilibrium_result,
)
from exogibbs.equilibrium.condensate.setup import (
    build_condensate_chemical_setup,
)
from exogibbs.equilibrium.condensate.types import AcceptedCondensateState
from exogibbs.thermo.models import ChemicalSetup


def test_result_builder_only_formats_an_accepted_state() -> None:
    gas_setup = ChemicalSetup(
        formula_matrix=jnp.asarray([[1.0]]),
        hvector_func=lambda temperature: jnp.asarray([0.0]),
        elements=("H",),
        species=("H",),
    )
    condensate_setup = ChemicalSetup(
        formula_matrix=jnp.asarray([[2.0]]),
        hvector_func=lambda temperature: jnp.asarray([0.0]),
        elements=("H",),
        species=("H2_s",),
    )
    setup = build_condensate_chemical_setup(
        gas_setup=gas_setup,
        condensate_setup=condensate_setup,
    )
    accepted_state = AcceptedCondensateState(
        gas_ln_n=jnp.asarray([0.0]),
        gas_n=jnp.asarray([1.0]),
        gas_x=jnp.asarray([1.0]),
        gas_ntot=jnp.asarray(1.0),
        condensate_amounts=jnp.asarray([0.25]),
        status="converged",
        acceptance_tier="fixed_support_v2_accepted",
        warning_messages=(),
        diagnostics={"gate_was_evaluated": True},
    )

    result = build_condensate_equilibrium_result(
        setup=setup,
        accepted_state=accepted_state,
        support_indices=(0,),
        selected_route="head_v2_fixed_support_lifecycle",
    )

    assert result.gas_ln_n is accepted_state.gas_ln_n
    assert result.condensate_amounts is accepted_state.condensate_amounts
    assert result.condensate_support_names == ("H2_s",)
    assert result.diagnostics["gate_was_evaluated"]
