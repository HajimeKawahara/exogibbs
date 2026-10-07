"""Gas seed ownership at the production condensate lifecycle boundary."""

from types import SimpleNamespace

import jax.numpy as jnp
import numpy as np
import pytest

from exogibbs.equilibrium.condensate.setup import (
    build_condensate_chemical_setup,
)
from exogibbs.equilibrium.condensate.solve import condensate_equilibrium_profile
from exogibbs.equilibrium.condensate.types import CondensateEquilibriumInit
from exogibbs.thermo.models import ChemicalSetup


@pytest.mark.parametrize(
    "initializer_fields",
    (
        {},
        {"gas_ntot": 7.0},
        {"element_potential": [2.0, 3.0]},
        {"gas_ln_n": [1.0, 2.0]},
        {"gas_ln_n": [1.0, 2.0], "gas_ntot": 7.0},
    ),
    ids=("cold", "total_only", "potential_only", "gas_only", "gas_and_total"),
)
def test_activity_gas_solve_is_reused_without_replacing_explicit_state(
    monkeypatch: pytest.MonkeyPatch,
    initializer_fields,
) -> None:
    gas_setup = ChemicalSetup(
        formula_matrix=jnp.eye(2, dtype=jnp.float64),
        hvector_func=lambda temperature: jnp.zeros(2, dtype=jnp.float64),
        elements=("H", "O"),
        species=("H", "O"),
        metadata={},
    )
    condensate_setup = ChemicalSetup(
        formula_matrix=jnp.asarray([[1.0], [0.0]], dtype=jnp.float64),
        hvector_func=lambda temperature: jnp.asarray([-5.0], dtype=jnp.float64),
        elements=gas_setup.elements,
        species=("H[s]",),
        metadata={},
    )
    setup = build_condensate_chemical_setup(
        gas_setup=gas_setup, condensate_setup=condensate_setup
    )
    temperatures = [900.0, 1100.0]
    pressures = [3.0, 7.0]
    inventory = jnp.asarray([6.0, 4.0], dtype=jnp.float64)
    gas_amounts = (jnp.asarray([0.55, 0.35]), jnp.asarray([0.45, 0.4]))
    calls = []
    captured = {}

    def lnphi_func(temperature, pressure, composition):
        assert composition is None
        return jnp.asarray([0.1, 0.2]) * pressure

    def gas_equilibrium(gas_setup_arg, temperature, pressure, target, **kwargs):
        index = len(calls)
        assert gas_setup_arg is gas_setup
        assert temperature == temperatures[index]
        assert pressure == pressures[index]
        np.testing.assert_allclose(target, [0.6, 0.4])
        assert kwargs["Pref"] == 2.0
        assert kwargs["lnphi_func"] is lnphi_func
        assert kwargs["options"].epsilon_crit == 1.0e-10
        calls.append(kwargs)
        return SimpleNamespace(
            ln_n=jnp.log(gas_amounts[index]),
            ntot=jnp.sum(gas_amounts[index]),
        )

    class InitialStatesCaptured(RuntimeError):
        pass

    def capture_initial_states(**kwargs):
        captured["bucket"] = kwargs["buckets"][0]
        raise InitialStatesCaptured

    monkeypatch.setattr(
        "exogibbs.equilibrium.gas.solve.equilibrium", gas_equilibrium
    )
    monkeypatch.setattr(
        "exogibbs.equilibrium.condensate.fixed_support.batch."
        "run_fixed_support_profile",
        capture_initial_states,
    )
    initial = CondensateEquilibriumInit(**initializer_fields)
    with pytest.raises(InitialStatesCaptured):
        condensate_equilibrium_profile(
            setup,
            T=temperatures,
            P=pressures,
            b=inventory,
            Pref=2.0,
            init=(initial, None),
            lnphi_func=lnphi_func,
        )

    assert len(calls) == 2
    bucket = captured["bucket"]
    expected_q = np.log(np.asarray(gas_amounts))
    expected_qtot = np.log(np.sum(np.asarray(gas_amounts), axis=1))
    if initial.gas_ln_n is not None:
        expected_q[0] = np.asarray(initial.gas_ln_n) - np.log(10.0)
        expected_qtot[0] = np.log(
            initial.gas_ntot / 10.0
            if initial.gas_ntot is not None
            else np.sum(np.exp(expected_q[0]))
        )
    np.testing.assert_allclose(bucket.ln_nk_init, expected_q)
    np.testing.assert_allclose(bucket.ln_ntot_init, expected_qtot)
    if initial.element_potential is not None:
        np.testing.assert_array_equal(
            bucket.element_potential_init[0], initial.element_potential
        )
    if initial.gas_ln_n is not None and initial.gas_ntot is not None:
        np.testing.assert_allclose(calls[0]["init"].ln_nk, expected_q[0])
        assert float(calls[0]["init"].ln_ntot) == pytest.approx(expected_qtot[0])
    else:
        assert calls[0]["init"] is None
    assert calls[1]["init"] is None
