"""A converged face can still omit a favorable pure phase."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from exogibbs.api import multiphase as mp

jax.config.update("jax_enable_x64", True)
R = 8.31446261815324


def problem():
    # One atom, two competing polymorphs. A bounded two-support search can
    # retain the initial phase before trying the lower-energy endpoint.
    phases = tuple(mp.Phase(name, (name,), [[1.]],
        lambda t, p, n, args, i=i: R*t*n[0]*args[i])
        for i, name in enumerate(("initial", "competitor")))
    return mp.Problem(("A",), phases)


def arguments():
    return dict(temperature_k=1200., pressure_pa=1e5,
        element_amounts_mol=np.array([3.]), init=np.array([3., 0.]),
        parameters=jnp.array([0., -1.]), options=mp.Options(max_iterations=200,max_supports=2))


def test_live_negative_pure_phase_is_inserted_and_history_retained():
    result = mp.solve_with_pure_phase_updates(problem(), **arguments())
    assert result.attempts[0].local_kkt_valid
    assert result.attempts[0].stability_status == "unstable"
    assert result.inserted_phases == ("competitor",)
    assert result.stop_reason == "no_negative_absent_pure_phase"
    assert result.result.stability_status == "certified"
    np.testing.assert_allclose(result.result.amounts_mol,[0.,3.],atol=1e-12)
    assert result.result.gibbs_j < result.attempts[0].gibbs_j
    np.testing.assert_allclose(result.result.conservation_residual_mol,0.,atol=1e-12)


def test_exhaustion_is_not_stability_and_standards_are_dynamic():
    data = arguments()
    limited = mp.solve_with_pure_phase_updates(problem(),max_updates=0,**data)
    assert limited.stop_reason == "update_limit"
    assert limited.result.stability_status == "unstable"
    assert len(limited.attempts) == 1
    data["parameters"] = jnp.array([0.,1.])
    stable = mp.solve_with_pure_phase_updates(problem(),**data)
    assert stable.inserted_phases == ()
    np.testing.assert_allclose(stable.result.amounts_mol,[3.,0.])


def test_update_limit_contract():
    for invalid in (-1,1.5,True):
        with pytest.raises(ValueError,match="max_updates"):
            mp.solve_with_pure_phase_updates(problem(),max_updates=invalid,**arguments())
