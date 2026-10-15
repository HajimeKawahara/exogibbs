"""Optional native-provider full-G parity, basis mapping and derivatives."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from scipy.optimize import brentq

from exogibbs.api import multiphase as mp
from exogibbs.equilibrium.multiphase.types import R_SCALE as R
from exogibbs.interop.exoeos import make_magma_liquid_phase, make_magma_pure_phase


@pytest.fixture
def magma():
    return pytest.importorskip("exoeos.magma")


def test_native_liquid_and_pure_phase_full_g_units_derivatives(magma):
    phase = make_magma_liquid_phase(elements=("Si", "O", "Mg", "He", "C"),
                                    components=("Mg2SiO4", "SiO2"))
    np.testing.assert_array_equal(phase.atom_matrix,
                                   [[1, 1], [4, 2], [2, 0], [0, 0], [0, 0]])
    state = jnp.array([2000., 5e7, .75, .25])

    def energy(state):
        return phase.gibbs_j(state[0], state[1], state[2:], ())

    np.testing.assert_allclose(jax.jit(energy)(state),
                               magma.liquid_gibbs(state[0], state[1], state[2:][::-1]), rtol=1e-14)
    derivatives = jax.grad(energy)(state)
    for index, step in enumerate([.01, 100., 1e-5, 1e-5]):
        estimates = []
        for factor in (1., .5):
            delta = jnp.zeros(4).at[index].set(step * factor)
            estimates.append((energy(state + delta) - energy(state - delta)) / (2 * step * factor))
        np.testing.assert_allclose(estimates[-1], derivatives[index], rtol=2e-7, atol=1e-6)
    n = state[2:]
    mu = jax.grad(lambda n: phase.gibbs_j(state[0], state[1], n, ()))(n)
    np.testing.assert_allclose(n @ mu, energy(state), rtol=1e-14)
    np.testing.assert_allclose(phase.gibbs_j(state[0], state[1], 1e24 * n, ()),
                               1e24 * energy(state), rtol=1e-14)
    crystal = make_magma_pure_phase(elements=("Mg", "Si", "O"), component="Mg2SiO4",
                                    molar_gibbs_j=magma.forsterite_gibbs, name="forsterite")
    np.testing.assert_allclose(crystal.gibbs_j(2000., 5e7, jnp.array([3.]), ()),
                               3 * magma.forsterite_gibbs(2000., 5e7), rtol=1e-14)


def test_native_fixed_support_matches_independent_same_scalar_reference(magma):
    elements = ("Mg", "Si", "O")
    liquid = make_magma_liquid_phase(elements=elements, components=magma.COMPONENTS)
    solid = make_magma_pure_phase(elements=elements, component="Mg2SiO4",
                                 molar_gibbs_j=magma.forsterite_gibbs, name="forsterite")
    problem = mp.Problem(elements, (liquid, solid))

    # One independent reaction coordinate; use the analytic binary potential
    # rather than the new optimizer's AD gradient or private kernel.
    def reaction(crystal, t, p):
        x_f = (.75 - crystal) / (1. - crystal)
        mu_f = magma.liquid_standard_gibbs(t, p)[1] + magma.R * t * np.log(x_f) + magma.W * (1 - x_f)**2
        return float(magma.forsterite_gibbs(t, p) - mu_f)

    solve = jax.jit(lambda t, p, b, n: mp.solve_fixed_support(
        problem, temperature_k=t, pressure_pa=p, element_amounts_mol=b,
        init=n, support=mp.Support((0, 1, 2)),
    ))
    for temperature, pressure in [(2000., 1e5), (1900., 5e7)]:
        crystal = brentq(reaction, 0., .75 - 1e-10, args=(temperature, pressure), xtol=1e-14)
        reference = jnp.array([.25, .75 - crystal, crystal])
        result = solve(temperature, pressure, jnp.array([1.5, 1., 3.5]),
                        jnp.array([.25, .45, .3]))
        assert result.converged and result.local_kkt_valid and result.minimum_valid
        np.testing.assert_allclose(result.amounts_mol, reference, atol=1e-9)
        reference_g = (magma.liquid_gibbs(temperature, pressure, reference[:2])
                       + crystal * magma.forsterite_gibbs(temperature, pressure))
        assert abs(float(result.gibbs_j - reference_g)) / (R * temperature * 6) < 1e-9
        assert result.conservation_error < 1e-9 and result.stationarity_error < 1e-8
        np.testing.assert_allclose(result.chemical_potentials_j_mol[1],
                                   result.chemical_potentials_j_mol[2], atol=1e-6)


def test_hydrous_basis_is_not_old_h2_he_recipe_and_rejects_missing_atoms(magma):
    phase = make_magma_liquid_phase(elements=magma.ELEMENTS + ("He",),
                                    components=magma.HYDROUS_COMPONENTS)
    n = jnp.linspace(.01, .12, 12)
    np.testing.assert_allclose(phase.gibbs_j(2000., 1e7, n, ()),
                               magma.liquid_gibbs(2000., 1e7, n), rtol=1e-14)
    assert np.all(np.asarray(phase.atom_matrix)[-1] == 0)
    with pytest.raises(ValueError, match="omits"):
        make_magma_liquid_phase(elements=("Mg", "Si"), components=magma.COMPONENTS)
    with pytest.raises(ValueError, match="complete supported"):
        make_magma_liquid_phase(elements=magma.ELEMENTS, components=("SiO2", "H2O"))


def test_missing_provider_does_not_break_core_import(monkeypatch):
    import sys

    monkeypatch.setitem(sys.modules, "exoeos", None)
    with pytest.raises(ImportError, match="require ExoEOS magma"):
        make_magma_liquid_phase(elements=("Mg", "Si", "O"), components=("SiO2", "Mg2SiO4"))
