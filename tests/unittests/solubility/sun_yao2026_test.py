"""Independent Eq. 7 evaluation, units, domains and composition derivatives."""

import math

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from exogibbs.solubility import (
    SUN_YAO2026_WATER_METADATA,
    h2o_sun_yao2026,
    h2o_sun_yao2026_terms,
)


def reference(pw, t, p, na, al):
    """Literal scalar evaluation of the pinned Julia Eq. 7, in wt%."""
    sm = math.exp(.807805*math.log(pw)-.700336*math.log(t)
                  +2.949559e-6*p*p/t-74.546597*math.sqrt(p)*na/t+7146.275452*na/t)
    oh = math.exp(.449387*math.log(pw)-.360194*math.log(t)
                  +(-.308259+2.984703*al-1.693710*na)*p/t)
    return sm+oh


@pytest.mark.parametrize("pw,t,p,na,al", [
    (.22, 2173.15, 61.114, .002, .025),
    (26., 2173.15, 439.785, .002, .025),
    (1., 1673.15, 1., .03, .1),
    (500., 1473.15, 2000., .05, .1),
    (5000., 1923.15, 10000., 0., .05),
])
def test_matches_pinned_equation_in_mass_fraction(pw, t, p, na, al):
    actual = h2o_sun_yao2026(pw, t, p, x_na2o=na, x_al2o3=al)
    assert float(actual)*100 == pytest.approx(reference(pw, t, p, na, al), rel=2e-6)


def test_jit_broadcast_and_composition_response():
    fn = jax.jit(lambda v: h2o_sun_yao2026(v[0], v[1], v[2], x_na2o=v[3], x_al2o3=v[4]))
    x = jnp.array([20., 2173.15, 440., .02, .08])
    grad = jax.grad(fn)(x)
    eps = np.asarray(x)*1e-2
    fd = np.array([(float(fn(x+jnp.eye(5)[i]*eps[i]))-float(fn(x-jnp.eye(5)[i]*eps[i])))/(2*eps[i])
                   for i in range(5)])
    np.testing.assert_allclose(grad, fd, rtol=.003, atol=1e-7)
    assert grad[2] != 0  # Total pressure is not replaced by water pressure.
    values = h2o_sun_yao2026(jnp.array([[1.], [10.]]), jnp.array([1400., 1900., 2173.15]),
                           440., x_na2o=.02, x_al2o3=.08)
    assert values.shape == (2, 3)
    c, a = h2o_sun_yao2026_terms(x[1], x[2], x_na2o=x[3], x_al2o3=x[4])
    assert float(jnp.sum(c*x[0]**a)) == pytest.approx(float(fn(x)), rel=1e-6)


@pytest.mark.parametrize("pw,t,p,na,al", [
    (-1., 2000., 440., .02, .08), (441., 2000., 440., .02, .08),
    (1., 0., 440., .02, .08), (1., 2000., 0., .02, .08),
    (1., 2000., 440., -.02, .08), (1., 2000., 440., .8, .8),
    (np.inf, 2000., 440., .02, .08), (1., 2000., 440., np.nan, .08),
    (60000., 2173.15, 60000., .002, .025),
])
def test_invalid_inputs_and_nonphysical_extrapolation(pw, t, p, na, al):
    assert np.isnan(h2o_sun_yao2026(pw, t, p, x_na2o=na, x_al2o3=al))


def test_zero_boundary_metadata_and_no_calibration_clipping():
    assert h2o_sun_yao2026(0., 2173.15, 440., x_na2o=0., x_al2o3=0.) == 0.
    assert np.isfinite(h2o_sun_yao2026(1., 2300., 440., x_na2o=.01, x_al2o3=.1))
    assert SUN_YAO2026_WATER_METADATA.calibration_temperature_k == (933.15, 2197.15)
    assert "not a rectangular" in SUN_YAO2026_WATER_METADATA.notes
