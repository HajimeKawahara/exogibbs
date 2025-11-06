import math

import jax.numpy as jnp
import pytest


def _logk(temp: float, coeffs) -> float:
    a1, a2, a3, a4, a5 = coeffs
    return a1 / temp + a2 * math.log(temp) + a3 + a4 * temp + a5 * temp * temp


def test_chemsetup_piecewise_segments():
    from exogibbs.presets.fastchem_cond import chemsetup

    cond = chemsetup()

    assert cond.species is not None
    species = list(cond.species)
    assert "Al(s)" in species

    target = "AlCl3(s,l)"
    assert target in species
    idx = species.index(target)

    coeff_low = (
        1.6752020623095072e05,
        -7.5508095875449666,
        -2.6523945119240171e01,
        2.2745417658541090e-02,
        -1.0189579863311542e-05,
    )
    coeff_high = (
        1.6529023277388894e05,
        2.4647025638091016,
        -7.5718997854319525e01,
        1.9312847893898392e-03,
        -2.8258846125321814e-07,
    )

    temp_low = 400.0
    temp_high = 600.0

    hv_low = cond.hvector_func(temp_low)
    hv_high = cond.hvector_func(temp_high)

    expected_low = -_logk(temp_low, coeff_low)
    expected_high = -_logk(temp_high, coeff_high)

    assert hv_low[idx].item() == pytest.approx(expected_low, rel=1e-5)
    assert hv_high[idx].item() == pytest.approx(expected_high, rel=1e-5)

    hv_stack = cond.hvector_func(jnp.array([temp_low, temp_high]))
    assert hv_stack.shape == (2, len(species))
    assert hv_stack[0, idx].item() == pytest.approx(expected_low, rel=1e-5)
    assert hv_stack[1, idx].item() == pytest.approx(expected_high, rel=1e-5)
