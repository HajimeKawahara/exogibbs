"""Check phase insertion sign, elemental gauge invariance and source bounds."""

from pathlib import Path
import runpy

import numpy as np
import pytest


ROOT = Path(__file__).resolve().parents[2]
STUDY = runpy.run_path(str(ROOT / "mceg/reviews/phosphorus_cold_phases_20261031.py"))


def test_phase_insertion_sign_and_reference_invariance():
    elements, composition = ["H", "O", "P"], {"H": 3, "O": 4, "P": 1}
    mu = np.array([[-1., -2., -3.]])
    g = np.array([-15.])
    result = STUDY["log_saturation"](mu, composition, elements, g)
    assert result[0] == pytest.approx(1.)  # Condensed insertion lowers Gibbs by RT.
    shift = np.array([2., -5., 7.])
    moved = STUDY["log_saturation"](mu+shift, composition, elements,
                                      g + shift @ np.array([3., 4., 1.]))
    np.testing.assert_allclose(moved, result)
    with pytest.raises(ValueError, match="element absent"):
        STUDY["log_saturation"](mu, {"Mg": 3}, elements, g)


def test_phase_polynomial_does_not_extrapolate_or_cross_phase_bounds():
    record = {"intervals": [{"bounds_K": [200., 315.5],
                             "coefficients": [0., 0., 3., 0., 0., 0., 0., -10., 2.]}]}
    t = np.array([199., 200., 298.15, 315.5, 316., np.nan])
    actual = STUDY["standard_gibbs_rt"](record, t)
    assert np.isnan(actual[[0, 4, 5]]).all()
    np.testing.assert_allclose(actual[1:4], 3.-10./t[1:4]-3.*np.log(t[1:4])-2.)
    assert STUDY["json_values"](actual)[0] is None
