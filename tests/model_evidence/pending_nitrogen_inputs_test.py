"""Check the source adapter without depending on any unmerged source packet."""

from copy import deepcopy
from pathlib import Path
import sys

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'mceg/reviews'))
import pending_nitrogen_inputs as inputs
from tio_candidate import _nasa


def synthetic_record():
    # Deliberately artificial inputs isolate units/sign/identity safeguards;
    # molecule-specific source proofs live with their own candidate packets.
    rows = {'AB': [12., -.3, 4.1, 1e-4, -2e-8, 3e-12, -4e-17, -55000., 7.],
            'A': [0., 0., 2.5, 0., 0., 0., 0., 9000., 2.],
            'B': [4., .1, 2.6, 2e-5, 0., 0., 0., 11000., 3.]}
    return {'molecule': 'AB', 'composition': {'A': 1, 'B': 1},
            'janaf_D0_J_mol': 800500., 'atct_D0_J_mol': 801375.,
            'delta_D0_J_mol': 875., 'nasa_D0_J_mol': 800000.,
            'original_coefficients': [12345., -2., 3., 1e-4, -1e-8],
            'nasa_same_identity': True,
            'nasa_records': {n: {'intervals': [{'bounds_K': [1000., 6000.],
                                                'coefficients': coefficients}]}
                             for n, coefficients in rows.items()}}


def test_retained_physical_shift_preserves_thermal_shape_and_calorics():
    record = synthetic_record()
    spec = inputs.retained_spec(record)
    assert spec['delta_GH_J_mol'] == -875.
    assert spec['coefficients'] == record['original_coefficients']
    old = deepcopy(record)
    old['delta_D0_J_mol'] = 0.
    for t in (1500., 1927.1, 2500., 3999.73, 4000.):
        before, after = [inputs.independent_properties(r, t, 'retained_energy') for r in (old, record)]
        for key in ('G_J_mol', 'H_J_mol'):
            assert after[key]-before[key] == pytest.approx(-875., rel=0, abs=1e-8)
        for key in ('S_J_mol_K', 'Cp_J_mol_K'):
            assert after[key] == before[key]


def test_nasa_native_energy_removal_and_si_insertion_are_separate():
    record = synthetic_record()
    spec = inputs.nasa_spec(record)
    assert spec['native_energy_removal_D0_over_R_K'] == 800000. / inputs.NASA_R
    assert spec['selected_energy_insertion_D0_over_R_K'] == 801375. / inputs.R
    for t in (1500., 1927.1, 2500., 3999.73, 4000.):
        g, slope, curvature = _nasa(t, spec['coefficients'], np)
        actual = np.asarray([inputs.R*g, inputs.R*(g-t*slope), -inputs.R*slope, -inputs.R*t*curvature])
        expected = np.asarray(list(inputs.independent_properties(record, t, 'nasa9').values()))
        np.testing.assert_allclose(actual, expected, rtol=0, atol=1e-8)
        assert abs(record['nasa_D0_J_mol']*(1-inputs.R/inputs.NASA_R)) > 1.
        with pytest.raises(AssertionError):
            np.testing.assert_allclose(actual+[1e-5, 0, 0, 0], expected, rtol=0, atol=1e-8)


def test_missing_zero_kelvin_or_incorrect_identity_cannot_be_silently_replaced():
    record = synthetic_record()
    record['janaf_D0_J_mol'] = None
    with pytest.raises(ValueError, match='missing JANAF'):
        inputs.retained_spec(record)
    record['nasa_same_identity'] = False
    with pytest.raises(ValueError, match='isomer-mixture'):
        inputs.nasa_spec(record)
    with pytest.raises(ValueError, match='outside'):
        inputs.independent_properties(synthetic_record(), 0., 'nasa9')
