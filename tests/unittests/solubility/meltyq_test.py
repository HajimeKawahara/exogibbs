"""Regression tests for the volatile-solubility laws adopted by MELTYQ."""

import math

import jax
import jax.numpy as jnp
import numpy as np
import pytest

import exogibbs.solubility as solubility
import exogibbs.solubility.meltyq as meltyq
from exogibbs.solubility import (
    MELTYQ_SOLUBILITY_METADATA,
    ch4_ardia2013,
    co2_lichtenberg2021,
    co_yoshioka2019,
    h2_hirschmann2012,
    h2o_lichtenberg2021,
    n2_dasgupta2022,
)


@pytest.mark.parametrize(
    ("calculated", "expected"),
    (
        (
            h2_hirschmann2012(1000.0, 1.0),
            0.0052200686751357186,
        ),
        (
            h2o_lichtenberg2021(1.0e7),
            0.010494680939906098,
        ),
        (
            co2_lichtenberg2021(1.0e8),
            0.0003101783014529868,
        ),
        (
            co_yoshioka2019(1.0),
            6.30957344480193e-8,
        ),
        (
            ch4_ardia2013(1.0, 1.0),
            7.263962399245182e-5,
        ),
        (
            n2_dasgupta2022(0.1, 2000.0, 4.0, -2.0),
            0.0028953374853364663,
        ),
    ),
)
def test_meltyq_laws_match_independent_reference_values(
    calculated: jnp.ndarray,
    expected: float,
) -> None:
    assert calculated.shape == ()
    np.testing.assert_allclose(calculated, expected, rtol=1.0e-6)


def test_n2_default_basalt_matches_explicit_composition() -> None:
    default = n2_dasgupta2022(0.1, 2000.0, 1.0, -2.0)
    explicit = n2_dasgupta2022(
        0.1,
        2000.0,
        1.0,
        -2.0,
        x_sio2=0.56,
        x_al2o3=0.11,
        x_tio2=0.01,
    )

    np.testing.assert_allclose(default, explicit, rtol=0.0, atol=0.0)


def test_laws_broadcast_array_inputs() -> None:
    fugacity = jnp.asarray([[100.0], [1000.0]])
    melt_pressure = jnp.asarray([0.7, 1.0, 3.0])

    calculated = h2_hirschmann2012(fugacity, melt_pressure)
    expected = np.asarray(fugacity) * np.exp(
        -11.403 - 0.76 * np.asarray(melt_pressure)
    )

    assert calculated.shape == (2, 3)
    np.testing.assert_allclose(calculated, expected, rtol=1.0e-6)


@pytest.mark.parametrize(
    "calculated",
    (
        h2_hirschmann2012(0.0, 1.0),
        h2o_lichtenberg2021(0.0),
        co2_lichtenberg2021(0.0),
        co_yoshioka2019(0.0),
        ch4_ardia2013(0.0, 1.0),
        n2_dasgupta2022(0.0, 2000.0, 1.0, 0.0),
    ),
)
def test_zero_driving_pressure_has_zero_solubility(
    calculated: jnp.ndarray,
) -> None:
    assert float(calculated) == 0.0


@pytest.mark.parametrize(
    "calculated",
    (
        h2_hirschmann2012(-1.0, 1.0),
        h2o_lichtenberg2021(-1.0),
        co2_lichtenberg2021(-1.0),
        co_yoshioka2019(-1.0),
        ch4_ardia2013(-1.0, 1.0),
        n2_dasgupta2022(-1.0, 2000.0, 1.0, 0.0),
    ),
)
def test_negative_driving_pressure_returns_nan(
    calculated: jnp.ndarray,
) -> None:
    assert math.isnan(float(calculated))


@pytest.mark.parametrize(
    "calculated",
    (
        h2_hirschmann2012(1.0, -1.0),
        h2o_lichtenberg2021(jnp.inf),
        co2_lichtenberg2021(jnp.nan),
        co_yoshioka2019(jnp.inf),
        ch4_ardia2013(1.0, -1.0),
        n2_dasgupta2022(0.1, 0.0, 1.0, 0.0),
        n2_dasgupta2022(0.1, 2000.0, -1.0, 0.0),
        n2_dasgupta2022(0.1, 2000.0, 1.0, jnp.nan),
        n2_dasgupta2022(0.1, 2000.0, 1.0, 0.0, x_sio2=-0.1),
        n2_dasgupta2022(0.1, 2000.0, 1.0, 0.0, x_sio2=1.1),
        n2_dasgupta2022(
            0.1,
            2000.0,
            1.0,
            0.0,
            x_sio2=0.8,
            x_al2o3=0.3,
        ),
    ),
)
def test_nonphysical_secondary_inputs_return_nan(
    calculated: jnp.ndarray,
) -> None:
    assert math.isnan(float(calculated))


@pytest.mark.parametrize(
    ("evaluator", "value"),
    (
        (lambda value: h2_hirschmann2012(value, 1.0), 1000.0),
        (h2o_lichtenberg2021, 1.0e7),
        (co2_lichtenberg2021, 1.0e8),
        (co_yoshioka2019, 1.0),
        (lambda value: ch4_ardia2013(value, 1.0), 1.0),
        (
            lambda value: n2_dasgupta2022(value, 2000.0, 4.0, -2.0),
            0.1,
        ),
    ),
)
def test_laws_support_jit_and_automatic_differentiation(
    evaluator,
    value: float,
) -> None:
    calculated = jax.jit(evaluator)(value)
    derivative = jax.grad(evaluator)(value)

    assert jnp.isfinite(calculated)
    assert jnp.isfinite(derivative)


@pytest.mark.parametrize(
    "evaluator",
    (
        h2o_lichtenberg2021,
        co_yoshioka2019,
        lambda value: n2_dasgupta2022(value, 2000.0, 1.0, 0.0),
    ),
)
def test_fractional_power_derivative_is_singular_at_zero(evaluator) -> None:
    assert jnp.isinf(jax.grad(evaluator)(0.0))


def test_metadata_records_native_bases_and_provenance() -> None:
    expected_bases = {
        "h2_hirschmann2012": "mole_fraction",
        "h2o_lichtenberg2021": "mass_fraction",
        "co2_lichtenberg2021": "mass_fraction",
        "co_yoshioka2019": "mass_fraction",
        "ch4_ardia2013": "mole_fraction",
        "n2_dasgupta2022": "mass_fraction",
    }

    assert {
        name: metadata.output_basis
        for name, metadata in MELTYQ_SOLUBILITY_METADATA.items()
    } == expected_bases
    assert all(
        metadata.experimental_doi
        and metadata.formulation_doi
        and metadata.implementation_doi
        for metadata in MELTYQ_SOLUBILITY_METADATA.values()
    )
    co_notes = MELTYQ_SOLUBILITY_METADATA["co_yoshioka2019"].notes
    ch4_notes = MELTYQ_SOLUBILITY_METADATA["ch4_ardia2013"].notes
    n2_notes = MELTYQ_SOLUBILITY_METADATA["n2_dasgupta2022"].notes
    assert "-7.2" in co_notes and "bar" in co_notes
    assert (
        MELTYQ_SOLUBILITY_METADATA["co_yoshioka2019"].output_quantity
        == "elemental C dissolved as CO"
    )
    assert "GPa" in ch4_notes
    assert "sqrt(melt pressure)" in n2_notes
    assert (
        MELTYQ_SOLUBILITY_METADATA["n2_dasgupta2022"].output_quantity
        == "total elemental N"
    )


def test_package_exports_are_explicit_and_unique() -> None:
    expected_exports = (
        "MELTYQ_SOLUBILITY_METADATA",
        "SolubilityMetadata",
        "ch4_ardia2013",
        "co2_lichtenberg2021",
        "co_yoshioka2019",
        "h2_hirschmann2012",
        "h2o_lichtenberg2021",
        "n2_dasgupta2022",
    )

    assert solubility.__all__ == expected_exports
    assert meltyq.__all__ == expected_exports


def test_metadata_mapping_is_read_only() -> None:
    with pytest.raises(TypeError):
        MELTYQ_SOLUBILITY_METADATA["new_law"] = MELTYQ_SOLUBILITY_METADATA[
            "h2_hirschmann2012"
        ]
