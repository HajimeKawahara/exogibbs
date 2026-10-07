from __future__ import annotations

import jax
import jax.numpy as jnp
import pytest

from exogibbs.utils.interpolation import Interpolator3D, interp1d


def _affine_interpolator(method: str, extrap):
    axis = jnp.asarray([0.0, 1.0])
    x, y, z = jnp.meshgrid(axis, axis, axis, indexing="ij")
    values = 2.0 * x + 3.0 * y + 5.0 * z
    return Interpolator3D(
        axis,
        axis,
        axis,
        values,
        method=method,
        extrap=extrap,
    )


@pytest.mark.parametrize("method", ("linear", "nearest"))
def test_interp1d_returns_nan_outside_bounds(method: str) -> None:
    x = jnp.asarray([0.0, 1.0])
    values = jnp.asarray([2.0, 4.0])

    result = interp1d(jnp.asarray([-1.0, 0.0, 1.0, 2.0]), x, values, method)

    assert jnp.isnan(result[0])
    assert result[1] == pytest.approx(2.0)
    assert result[2] == pytest.approx(4.0)
    assert jnp.isnan(result[3])


@pytest.mark.parametrize("method", ("linear", "nearest"))
def test_interp1d_rejects_mismatched_knots_and_values(method: str) -> None:
    with pytest.raises(ValueError, match="match the first dimension"):
        interp1d(
            jnp.asarray(0.5),
            jnp.asarray([0.0, 1.0]),
            jnp.asarray([2.0]),
            method,
        )


@pytest.mark.parametrize("method", ("linear", "nearest"))
def test_interpolator3d_extrap_false_returns_nan(method: str) -> None:
    interpolator = _affine_interpolator(method, False)

    assert jnp.isnan(interpolator(-0.25, 0.2, 0.7))


@pytest.mark.parametrize("method", ("linear", "nearest"))
def test_interpolator3d_scalar_extrapolation_value_fills_outside(method: str) -> None:
    interpolator = _affine_interpolator(method, 99.0)

    assert interpolator(-0.25, 0.2, 0.7) == pytest.approx(99.0)


def test_interpolator3d_linear_extrap_true_uses_boundary_slope() -> None:
    interpolator = _affine_interpolator("linear", True)

    result = interpolator(-0.25, 0.2, 0.7)

    assert result == pytest.approx(3.6)


def test_interpolator3d_nearest_extrap_true_uses_nearest_endpoint() -> None:
    interpolator = _affine_interpolator("nearest", True)

    result = interpolator(-0.25, 0.2, 0.7)

    assert result == pytest.approx(5.0)


def test_interpolator3d_nearest_midpoint_tie_uses_upper_knot() -> None:
    interpolator = _affine_interpolator("nearest", False)

    assert interpolator(0.5, 0.5, 0.5) == pytest.approx(10.0)


@pytest.mark.parametrize("method", ("linear", "nearest"))
@pytest.mark.parametrize(
    ("query", "expected"),
    (
        ((-1.0, 0.2, 0.3), 10.0),
        ((2.0, 0.2, 0.3), 11.0),
        ((0.2, -1.0, 0.3), 20.0),
        ((0.2, 2.0, 0.3), 21.0),
        ((0.2, 0.3, -1.0), 30.0),
        ((0.2, 0.3, 2.0), 31.0),
    ),
)
def test_interpolator3d_nested_extrapolation_fills_each_axis_side(
    method: str,
    query: tuple[float, float, float],
    expected: float,
) -> None:
    extrap = ((10.0, 11.0), (20.0, 21.0), (30.0, 31.0))
    interpolator = _affine_interpolator(method, extrap)

    assert interpolator(*query) == pytest.approx(expected)


def test_interpolator3d_extrapolation_pair_applies_to_every_axis() -> None:
    interpolator = _affine_interpolator("linear", (10.0, 11.0))

    assert interpolator(0.2, -1.0, 0.3) == pytest.approx(10.0)
    assert interpolator(0.2, 0.3, 2.0) == pytest.approx(11.0)


def test_interpolator3d_applies_multiaxis_fill_in_xyz_order() -> None:
    extrap = ((10.0, 11.0), (20.0, 21.0), (30.0, 31.0))
    interpolator = _affine_interpolator("linear", extrap)

    assert interpolator(-1.0, -1.0, -1.0) == pytest.approx(30.0)


@pytest.mark.parametrize(
    "extrap",
    ((), (1.0,), (1.0, 2.0, 3.0), ((1.0, 2.0), (3.0, 4.0))),
)
def test_interpolator3d_rejects_invalid_extrapolation_shape(extrap) -> None:
    with pytest.raises(ValueError, match="extrap must"):
        _affine_interpolator("linear", extrap)


def test_interpolator3d_rejects_mismatched_value_shape() -> None:
    axis = jnp.asarray([0.0, 1.0])

    with pytest.raises(ValueError, match="leading dimensions"):
        Interpolator3D(
            axis,
            axis,
            axis,
            jnp.zeros((1, 2, 2)),
        )


def test_interpolator3d_preserves_singleton_axis_behavior() -> None:
    x = jnp.asarray([0.0])
    yz = jnp.asarray([0.0, 1.0])
    _, y, z = jnp.meshgrid(x, yz, yz, indexing="ij")
    values = 3.0 * y + 5.0 * z

    disabled = Interpolator3D(x, yz, yz, values, extrap=False)
    enabled = Interpolator3D(x, yz, yz, values, extrap=True)

    assert jnp.isnan(disabled(1.0, 0.2, 0.7))
    assert enabled(1.0, 0.2, 0.7) == pytest.approx(4.1)


def test_interpolator3d_extrapolation_is_jittable() -> None:
    interpolator = _affine_interpolator("linear", True)
    compiled = jax.jit(interpolator)

    assert compiled(-0.25, 0.2, 0.7) == pytest.approx(3.6)
