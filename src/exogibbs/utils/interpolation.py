from __future__ import annotations

from typing import Tuple, Union

import jax.numpy as jnp

Array = jnp.ndarray


def _is_bool(value: object) -> bool:
    return isinstance(value, bool) or (
        hasattr(value, "dtype") and value.dtype == bool
    )


def _parse_extrap(
    extrap: Union[bool, float, Tuple[object, ...]],
) -> Tuple[object, ...]:
    """Normalize extrapolation options to lower/upper policies per axis."""
    if _is_bool(extrap) or jnp.isscalar(extrap):
        return (extrap,) * 6
    try:
        extrap_length = len(extrap)
    except TypeError as exc:
        raise ValueError(
            "extrap must be a scalar, a (low, high) pair, or three "
            "(low, high) pairs."
        ) from exc
    if extrap_length == 2 and all(jnp.isscalar(value) for value in extrap):
        return tuple(value for _ in range(3) for value in extrap)
    if extrap_length == 3:
        try:
            is_axis_pair = all(len(axis_extrap) == 2 for axis_extrap in extrap)
        except TypeError:
            is_axis_pair = False
        if is_axis_pair:
            return tuple(value for axis_extrap in extrap for value in axis_extrap)
    raise ValueError(
        "extrap must be a scalar, a (low, high) pair, or three "
        "(low, high) pairs."
    )


def interp1d(xq: Array, x: Array, y: Array, method: str = "linear") -> Array:
    """JAX-compatible 1D interpolation for scalar or array queries."""
    xq = jnp.asarray(xq)
    x = jnp.asarray(x)
    y = jnp.asarray(y)
    if x.ndim != 1 or y.ndim == 0 or y.shape[0] != x.shape[0]:
        raise ValueError("x must be one-dimensional and match the first dimension of y.")
    if x.shape[0] == 0:
        raise ValueError("x and y must contain at least one knot.")
    if method == "linear":
        out = jnp.interp(xq, x, y)
    elif method == "nearest":
        idx = jnp.argmin(jnp.abs(x - xq[..., None]), axis=-1)
        out = y[idx]
    else:
        raise NotImplementedError(f"Unsupported interpolation method: {method!r}")
    outside = (xq < x[0]) | (xq > x[-1])
    return jnp.where(outside, jnp.nan, out)


def _bracket(axis: Array, query: Array):
    axis = jnp.asarray(axis)
    query = jnp.asarray(query)
    if axis.shape[0] == 1:
        index = jnp.asarray(0, dtype=jnp.int32)
        return index, index, jnp.zeros_like(query)
    upper = jnp.searchsorted(axis, query, side="right")
    upper = jnp.clip(upper, 1, axis.shape[0] - 1)
    lower = upper - 1
    x0 = axis[lower]
    x1 = axis[upper]
    delta = x1 - x0
    weight = jnp.where(delta == 0, 0.0, (query - x0) / delta)
    return lower, upper, weight


def _nearest_index(axis: Array, query: Array) -> Array:
    lower, upper, _ = _bracket(axis, query)
    lower_distance = jnp.abs(query - axis[lower])
    upper_distance = jnp.abs(axis[upper] - query)
    return jnp.where(lower_distance < upper_distance, lower, upper)


def _apply_axis_extrapolation(
    out: Array,
    query: Array,
    axis: Array,
    low: object,
    high: object,
) -> Array:
    for outside, policy in (
        (query < axis[0], low),
        (query > axis[-1], high),
    ):
        if _is_bool(policy) and bool(policy):
            continue
        fill_value = jnp.nan if _is_bool(policy) else policy
        out = jnp.where(outside, fill_value, out)
    return out


class Interpolator3D:
    """Minimal trilinear interpolator for scalar equilibrium grid lookups."""

    def __init__(
        self,
        x: Array,
        y: Array,
        z: Array,
        f: Array,
        method: str = "linear",
        extrap: Union[bool, float, Tuple[object, ...]] = False,
        **kwargs: object,
    ) -> None:
        if kwargs:
            raise NotImplementedError(
                f"Unsupported interpolation options: {sorted(kwargs)}"
            )
        if method not in {"linear", "nearest"}:
            raise NotImplementedError(f"Unsupported interpolation method: {method!r}")
        self.x = jnp.asarray(x)
        self.y = jnp.asarray(y)
        self.z = jnp.asarray(z)
        self.f = jnp.asarray(f)
        if any(axis.ndim != 1 for axis in (self.x, self.y, self.z)):
            raise ValueError("Interpolation axes must be one-dimensional.")
        if any(axis.shape[0] == 0 for axis in (self.x, self.y, self.z)):
            raise ValueError("Interpolation axes must contain at least one knot.")
        expected_shape = (self.x.shape[0], self.y.shape[0], self.z.shape[0])
        if self.f.ndim < 3 or self.f.shape[:3] != expected_shape:
            raise ValueError(
                "f must have leading dimensions matching the interpolation axes; "
                f"expected {expected_shape}, got {self.f.shape}."
            )
        self.method = method
        self.extrap = extrap
        self._extrap = _parse_extrap(extrap)

    def __call__(
        self,
        xq: Array,
        yq: Array,
        zq: Array,
        dx: int = 0,
        dy: int = 0,
        dz: int = 0,
    ) -> Array:
        if dx or dy or dz:
            raise NotImplementedError("Derivative interpolation is not implemented.")
        if self.method == "nearest":
            out = self.f[
                _nearest_index(self.x, xq),
                _nearest_index(self.y, yq),
                _nearest_index(self.z, zq),
            ]
        else:
            out = self._linear(xq, yq, zq)
        return self._apply_extrapolation(out, xq, yq, zq)

    def _linear(self, xq: Array, yq: Array, zq: Array) -> Array:
        ix0, ix1, wx = _bracket(self.x, xq)
        iy0, iy1, wy = _bracket(self.y, yq)
        iz0, iz1, wz = _bracket(self.z, zq)

        c000 = self.f[ix0, iy0, iz0]
        c001 = self.f[ix0, iy0, iz1]
        c010 = self.f[ix0, iy1, iz0]
        c011 = self.f[ix0, iy1, iz1]
        c100 = self.f[ix1, iy0, iz0]
        c101 = self.f[ix1, iy0, iz1]
        c110 = self.f[ix1, iy1, iz0]
        c111 = self.f[ix1, iy1, iz1]

        c00 = c000 * (1.0 - wx) + c100 * wx
        c01 = c001 * (1.0 - wx) + c101 * wx
        c10 = c010 * (1.0 - wx) + c110 * wx
        c11 = c011 * (1.0 - wx) + c111 * wx
        c0 = c00 * (1.0 - wy) + c10 * wy
        c1 = c01 * (1.0 - wy) + c11 * wy
        return c0 * (1.0 - wz) + c1 * wz

    def _apply_extrapolation(
        self,
        out: Array,
        xq: Array,
        yq: Array,
        zq: Array,
    ) -> Array:
        lowx, highx, lowy, highy, lowz, highz = self._extrap
        out = _apply_axis_extrapolation(out, xq, self.x, lowx, highx)
        out = _apply_axis_extrapolation(out, yq, self.y, lowy, highy)
        return _apply_axis_extrapolation(out, zq, self.z, lowz, highz)
