from __future__ import annotations

from typing import Tuple, Union

import jax.numpy as jnp

Array = jnp.ndarray


def interp1d(xq: Array, x: Array, y: Array, method: str = "linear") -> Array:
    """JAX-compatible 1D interpolation for scalar or array queries."""
    xq = jnp.asarray(xq)
    x = jnp.asarray(x)
    y = jnp.asarray(y)
    if method == "linear":
        return jnp.interp(xq, x, y)
    if method == "nearest":
        idx = jnp.argmin(jnp.abs(x - xq[..., None]), axis=-1)
        return y[idx]
    raise NotImplementedError(f"Unsupported interpolation method: {method!r}")


def _bracket(axis: Array, query: Array, extrap: Union[bool, float, Tuple[object, ...]]):
    axis = jnp.asarray(axis)
    query = jnp.asarray(query)
    if axis.shape[0] == 1:
        index = jnp.asarray(0, dtype=jnp.int32)
        outside = (query != axis[0]) if extrap is False else jnp.asarray(False)
        return index, index, jnp.zeros_like(query), outside
    upper = jnp.searchsorted(axis, query, side="right")
    upper = jnp.clip(upper, 1, axis.shape[0] - 1)
    lower = upper - 1
    x0 = axis[lower]
    x1 = axis[upper]
    weight = (query - x0) / jnp.clip(x1 - x0, 1e-300)
    if extrap is False:
        outside = (query < axis[0]) | (query > axis[-1])
    else:
        outside = jnp.asarray(False)
        weight = jnp.clip(weight, 0.0, 1.0)
    return lower, upper, weight, outside


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
            raise NotImplementedError(f"Unsupported interpolation options: {sorted(kwargs)}")
        if method not in {"linear", "nearest"}:
            raise NotImplementedError(f"Unsupported interpolation method: {method!r}")
        self.x = jnp.asarray(x)
        self.y = jnp.asarray(y)
        self.z = jnp.asarray(z)
        self.f = jnp.asarray(f)
        self.method = method
        self.extrap = extrap

    def __call__(self, xq: Array, yq: Array, zq: Array, dx: int = 0, dy: int = 0, dz: int = 0) -> Array:
        if dx or dy or dz:
            raise NotImplementedError("Derivative interpolation is not implemented.")
        if self.method == "nearest":
            ix = jnp.argmin(jnp.abs(self.x - xq))
            iy = jnp.argmin(jnp.abs(self.y - yq))
            iz = jnp.argmin(jnp.abs(self.z - zq))
            return self.f[ix, iy, iz]
        return self._linear(xq, yq, zq)

    def _linear(self, xq: Array, yq: Array, zq: Array) -> Array:
        ix0, ix1, wx, ox = _bracket(self.x, xq, self.extrap)
        iy0, iy1, wy, oy = _bracket(self.y, yq, self.extrap)
        iz0, iz1, wz, oz = _bracket(self.z, zq, self.extrap)

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
        out = c0 * (1.0 - wz) + c1 * wz

        outside = ox | oy | oz
        return jnp.where(outside, jnp.full_like(out, jnp.nan), out)
