"""Pinned JANAF atomic G=Hf(298)+delta H-T*S with JAX Hermite interpolation."""

import json
from pathlib import Path

import jax
import jax.numpy as jnp
from jax.typing import ArrayLike
import numpy as np

DATA = json.loads((Path(__file__).parents[1] / "data/janaf/atomic.json").read_text())
ELEMENTS = tuple(DATA["rows"])


def standard_potentials_rt(temperature_k: ArrayLike) -> jax.Array:
    """Return atomic standards in ELEMENTS order, only over 2100--2400 K.

    Cubic Hermite G uses the measured/tabulated -S slopes. It does not
    extrapolate or replace G by a temperature-dependent formation energy.
    """
    t = jnp.asarray(temperature_k)
    values = []
    for row in DATA["rows"].values():
        grid, entropy, _, dh = np.asarray(row["data"]).T
        energy = row["formation_enthalpy_298_kJ_mol"]*1000+dh*1000-grid*entropy
        index = jnp.clip(jnp.searchsorted(jnp.asarray(grid), t, side="right")-1, 0, len(grid)-2)
        width = jnp.asarray(np.diff(grid))[index]
        z = (t-jnp.asarray(grid)[index])/width
        g0, g1 = jnp.asarray(energy)[index], jnp.asarray(energy)[index+1]
        m0, m1 = -jnp.asarray(entropy)[index], -jnp.asarray(entropy)[index+1]
        value = (2*z**3-3*z**2+1)*g0+(z**3-2*z**2+z)*width*m0
        value += (-2*z**3+3*z**2)*g1+(z**3-z**2)*width*m1
        values.append(value/(DATA["common_R_J_mol_K"]*t))
    lower, upper = DATA["evaluation_temperature_interval_K"]
    return jnp.where((t >= lower) & (t <= upper), jnp.stack(values), jnp.nan)
