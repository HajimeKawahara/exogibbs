"""Sun & Yao (2026) water solubility, following the pinned MAGEMin port.

Equation 7 and Table 1 are transcribed from MAGEMin_C.jl commit
47e2d12bfec0f799acf030b285fa3b1e8eb32098, julia/TE_saturation_models.jl
(GPL-3.0; ComputationalThermodynamics contributors). The original MATLAB
program and experiment table have not been independently replayed here.
"""

from __future__ import annotations

from typing import Tuple

import jax.numpy as jnp
from jax.typing import ArrayLike

from .volatile import SolubilityMetadata


SUN_YAO2026_WATER_METADATA = SolubilityMetadata(
    species="H2O",
    output_quantity="total H2O equivalent (molecular H2O plus hydroxyl)",
    output_basis="mass_fraction",
    calibration_temperature_k=(933.15, 2197.15),
    calibration_total_pressure_gpa=(1.e-4, 6.),
    experimental_reference="Sun & Yao (2026), 2090 joint H2O-CO2 experiments",
    experimental_doi="10.1016/j.epsl.2026.120115",
    formulation_reference="Sun & Yao (2026), Eq. 7 and Table 1",
    formulation_doi="10.1016/j.epsl.2026.120115",
    implementation_reference=(
        "MAGEMin_C.jl 47e2d12bfec0f799acf030b285fa3b1e8eb32098, "
        "julia/TE_saturation_models.jl, volatile_saturation_SY26"
    ),
    implementation_doi="",
    notes=(
        "The ranges describe the entire joint dataset, not a rectangular "
        "water-only calibration domain. Dry oxide mole fractions are required. "
        "Partial pressure is not fugacity. H2/He-rich fluids and BSE at 2173 K "
        "are not independently validated. No CO2 branch or experiment replay."
    ),
)

# Eq. 7 / Table 1 as reported by the pinned public implementation.
_A = (.807805, -.700336, 2.949559e-6, -74.546597, 7146.275452,
      .449387, -.360194, -.308259, 2.984703, -1.693710)


def h2o_sun_yao2026_terms(
    temperature_k: ArrayLike,
    melt_pressure_bar: ArrayLike,
    *,
    x_na2o: ArrayLike,
    x_al2o3: ArrayLike,
) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """Return mass-fraction capacities C and exponents a for w=sum(C*p**a).

    The final axis is (molecular H2O, hydroxyl expressed as H2O); p is H2O
    partial pressure / 1 bar. The oxide fractions are on a volatile-free
    mole basis. Callers normalize their full dry oxide inventory; this
    function neither renormalizes Na+Al nor specifies an iron-redox basis.
    Capacities retain T, total-P, and composition derivatives for consumers
    that integrate this empirical isotherm. This function supplies no Gibbs
    energy and does not establish multicomponent thermodynamic consistency.

    Finite positive T/P and nonnegative Na/Al with Na+Al<=1 are required.
    Invalid inputs return NaN; calibration ranges are not clipped.
    """
    t, p, na, al = jnp.broadcast_arrays(
        temperature_k, melt_pressure_bar, x_na2o, x_al2o3)
    a = _A
    molecular = a[1]*jnp.log(t) + a[2]*p*p/t + (a[3]*jnp.sqrt(p)+a[4])*na/t
    hydroxyl = a[6]*jnp.log(t) + (a[7]+a[8]*al+a[9]*na)*p/t
    valid = (jnp.isfinite(t) & jnp.isfinite(p) & jnp.isfinite(na) & jnp.isfinite(al)
             & (t > 0) & (p > 0) & (na >= 0) & (al >= 0) & (na+al <= 1))
    # The source reports wt%, including the OH contribution as H2O equivalent.
    capacity = jnp.exp(jnp.stack((molecular, hydroxyl), axis=-1))/100.
    return jnp.where(valid[..., None], capacity, jnp.nan), jnp.array([a[0], a[5]])


def h2o_sun_yao2026(
    water_partial_pressure_bar: ArrayLike,
    temperature_k: ArrayLike,
    melt_pressure_bar: ArrayLike,
    *,
    x_na2o: ArrayLike,
    x_al2o3: ArrayLike,
) -> jnp.ndarray:
    """Return total dissolved H2O-equivalent mass fraction (not wt%).

    T is in K; both pressures are in bar. Total melt pressure remains a
    separate input from water partial pressure. The binary dry-host/water
    basis is mass_H2O/(mass_dry+mass_H2O); additional dissolved species need
    an explicit consumer convention. Zero water pressure gives zero, with
    a singular fractional-power derivative. Negative/nonfinite pressure,
    partial pressure above total pressure, or predicted w>=1 returns NaN.
    No empirical extrapolation is clipped into a physical result.
    """
    c, a = h2o_sun_yao2026_terms(temperature_k, melt_pressure_bar,
                               x_na2o=x_na2o, x_al2o3=x_al2o3)
    p = jnp.asarray(water_partial_pressure_bar)
    value = jnp.sum(c*p[..., None]**a, axis=-1)
    valid = (jnp.isfinite(p) & (p >= 0) & (p <= jnp.asarray(melt_pressure_bar))
             & jnp.isfinite(value) & (value < 1))
    return jnp.where(valid, value, jnp.nan)
