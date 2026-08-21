"""Pure-component fugacity corrections with ExoEOS
====================================================

This example connects the optional `ExoEOS <https://github.com/HajimeKawahara/exoeos>`_
package to the gas-equilibrium solver through ``lnphi_func``.  ExoGibbs keeps
pressure in bar, while ExoEOS expects Pa, so the unit conversion belongs in the
adapter.  The callback returns natural-log, dimensionless fugacity coefficients
in exactly the same order as ``ChemicalSetup.species``.

The seven-species setup below deliberately uses zero standard-state chemical
potentials.  It is a compact demonstration of the fugacity interface, not a
thermochemical abundance model.  Consequently, the difference between the two
solutions isolates the pure-component fugacity correction.
"""

from __future__ import annotations

from typing import Any, Optional

import jax
from jax import config
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np

from exogibbs.api.chemistry import ChemicalSetup
from exogibbs.api.gas import (
    EquilibriumOptions,
    LogFugacityCoefficientFunction,
    solve,
)
from exogibbs.interop.exoeos import make_pure_lnphi_func


config.update("jax_enable_x64", True)

SPECIES = ("CH4", "H2O", "CO2", "H2", "CO", "O2", "C2H6")
ELEMENTS = ("C", "H", "O")
FORMULA_MATRIX = jnp.asarray(
    [
        [1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 2.0],
        [4.0, 2.0, 0.0, 2.0, 0.0, 0.0, 6.0],
        [0.0, 1.0, 2.0, 0.0, 1.0, 2.0, 0.0],
    ],
    dtype=jnp.float64,
)
ELEMENT_INVENTORY = jnp.asarray([1.0, 4.0, 2.0], dtype=jnp.float64)
TEMPERATURE_K = 1500.0
PRESSURES_BAR = jnp.geomspace(100.0, 5000.0, 6)

_EXOEOS_INSTALL_HINT = (
    "Install a current ExoEOS checkout with "
    "`python -m pip install -e /path/to/exoeos` to run this optional example."
)

try:
    from exoeos import ZhangDuanEOS
except ImportError as exc:
    ZhangDuanEOS = None
    _EXOEOS_IMPORT_ERROR: Optional[ImportError] = exc
else:
    _EXOEOS_IMPORT_ERROR = None


# %%
# Build a deliberately small ExoGibbs problem
# --------------------------------------------
#
# The thermochemical vector follows ``SPECIES`` order. Later, ``eos_by_species``
# explicitly labels each one-component EOS because ExoEOS models store numerical
# component arrays rather than species names.


def build_reduced_setup() -> ChemicalSetup:
    """Return the seven-species illustrative C-H-O setup."""

    def hvector_func(temperature: Any) -> jax.Array:
        dtype = jnp.result_type(jnp.asarray(temperature), jnp.float64)
        return jnp.zeros((len(SPECIES),), dtype=dtype)

    return ChemicalSetup(
        formula_matrix=FORMULA_MATRIX,
        hvector_func=hvector_func,
        elements=ELEMENTS,
        species=SPECIES,
        element_vector_reference=ELEMENT_INVENTORY,
        metadata={
            "source": "illustrative zero standard-state chemical potentials",
            "nonideal_backend": "optional ExoEOS Zhang-Duan 2009",
        },
    )


# %%
# Adapt the ExoEOS pure-component states
# --------------------------------------
#
# Each mapping value is a one-component EOS. The adapter evaluates ``x=[1]``,
# converts pressure from bar to Pa, and restores the source species order.


def solve_pressure_profile(
    setup: ChemicalSetup,
    pressures_bar: jax.Array,
    *,
    lnphi_func: Optional[LogFugacityCoefficientFunction] = None,
) -> jax.Array:
    """Solve independent layers and return their gas mole fractions."""

    options = EquilibriumOptions(epsilon_crit=1.0e-11, max_iter=1000)
    compositions = []
    for pressure_bar in np.asarray(pressures_bar, dtype=np.float64):
        result = solve(
            setup,
            TEMPERATURE_K,
            pressure_bar,
            ELEMENT_INVENTORY,
            Pref=1.0,
            options=options,
            lnphi_func=lnphi_func,
        )
        compositions.append(result.x)
    return jnp.stack(compositions)


# %%
# Compare ideal and pure-fugacity solutions
# -----------------------------------------


def main() -> None:
    """Run the optional ExoEOS comparison and draw the gallery figure."""

    if _EXOEOS_IMPORT_ERROR is not None:
        print(f"Skipping the ExoEOS fugacity example. {_EXOEOS_INSTALL_HINT}")
        print(f"Original import error: {_EXOEOS_IMPORT_ERROR}")
        return

    setup = build_reduced_setup()
    eos_by_species = {
        species: ZhangDuanEOS.from_species((species,)) for species in SPECIES
    }
    lnphi_func = make_pure_lnphi_func(
        source_species=setup.species,
        eos_by_species=eos_by_species,
        phase="vapor",
    )

    ideal_x = solve_pressure_profile(setup, PRESSURES_BAR)
    nonideal_x = solve_pressure_profile(
        setup,
        PRESSURES_BAR,
        lnphi_func=lnphi_func,
    )
    pure_lnphi = jax.vmap(
        lambda pressure_bar: lnphi_func(
            TEMPERATURE_K,
            pressure_bar,
            None,
        )
    )(PRESSURES_BAR)

    highest_pressure = float(PRESSURES_BAR[-1])
    print(f"T = {TEMPERATURE_K:.0f} K, P = {highest_pressure:.0f} bar")
    print("species      ideal x       pure-fugacity x       ln(phi_pure)")
    for index, species in enumerate(SPECIES):
        print(
            f"{species:>5s}  "
            f"{float(ideal_x[-1, index]):12.5e}  "
            f"{float(nonideal_x[-1, index]):17.5e}  "
            f"{float(pure_lnphi[-1, index]):13.6f}"
        )

    figure, axes = plt.subplots(1, 2, figsize=(11.0, 4.2))
    colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]
    for index, species in enumerate(SPECIES):
        color = colors[index % len(colors)]
        axes[0].plot(
            PRESSURES_BAR,
            ideal_x[:, index],
            linestyle="--",
            color=color,
            alpha=0.7,
        )
        axes[0].plot(
            PRESSURES_BAR,
            nonideal_x[:, index],
            label=species,
            color=color,
        )
        axes[1].plot(
            PRESSURES_BAR,
            pure_lnphi[:, index],
            label=species,
            color=color,
        )

    axes[0].plot([], [], "k--", label="ideal")
    axes[0].plot([], [], "k-", label="pure fugacity")
    axes[0].set_xscale("log")
    axes[0].set_yscale("log")
    axes[0].set_xlabel("Pressure (bar)")
    axes[0].set_ylabel("Gas mole fraction")
    axes[0].set_title("ExoGibbs equilibrium")
    axes[0].legend(fontsize=8, ncol=3)

    axes[1].set_xscale("log")
    axes[1].axhline(0.0, color="0.5", linewidth=0.8)
    axes[1].set_xlabel("Pressure (bar)")
    axes[1].set_ylabel(r"$\ln(\phi_i^{\mathrm{pure}})$")
    axes[1].set_title("Zhang-Duan pure-component correction")
    axes[1].legend(fontsize=8, ncol=2)
    figure.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
