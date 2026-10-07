"""Plot the stable phases from a solar SiO(s)-seeded profile."""

from __future__ import annotations

from _curated_demo_common import curated_output_path, plot_curated_family

FAMILY = "SiO_s_condensate_window"
DISPLAY_NAME = "Solar Mg-silicate/Fe/feldspar profile"
GAS_SPECIES = ("H2", "Si1", "O1", "H2O1", "C1O1", "Mg1", "Fe1")
CONDENSATES = (
    "MgSiO3(s,l)",
    "Mg2SiO4(s,l)",
    "Fe(s,l)",
    "CaMgSi2O6(s)",
    "NaAlSi3O8(s)",
    "CaSiO3(s)",
)


def main() -> None:
    output_path = plot_curated_family(
        family=FAMILY,
        display_name=DISPLAY_NAME,
        preferred_gas_species=GAS_SPECIES,
        preferred_condensates=CONDENSATES,
        output_path=curated_output_path(__file__),
        title_suffix="SiO(s) seed support; equilibrium phases shown",
    )
    print(f"wrote {output_path}")


if __name__ == "__main__":
    main()
