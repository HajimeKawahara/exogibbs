"""Plot the stable phases from a carbon-rich CaS/MgS/AlN-seeded profile."""

from __future__ import annotations

from _curated_demo_common import curated_output_path, plot_curated_family

FAMILY = "carbon_rich_CaS_MgS_AlN_window"
DISPLAY_NAME = "Carbon-rich C/Fe/Mg-silicate/sulfide profile"
GAS_SPECIES = ("H2", "C1O1", "C1H4", "Ca1", "Mg1", "Al1", "N1", "S1")
CONDENSATES = (
    "C(s)",
    "Fe(s,l)",
    "MgSiO3(s,l)",
    "FeS(s,l)",
    "MgAl2O4(s,l)",
    "MnS(s)",
)


def main() -> None:
    output_path = plot_curated_family(
        family=FAMILY,
        display_name=DISPLAY_NAME,
        preferred_gas_species=GAS_SPECIES,
        preferred_condensates=CONDENSATES,
        output_path=curated_output_path(__file__),
        title_suffix="C/O = 2; CaS/MgS/AlN seed support",
    )
    print(f"wrote {output_path}")


if __name__ == "__main__":
    main()
