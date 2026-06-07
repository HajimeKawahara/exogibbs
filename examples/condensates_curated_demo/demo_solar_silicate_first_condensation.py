"""Plot ExoGibbs-only curated output for the solar silicate first-condensation family."""

from __future__ import annotations

from pathlib import Path

from _curated_demo_common import plot_curated_family

FAMILY = "solar_silicate_first_condensation"
GAS_SPECIES = ("H2", "H2O1", "C1O1", "Mg1", "Si1", "O1", "Fe1")
CONDENSATES = ("Al6Si2O13(s)", "MgSiO3(s,l)", "Mg2SiO4(s,l)", "SiO2(s,l)", "Fe2SiO4(s)")


def main() -> None:
    output_path = plot_curated_family(
        family=FAMILY,
        preferred_gas_species=GAS_SPECIES,
        preferred_condensates=CONDENSATES,
        output_path=Path(__file__).with_suffix(".png"),
        title_suffix="fresh HEAD route profile",
    )
    print(f"wrote {output_path}")


if __name__ == "__main__":
    main()
