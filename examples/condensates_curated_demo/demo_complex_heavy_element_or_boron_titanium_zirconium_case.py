"""Plot ExoGibbs-only curated output for the heavy-element oxide family."""

from __future__ import annotations

from pathlib import Path

from _curated_demo_common import plot_curated_family

FAMILY = "complex_heavy_element_or_boron_titanium_zirconium_case"
GAS_SPECIES = ("H2", "H2O1", "C1O1", "Ti1", "V1", "Cr1", "Fe1", "Al1")
CONDENSATES = ("TiO(s,l)", "TiO2(s,l)", "Ti2O3(s,l)", "Ti3O5(s,l)", "CaTiO3(s)", "Cr2O3(s,l)", "Al2O3(s,l)")


def main() -> None:
    output_path = plot_curated_family(
        family=FAMILY,
        preferred_gas_species=GAS_SPECIES,
        preferred_condensates=CONDENSATES,
        output_path=Path(__file__).with_suffix(".png"),
        title_suffix="HEAD route curated rows",
    )
    print(f"wrote {output_path}")


if __name__ == "__main__":
    main()
