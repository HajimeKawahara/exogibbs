"""Plot ExoGibbs-only curated output for the solar water-condensation family."""

from __future__ import annotations

from pathlib import Path

from _curated_demo_common import plot_curated_family

FAMILY = "solar_water_condensation"
GAS_SPECIES = ("H2", "H1", "H2O1", "O1", "C1O1", "C1O2", "C1H4")
CONDENSATES = ("H2O(s,l)", "H2SO4.H2O(s,l)", "H2SO4.2H2O(s,l)", "O2S(OH)2(s,l)")


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
