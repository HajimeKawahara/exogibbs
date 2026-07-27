"""Plot ExoGibbs-only curated output for the carbon-rich CaS/MgS/AlN family."""

from __future__ import annotations

from pathlib import Path

from _curated_demo_common import plot_curated_family

FAMILY = "carbon_rich_CaS_MgS_AlN_window"
GAS_SPECIES = ("H2", "C1O1", "C1H4", "Ca1", "Mg1", "Al1", "N1", "S1")
CONDENSATES = ("CaS(s)", "MgS(s)", "AlN(s)", "C(s)", "SiC(s)")


def main() -> None:
    output_path = plot_curated_family(
        family=FAMILY,
        preferred_gas_species=GAS_SPECIES,
        preferred_condensates=CONDENSATES,
        output_path=Path(__file__).with_suffix(".png"),
        title_suffix="production fixed-support v2 profile, C/O = 2",
    )
    print(f"wrote {output_path}")


if __name__ == "__main__":
    main()
