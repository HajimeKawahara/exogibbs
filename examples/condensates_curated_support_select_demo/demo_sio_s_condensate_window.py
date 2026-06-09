"""Plot support-selection output for the SiO(s) condensate family."""

from __future__ import annotations

from pathlib import Path

from _support_select_demo_common import plot_support_select_family

FAMILY = "SiO_s_condensate_window"
GAS_SPECIES = ("H2", "Si1", "O1", "H2O1", "C1O1", "Mg1", "Fe1")
CONDENSATES = ("SiO(s)", "MgSiO3(s,l)", "Mg2SiO4(s,l)", "Fe(s,l)", "TiO2(s,l)")


def main() -> None:
    output_path, summary_path = plot_support_select_family(
        family=FAMILY,
        preferred_gas_species=GAS_SPECIES,
        preferred_condensates=CONDENSATES,
        output_path=Path(__file__).with_suffix(".png"),
        title_suffix="native support selection",
    )
    print(f"wrote {output_path}")
    print(f"wrote {summary_path}")


if __name__ == "__main__":
    main()
