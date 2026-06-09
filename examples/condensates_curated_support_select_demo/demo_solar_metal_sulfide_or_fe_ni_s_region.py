"""Plot support-selection output for the solar metal-sulfide family."""

from __future__ import annotations

from pathlib import Path

from _support_select_demo_common import plot_support_select_family

FAMILY = "solar_metal_sulfide_or_Fe_Ni_S_region"
GAS_SPECIES = ("H2", "H2S1", "S1", "Fe1", "Ni1", "Mg1", "C1O1", "H2O1")
CONDENSATES = ("Fe(s,l)", "FeS(s,l)", "Ni(s,l)", "NiS(s,l)", "MgSiO3(s,l)", "MnS(s)")


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
