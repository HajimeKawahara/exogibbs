"""Plot support-selection output for the high-temperature gas regression family."""

from __future__ import annotations

from pathlib import Path

from _support_select_demo_common import plot_support_select_family

FAMILY = "solar_highT_no_condensate_gas_regression"
GAS_SPECIES = ("H2", "H1", "H2O1", "C1O1", "C1O2", "C1H4", "Mg1", "Si1", "Fe1")
CONDENSATES = ("Fe(s,l)", "TiN(s,l)", "Al6Si2O13(s)")


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
