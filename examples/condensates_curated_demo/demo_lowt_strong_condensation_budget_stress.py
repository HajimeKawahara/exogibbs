"""Plot ExoGibbs-only curated output for the low-temperature budget-stress family."""

from __future__ import annotations

from pathlib import Path

from _curated_demo_common import plot_curated_family

FAMILY = "lowT_strong_condensation_budget_stress"
GAS_SPECIES = ("H2", "H2O1", "C1O1", "Mg1", "Si1", "Fe1", "S1", "e1-")
CONDENSATES = ("H2O(s,l)", "MgSiO3(s,l)", "Mg2SiO4(s,l)", "Fe(s,l)", "FeS(s,l)", "SiO(s)")


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
