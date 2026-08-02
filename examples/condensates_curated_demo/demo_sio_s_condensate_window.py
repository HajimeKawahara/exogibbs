"""Plot ExoGibbs-only curated output for the SiO(s) condensate family."""

from __future__ import annotations

from _curated_demo_common import curated_output_path, plot_curated_family

FAMILY = "SiO_s_condensate_window"
GAS_SPECIES = ("H2", "Si1", "O1", "H2O1", "C1O1", "Mg1", "Fe1")
CONDENSATES = ("SiO(s)", "SiO2(s,l)", "MgSiO3(s,l)", "Mg2SiO4(s,l)", "CaSiO3(s)")


def main() -> None:
    output_path = plot_curated_family(
        family=FAMILY,
        preferred_gas_species=GAS_SPECIES,
        preferred_condensates=CONDENSATES,
        output_path=curated_output_path(__file__),
        title_suffix="production fixed-support v2 profile",
    )
    print(f"wrote {output_path}")


if __name__ == "__main__":
    main()
