"""Plot ExoGibbs-only curated output for the near-phase-boundary support family."""

from __future__ import annotations

from _curated_demo_common import curated_output_path, plot_curated_family

FAMILY = "near_phase_boundary_support_sensitivity"
GAS_SPECIES = ("H2", "H2O1", "C1O1", "Mg1", "Si1", "Fe1", "Ca1", "Ti1")
CONDENSATES = ("MgSiO3(s,l)", "Mg2SiO4(s,l)", "Fe(s,l)", "CaTiO3(s)", "TiO2(s,l)")


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
