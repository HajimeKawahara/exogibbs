"""Plot ExoGibbs-only curated output for the solar metal-sulfide and Fe/Ni/S family."""

from __future__ import annotations

from pathlib import Path

from _curated_demo_common import plot_curated_family

FAMILY = "solar_metal_sulfide_or_Fe_Ni_S_region"
GAS_SPECIES = ("H2", "H2S1", "S1", "Fe1", "Ni1", "Mg1", "C1O1", "H2O1")
CONDENSATES = ("Fe(s,l)", "FeS(s,l)", "FeS2(s)", "Ni(s,l)", "NiS(s,l)", "Ni3S2(s,l)", "MgS(s)")


def main() -> None:
    output_path = plot_curated_family(
        family=FAMILY,
        preferred_gas_species=GAS_SPECIES,
        preferred_condensates=CONDENSATES,
        output_path=Path(__file__).with_suffix(".png"),
        title_suffix="production fixed-support v2 profile",
    )
    print(f"wrote {output_path}")


if __name__ == "__main__":
    main()
