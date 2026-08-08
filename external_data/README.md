# External validation data

## `Ito_2025.xlsx`

This workbook was provided by Yuichi Ito and is redistributed in this
repository with his permission for the ExoGibbs comparison with:

> Yuichi Ito, Tadahiro Kimura, Kazumasa Ohno, Yuka Fujii, and Masahiro Ikoma,
> “Monosilane Worlds: Sub-Neptunes with Atmospheres Shaped by Reduced Magma
> Oceans,” *The Astrophysical Journal* 987, 174 (2025),
> <https://doi.org/10.3847/1538-4357/add3fe>.

It is an author-supplied export for the published calculation, not an official
journal supplementary file. Scientific use should cite the paper. No new
license is asserted here for the workbook.

The file contains 856 ground-to-top layers for the `P > 10 bar` part of the
profile associated with Figure 2(a) of the paper. Layer 1 uses the separate
magma-contact equilibrium and water-solubility condition. Layers 2–856 use
the above-ground H–O–Si equilibrium and rainout system. The workbook contains
pressure, temperature, and gas fractions but does not contain local
condensate amounts or stored layer-by-layer elemental inventories. The
original `Fracton of O2` column spelling is preserved.

SHA256:

```text
5029bd874e813d3cd43407d551a2b693c4d0e52a8b0ab2f7f94a907b21e44bb1
```

The workbook is source-validation data under `external_data/`; it is not
installed as part of the `exogibbs` Python package.
