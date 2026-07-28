# Historical YK B4 regression data

`p10.txt` is a frozen gas-composition snapshot from the historical YK B4
calculation used during early ExoGibbs development. It contains 160
comma-separated equilibrium amounts for 500 K and 10 bar. Values are ordered
exactly like `exogibbs.presets.ykb4.chemsetup().species`; 16 entries exceed
the original comparison floor of `1e-14`.

The historical script compared these values directly with the solver's
species amounts (`EquilibriumResult.n`). Although the file happens to sum
very nearly to one, that does not establish a mole-fraction convention, so
the restored regression deliberately does not renormalize either side.

The corresponding element budget is retained verbatim in
`examples/comparisons/comparison_with_ykcode.py`, in this element order:

```text
C, H, He, K, N, Na, O, P, S, Ti, V, e-
```

## Provenance

- The passing historical comparison and this data payload were recorded in
  commit `f6a69d9db6e46dc1fe11ecc40deeb0d599cb72db` on 2025-08-09
  (`comparison with ykB4 passed`), originally as `examples/p10.txt`.
- The unchanged file was moved to `examples/data/p10.txt` in commit
  `fac1890cf1f05e7811b834ead805a9cfe1c4d91a` on 2025-08-12.
- The comparison and data were removed from the develop history by commit
  `8d8c68ab61f7c149e29883bd2579bcdc5b799d38` on 2026-06-06.
- The exact data Git blob is
  `2ae55ed726a89de7c034ef4412023a93afd1f65e`.
- The exact file SHA-256 is
  `062a0d21768f85871b7980ae3883d34f2466b9e11618255060e19d32c4a8612b`.
- The expected ordered species catalog SHA-256 is
  `3f020b61342d0034c7d01b110fca2e62f63ea135e3cf7045c99a551c739f492a`,
  computed from UTF-8 species names separated by a NUL byte.

The historical `yk.list` copy is not restored because it contains the same
160 numeric values as `p10.txt` (apart from the final newline) and was unused
by the comparison. The packaged YKB4 preset is the canonical catalog source,
and the example checks its length and full order fingerprint before aligning
the 160 reference values.

## Scope

This asset supports a traceable regression, not an independently rerunnable
external-code oracle. The original YK B4 program, its executable, and a
machine-readable run configuration are not included. Therefore the snapshot
can detect changes relative to the historical passing result, but it cannot
by itself establish current cross-code agreement.

Run the modernized regression from the repository root:

```bash
python examples/comparisons/comparison_with_ykcode.py
```

The script uses the current public `exogibbs.api.gas.solve` API, requires
solver convergence, compares `EquilibriumResult.n` with all historical major
entries, and writes a readable plot to
`results/ykb4/comparison_with_ykcode.png` by default.
