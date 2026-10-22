# Archived metal--silicate data

Version 0.7 retired the example-local multiphase solvers and their M1/M2
workflows. The JSON reference data and validation records in this directory
retain their original content. Current calculations use
`exogibbs.api.multiphase`; see the [multiphase guide](../../documents/multiphase.rst).

A pre-removal checkout containing the complete legacy workflows is
[`64325fd347b2f1d77237ce5afae4aa43b2e97d08`](https://github.com/HajimeKawahara/exogibbs/tree/64325fd347b2f1d77237ce5afae4aa43b2e97d08/examples/metal_silicate).
It contains the old code, tests, instructions and material-specific proof
workflows. For a recorded calculation, use its recorded source commit and
provider versions. Old ExoInventory runners must select that historical
ExoGibbs checkout instead of the current release.

```console
git worktree add --detach ../exogibbs-legacy-m2 64325fd347b2f1d77237ce5afae4aa43b2e97d08
```

Run the historical instructions with that worktree's `src` on `PYTHONPATH`.
Preserved validation documents describe their original executions; their
example scripts are available at the historical commit.
