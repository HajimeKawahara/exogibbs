# Equilibrium Grid Smoke Test By Hand

This note is for a quick manual check of the equilibrium-grid workflow after local changes.

## Scripts

`examples/works/grid_generation_by_hand.py`
- Builds a small ExoGibbs-backed grid on the FastChem preset
- Builds a small FastChem-backed comparison grid
- Saves and reloads the ExoGibbs grid
- Runs compatibility validation on the loaded grid
- Checks that `GridEquilibriumInitializer` returns an initialization state on a matching request

`examples/works/grid_interpolation_by_hand.py`
- Loads the packaged default FastChem grid from `exogibbs.data.grids.fastchem/grid_exogibbs.nc`
- Interpolates one state at a chosen `(T, P, log10(Z/Zsun))`
- Prints `ntot`, `ln_ntot`, `sum(x)`, and the first few species values for visual inspection

## How To Run

From the repository root:

```bash
python examples/works/grid_generation_by_hand.py
python examples/works/grid_interpolation_by_hand.py
```

## What To Confirm Manually

For `grid_generation_by_hand.py`:
- The script completes without error.
- ExoGibbs grid build and FastChem grid build both complete.
- Save/load works and the loaded grid reports the expected composition-axis name.
- Compatibility validation passes.
- `GridEquilibriumInitializer` returns a finite `ln_ntot` and a species-shaped `ln_nk`.
- The reported species-fraction sum min/max is close to `1`.

For `grid_interpolation_by_hand.py`:
- The script loads the saved grid without error.
- Interpolation returns finite `ntot` and `ln_ntot`.
- `sum(x)` is close to `1`.
- The printed species rows look numerically reasonable and non-pathological.

## Manual Record

Date:

Environment:

Commands run:

Generation script result:

Interpolation script result:

Notes / anomalies:
