# Approximate source boundary validation

`validation.json` contains fresh source and upper condensate solves for the
vendored absolute low-oxygen inputs at 2350 K and 20.42834233241181 / 54.15523948877404 bar.
The implementation commit is `a4937e6`; the record includes actual import paths,
source/data/fixture/evaluation-script hashes and unchanged tolerances
(element/mass 1e-9, chemical 1e-8).

Both approximate transfer contracts are met. The source gas Si cloud fractions
are 0.5576382769419852 and 0.6615321327209696; O fractions are 0.4340434405736281
and 0.4641947934378124. The largest fixed-deep source reaction changes are
0.8265632899295925 and 1.0969342008470653 /RT. Common equilibrium is not certified
and M1-A is not accepted.

Reproduce offline from the repository root, choosing a new output path:

```bash
JAX_PLATFORMS=cpu JAX_PLATFORM_NAME=cpu JAX_ENABLE_X64=1 PYTHONPATH=src \
python examples/metal_silicate/m1_boundary_validation.py \
    --output results/m1_boundary_contract/new_validation.json
```
