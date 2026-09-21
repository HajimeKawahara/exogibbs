# Independent standard and hydrogen-basis audit

`m2_standards_audit.py` compares the standards used by the conditional BSE
runner with the independent FastChem4 table used by M1. It removes only an
elemental reference gauge and reports the remaining species differences,
along with four independent, atom-balanced gas reactions. Changing the
elemental reference cannot change those reactions. No molecular or phase
offset is fitted to force agreement.

At 2173.15 K the lower-minus-upper reaction-energy differences are:

| Reaction | Difference in Delta G / RT |
| --- | ---: |
| H2 -> 2 H | -0.00278721138 |
| H2 + 0.5 O2 -> H2O | -0.00024310316 |
| 0.5 H2 + 0.5 O2 -> OH | -0.09313289842 |
| SiO + 2 H2 -> SiH4 + 0.5 O2 | -0.01576341580 |

These differences exceed the numerical contact tolerance. Neither source is
declared the correct BSE calibration by this comparison. The 2350 K comparison
is recorded separately. The existing `m1_thermochemical_control.py` changes
H2O and SiH4 reaction data at 2350 K and cannot reconcile a new temperature.

The native audit evaluates the initial dry BSE composition once with the
pinned MELTS runtime. It records pure-endmember standards and four balanced
vaporization reactions. Source sums such as `2 FeO + SiO2` are explicitly
virtual combinations, not measured standards of a MELTS `Fe2SiO4` endmember.
Their differences expose a change in material model; they do not justify a
phase correction. Native Ma convention shifts and inherited source alloy
standards are recorded separately. Independent calibration of MELTS/alloy/gas
exchange, including alloy H, remains unavailable.

## Hydrogen concentration basis

The old sensitivity law remains unchanged:

```text
x_H2 = (f_H2 / bar) exp(-11.403 - 0.76 P_melt/GPa).
```

Its use of `sum(n_MELTS_endmember) + n_H2` is a model assumption; the
experimental-to-endmember denominator conversion has not been verified.
[Seo et al. (2024)](https://doi.org/10.3847/1538-4357/ad7461) describe a mole
fraction and use an approximate magma molar mass of 0.06 kg/mol in their
reservoir estimate. The adopted BSE ledger gives about 0.1252014 kg per mole
of MELTS endmembers. Equating these counts would change the physical H2
amount by about a factor of 2.09 in the dilute limit. This does not establish
the denominator in the original
[Hirschmann et al. (2012) experiment](https://doi.org/10.1016/j.epsl.2012.06.031).

[Chaudhari et al. (2025), Tables 1 and 2](https://epub.uni-bayreuth.de/id/eprint/8940/1/s00410-025-02272-y.pdf)
are publicly available and report molecular H2 in ppm by weight. Interpreting
this as H2 mass per total glass mass follows the analytical method; it is not
stated as a normalization to dry oxide mass. The hosts are Fe-free compositions.
Basalt and andesite were measured at 1673.15 K; haplogranite points span
1473.15--1573.15 K. They do not calibrate the project's 2173.15 K BSE liquid.

For H2 mass fraction `w`, host mass `m`, host endmember amount `S`, and H2
molar mass `M_H2`, the exact amount conversion is

```text
h = w / (1 - w) * m / M_H2
x_MELTS = h / (S + h).
```

The host excludes added H2 and includes native H2O when present.
`h2_mass_fraction_to_amount` implements only this reversible conversion.
Saved 10, 100 and 1000 ppm values are diagnostic inputs, not observations
or calibrated predictions. A composition-dependent denominator requires a
scalar free energy with the corresponding host derivatives before replacing
the dilution model. A varying shift applied to H2 alone is insufficient.

## Reproduction and column interface

Run from the selected ExoGibbs checkout with the matching provider:

```sh
JAX_PLATFORMS=cpu JAX_ENABLE_X64=1 PYTHONPATH=src:/path/to/exoeos/src \
python examples/metal_silicate/m2_standards_audit.py \
  --inventory /path/to/exoeos/examples/m2_material/bse_inventory.json \
  --exoeos-checkout /path/to/exoeos \
  --runtime /path/to/alphamelts-py-2.3.2-ubuntu_22_04-x86_64 \
  --python /path/to/native-worker/python \
  --output /tmp/m2-standards-audit.json
```

Omit the four provider/input options for gas comparisons alone. Existing
outputs are never overwritten. The documented-example job requires fresh
native properties and both gas comparisons. Completion means the requested
diagnostics ran; it does not mean standard agreement or M2-A/B acceptance.

`audit_shared_standards(A, lower, upper, species=..., elements=...)` accepts
an element-by-species matrix and aligned standard potentials on the same R
and standard pressure. Rows for absent background elements are allowed;
their gauge is unobservable. `audit_contact(lower, upper, species=...)`
compares aligned partial pressures in bar. Both exact zeros are retained;
a zero on only one side fails explicitly. A column must separately require
both audits and inspect additional upper species and condensates.

The BSE calculation still evaluates the fixed 2350 K source branch at
2173.15 K; the source MgO liquid fit lists 3105--5000 K. There is no common
calibrated material domain. M2-A/B remain pending and archived conditional
results retain their original meaning.
