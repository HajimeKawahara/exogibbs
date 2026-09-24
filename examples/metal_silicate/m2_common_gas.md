# Common gas reactions at the finite BSE boundary

The opt-in `gas_model="m1_shared"` argument to `build_bse_problem` adopts
the packaged FastChem4 temperature-dependent reaction data already used by
the M1 upper atmosphere. The historical default `gas_model="source"`, its
fixed source branch, and archived results remain available unchanged.

The eleven shared species are H2, O2, H2O, Fe, Mg, SiO, Na, H, He, OH and
SiH4. Each standard is evaluated at the requested temperature, with a common
1 bar pressure standard. The ideal gas pressure term occurs once. The
solubility sensitivity law uses the same gas H2 standard; its experimental
host calibration remains unverified.

## Element reference and acceptance

FastChem4 sets atomic gas reference energies to zero. Substituting those
values directly into only the lower gas phase would change its energy
reference relative to MELTS and the alloy. Instead, seven independent
existing lower standards anchor the reference: H, He, O2, Mg, SiO, Fe and Na.
For the formula matrix `A` (elements by species) and those columns `J`, solve

```text
A[:, J].T c(T) = h_source[J](T) - h_FastChem[J](T)
h_common(T) = h_FastChem(T) + A.T c(T).
```

This square system preserves the selected lower anchors exactly and leaves
every FastChem balanced gas reaction unchanged. It does not fit molecular
offsets or contact outputs. It retains the existing formal source reference;
it does not independently align MELTS/alloy reaction energies or establish
a calibrated common material domain. In particular, the inherited 2350 K
source branch and its extrapolation limits remain explicit.

Upper parcels retain the original FastChem gauge for both gas and pure
condensates. Conserved equilibrium is invariant to the common elemental
shift: its change in `G/(RT)` is the constant `b.T c`. Comparisons of absolute
cross-boundary chemical potentials must include that gauge; comparing raw
absolute values would be incorrect.

`run_m2_contact.py` freshly solves a finite 13-element BSE inventory with
metal explicitly suppressed, independently recounts its atoms, and then
solves three parcels at the same temperature and pressure:

| Parcel | Purpose | Numerical requirement |
| --- | --- | --- |
| Same 11 gases | Matched contact control | Atom/KKT audits, standards modulo one element gauge, and shared log partial-pressure residual below `1e-8` |
| Expanded 35 gases | Additional gas species diagnostic | Independent parcel audits; record the remaining contact difference |
| 35 gases + 26 condensates | Added retained-cloud diagnostic | Independent gas/cloud audits; record the remaining contact difference |

The transferred budget is reconstructed from source gas components only.
Deep melt and metal remain fixed. Additional species and cloud formation
can change a parcel's equilibrium, so a matched control does not certify
contact with the expanded atmosphere. M2-A/B/C remain pending. A successful
process exit means the matched control and all requested numerical
diagnostics completed, not that an expanded atmosphere passed contact.

## Reproduction

```sh
JAX_PLATFORMS=cpu JAX_ENABLE_X64=1 PYTHONPATH=src:/path/to/exoeos/src \
python examples/metal_silicate/run_m2_contact.py \
  --inventory /path/to/exoeos/examples/m2_material/bse_inventory.json \
  --exoeos-checkout /path/to/exoeos \
  --runtime /path/to/alphamelts-py-2.3.2-ubuntu_22_04-x86_64 \
  --python /path/to/native-worker/python \
  --output /tmp/m2-common-gas-contact.json
```

Existing outputs are never overwritten. The record includes the canonical
input, source/provider/data hashes, native call count, full source component
amounts, independently audited parcels, contact residuals, and preserved
failure diagnostics. `metal_m2_contact` registers this full native run in the
documented-example harness; a successful exit without the evidence fails
artifact acceptance.

The [2026-09-22 native validation](../../results/m2_contact/20260922/README.md)
passes the same-catalog contact at `3.91e-14` maximum log partial-pressure
residual. Expanding to 35 gases gives `0.01294`; including the 26-entry
retained-condensate catalog gives `1.29279`. Only the pure Fe condensate is
present at this point. All three local parcel audits pass.
The archived clean source/provider revisions, full component amounts,
registered-job evidence, and earlier iteration-limit failure are retained.

## Full retained-atmosphere contact

`gas_model="m1_expanded"` includes all 35 atmospheric gas species in the
finite source and audits every gas at contact. It retains the same seven
reference anchors. Gas aliases and their formulas are recorded explicitly;
new species are not dropped from the pressure or atom totals.

`build_expanded_bse_problem` in `m2_expanded_source.py` additionally couples
all 26 atmospheric condensates through the existing retained-parcel solver.
At each finite atmospheric atomic allocation `b_atm`, the callback minimizes

```text
G_atm/(RT) = min_[n_g,n_c >= 0; A_g n_g + A_c n_c = b_atm]
  sum_i n_g[i] (h_g[i] + log(n_g[i]/sum(n_g)) + log(P/1 bar))
  + sum_j n_c[j] h_c[j] + b_atm.T c(T).
```

Its amount derivative is the atmospheric elemental potential plus the same
reference gauge. The outer finite BSE solve varies seven atomic amounts
alongside MELTS and the optional alloy. These internal coordinates are
neither extra gas species nor extra matter. The source is solved again with
this atmosphere energy; clouds are not appended after freezing a gas-only
source. Pure condensates have no ideal-solution mixing entropy and contribute
no gas pressure. Every condensate uses its packaged temperature eligibility.

`unpack_expanded_source` reconstructs all 35 gas and 26 cloud amounts for the
consumer, keeping the internal solver record separate. Explicit mappings
identify each gas and retained condensate. Atmospheric Fe cloud is distinct
from the deep Fe-Si-O-H alloy. The atmospheric transfer budget includes both
gas and retained cloud atoms; deep silicate and alloy remain separate.
Exact-zero atmospheric elements remove only species that contain them, with
exact-zero primitive amounts and no element floor.

`audit_expanded_contact` independently recounts both source and upper
primitive amounts, tests gas stationarity and present/absent condensate KKT,
checks a single elemental reference across all gas and eligible condensate
standards, compares all 35 partial pressures, and checks cloud atom totals.
An accepted flag or a match among only the historical eleven gases is
insufficient. Nonunique pure-phase partitions are assessed by their common
chemical potentials and cloud atom totals, not an arbitrary component split.

Run the finite source with the full retained atmosphere by adding
`--gas-model m1_retained` to the command above. `--metal-mode suppressed`
retains the explicit metal-suppressed control; `--metal-mode select` runs
the constrained alloy selection. The registered native jobs
`metal_m2_retained_suppressed` and `metal_m2_retained_select` preserve each
source solve, its primitive expansion, native calls and complete contact
audit. Both retain pending M2-A/B/C: local contact does not establish a
calibrated BSE material domain, global MELTS phase stability or a planetary
pressure/inventory closure.
