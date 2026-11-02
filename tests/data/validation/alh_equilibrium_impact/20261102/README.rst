AlH source-sensitivity diagnostic, 2026-11-02
==========================================

This packet compares the current ``curated-c2h-v1`` model against a diagnostic
that replaces only AlH with the unadjusted NASA ``ALH - AL - H`` reaction.
NASA coefficients come from the pinned baseline source records, using the
1000--6000 K intervals only over the declared 1500--4000 K diagnostic domain.
The probe has the distinct ID ``curated-c2h-v1+raw-nasa-alh-diagnostic``.
It is not an adopted correction, preferred-source ranking, uncertainty interval
or validation of the NASA energy scale. Runtime constants remain unchanged.

``prioritization.json`` reproduces a screen of saved current-curated abundances.
AlH holds up to 50.97% of Al in this grid and has a provisional source choice.
PO2, the most abundant unresolved P1 candidate by its own elemental share here,
holds only 0.0743% of P. The screening tier is therefore not treated as an
absolute physical priority. C3 remains a useful follow-up, while reviewed
choices for VO and other major carriers are not reset solely for differing
from NASA. This is a selected candidate comparison, not an exhaustive ranking
or a claim about observed spectra. Input hashes and peak conditions are saved.

The 537-species ion-inclusive network, formula matrix and budget are identical
within each source pair. There are 75 pairs: five temperatures (1500, 2000,
2500, 3000 and 4000 K), five pressures (1e-6, 1e-4, 0.01, 1 and 100 bar), and
three C/O values (reference, 1 and 1.5). Only C/H changes between compositions;
every other element/H ratio is preserved before normalization.

``equilibria.npz`` retains every species' n, ln(n), x, ln(x), delta-x,
delta-log10(x), Al/C/O/N/P shares, both h vectors, A and b. It also holds the
150 tighter cold solutions in addition to the 150 primary solutions. Both
tolerances, 1e-13 and 5e-14, pass conservation and mass-action checks. No
concentration floors or omitted grid conditions are used. The summary's 1e-12
threshold applies only to an additional display statistic.

At C/O = 1.5, 2500 K and 100 bar, AlH holds 50.9702% of Al in the baseline and
63.4698% in the probe. Atomic Al falls from 47.4512% to 35.3773%; AlOH and other
Al carriers also change. AlH mole fraction increases from 2.35246e-6 to
2.92937e-6, about 24.52%. Both AlOH connectivity records remain distinct.
The summary stores every Al carrier at this point, and the NPZ holds complete
Al allocation at every condition. The largest absolute AlH mole-fraction
change is 6.93212e-7 at C/O = 1.5, 2000 K and 100 bar (probe/current = 1.59296).

The independent FastChem check uses C/O = 1.5, 3000 K and 100 bar: the largest
baseline AlH share among the temperatures supported by the existing exporter
(1500, 2000, 3000 and 4000 K). It is a different point from the 2500 K highlight.
Both models agree with FastChem within 4.43e-8 dex over all 537 mole fractions.
The executable digest and condition are recorded. This numerical agreement
does not establish thermochemical or atmospheric accuracy; neutral-only
sensitivity, condensation, nonideality, quenching and spectra are outside scope.

Replay into a new directory::

   JAX_PLATFORMS=cpu JAX_ENABLE_X64=1 PYTHONPATH=src \
     python mceg/reviews/alh_equilibrium_impact.py \
     --output-dir results/alh-impact-replay

Add ``--fastchem-executable /path/to/fastchem`` for an independent check; without
it, that check is explicitly ``not_run``. Ordinary evidence tests need neither
FastChem nor MCEG. The receipt pins selected input/output bytes. Reconstruction
checks permit ordinary floating-point roundoff while separately checking the
immutable archive hashes. The implementation depends only on helpers already
in develop, not the separate PH2 investigation PR.
