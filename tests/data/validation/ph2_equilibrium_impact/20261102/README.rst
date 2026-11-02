PH2 source-sensitivity diagnostic, 2026-11-02
==========================================

This archive compares the current ``curated-c2h-v1`` model against a diagnostic
that replaces only PH2 with the unadjusted NASA ``PH2 - P - 2H`` reaction.
The NASA coefficients come from the pinned baseline source excerpt, using its
1000--6000 K intervals only over the declared 1500--4000 K diagnostic domain.
The probe has a distinct dataset ID, ``curated-c2h-v1+raw-nasa-ph2-diagnostic``.
It is not a curated update, source recommendation, uncertainty interval, or
validation of the NASA absolute energy scale. Runtime constants are unchanged.

The 537-species ion-inclusive network, formula matrix and budget are identical
within each pair. The grid has 75 pairs: five temperatures (1500, 2000, 2500,
3000 and 4000 K), five pressures (1e-6, 1e-4, 0.01, 1 and 100 bar), and three
C/O values (the reference value, 1 and 1.5). Only C/H varies between C/O settings;
every other element/H ratio is preserved before normalization.

``equilibria.npz`` retains every species' n, ln(n), x, ln(x), delta-x,
delta-log10(x), phosphorus/carbon/nitrogen shares, both h vectors, A and b.
It includes 150 tighter cold solutions in addition to the 150 primary solutions.
Both tolerances, 1e-13 and 5e-14, pass conservation and mass-action checks.
No concentration floors or omitted grid conditions are used. The summary's
1e-12 threshold is used only for one additional display statistic.

At 2500 K, 100 bar and reference C/O, the baseline places 64.2509% of phosphorus
in PH2. The diagnostic lowers this to 50.3632%, while PH3 rises from 14.8918%
to 20.6648%. PH and HCP also take substantial shares; the summary preserves
the complete phosphorus partition, not just the selected target species.
These shifts measure sensitivity to this particular source substitution.

The independent FastChem check uses 3000 K, 100 bar and reference C/O, the
largest baseline PH2 phosphorus share among temperatures supported by the
existing single-node exporter (1500, 2000, 3000 and 4000 K). It therefore
checks a different point from the highlighted 2500 K case. Both models agree
with FastChem to within 5.69e-7 dex over all 537 mole fractions at that point.
The binary digest and the exact check condition are recorded. Numerical
agreement does not resolve the thermochemical source choice or validate
physical atmospheres; condensation, quenching and spectra are outside scope.

Replay into a new directory::

   JAX_PLATFORMS=cpu JAX_ENABLE_X64=1 PYTHONPATH=src \
     python mceg/reviews/ph2_equilibrium_impact.py \
     --output-dir results/ph2-impact-replay

Add ``--fastchem-executable /path/to/fastchem`` for the independent check.
Without it, that check is explicitly ``not_run``. Ordinary evidence tests
need neither FastChem nor MCEG. The receipt pins selected repository inputs
and output bytes. Numerical reconstruction tests permit ordinary floating
point roundoff while separately checking the immutable archive hashes.
