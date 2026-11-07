C3 source-sensitivity diagnostic, 2026-11-02
==========================================

This packet compares ``curated-c2h-v1`` with an explicitly diagnostic model
that replaces only C3 with the frozen NASA ``C3 - 3 C`` reaction. Coefficients
come from the pinned source-comparison baseline. The 1000--6000 K NASA interval
is evaluated only within the declared 1500--4000 K diagnostic domain. The
probe has the distinct ID ``curated-c2h-v1+raw-nasa-c3-diagnostic``; it is not
an adopted correction, source preference, uncertainty interval or physical
validation. Runtime constants and previous snapshots are unchanged.

``prioritization.json`` replays the saved current-curated C3 abundance and
carbon share. Its inherited JANAF source is unreviewed; at C/O = 1.5, 2000 K
and 1e-6 bar, C3 holds 2.25410% of carbon with mole fraction 6.08215e-6.
This finite-grid result motivates source investigation without claiming an
exhaustive priority ranking or importance for spectra or nonequilibrium chemistry.

The 537-species ion-inclusive network, formula matrix and elemental budget
are identical within each source pair. The 75 pairs span five temperatures
(1500, 2000, 2500, 3000, 4000 K), five pressures (1e-6, 1e-4, 0.01, 1, 100 bar),
and three C/O values (reference, 1, 1.5). Only C/H changes between compositions;
every other element/H ratio is preserved before normalization.

``equilibria.npz`` retains n, ln(n), x, ln(x), delta-x, delta-log10(x), C/O/N
allocations for all species, both h vectors, A and b. It also holds 150 tighter
cold solutions in addition to the 150 primary solutions. Both tolerances,
1e-13 and 5e-14, pass convergence, conservation and mass-action checks.
No concentration floors or omitted conditions are used; the summary's 1e-12
threshold is only for an additional display statistic.

At C/O = 1.5, 2000 K, 1e-6 bar, C3 mole fraction falls from 6.08215e-6 to
4.19707e-6 (about 30.99%) and its carbon share falls from 2.25410% to 1.55547%.
This point also has the largest absolute C3 mole-fraction change, -1.88508e-6.
C3H is the largest recipient of the released carbon: its carbon share rises
from 11.64305% to 12.02252%, with mole fraction increasing by 3.25936%.
C2H2 rises from 4.67992e-5 to 4.78106e-5 (2.16118%) and HCN rises from
2.38136e-5 to 2.40125e-5 (0.83547%). Atomic carbon, C2H, C2, C4, C5 and other
carriers respond while their thermochemistry remains fixed. CO changes by
+1.16061e-9 (0.0002151%). Complete carbon allocation is saved, so these
selected species should not be interpreted as the whole redistribution.
The source K ratio changes with temperature; this point's decrease is not
a claim that C3 decreases at every condition.

The maximum tolerance differences are 9.21e-14 in x and 2.37e-5 in ln(x).
The independent standalone FastChem check uses the same highlighted condition
and identical h/A/b for each endpoint, agreeing within 9.78e-7 dex over all
537 species. Its executable digest is recorded. Numerical agreement does
not establish thermochemical or atmospheric accuracy. Neutral-only sensitivity,
condensation, nonideality, quenching and spectra are outside scope.

Replay into a new directory::

   JAX_PLATFORMS=cpu JAX_ENABLE_X64=1 PYTHONPATH=src \
     python mceg/reviews/c3_equilibrium_impact.py \
     --output-dir results/c3-impact-replay

Add ``--fastchem-executable /path/to/fastchem`` for the independent check;
without it, that check is explicitly ``not_run``. The frozen run used the
executable and was copied unchanged from the command's original results
directory into this archive. The receipt identifies selected implementation,
input and output bytes. Ordinary offline evidence tests require neither
FastChem nor MCEG. The implementation reuses only helpers in develop and has
no dependency on the separate PH2 or AlH investigation PRs.
