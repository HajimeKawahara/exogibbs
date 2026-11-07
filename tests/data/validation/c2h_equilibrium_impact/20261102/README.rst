C2H finite substitution, 2026-11-02
==================================

The 537-species ion-inclusive network, formula matrix and budget are identical
within each of the 75 pairs. Only C/H varies between the three C/O settings;
all other element/H ratios are preserved before renormalization.

``equilibria.npz`` retains every species' n, ln(n), x, ln(x), delta-x,
delta-log10(x), carbon share, both h vectors, A and b. Other elemental shares
are A[e,i]*n[i]/b[e]. It also retains 150 tighter cold solutions and a five-point
finite substitution. The summary threshold is for display, never a concentration
floor. The receipt pins selected inputs and execution outputs. FastChem is
optional; its binary digest and two successful comparisons are recorded.

``precision_probe.json`` retains a rejected 2e-15 tolerance attempt. The full
grid subsequently passed at 1e-13 and 5e-14; no conditions were dropped.

Replay into a new directory::

   JAX_PLATFORMS=cpu JAX_ENABLE_X64=1 PYTHONPATH=src \
     python mceg/reviews/c2h_equilibrium_impact.py \
     --output-dir results/c2h-impact-replay

Add ``--fastchem-executable /path/to/fastchem`` for independent checks.
Without it, that check is ``not_run``. Ordinary evidence tests need neither
FastChem nor MCEG. Numerical agreement does not validate physical atmospheres.

The archived bytes are verified with their recorded SHA-256 hashes. Rebuilding
the coefficient arrays permits only last-bit roundoff: Python 3.12 changed
float summation, so Python 3.11 can serialize the same sum differently.
All remaining rebuilt fields are compared exactly. Mole fractions are checked
against their saved log fractions separately from the logsumexp reconstruction,
avoiding amplification of an already accepted logarithmic rounding difference.
These checks do not change source constants or conservation/mass-action bounds.
