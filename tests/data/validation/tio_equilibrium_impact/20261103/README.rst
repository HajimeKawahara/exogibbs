TiO candidate finite substitution
=================================

This packet compares current curated, TOTO/Jain and NASA-thermal/Jain on
identical 75-case, 537-species ion-inclusive gas problems. Only TiO's formation
potential changes. The current solutions are reused from the hashed
curated_default_adoption archive after matching species, elements, budgets,
formula matrix and chemical potentials. The two candidates add 150 cold solves.
Production curated is unchanged.

``equilibria.npz`` retains every species, with no abundance floor. It contains
the three endpoint arrays, six tighter solves at three selected cases, a
five-point finite substitution and two D0-only perturbations at the largest
Ti-allocation response. The central thermal inventory remains fixed for those
energy perturbations. ``diagnostics.json`` and ``selected_diagnostics.json``
record convergence and constraints; ``summary.json`` records abundance changes,
titanium allocation, same-model Gibbs comparisons and three independent
FastChem checks at the highlighted condition. These are numerical comparisons,
not measured atmospheric errors.

The receipt pins execution inputs, output bytes and the optional FastChem
executable. The native backend is CPU, with JAX float64. Replay the archived
evidence without MCEG, FastChem or network access::

   PYTHONPATH=src pytest tests/model_evidence/tio_equilibrium_impact_test.py

Recompute the experiment into a new ignored directory::

   JAX_PLATFORMS=cpu PYTHONPATH=src python mceg/reviews/tio_equilibrium_impact.py \
       --output-dir results/tio-impact-new --fastchem-executable /path/to/fastchem

Omitting the executable leaves the independent-solver check explicitly
unperformed. Plot the archived 75-case response with::

   python mceg/reviews/plot_tio_candidate.py --output results/tio-response.png

The raw Gibbs-exchange allowance is 2e-12 per normalized elemental budget.
The independent offline test also removes the first-order contribution of
small numerical budget differences, using the solved elemental potentials.
Neither comparison ranks the physical correctness of different source models.
The candidate construction and its isotope/completeness limits are documented
in ``documents/mceg/tio_candidate.rst`` and the separate candidate packet.
