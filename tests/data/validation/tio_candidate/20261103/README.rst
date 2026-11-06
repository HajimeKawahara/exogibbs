TiO representative-isotope candidate
====================================

This is an offline candidate for a controlled TiO substitution. It does not
modify the production curated preset or establish complete physical accuracy.
The existing element-aggregated TiO species is represented by 48Ti16O internal
levels, molecular translation at 63.8664 u (NASA Ti + O masses), nuclear-spin
factor one, Jain D0 = 6.824 eV and the existing NASA neutral-atom thermal gauge.
The standard pressure is 1 bar and the domain is 1500-4000 K. This is neither
a complete natural-isotope mixture nor an isotope-resolved reaction network.

State selection and continuous functions
----------------------------------------

The official TOTO version 20240509 file has 301245 levels, all with weight
2J+1. Its two ground parity components have energy zero. Levels above the
central D0 threshold, 55039.271828 cm^-1, are excluded: 300670 levels remain.
No additional zero-point or theoretical spin-orbit correction is applied.
The central state inventory stays fixed for D0-only uncertainty variants.

All 16 electronic manifolds are included before that cutoff. The approximate
D,F,g,h,i manifolds already occur in the current file; adding their published
partition supplement would double count them. ``scientific_scope.json`` records
isotope and represented-shell sensitivities. Those diagnostics do not bound
unrepresented high states or isotope-dependent atomization uncertainties.

At six knots, direct Boltzmann sums provide Q, mean epsilon and its variance,
where epsilon=E/k_B. Translation is included analytically at each knot:

* H-H0 = R*(2.5*T + mean(epsilon)).
* Cp = R*(2.5 + Var(epsilon)/T^2).
* G-H0 = -R*T*ln(Q*q_translation), including k_B*T/p0 in q_translation.
* S = (H-G)/T.

A single quintic Hermite representation of f=(G-H0)/R matches f, -S/R and
-Cp/(R*T) at 1500, 2000, 2500, 3000, 3500 and 4000 K. It is C2 across joins;
G, H, S and Cp are derived from this same polynomial. No independent fits of
the caloric quantities are combined. ``off_knot_reference.json`` freezes
45 direct state sums excluded from the knot fit. ``validation.json`` records
maximum errors in ln K, H, S and Cp, and the minimum Cp obtained from every
interval's analytic stationary points. The achieved maximum errors are
1.024e-6 in ln K and 0.002492 J/mol/K in Cp, below targets of 1e-5 and 0.01.
These are interpolation errors within the specified level model, not physical
error bars or abundance responses.

``cutoff_sensitivity.json`` compares the full file against the bound-level
selection, and compares its Q with the separately distributed PF. The cutoff
removes at most 3.836e-9 of Q and changes Cp by less than 1e-5 J/mol/K. The
distributed PF is a version check; its values are not mixed with derivatives
from the current states.

Evaluation and regeneration
----------------------------

``mceg/reviews/tio_candidate.py`` exposes:

* ``formation_h(T, thermal="states", d0_eV=6.824)``: scalar Gformation/(RT),
  or minus ln K, in the existing elemental solver gauge.
* ``formation_h_jax(...)``: the same function for traced scalars or arrays;
  JIT, gradients and vmap are supported. Use JAX 64-bit for recorded precision.
* ``molecular_properties(T, thermal="states")``: G-H0 and H-H0 in J/mol,
  S and Cp in J/mol/K. The reference zero is explicit.

``thermal="nasa"`` provides the historical NASA molecular thermal shape with
the same chosen D0 and atomic background. It is an independently named proxy,
not an assumption that NASA's historical equilibrium constant is correct.
Changing ``d0_eV`` shifts only the energy term. The scalar API rejects
temperatures outside the domain; its traced counterpart returns NaN.

Ordinary evaluation and tests need only the frozen packet::

   PYTHONPATH=src python mceg/reviews/tio_candidate.py --temperature 2500
   JAX_PLATFORMS=cpu PYTHONPATH=src pytest tests/model_evidence/tio_candidate_test.py

Given a directory containing the official compressed states and all five
Ti16O PF/definition files, regenerate the packet with no network access::

   PYTHONPATH=src python mceg/reviews/tio_candidate.py --extract ORIGINAL_SOURCE_DIRECTORY

The extractor rejects source bytes differing from the reviewed hashes.
``candidate.json`` and ``scientific_scope.json`` contain the original URLs and
hashes; complete source files are not redistributed. ``receipt.json`` pins
the extraction implementation, inherited source inputs and generated outputs.
The DOI and inspection record for the D0 measurement and line lists remain in
``tests/data/validation/tio_source_review/20261103``. CIAAW isotope ratios are
terrestrial reference sensitivities, not measured exoplanet compositions.
