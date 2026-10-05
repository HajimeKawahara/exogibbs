Standard thermodynamics for a hot gas model
===========================================

``prepare_fastchem_thermodynamics`` supplies analytic, mutually consistent
standard Gibbs energy, entropy, and heat capacity for an explicit gas model
intended for equilibrium-entropy convection. It adds NASA atomic reference
functions to FastChem reaction data; differentiating reaction log K alone
would omit the atomic heat capacities. It does not differentiate a minimized
Gibbs energy or require second derivatives of equilibrium composition.

This is an **opt-in change of thermochemical model**. Of the 539 species in the
ion-inclusive FastChem4 gas table with its default elemental set, 34 use explicitly identified NASA9 replacements
for both chemical potentials and caloric properties. Diacetyl peroxide
(``C4H6O4``) is omitted because its reconstructed heat capacity becomes negative
and the NASA source has no matching species. The resulting 538-species setup
retains every conservation row. The original FastChem4 preset remains
unchanged. Using its chemistry with the replacement entropy is inconsistent.

.. code-block:: python

   from exogibbs.thermo.standard import prepare_fastchem_thermodynamics

   thermo = prepare_fastchem_thermodynamics()
   setup = thermo.chemical_setup
   entropy_over_R = thermo.standard_entropy_r(3000.0)
   cp_over_R = thermo.standard_cp_r(3000.0)
   gibbs_over_RT = thermo.standard_gibbs_rt(3000.0)

Functions accept a scalar or an array of temperatures in K and append the
species axis. Standard pressure is 1 bar. The optional ``temperature_range``
can restrict 1500--4500 K. The entire domain lies inside one smooth NASA9
coefficient interval. There is no temperature clipping or extrapolation;
out-of-domain values return NaN. All species have analytic heat-capacity
minima checked on the complete interval, allowing 0.01 in ``cp/R`` for
polynomial discrepancies below the ideal-gas translational limit 2.5.

``hvector_func`` is the exact same callable as
``chemical_setup.hvector_func``. It removes the common atomic Gibbs reference
for solver conditioning. The absolute ``standard_gibbs_rt`` restores it; the
difference is a conserved elemental gauge and does not change equilibrium.
The species functions obey ``s/R = -g/(RT) - T d[g/(RT)]/dT`` and
``cp/R = T d[s/R]/dT``. The isotope convention and atomic weights are exposed
with the provider so that downstream specific entropy uses a consistent
conserved mass, including the electron charge row.

Mixing and pressure entropy must still be added to the standard entropy.
For species amounts ``n``, number fractions ``x``, molar masses ``M`` in
kg/mol, and standard entropy divided by R, use
``R * sum(n * (s_over_R - log(x * P / P0))) / sum(n * M)``.
Treat zero species amounts using their zero entropy-contribution limit.
All adjacent atmospheric points must share the same conserved elemental
amounts. Chemistry convergence, conservation, and finite state checks remain
the caller's responsibility.

Comparison with the unmodified network
--------------------------------------

An 80-state comparison used temperatures 1500, 2200, 3000, and 4500 K;
pressures 1e-5, 0.01, 1, and 100 bar; and ``(log_metal_scale, C/O)`` pairs
``(0, 0.6)``, ``(-1, 0.3)``, ``(-1, 1.2)``, ``(1, 0.3)``, ``(1, 1.2)``.
The solver used ``epsilon_crit=1e-14`` and at most 2000 iterations. It converged
at 78/80 original, 77/80 omission-only, and 76/80 hybrid-model states. Comparisons
used only jointly converged states passing relative elemental and charge
conservation thresholds of 3e-6. The original, omission-only, and hybrid models
passed these joint gates at 77, 76, and 74 states, respectively. That audit threshold is not a recommended retrieval
tolerance; a downstream calculation must apply its own stricter acceptance
criteria. These runs demonstrate neither universal convergence nor the
accuracy of retrieved parameters.

Omitting only diacetyl peroxide changed any mole fraction by at most
8.72e-15 and mean molecular weight by at most 9.10e-14 relative on the 76
common valid states. Its largest original mole fraction was 6.92e-39 on
this grid. These small omission effects do not justify untested abundance
patterns or a larger temperature range.

The 34 NASA replacements have a measurable effect: on 73 common valid states,
the maximum change of any mole fraction was 1.163e-5, and the maximum relative
change of mean molecular weight was 8.35e-6. Electron mole fraction changed by
up to **11.75 percent relative**. The replacement model therefore needs its
own matched TCE/RCE comparison; it is not a numerically interchangeable
version of the original FastChem4 chemistry. In particular, an H-minus
opacity calculation must use the matching electron abundance.

Run ``PYTHONPATH=src python examples/audit_rce_thermodynamics.py`` to reproduce
the comparison. The recorded grid, per-state convergence and conservation
diagnostics, and gated comparisons are in
``results/rce_thermodynamics_audit.json``. No observational data are required.

The packaged ``data/nasa9/README.md`` and ``fastchem4_hot.json`` contain source
hashes, all replacement identities, the omitted species, atomic weights,
and original record references. These checks validate the specified model;
they do not certify the original full table at high temperature or retrieval
accuracy. The primary coefficient source is
`NASA CEA <https://github.com/nasa/cea/blob/2f79a647e85737742fb08a127da00d1ea1d069da/data/thermo.inp>`_,
using the functions described in
`NASA/TP-2002-211556 <https://ntrs.nasa.gov/citations/20020085330>`_.
