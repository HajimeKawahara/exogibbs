Ni and Co ionization: FastChem, JANAF, and Saha
===============================================

The packaged FastChem fits overestimate the first-ionization equilibrium
constant relative to Saha with Barklem & Collet (2016) partition functions.
At 3000 K the factors are about 287 for Ni and 1.90 for Co; at 6000 K they
are about 280 and 12.6. Direct JANAF evaluation reproduces almost all of
these discrepancies. The problem is already present in the published JANAF
ionic thermochemistry, rather than an ExoGibbs sign or unit conversion.

The original FastChem presets and tables are retained. The follow-up
implementation adds an explicit NASA Ni/Co ionization preset and adds Co+
to the separate caloric model, as described below. This investigation does
not identify the exact historical calculation error or claim that all JANAF
data are invalid.

Comparison conventions
----------------------

For :math:`M \rightleftharpoons M^+ + e^-`, all comparisons use
dimensionless :math:`K_p` with :math:`p^\circ=1` bar:

.. math::

   K_p = \frac{(p_{M^+}/p^\circ)(p_e/p^\circ)}{p_M/p^\circ},
   \qquad K_n = \frac{n_{M^+} n_e}{n_M},
   \qquad K_p = K_n\frac{k_B T}{p^\circ}.

Saha gives, in SI units,

.. math::

   K_n = 2\frac{U_{M^+}(T)}{U_M(T)}
   \left(\frac{2\pi m_e k_B T}{h^2}\right)^{3/2}
   \exp\left(-\frac{\chi}{k_B T}\right).

The factor 2 is the electron spin degeneracy. Atomic partition functions
include excited states. The negligible ion/atom translational mass correction
is omitted, as in the usual Saha approximation. Converting the numerical
value of :math:`K_n` from m\ :sup:`-3` to cm\ :sup:`-3` divides it by
:math:`10^6`.

FastChem's five-coefficient polynomial is **ln Kp**, despite the ``logK``
filename. ExoGibbs assigns ``h(M+) = -ln(Kp)`` and uses zero atomic and
electron reference functions. Its reaction chemical potential therefore
gives the same constant. The Ni1+ and Co1+ coefficients are identical in
the packaged v3 standard/extended and v4 standard/extended/Hoeijmakers files.

For JANAF, the existing ExoGibbs raw-table loader gives formation Gibbs
energies in kJ/mol:

.. math::

   \ln K_p = -\frac{1000\,[\Delta_fG^\circ(M^+)
                +\Delta_fG^\circ(e^-)-\Delta_fG^\circ(M)]}{RT}.

JANAF's electron *formation* Gibbs energy is zero. NASA9 provides absolute
standard functions, so its electron Gibbs term must be included explicitly.
Mixing these two electron conventions would give an incorrect comparison.

Numerical results
-----------------

Values below use exact partition-table nodes and NIST ionization energies
Ni = 7.639878 eV and Co = 7.88101 eV from the locally available January 2022
snapshot. The reference JSON records the source hashes and uncertainties.

.. list-table:: Ratios of ionization equilibrium constants
   :header-rows: 1

   * - T [K]
     - Ni FastChem/Saha
     - Ni JANAF/Saha
     - Co FastChem/Saha
     - Co JANAF/Saha
   * - 1000
     - 83.18
     - 87.96
     - 1.135
     - 1.214
   * - 2000
     - 226.23
     - 222.34
     - 1.142
     - 1.137
   * - 3000
     - 286.97
     - 282.65
     - 1.901
     - 1.750
   * - 4000
     - 293.40
     - 297.24
     - 3.669
     - 3.863
   * - 5000
     - 283.90
     - 288.59
     - 7.070
     - 7.373
   * - 6000
     - 280.50
     - 270.85
     - 12.644
     - 11.634

FastChem differs from direct JANAF by at most 8.7% at these nodes. Independent
NASA CEA data are much closer to Saha: within 3% for Ni over this grid, and
within 11% for Co at 3000--6000 K (27% at 1000 K). NASA is a second reference,
not an exact replacement for the adopted Saha inputs.

These results reproduce the scale and temperature dependence in the supplied
``fig1_ionK_ratio.png``. That figure labels its ionization energies as Barklem
& Collet; this audit uses the explicit NIST snapshot above. Its CEA refit
curves were not reproduced because the fitted coefficients were not supplied.

Locating the discrepancy within JANAF
-------------------------------------

The tabulated reaction energies satisfy :math:`\Delta G=\Delta H-T\Delta S`
within 0.003 kJ/mol at the sampled temperatures. Thus the discrepancy is not
an isolated typo in the formation-Gibbs column. An independent diagnostic
recovers the internal partition function from the absolute entropy and
thermal enthalpy:

.. math::

   \ln U(T) = \frac{S^\circ(T)}{R}
       - \frac{H^\circ(T)-H^\circ(0)}{RT}
       - \ln\left[\left(\frac{2\pi m k_BT}{h^2}\right)^{3/2}
                   \frac{k_BT}{p^\circ}\right].

Neutral Ni and Co agree with Barklem & Collet to within 0.04% on this grid.
At 3000 K the JANAF Ni+ table instead implies :math:`U\simeq2323`, compared
with 8.29948 in the spectroscopic table. Co+ gives about 34.37 versus 20.9531
at 3000 K; by 6000 K its partition function is about 11.26 times larger.
The large discrepancy is localized to the **ionic electronic thermal
contribution**.

Using JANAF's own recovered partition functions and ionization energies in
Saha restores its equilibrium constants (within about 0.0001 dex at the
checked 1000, 3000 and 6000 K points). Saha's law itself is consistent with
JANAF; the disagreement concerns the ionic electronic data supplied to it.

The 0 K JANAF formation-enthalpy differences imply ionization energies of
7.63743 eV for Ni and 7.86432 eV for Co. Their differences from the adopted
NIST values are too small to explain the large discrepancies. For example,
the Co energy difference changes K by about 3.3% at 6000 K, not a factor 12.

The anomalous ionic heat capacities are also present in the printed fourth
edition: Ni+ has 121.588 J/mol/K at 298.15 K, and Co+ has 94.494 J/mol/K at
2800 K. The original PDF pages reproduce the online S, H, G and log K values;
this is not just an online transcription issue. Their notes describe using
both observed and estimated electronic levels. The abbreviated listings do
not establish which level, extrapolation, or computational step caused the
problem.

Implications for ExoGibbs
-------------------------

The generic JANAF loader supports these tables, but the packaged ``ykb4``
preset does not contain Ni or Co. Selecting JANAF instead of FastChem would
therefore require supplying these species, and using these original tables
would retain the discrepancy.

The separate opt-in ``prepare_fastchem_thermodynamics()`` model already used
NASA9 for Ni1+, following a previous heat-capacity audit. The follow-up
implementation adds Co1+ to that model's replacements, updating chemical
potentials, entropy and heat capacity together. Its public domain remains
1500--4500 K; this investigation directly evaluates NASA records valid at
1000--6000 K.

At fixed electron density, an overestimated K increases :math:`n_{M^+}/n_M`
and reduces the neutral line absorber. This agrees with the direction of
the supplied ``fig2_profile_Ni.png``. Its neutral-Ni factors of roughly 3--30
cannot be independently reproduced from the image alone: they depend on
the pressure-temperature profile, abundances and self-consistent electrons,
and are not simply the K ratio. Those atmospheric calculations were not run.

`PR #280 <https://github.com/HajimeKawahara/exogibbs/pull/280>`_ introduced
``fastchem4_nasa9_ions``, a 539-species preset replacing only Ni1+ and Co1+
with NASA9 reaction potentials on 1000--6000 K. That preset and its dedicated
API tests have been retired in favor of :doc:`presets/curated`, which applies
the subsequent atomic-ion and major-molecule decisions on 1500--4000 K.
The old implementation remains available in the linked PR's history.

This audit evaluates the original source data directly and remains
reproducible without the retired preset. Its regression tests retain the
independent NASA values, Saha conventions and JANAF discrepancy checks.
The shared NASA data and the separate caloric model remain available.

Reproduction and sources
------------------------

Run from the repository root:

.. code-block:: console

   JAX_PLATFORMS=cpu python examples/comparisons/compare_ni_co_ionization.py --output-dir results/ni_co_ionization
   JAX_PLATFORMS=cpu pytest -q tests/unittests/examples/ni_co_ionization_test.py

The script writes ``comparison.csv``, ``comparison.png`` (equilibrium constants)
and ``partition_functions.png`` (JANAF/reference electronic partition ratios).
It uses existing
ExoGibbs FastChem, JANAF and NASA evaluators and the small offline excerpts in
``examples/comparisons/data``. JANAF excerpts contain selected original rows
(including 0 K enthalpy references for the metals); they are not new fits.
Their source URLs are in the headers. Other inputs and source hashes are in
``ni_co_ionization_reference.json``. Runtime and tests need neither ExoJAX
nor network access. Matplotlib is only needed to produce the figure.

* JANAF original tables:
  `Ni <https://janaf.nist.gov/tables/Ni-005.html>`_,
  `Ni+ <https://janaf.nist.gov/tables/Ni-006.html>`_,
  `Co <https://janaf.nist.gov/tables/Co-005.html>`_,
  `Co+ <https://janaf.nist.gov/tables/Co-006.html>`_,
  `electron <https://janaf.nist.gov/tables/D-020.html>`_.
* Printed fourth edition:
  `Nickel PDF, page 5 / printed p. 1702 <https://janaf.nist.gov/pdf/JANAF-FourthEd-1998-Nickel.pdf>`_;
  `Cobalt PDF, page 6 / printed p. 948 <https://janaf.nist.gov/pdf/JANAF-FourthEd-1998-Cobalt.pdf>`_.
* `Barklem & Collet (2016), A&A 588, A96, Table 8 <https://doi.org/10.1051/0004-6361/201526961>`_
  and `NIST ASD ionization energies <https://physics.nist.gov/PhysRefData/ASD/ionEnergy.html>`_.
* `NASA CEA pinned source <https://github.com/nasa/cea/blob/2f79a647e85737742fb08a127da00d1ea1d069da/data/thermo.inp>`_.
  Existing packaged NASA data supply Ni, Ni+, Co and e-; the additional Co+
  excerpt comes from the identical source snapshot. The accompanying NASA
  license and notice are in ``src/exogibbs/data/nasa9``.
