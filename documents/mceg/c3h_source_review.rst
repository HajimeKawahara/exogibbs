C3H source, identity and caloric review
=======================================

The inherited C3H record reproduces Tsuji's 1973 table closely, but it is not yet
a defensible modern thermal model. For explicitly identified linear CCCH, the
ATcT energy evaluation is the preferred starting point. A production replacement
requires resolving the historical species definition and the high-temperature
state sum. This review retains the current coefficients operationally and makes
no abundance-substitution claim.

Why this molecule matters
-------------------------

The existing ``curated-c2h-v1`` archive contains 75 ideal-gas cases with 537
ion-inclusive species. Its maximum C3H1 carbon share is **11.6431%** at 2000 K,
10^-6 bar and C/O = 1.5, with mole fraction 3.14160e-5. This is a model result,
not an observed abundance. The screen spans 1500--4000 K, five pressures and
three C/O values; it cannot establish a universal ranking. C3H was outside the
original 93-species NASA/JANAF comparison, so that frozen comparison is unchanged.

What is identified
------------------

The current FastChem row is ``C3H1 2-Proynylidene`` with formula C3H, citing
`Tsuji (1973), A&A 23, 411--431
<https://articles.adsabs.harvard.edu/pdf/1973A%26A....23..411T>`_. The inspected
Table 4 names only C3H. The label and formula alone do not recover its isomer,
electronic state or isotope convention. The earlier Tsuji 1964 construction
remains unread after unsuccessful public-access attempts.

Modern identities must be kept separate:

* Linear CCCH, ATcT ID ``53590-28-6*0``, is 2-propynylidyne.
* Cyclic CH(CC), ID ``119178-99-3*0``, is cyclopropynyl.
* HCCCH, often called propynylidene, is **C3H2**. Propargyl is C3H3.
  Neither is a formula match, and the charged C3H species are also distinct.

Recovering the printed source
-----------------------------

Tsuji's Eq. (7) uses log10 of the dissociation constant, with theta = 5040/T.
The pressure convention is dyn/cm2, as identified in the printed pressure table.
For C3H -> 3 C + H, conversion to dimensionless formation K at 1 bar is

.. math::

   \ln K_\mathrm{form}^{\circ}
     = \ln(10)\left[18-\sum_{i=0}^{4}a_i(5040/T)^i\right].

The 18 is 3 log10(10^6): the dissociation increases particle number by three,
and 1 bar = 10^6 dyn/cm2. Reversing the reaction supplies the other sign.
The five printed coefficients are 40.791, -21.762, 0.93377, -0.13863 and
0.0074549. At the five reviewed 1500--4000 K nodes, the maximum absolute
current-minus-printed residual is **0.023871 in ln K**, about 2.42% in K.
The audit records the separate bound from displayed coefficient rounding.
That bound is only 0.0026--0.0056 in ln K at these nodes, so rounding alone does
not account for the residual; the two fitted representations agree approximately.
Agreement supports the cited ancestry, not the physical accuracy of the data
or recovery of the original fit-generation procedure.

Energy and thermal functions are separate decisions
---------------------------------------------------

`ATcT 1.172 linear CCCH
<https://atct.anl.gov/Thermochemical%20Data/version%201.172/species/?species_number=360>`_
gives Hf(0 K) = 718.75 and Hf(298.15 K) = 723.10 +/- 0.71 kJ/mol.
`The cyclic entry
<https://atct.anl.gov/Thermochemical%20Data/version%201.172/species/?species_number=750>`_
gives 711.24 and 715.37 +/- 0.71 kJ/mol. The uncertainties are estimated 95%
confidence limits. The cyclic species lies lower by 7.51 kJ/mol at 0 K; this
does not turn it into an interchangeable linear-species record.

Using C and H from the same ATcT version, the balanced linear reaction
3 C(g) + H(g) -> CCCH(g) has H(0 K) = -1631.427 and H(298.15 K) =
-1645.496 kJ/mol. No reaction uncertainty is obtained by independent quadrature:
ATcT reports shared correlations, including linear/cyclic 0.324 and linear/C
0.144. The network combines crossed-beam measurements, photoelectron data and
theory. These contributing studies are not independent confirmations of the
network evaluation. Its Costes 2006 input explicitly has an enlarged uncertainty.

The inspected `Burcat and Ruscic (2005) author-hosted compilation
<https://atct.anl.gov/~ruscic/reprints/Burcat%202005.pdf>`_ supplies a separate
linear candidate, A7/05, with G3B3 energy and experimental/calculated vibrational
inputs. Its NASA9 header Hf(298.15 K) is 719392.776 J/mol. It also documents
older symmetry-number disagreements. A modern compilation date therefore does
not establish a complete treatment of the electronic and vibronic states.

The following **conditional caloric comparison** uses the inspected Burcat
polynomial and the same frozen NASA C/H atomic functions for both sides.
The current reaction H and Cp follow analytically from its ln K fit. Restoring
those common atomic functions gives the molecular Cp below; it is not an
independent measurement or proof of the original Tsuji thermal construction.

.. list-table:: Caloric disagreement within the review range
   :header-rows: 1

   * - T (K)
     - Current minus Burcat reaction H (kJ/mol)
     - Current Cp (J/mol/K)
     - Burcat Cp (J/mol/K)
   * - 1500
     - -173.694
     - 59.460
     - 75.845
   * - 3000
     - -219.333
     - 43.903
     - 80.901
   * - 4000
     - -253.642
     - 52.946
     - 82.032

This exposes a large disagreement in both reaction enthalpy and thermal shape.
It is not a measured error bar, and the energy offset cannot be assigned solely
to a historical Hf value without the original construction. The current fit is
not extrapolated to 0 or 298 K. Burcat S/G/K are not compared because its standard
pressure convention has not been independently established in the inspected
record; ideal-gas H and Cp do not depend on that choice.

`CDMS entry 037501
<https://cdms.astro.uni-koeln.de/classic/entries/e037501.cat>`_ identifies linear
X2Pi_r and a Renner--Teller Sigma component only 27.20 cm^-1 above the lowest
Pi1/2 rotational state. Its tabulated partition function covers 9.375--300 K
and selected states. It cannot be extended into a complete 1500--4000 K model
without examining higher vibrational/electronic states and normalization.

Decision, MCEG and next evidence
--------------------------------

The source-replay and archived carbon-accounting claims are supported by offline
numerical checks. The conditional ATcT energy preference rests on the explicit
modern species and its evaluated experimental/theoretical network. The complete
replacement and atmospheric-accuracy claims remain unknown in
``mceg/contracts/c3h_source_review.yaml``.

Next, recover the Tsuji 1964 state/geometry and atomization inputs, reconcile the
linear vibronic partition with modern spectroscopy, and assess the cyclic
species separately. A justified candidate can then be evaluated in a fixed
network substitution. Adding an isomer is a different network change. No new
equilibrium solve, coefficient replacement, spectrum, condensation or kinetic
calculation is part of this investigation.

The packet lives in ``tests/data/validation/c3h_source_review/20261102``. Its
README documents the inspected pages and offline replay command. This PR's
validation includes the targeted model-evidence checks and MCEG assessments;
numerical agreement is kept separate from physical validation.
