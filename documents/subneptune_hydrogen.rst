Common hydrogen standards and host dilution
=============================================

``examples/metal_silicate/hydrogen.py`` introduces a separate conditional
model, ``source_host_ma_fe_si_o_h_hirschmann2012_pressure_control_v1``. It
uses the native ExoEOS ``MaFeSiOHLiquid`` control, finite elemental amounts,
and the source's ideal silicate host. The archived ``source_full``,
``completed_dry``, and ``gas_exchange`` cases remain their original models.
This control is not a calibrated melt/metal/gas benchmark or a phase search.

Pressure and common species standards
---------------------------------------

The named older sensitivity law is

.. math::

   x_{H_2,\ell}=(f_{H_2}/\mathrm{bar})
       \exp[-11.403-0.76(P_\ell/\mathrm{GPa})].

This is the Hirschmann (2012) formulation used through Seo (2024), also
available as ``exogibbs.solubility.h2_hirschmann2012``. Its experimental
metadata span approximately 1673--1773 K and 0.7--3 GPa. Fugacity is H2
fugacity, whereas the exponential pressure term uses total melt pressure.
The coefficient must not be applied against arbitrary total fluid pressure.
A source association-component mole fraction is a formal concentration
basis; matching a real host's experimental normalization remains a gate.

For ``ln_K = ln(x_H2 / (f_H2/bar))``, the example's
``dissolved_h2_standard_rt(mu0_g_rt, ln_K, standard_pressure_bar=1.0)`` returns

.. math::

   \frac{\mu^\circ_{H_2,\ell}}{RT}=
   \frac{\mu^\circ_{H_2,g}}{RT}-\ln(KP^\circ),\qquad
   0=\frac{\mu^\circ_{H_2,\ell}-\mu^\circ_{H_2,g}}{RT}
       +\ln x_{H_2,\ell}-\ln(f_{H_2}/P^\circ).

Pressure work is already contained in this selected law. No extra volume
integral or source R14 pressure offset is added. Replacing the H2 species
standard changes every affected reaction, including R4, R6, and R14.
``make_common_standard_potentials_rt`` also adds the native alloy's Fe/Si/O/H
standard shifts. Its H shift is zero; the absolute H standard is inherited
from the source's Okuchi-based reconstruction and is generally nonzero.
The source R14 offset remains confined to source reproduction.

The revised Chaudhari et al. (2025), doi:10.1007/s00410-025-02272-y,
tabulated data and host-specific fugacity calibration are not vendored.
The design identifies Fe-free basalt/andesite direct experiments through
1673.15 K. No new universal fit or factor-of-ten rescaling is supplied.
``H2_CALIBRATION`` records the missing-data gate. An old/new comparison,
experimental partition residuals, and resulting partition uncertainty remain
unavailable until the actual host, concentration basis, gas EOS, and tables
are supplied. The compatibility solubility function is unchanged.

Adding molecular H2 to an already mixed host
----------------------------------------------

``ideal_host_h2_dilution(host_gibbs_rt, host_mu_rt, host_amounts, h2_amount,
h2_standard_rt)`` returns ``(gibbs_rt, host_mu_rt, h2_mu_rt)``. All potentials
must first be converted to one common gas constant: divide the backend's
G in J and chemical potentials in J/mol by that common ``R*T``. In particular,
MELTS' backend activity logarithms are not automatically common-R logarithms.

For host endmember amounts totaling :math:`N_d`, molecular H2 amount
:math:`n_H`, and :math:`N=N_d+n_H`, the construction is

.. math::

   G/(RT)=G_{host}/(RT)+n_H\mu^\circ_{H_2}/(RT)
       +N_d\ln(N_d/N)+n_H\ln(n_H/N),

   \mu_{i,host}/(RT)=\mu_{i,host}^{native}/(RT)+\ln(N_d/N),\qquad
   \mu_{H_2}/(RT)=\mu^\circ_{H_2}/(RT)+\ln(n_H/N).

Thus all host components receive the reciprocal dilution correction. The
already evaluated host ideal/excess mixing and pressure work are used once.
``host_amounts`` are moles of the declared host endmembers, not oxide or atom
moles. The H2 standard must be independent of host composition; a
composition-dependent fit needs its own scalar energy and reciprocal
host-potential derivatives. MELTS H2O potentials must not be accompanied by
a second independent H2O-solubility constraint.

The mathematical domain requires positive total host, nonnegative component
amounts, and nonnegative H2. Exact zero H2 preserves the host G and potentials;
its absent-component chemical potential is ``-inf``. Pure H2 has no host
endpoint. Derivatives are tested on positive interior compositions.

Runnable finite-inventory control
-----------------------------------

Select both source checkouts explicitly so an older installed ExoEOS is not
used. The output records actual import paths, commits, changed tracked-file
hashes, source/fixture hashes, component formulas, amount conventions,
versions, dtype, inputs, and independently recomputed residuals.

.. code-block:: console

   JAX_PLATFORMS=cpu JAX_ENABLE_X64=1 PYTHONPATH=src:/path/to/exoeos/src python examples/metal_silicate/hydrogen.py
   JAX_PLATFORMS=cpu JAX_ENABLE_X64=1 PYTHONPATH=src:/path/to/exoeos/src python examples/metal_silicate/hydrogen.py --hydrogen-scale 0
   PYTHONPATH=src:/path/to/exoeos/src pytest tests/unittests/examples/metal_silicate_hydrogen_test.py

The numerical 2350 K, 1 bar control retains finite Mg/Si/Fe/O/H/Na/C and
inert He, initialized from the source reference. H2, H2O, CO, and CO2 remain
source silicate components. This state extrapolates the old H2 law and source
MgO liquid fit. No joint liquid stability evidence exists here; a root is
accepted only for the prescribed conditional assemblage. SiH4 and H/OH
species convergence, MELTS integration, competing phases, metal S/C/N,
and the missing alloy pressure response remain separate work.

``build_source_control`` accepts ``pure_lnphi_func(T_K, P_bar, None)`` in
source gas order followed by He. This supports the existing ExoEOS pure
fugacity adapter under the explicitly chosen pure-component/ideal-mixture
approximation. The callback does not implement a composition-dependent
mixture EOS. Regression controls compare ideal and specified fugacity
corrections at identical inputs; they do not establish a validated real-gas
calibration.

The tests cover finite elemental/H/He closure, exact zero H, amount scaling,
pressure/fugacity normalization, common reaction cycles and Okuchi inheritance,
all-host dilution, Euler's relation, symmetric amount derivatives,
finite-difference chemical potentials, and JIT. Acceptance requires relative
element residual below ``1e-9`` and dimensionless reaction residual below
``1e-8``. Numerical agreement is distinct from the missing empirical and
stable-phase acceptance gates.
