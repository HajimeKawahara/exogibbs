Auditing a gas amount below float64 range
=========================================

An ideal-gas chemical potential depends on ``log(n)``, even when ``n`` is too
small to represent as a positive float64 number. Pass the optional
``gas_log_amounts_mol`` array to retain this information. These are natural
logarithms of **absolute mol amounts**, in the same species order as
``gas_amounts_mol``. The audit evaluates

.. math::

   \mu_i/(RT)=g_i^\circ/(RT)+\ell_i
     -\operatorname{logsumexp}(\ell)+\log(P/\mathrm{bar}),
   \qquad \ell_i=\log(n_i/\mathrm{mol}).

The elemental potentials are still fitted independently from these chemical
potentials; gas stationarity, present/absent pure-condensate insertion,
temperature eligibility and elemental conservation retain their original
tolerances. A caller evaluating a solver result must also require the
solver's separate convergence flag. Finite logs alone do not establish an
equilibrium.

Reachable species require finite logs. Species excluded by an exactly zero
element budget require both zero linear amounts and ``-inf`` logs. For a
reachable species, a zero linear amount is accepted only if ``exp(log_amount)``
also underflows float64. Thus dropping a representable trace species or
introducing a composition floor is rejected. Positive normal amounts must
agree with their logs within ``64*eps*max(1, abs(log_amount))`` in log space;
positive subnormal amounts allow one subnormal quantum of rounding difference.
The representation check is separate from the unchanged chemical and element
residual tolerances.

For each underflowed species its exact positive amount is smaller than one
float64 subnormal quantum, :math:`q=\operatorname{nextafter}(0,1)` mol. The
audit adds an outward-rounded bound

.. math::

   u_e \ge q\sum_{i\in U}|A_{ei}|,
   \qquad
   \frac{|(A_g n_g+A_c n_c-b)_e|+u_e}{b_e}\le 10^{-9}

for every nonzero elemental budget. Zero-budget rows remain exactly zero.
``gas_underflow_species`` and ``gas_underflow_element_bound_mol`` report the
omitted species and this bound; ``relative_element_residual_upper_bound``
reports the quantity used in the element gate. The bound concerns only
underflowed gas amounts, not all floating-point error. Passing the audit does
not certify the empirical thermodynamic data or an unlisted phase.
``gibbs_rt`` still uses the linear mol amounts; it is not an interval bound
on the omitted trace contribution to Gibbs energy.

This standalone ExoGibbs example uses a synthetic H/H2 gas with dimensionless
standards 0 and 800 and an unfavorable pure solid. Its trace H2 amount has a
finite log of -800, although its linear representation is zero:

.. code-block:: python

   import numpy as np
   from exogibbs.api import condensate
   from exogibbs.thermo.models import ChemicalSetup

   gas = ChemicalSetup(
       formula_matrix=np.array([[1., 2.]]),
       elements=("H",), species=("H", "H2"),
       hvector_func=lambda t: np.array([0., 800.]))
   solid = ChemicalSetup(
       formula_matrix=np.array([[1.]]),
       elements=("H",), species=("H(s)",),
       hvector_func=lambda t: np.array([1.]))
   setup = condensate.build_condensate_chemical_setup(
       gas_setup=gas, condensate_setup=solid)
   log_mol = np.array([0., -800.])
   report = condensate.audit(
       setup, 300., 1., [1.], np.exp(log_mol), [0.],
       gas_log_amounts_mol=log_mol)
   assert report["accepted"]
   assert report["gas_underflow_species"] == ["H2"]

Omitting ``gas_log_amounts_mol`` retains the strict positive-linear-amount
contract, so the same state raises ``ValueError``. Replacing -800 by -500
while leaving the second linear amount zero also raises ``ValueError``:
that amount is representable and was incorrectly truncated.

If a solve used a normalized atom budget ``b / scale``, convert its returned
gas logs with ``log_mol = result.gas_ln_n + np.log(scale)`` and construct the
absolute gas amounts as ``np.exp(log_mol)``. This also restores a trace amount
that underflowed on the normalized basis but is representable on the absolute
basis. Multiply condensate amounts by the same ``scale``. Do not pass
normalized gas logs together with absolute mol amounts.

