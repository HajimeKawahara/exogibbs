"""ExoJAX NUTS retrieval through the gas VJP with a grid initializer.

The likelihood, priors, mock data, and reverse-mode NUTS settings are shared
with ``exojax_nuts_gas_no_grid.py``.  The only algorithmic difference is the
packaged FastChem ``GridEquilibriumInitializer`` supplied to each gas solve.
The custom VJP stops initialization gradients, so converged spectra and
posterior derivatives should agree while the primal iteration count drops.
"""

from _exojax_nuts_common import run_gas_demo


if __name__ == "__main__":
    raise SystemExit(
        run_gas_demo(
            use_grid_initializer=True,
            case_name="gas_grid",
        )
    )
