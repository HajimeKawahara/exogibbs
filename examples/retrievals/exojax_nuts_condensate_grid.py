"""ExoJAX NUTS retrieval with a fixed-support condensate grid initializer.

The chemistry, fixed graphite support, priors, mock data, and reverse-mode
NUTS settings are shared with ``exojax_nuts_condensate_fixed_support.py``.
This wrapper additionally precomputes a shared gas grid and local graphite
fixed-support grids, then interpolates gas and condensate initial values at
every NUTS evaluation.  Support discovery, phase transitions, the production
lifecycle, and rainout remain outside AD.
"""

from exojax_nuts_condensate_fixed_support import run_condensate_demo


if __name__ == "__main__":
    run_condensate_demo(
        use_grid_initializer=True,
        case_name="condensate_grid",
    )
