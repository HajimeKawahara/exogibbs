"""ExoJAX NUTS retrieval through the gas-equilibrium VJP, without a grid.

This is the direct analogue of the retrieval section in ExoJAX's
``equilibrium_chemistry.ipynb``: every atmospheric layer starts from the
default uniform gas initialization.  Use ``--preflight-only`` before a long
GPU run and ``--quick`` for a short end-to-end exercise.
"""

from _exojax_nuts_common import run_gas_demo


if __name__ == "__main__":
    raise SystemExit(
        run_gas_demo(
            use_grid_initializer=False,
            case_name="gas_no_grid",
        )
    )
