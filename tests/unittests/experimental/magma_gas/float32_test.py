"""Default-precision regression for the documented magma--gas solve."""

from __future__ import annotations

import os
from pathlib import Path
import subprocess
import sys
import textwrap


_FLOAT32_SMOKE = textwrap.dedent(
    """
    import jax.numpy as jnp

    from exogibbs.experimental.magma_gas import (
        prepare_meltyq_chemistry,
        solve_magma_atmosphere_interface,
    )
    from exogibbs.presets.ykb4 import chemsetup

    chemistry = prepare_meltyq_chemistry(
        chemsetup(),
        species_map={
            "He": "He1",
            "H2O": "H2O1",
            "CO": "C1O1",
            "CO2": "C1O2",
            "CH4": "C1H4",
            "NH3": "H3N1",
        },
    )
    state = solve_magma_atmosphere_interface(
        chemistry,
        temperature_melt_k=1700.0,
        pressure_melt_bar=7000.0,
        oxygen_fugacity_bar=1.0e-10,
        co_melt_mole_ratio=5.0e-5,
        n_melt_mole_ratio=1.0e-4,
    )
    diagnostics = state.diagnostics
    assert state.root_variables.dtype == jnp.float32
    assert bool(diagnostics.outer_converged), diagnostics
    assert bool(diagnostics.inner_converged), diagnostics
    assert bool(diagnostics.converged), diagnostics
    assert diagnostics.residual_norm <= diagnostics.root_tolerance
    assert diagnostics.inner_residual_norm <= diagnostics.inner_tolerance
    expected_tolerance = 8.0 * jnp.finfo(jnp.float32).eps
    assert diagnostics.inner_tolerance == expected_tolerance
    """
)


def test_documented_ykb4_point_converges_with_default_float32() -> None:
    repository_root = Path(__file__).resolve().parents[4]
    environment = os.environ.copy()
    environment["JAX_ENABLE_X64"] = "0"
    environment["JAX_PLATFORMS"] = "cpu"
    environment["JAX_PLATFORM_NAME"] = "cpu"
    environment["PYTHONPATH"] = os.pathsep.join(
        (str(repository_root / "src"), environment.get("PYTHONPATH", ""))
    )
    completed = subprocess.run(
        [sys.executable, "-c", _FLOAT32_SMOKE],
        cwd=repository_root,
        env=environment,
        capture_output=True,
        text=True,
        timeout=120,
        check=False,
    )

    assert completed.returncode == 0, completed.stdout + completed.stderr
