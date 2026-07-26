"""Import-boundary tests for the legacy condensate compatibility facade."""

from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess
import sys
import textwrap


REPOSITORY_ROOT = Path(__file__).resolve().parents[3]
SOURCE_ROOT = REPOSITORY_ROOT / "src"


def test_minimize_cond_import_does_not_load_raw_pipm_diagnostics() -> None:
    script = textwrap.dedent(
        """
        import importlib
        import json
        import sys

        import jax.numpy as jnp

        raw_module = "exogibbs.optimize.pipm_rgie_cond"
        loaded_before = raw_module in sys.modules
        module = importlib.import_module("exogibbs.optimize.minimize_cond")
        loaded_after_import = raw_module in sys.modules
        startup = module.build_rgie_condensate_init_from_policy(
            epsilon=-10.0,
            support_indices=jnp.asarray([0, 1], dtype=jnp.int32),
            startup_policy="ratio_uniform_r0",
            r0=1.0e-3,
        )
        pi = module._recompute_pi_for_residual(
            nk=jnp.asarray([0.8]),
            mk=jnp.asarray([0.2]),
            ntot=0.8,
            formula_matrix=jnp.asarray([[1.0]]),
            formula_matrix_cond=jnp.asarray([[1.0]]),
            b=jnp.asarray([1.0]),
            gk=jnp.asarray([0.0]),
            hvector_cond=jnp.asarray([0.0]),
            epsilon=-10.0,
        )
        inactive = module.summarize_rgie_inactive_driving(
            jnp.asarray([0.0, 0.25]),
            jnp.asarray([0], dtype=jnp.int32),
        )
        print(
            json.dumps(
                {
                    "loaded_before": loaded_before,
                    "loaded_after_import": loaded_after_import,
                    "loaded_after_small_helper": raw_module in sys.modules,
                    "startup_shape": list(startup.shape),
                    "pi_shape": list(pi.shape),
                    "inactive_positive_count": inactive["inactive_positive_count"],
                    "module_file": module.__file__,
                    "exports_restricted_solver": (
                        "solve_restricted_support_condensate_layer"
                        in module.__all__
                    ),
                    "exports_diagnostic_wrapper": (
                        "trace_condensate_reduced_solver_backends"
                        in module.__all__
                    ),
                }
            )
        )
        """
    )
    environment = os.environ.copy()
    environment["PYTHONPATH"] = str(SOURCE_ROOT)
    completed = subprocess.run(
        [sys.executable, "-c", script],
        check=True,
        capture_output=True,
        cwd=REPOSITORY_ROOT,
        env=environment,
        text=True,
    )
    payload = json.loads(completed.stdout.splitlines()[-1])

    assert payload["loaded_before"] is False
    assert payload["loaded_after_import"] is False
    assert payload["loaded_after_small_helper"] is False
    assert payload["startup_shape"] == [2]
    assert payload["pi_shape"] == [1]
    assert payload["inactive_positive_count"] == 1
    assert Path(payload["module_file"]).resolve() == (
        SOURCE_ROOT / "exogibbs" / "optimize" / "minimize_cond.py"
    ).resolve()
    assert payload["exports_restricted_solver"] is True
    assert payload["exports_diagnostic_wrapper"] is True
