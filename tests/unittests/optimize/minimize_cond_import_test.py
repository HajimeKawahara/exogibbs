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

        raw_module = "exogibbs.optimize.pipm_rgie_cond"
        loaded_before = raw_module in sys.modules
        module = importlib.import_module("exogibbs.optimize.minimize_cond")
        print(
            json.dumps(
                {
                    "loaded_before": loaded_before,
                    "loaded_after": raw_module in sys.modules,
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
    assert payload["loaded_after"] is False
    assert Path(payload["module_file"]).resolve() == (
        SOURCE_ROOT / "exogibbs" / "optimize" / "minimize_cond.py"
    ).resolve()
    assert payload["exports_restricted_solver"] is True
    assert payload["exports_diagnostic_wrapper"] is True
