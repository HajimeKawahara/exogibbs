"""Backward-compatible import path for condensate minimization.

This module preserves the legacy ``minimize_gibbs_cond_core`` import while also
exposing a diagnostics-returning wrapper around the active-in-practice RGIE
condensate solver used by the repository examples.
"""

from exogibbs.optimize.pdipm_cond import minimize_gibbs_cond_core
from exogibbs.optimize.pipm_rgie_cond import minimize_gibbs_cond_with_diagnostics

__all__ = ["minimize_gibbs_cond_core", "minimize_gibbs_cond_with_diagnostics"]
