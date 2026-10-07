"""Results of one numerical solve on an unchanged zero-barrier support."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np


@dataclass(frozen=True)
class FixedSupportState:
    """Physical coordinates returned without clipping signed phase amounts."""

    gas_log_amounts: np.ndarray
    condensate_amounts: np.ndarray
    total_gas_log_amount: float
    element_potential: np.ndarray


@dataclass(frozen=True)
class FixedSupportAttempt:
    """One solve, without physical certification or a support transition.

    The requested support is preserved even when an active amount is negative
    or reaches a numerical bound. The controller audits the returned state
    before deciding whether to retry its representation or change support.
    A missing state denotes an ineligible, exhausted, or exceptional attempt.
    Evaluation charges include conservative charges for optimizer exceptions.
    """

    support_indices: tuple[int, ...]
    state: FixedSupportState | None
    optimizer_success: bool
    optimizer_status: int | None
    optimizer_message: str
    function_evaluations: int
    report: dict[str, Any]
    amount_scales: np.ndarray | None = None
    phase_coordinates: np.ndarray | None = None
    lower_bound_support_indices: tuple[int, ...] = ()


@dataclass(frozen=True)
class NumericalRestart:
    """Initial values and scaling for a specified ordered support.

    A restart cannot infer another support from diagnostics. If a preceding
    controller dropped a phase, its returned support must be supplied explicitly.
    """

    initializer: str
    variable_scaling: str
    restart_from_terminal_state: bool
    support_indices: tuple[int, ...]
    state: FixedSupportState
