"""Control results kept separate from zero-barrier diagnostic serialization."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np


@dataclass(frozen=True)
class ZeroBarrierAudit:
    """Independent acceptance and transition permissions for one exact state.

    A certified terminal root and a root allowed to change support are distinct:
    optimizer status zero can authorize the former only. Signed active amounts
    may authorize deletion when every equality block passes. Diagnostics are an
    output of this decision, never its mutable control state.
    """

    accepted: bool
    local_kkt_failure_reasons: tuple[str, ...]
    root_blocks_passed: bool
    full_driving: np.ndarray
    _diagnostics: dict[str, Any] = field(repr=False)

    @property
    def local_kkt_passed(self) -> bool:
        return not self.local_kkt_failure_reasons

    def to_report(self) -> dict[str, Any]:
        """Export the historical audit schema without exposing control arrays."""

        return {
            key: value.copy() if isinstance(value, np.ndarray) else value
            for key, value in self._diagnostics.items()
        }
