from __future__ import annotations

from dataclasses import asdict
from dataclasses import dataclass
from typing import Any
from typing import Dict
from typing import Optional


JsonDict = Dict[str, Any]


@dataclass(frozen=True)
class ExecutionConfig:
    backend: Optional[str]
    platform: Optional[str]
    dtype: Optional[str]
    warmup_count: int
    repeat_count: int
    initializer_mode: str = "none"
    initializer_grid_path: Optional[str] = None
    initializer_preset_name: Optional[str] = None

    def to_dict(self) -> JsonDict:
        return asdict(self)


@dataclass(frozen=True)
class BenchmarkCase:
    case_id: str
    category: str
    setup_metadata: JsonDict
    axes: JsonDict
    solver_options: JsonDict
    pref_bar: float
    setup_kwargs: JsonDict

    def to_dict(self) -> JsonDict:
        return asdict(self)


@dataclass(frozen=True)
class BenchmarkResult:
    benchmark_version: str
    case_id: str
    category: str
    timestamp_utc: str
    setup_metadata: JsonDict
    axes: JsonDict
    solver_options: JsonDict
    execution_config: ExecutionConfig
    metrics: JsonDict
    status: str

    def to_dict(self) -> JsonDict:
        payload = asdict(self)
        payload["execution_config"] = self.execution_config.to_dict()
        return payload
