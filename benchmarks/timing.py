from __future__ import annotations

from dataclasses import dataclass
import math
import statistics
import time
from typing import Any
from typing import Callable
from typing import Sequence

import jax


def block_tree(tree: Any) -> Any:
    """Synchronize all array leaves before reading host timing."""
    return jax.tree_util.tree_map(
        lambda x: x.block_until_ready() if hasattr(x, "block_until_ready") else x,
        tree,
    )


@dataclass(frozen=True)
class TimingResult:
    first_call_wall_s: float
    warm_call_wall_s: list[float]
    warm_call_mean_s: float
    warm_call_median_s: float
    warm_call_p95_s: float
    warm_call_min_s: float
    last_result: Any


def _percentile(values: Sequence[float], q: float) -> float:
    if not values:
        raise ValueError("values must not be empty")
    if len(values) == 1:
        return float(values[0])
    rank = (len(values) - 1) * q
    lower = math.floor(rank)
    upper = math.ceil(rank)
    if lower == upper:
        return float(values[lower])
    weight = rank - lower
    return float(values[lower] * (1.0 - weight) + values[upper] * weight)


def time_first_and_repeated_calls(
    fn: Callable[..., Any],
    *args: Any,
    warmup_count: int,
    repeat_count: int,
) -> TimingResult:
    """Measure one first call, then warmups, then repeated warm calls."""
    first_t0 = time.perf_counter()
    first_result = fn(*args)
    block_tree(first_result)
    first_call_wall_s = time.perf_counter() - first_t0

    for _ in range(warmup_count):
        block_tree(fn(*args))

    warm_call_wall_s = []
    last_result = first_result
    for _ in range(repeat_count):
        t0 = time.perf_counter()
        last_result = fn(*args)
        block_tree(last_result)
        warm_call_wall_s.append(time.perf_counter() - t0)

    if not warm_call_wall_s:
        raise ValueError("repeat_count must be at least 1")

    sorted_timings = sorted(warm_call_wall_s)
    return TimingResult(
        first_call_wall_s=first_call_wall_s,
        warm_call_wall_s=warm_call_wall_s,
        warm_call_mean_s=statistics.mean(warm_call_wall_s),
        warm_call_median_s=statistics.median(warm_call_wall_s),
        warm_call_p95_s=_percentile(sorted_timings, 0.95),
        warm_call_min_s=min(warm_call_wall_s),
        last_result=last_result,
    )

