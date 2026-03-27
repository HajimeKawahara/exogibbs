from __future__ import annotations

from datetime import datetime
from datetime import timezone
import math
from pathlib import Path
from typing import Any
from typing import Mapping
from typing import Optional

import jax
import jax.numpy as jnp

from exogibbs.io.load_data import get_data_filepath


def device_for_platform(platform: Optional[str]) -> tuple[str, jax.Device]:
    if platform is None:
        return jax.default_backend(), jax.devices()[0]
    devices = jax.devices(platform)
    if not devices:
        raise RuntimeError(f"No JAX device found for platform={platform!r}.")
    return platform, devices[0]


def to_python(value: Any) -> Any:
    if value is None:
        return None
    if isinstance(value, dict):
        return {str(key): to_python(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [to_python(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    if hasattr(value, "tolist"):
        converted = value.tolist()
        return to_python(converted)
    if isinstance(value, (bool, int, float, str)):
        return value
    return value


def has_nan_tree(tree: Any) -> bool:
    leaves = jax.tree_util.tree_leaves(tree)
    for leaf in leaves:
        if hasattr(leaf, "dtype") and jnp.issubdtype(leaf.dtype, jnp.inexact):
            if bool(jnp.any(jnp.isnan(leaf)).item()):
                return True
    return False


def extract_diag_value(diagnostics: Optional[Mapping[str, Any]], key: str) -> Any:
    if diagnostics is None:
        return None
    if key not in diagnostics:
        return None
    return to_python(jax.device_get(diagnostics[key]))


def load_normalized_element_abundances(
    element_file: str,
    element_order: tuple[str, ...],
) -> jax.Array:
    """Convert FastChem-style A(X)=log10(n_X/n_H)+12 abundances to normalized linear abundances."""
    path = get_data_filepath(element_file)
    log_abundances = {}
    with open(path, "r", encoding="utf-8") as handle:
        for raw_line in handle:
            line = raw_line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split()
            if len(parts) < 2:
                continue
            log_abundances[parts[0]] = float(parts[1])

    linear_abundances = []
    for element in element_order:
        if element == "e-":
            linear_abundances.append(0.0)
            continue
        if element not in log_abundances:
            linear_abundances.append(1.0e-14)
            continue
        linear_abundances.append(math.pow(10.0, log_abundances[element] - 12.0))

    abundances = jnp.asarray(linear_abundances)
    total = jnp.sum(abundances)
    return abundances / jnp.where(total > 0.0, total, 1.0)


def current_timestamp_utc() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")
