"""Shared validation for condensate thermochemistry presets."""

from typing import Iterable, Mapping, Sequence, Tuple


def validate_condensate_elements(
    components: Iterable[Tuple[str, Mapping[str, int]]],
    elements: Sequence[str],
) -> None:
    """Reject species whose elemental inventories the gas setup cannot track."""
    available = set(elements)
    incompatible = [
        (name, sorted(set(counts) - available))
        for name, counts in components
        if set(counts) - available
    ]
    if not incompatible:
        return
    missing = sorted({element for _, values in incompatible for element in values})
    examples = ", ".join(
        f"{name} ({'/'.join(values)})" for name, values in incompatible[:3]
    )
    raise ValueError(
        "Condensate data are incompatible with gas_setup.elements; "
        f"missing elements: {', '.join(missing)}. "
        f"Affected species include: {examples}."
    )
