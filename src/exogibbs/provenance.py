"""Find optional scientific records using declared setup metadata.

This module uses only the standard library. It neither imports MCEG nor
evaluates thermodynamics. Matching metadata is not execution verification.
"""

import json
from pathlib import Path
import shlex
from typing import Any, Dict, TYPE_CHECKING

from exogibbs import __version__

if TYPE_CHECKING:
    from exogibbs.thermo.models import ChemicalSetup


def evidence_catalog() -> Path:
    """Return the bundled catalog path for optional ``mceg explain --catalog``.

    This also works in an editable checkout; MCEG does not need to import
    ExoGibbs to read the returned file.
    """
    return Path(__file__).with_name("data") / "mceg" / "index.json"


def describe(setup: "ChemicalSetup") -> Dict[str, Any]:
    """Describe declared model provenance and the optional evidence entry point.

    The returned dictionary is JSON-serializable. It matches the dataset hash
    and choices recorded in ``setup.metadata``, not the callable, numerical
    results or user's conditions. Custom setups with copied metadata are not
    independently identified or validated by this helper.
    Unrecognized metadata field types are reported as ``None``.

    The catalog distinguishes the frozen ion-inclusive curated model from its
    opt-in C2H correction. Missing records do not establish physical reliability.
    """
    catalog_path = evidence_catalog()
    catalog = json.loads(catalog_path.read_text())
    metadata = setup.metadata or {}
    dataset = metadata.get("dataset")
    dataset = dataset if isinstance(dataset, str) else None
    digest = metadata.get("data_sha256")
    digest = digest if isinstance(digest, str) else None
    ions = metadata.get("ions")
    settings = {"ions": ions if type(ions) is bool else None}
    model = next((item for item in catalog["models"]
                  if item["dataset"] == dataset
                  and item["identity"]["data_sha256"] == digest
                  and all(type(settings.get(key)) is type(value) and settings.get(key) == value
                          for key, value in item["identity"]["settings"].items())), None)
    return {
        "provider": "exogibbs",
        "package_version": __version__,
        "dataset": dataset,
        "data_sha256": digest,
        "settings": settings,
        "model_id": model["id"] if model else None,
        "record_match": "declared_metadata" if model else "no_matching_record",
        "record_version": catalog["record_version"],
        "catalog": str(catalog_path),
        "notice": (
            "Scientific records and known thermochemical differences are available; "
            "physical accuracy and your calculation conditions are not evaluated here. "
            "Matching user-editable metadata does not verify a custom setup or a calculation."
            if model else "No bundled record matches the declared dataset and settings. "
            "This does not establish physical reliability."),
        "explain_command": shlex.join([
            "mceg", "explain", "--catalog", str(catalog_path), "--model", model["id"],
        ]) if model else None,
    }


__all__ = ("describe", "evidence_catalog")
