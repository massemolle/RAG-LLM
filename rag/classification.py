"""
Data Classification for RAG Document Ingestion

Implements a 5-level classification policy:
  - public            : openly available
  - entity_internal   : internal to the entity (default)
  - group_internal    : internal to the group
  - classified        : restricted access — REJECTED at ingestion
  - secret            : top secret — REJECTED at ingestion

Classification can be set explicitly (e.g. from a UI dropdown or
Confluence tag) or inferred from the folder path.
"""

import logging
import os
from enum import Enum
from typing import Optional

logger = logging.getLogger(__name__)


class DataClassification(Enum):
    PUBLIC = "public"
    ENTITY_INTERNAL = "entity_internal"
    GROUP_INTERNAL = "group_internal"
    CLASSIFIED = "classified"
    SECRET = "secret"


# Only these levels are allowed into the knowledge base
INGESTIBLE = {
    DataClassification.PUBLIC,
    DataClassification.ENTITY_INTERNAL,
    DataClassification.GROUP_INTERNAL,
}

# Folder-name hints (case-insensitive) that map to classification levels
_FOLDER_HINTS = {
    "public": DataClassification.PUBLIC,
    "entity_internal": DataClassification.ENTITY_INTERNAL,
    "entity-internal": DataClassification.ENTITY_INTERNAL,
    "internal": DataClassification.ENTITY_INTERNAL,
    "group_internal": DataClassification.GROUP_INTERNAL,
    "group-internal": DataClassification.GROUP_INTERNAL,
    "group": DataClassification.GROUP_INTERNAL,
    "classified": DataClassification.CLASSIFIED,
    "secret": DataClassification.SECRET,
    "top_secret": DataClassification.SECRET,
    "top-secret": DataClassification.SECRET,
    "restricted": DataClassification.CLASSIFIED,
    "confidential": DataClassification.CLASSIFIED,
}

# Default level when nothing else matches
DEFAULT_CLASSIFICATION = DataClassification.ENTITY_INTERNAL


def classify_document(
    filepath: str,
    explicit_level: Optional[str] = None,
) -> DataClassification:
    """
    Determine the classification level of a document.

    Priority:
        1. explicit_level (from UI, API, or Confluence tag)
        2. Folder-name inference
        3. DEFAULT_CLASSIFICATION (entity_internal)

    Args:
        filepath: Full or relative path to the document.
        explicit_level: Optional string matching a DataClassification value
                        (e.g. "public", "secret").

    Returns:
        DataClassification enum value.
    """
    # 1. Explicit level takes priority
    if explicit_level:
        normalized = explicit_level.strip().lower().replace("-", "_")
        for dc in DataClassification:
            if dc.value == normalized:
                return dc
        logger.warning(
            f"Unknown explicit classification '{explicit_level}' for {filepath}; "
            f"falling back to folder inference."
        )

    # 2. Infer from folder names in the path
    parts = os.path.normpath(filepath).lower().replace("\\", "/").split("/")
    for part in parts:
        if part in _FOLDER_HINTS:
            inferred = _FOLDER_HINTS[part]
            logger.debug(f"Classified '{filepath}' as {inferred.value} (from folder '{part}')")
            return inferred

    # 3. Default
    return DEFAULT_CLASSIFICATION


def is_ingestible(classification: DataClassification) -> bool:
    """Return True if the classification level allows ingestion."""
    return classification in INGESTIBLE


def rejection_reason(classification: DataClassification) -> str:
    """Human-readable reason for rejecting a document."""
    return (
        f"Document classified as '{classification.value}' — "
        f"only public, entity_internal, and group_internal documents "
        f"are allowed in the knowledge base."
    )
