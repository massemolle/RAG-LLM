"""
Index Versioning and Rollback for RAG Knowledge Base

Keeps timestamped snapshots of the safe_index so that poisoning
incidents or bad ingestions can be reverted.
"""

import json
import logging
import os
import shutil
from datetime import datetime
from typing import Dict, List, Optional

logger = logging.getLogger(__name__)

INDEX_DIR = "./rag/index"
VERSIONS_DIR = os.path.join(INDEX_DIR, "versions")
INDEX_FILE = os.path.join(INDEX_DIR, "index.jsonl")
MANIFEST_FILE = os.path.join(INDEX_DIR, "manifest.json")
MAX_VERSIONS = 5  # keep last N snapshots


def _timestamp_label() -> str:
    """Generate a filesystem-safe timestamp string."""
    return datetime.utcnow().strftime("%Y%m%d_%H%M%S")


def snapshot_current() -> Optional[str]:
    """
    Save a snapshot of the current index and manifest.

    Returns:
        The version label (timestamp string) if a snapshot was created,
        or None if there was nothing to snapshot.
    """
    if not os.path.exists(INDEX_FILE):
        logger.debug("No index file to snapshot.")
        return None

    label = _timestamp_label()
    dest = os.path.join(VERSIONS_DIR, label)
    os.makedirs(dest, exist_ok=True)

    shutil.copy2(INDEX_FILE, os.path.join(dest, "index.jsonl"))
    if os.path.exists(MANIFEST_FILE):
        shutil.copy2(MANIFEST_FILE, os.path.join(dest, "manifest.json"))

    # Write version metadata
    meta = {
        "label": label,
        "created": datetime.utcnow().isoformat() + "Z",
        "index_size_bytes": os.path.getsize(INDEX_FILE),
    }
    # Count chunks/files from manifest if available
    if os.path.exists(MANIFEST_FILE):
        try:
            with open(MANIFEST_FILE, encoding="utf-8") as f:
                manifest = json.load(f)
            meta["files"] = len(manifest.get("files", []))
            meta["collection"] = manifest.get("collection", "")
        except Exception:
            pass
    # Count chunks from index
    try:
        with open(INDEX_FILE, encoding="utf-8") as f:
            meta["chunks"] = sum(1 for _ in f)
    except Exception:
        pass

    with open(os.path.join(dest, "version_meta.json"), "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)

    logger.info(f"Index snapshot saved: {label} ({meta.get('chunks', '?')} chunks)")

    # Prune old versions
    _prune_old_versions()

    return label


def _prune_old_versions():
    """Remove oldest versions beyond MAX_VERSIONS."""
    if not os.path.exists(VERSIONS_DIR):
        return
    versions = sorted(os.listdir(VERSIONS_DIR))
    while len(versions) > MAX_VERSIONS:
        oldest = versions.pop(0)
        oldest_path = os.path.join(VERSIONS_DIR, oldest)
        shutil.rmtree(oldest_path, ignore_errors=True)
        logger.info(f"Pruned old index version: {oldest}")


def list_versions() -> List[Dict]:
    """
    List available index snapshots.

    Returns:
        List of dicts with version metadata, sorted newest first.
    """
    if not os.path.exists(VERSIONS_DIR):
        return []

    result = []
    for entry in sorted(os.listdir(VERSIONS_DIR), reverse=True):
        meta_path = os.path.join(VERSIONS_DIR, entry, "version_meta.json")
        if os.path.exists(meta_path):
            try:
                with open(meta_path, encoding="utf-8") as f:
                    meta = json.load(f)
                result.append(meta)
            except Exception:
                result.append({"label": entry, "error": "metadata unreadable"})
        else:
            result.append({"label": entry, "created": "unknown"})
    return result


def rollback_index(version_label: str) -> bool:
    """
    Restore a previous index snapshot.

    This first snapshots the current state (so the rollback itself is
    reversible), then overwrites the active index with the chosen version.

    Args:
        version_label: Timestamp label of the version to restore.

    Returns:
        True if rollback succeeded.
    """
    src = os.path.join(VERSIONS_DIR, version_label)
    src_index = os.path.join(src, "index.jsonl")

    if not os.path.exists(src_index):
        logger.error(f"Version '{version_label}' not found or has no index file.")
        return False

    # Snapshot current state first (so we can undo the rollback)
    snapshot_current()

    # Restore
    shutil.copy2(src_index, INDEX_FILE)
    src_manifest = os.path.join(src, "manifest.json")
    if os.path.exists(src_manifest):
        shutil.copy2(src_manifest, MANIFEST_FILE)

    logger.info(f"Index rolled back to version: {version_label}")
    return True
