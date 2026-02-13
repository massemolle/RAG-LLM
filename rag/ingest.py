"""
Secure Document Ingestion Pipeline for RAG

Reads documents, classifies them, scans for threats, sanitizes,
chunks, and writes a versioned JSONL index with rich provenance
metadata per chunk.
"""

import os
import re
import json
import logging
import datetime
from typing import List, Optional, Tuple

from rag.content_scanner import scan_document, ScanResult, FlagSeverity
from rag.classification import (
    classify_document, is_ingestible, rejection_reason, DataClassification,
)

# Shared utilities — hashing, file I/O, chunking
from utils.text import sha256_hash as sha256, read_document as _read_text, chunk_text as _chunk

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Main ingestion entry point
# ---------------------------------------------------------------------------

def run_ingest(
    src: str = "./rag/data",
    collection: str = "grid_ops",
    classification_level: Optional[str] = None,
    chunk_size: int = 800,
    chunk_overlap: int = 120,
) -> dict:
    """
    Run the full secure ingestion pipeline.

    Steps per file:
        1. Classify document -> reject if classified/secret
        2. Read text from file
        3. Run deep content scanner -> quarantine if blocking flags
        4. Sanitize + chunk
        5. Write to JSONL index with provenance metadata

    Args:
        src: Directory containing raw documents.
        collection: Collection name tag for the index.
        classification_level: Optional explicit classification for all files
                              (overrides folder inference). One of:
                              public, entity_internal, group_internal,
                              classified, secret.
        chunk_size: Characters per chunk.
        chunk_overlap: Overlap between consecutive chunks.

    Returns:
        dict with keys: files, chunks, collection, rejected, quarantined,
        scan_summary (list of per-file scan info).
    """
    # Version the existing index before overwriting
    try:
        from rag.index_versioning import snapshot_current
        snapshot_current()
    except Exception as e:
        logger.debug(f"Index versioning skipped: {e}")

    os.makedirs("./rag/index", exist_ok=True)

    now_iso = datetime.datetime.utcnow().isoformat() + "Z"
    manifest = {
        "created": now_iso,
        "collection": collection,
        "classification_default": classification_level or "auto",
        "files": [],
    }

    written = 0
    rejected_files = []
    quarantined_files = []
    scan_summary = []

    with open("./rag/index/index.jsonl", "w", encoding="utf-8") as idx:
        for root, _, files in os.walk(src):
            for f in files:
                path = os.path.join(root, f)
                file_info = {"file": f, "path": path}

                # --- Step 1: Classification ---
                doc_class = classify_document(path, explicit_level=classification_level)
                file_info["classification"] = doc_class.value

                if not is_ingestible(doc_class):
                    reason = rejection_reason(doc_class)
                    file_info["status"] = "rejected"
                    file_info["reason"] = reason
                    rejected_files.append(file_info)
                    scan_summary.append(file_info)
                    logger.warning(f"REJECTED '{f}': {reason}")
                    continue

                # --- Step 2: Read text ---
                txt = _read_text(path)
                if not txt.strip():
                    file_info["status"] = "skipped_empty"
                    scan_summary.append(file_info)
                    continue

                # --- Step 3: Deep content scan ---
                scan_result: ScanResult = scan_document(txt, filename=f)
                file_info["scan_flags"] = len(scan_result.flags)
                file_info["scan_blocks"] = scan_result.stats.get("block_flags", 0)
                file_info["scan_warns"] = scan_result.stats.get("warn_flags", 0)
                file_info["scan_info"] = scan_result.stats.get("info_flags", 0)
                # Serialize flags for UI display
                file_info["flags_detail"] = [
                    {
                        "category": flag.category.value,
                        "severity": flag.severity.value,
                        "description": flag.description,
                        "matched_text": flag.matched_text[:120],
                        "pattern_name": flag.pattern_name,
                    }
                    for flag in scan_result.flags
                ]
                file_info["stats"] = scan_result.stats

                if not scan_result.is_clean:
                    # Quarantine: write sanitized text for human review
                    os.makedirs("./rag/index/quarantine", exist_ok=True)
                    qpath = os.path.join("./rag/index/quarantine", f + ".txt")
                    with open(qpath, "w", encoding="utf-8") as qf:
                        qf.write(f"# Quarantined: {f}\n")
                        qf.write(f"# Classification: {doc_class.value}\n")
                        qf.write(f"# Flags:\n")
                        for flag in scan_result.flags:
                            qf.write(f"#   [{flag.severity.value}] {flag.category.value}: "
                                     f"{flag.description}\n")
                        qf.write(f"\n{scan_result.sanitized_text}")
                    file_info["status"] = "quarantined"
                    file_info["quarantine_path"] = qpath
                    quarantined_files.append(file_info)
                    scan_summary.append(file_info)
                    logger.warning(
                        f"QUARANTINED '{f}': {scan_result.stats.get('block_flags', 0)} blocking flags"
                    )
                    continue

                # --- Step 4: Chunk sanitized text ---
                clean = scan_result.sanitized_text
                chunks = _chunk(clean, size=chunk_size, overlap=chunk_overlap)
                fid = sha256(path)
                content_sha = sha256(clean)

                manifest["files"].append({
                    "path": os.path.abspath(path),
                    "sha256": fid,
                    "content_sha256": content_sha,
                    "chunks": len(chunks),
                    "classification": doc_class.value,
                    "scan_warns": scan_result.stats.get("warn_flags", 0),
                })

                # --- Step 5: Write chunks with provenance ---
                for j, chunk_text in enumerate(chunks):
                    rec = {
                        "collection": collection,
                        "doc": os.path.basename(path),
                        "doc_sha": fid,
                        "chunk": j,
                        "text": chunk_text,
                        # Provenance metadata
                        "source_file": os.path.abspath(path),
                        "classification": doc_class.value,
                        "ingested_at": now_iso,
                        "content_sha256": sha256(chunk_text),
                        "scanner_warns": scan_result.stats.get("warn_flags", 0),
                    }
                    idx.write(json.dumps(rec, ensure_ascii=False) + "\n")
                    written += 1

                file_info["status"] = "ingested"
                file_info["chunks"] = len(chunks)
                scan_summary.append(file_info)

    # Write manifest
    with open("./rag/index/manifest.json", "w", encoding="utf-8") as mf:
        mf.write(json.dumps(manifest, indent=2))

    result = {
        "files": len(manifest["files"]),
        "chunks": written,
        "collection": collection,
        "rejected": len(rejected_files),
        "quarantined": len(quarantined_files),
        "scan_summary": scan_summary,
    }

    logger.info(
        f"Ingestion complete: {result['files']} files, {written} chunks, "
        f"{result['rejected']} rejected, {result['quarantined']} quarantined"
    )
    return result
