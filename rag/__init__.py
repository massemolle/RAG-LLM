"""
RAG Module
==========

Consolidated security and data layer for RAG document ingestion,
retrieval, and output integrity.

Sub-modules:
    content_scanner     – Deep ingestion-time threat scanning
    classification      – Data classification and access control
    safe_retrieval      – BM25-based retrieval over sanitized index
    ingest              – Secure ingestion pipeline
    index_versioning    – Snapshot and rollback for knowledge base
    retrieval_monitor   – Anomaly detection on retrieval patterns
    consistency         – Cross-chunk contradiction flagging

Data directories:
    rag/data/           – Raw documents for ingestion
    rag/index/          – Sanitized JSONL index and versions
"""

from rag.content_scanner import scan_document, ScanResult, FlagSeverity, FlagCategory
from rag.classification import (
    DataClassification, classify_document, is_ingestible, rejection_reason,
    INGESTIBLE,
)
from rag.safe_retrieval import SafeIndex
from rag.ingest import run_ingest
from rag.index_versioning import snapshot_current, list_versions, rollback_index
from rag.retrieval_monitor import get_retrieval_monitor, RetrievalMonitor
from rag.consistency import flag_inconsistencies, ConsistencyFlag

__all__ = [
    # Scanner
    "scan_document", "ScanResult", "FlagSeverity", "FlagCategory",
    # Classification
    "DataClassification", "classify_document", "is_ingestible",
    "rejection_reason", "INGESTIBLE",
    # Retrieval
    "SafeIndex",
    # Ingestion
    "run_ingest",
    # Versioning
    "snapshot_current", "list_versions", "rollback_index",
    # Monitoring
    "get_retrieval_monitor", "RetrievalMonitor",
    # Consistency
    "flag_inconsistencies", "ConsistencyFlag",
]
