"""
RAG Defense Module

Consolidated security layer for RAG document ingestion, retrieval,
and output integrity. Includes:

- content_scanner:     Deep ingestion-time threat scanning
- classification:      Data classification and access control
- safe_retrieval:      BM25-based retrieval over sanitized index
- ingest:              Secure ingestion pipeline
- index_versioning:    Snapshot and rollback for knowledge base
- retrieval_monitor:   Anomaly detection on retrieval patterns
- consistency:         Cross-chunk contradiction flagging
"""

from rag_defense.content_scanner import scan_document, ScanResult, FlagSeverity, FlagCategory
from rag_defense.classification import (
    DataClassification, classify_document, is_ingestible, rejection_reason,
    INGESTIBLE,
)
from rag_defense.safe_retrieval import SafeIndex
from rag_defense.ingest import run_ingest
from rag_defense.index_versioning import snapshot_current, list_versions, rollback_index
from rag_defense.retrieval_monitor import get_retrieval_monitor, RetrievalMonitor
from rag_defense.consistency import flag_inconsistencies, ConsistencyFlag

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
