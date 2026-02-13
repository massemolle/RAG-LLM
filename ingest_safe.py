# Backward-compatibility re-export
# Ingestion logic has moved to rag/ingest.py
from rag.ingest import run_ingest  # noqa: F401

__all__ = ["run_ingest"]
