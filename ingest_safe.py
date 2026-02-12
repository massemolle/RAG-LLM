# Backward-compatibility re-export
# Ingestion logic has moved to rag_defense/ingest.py
from rag_defense.ingest import run_ingest  # noqa: F401

__all__ = ["run_ingest"]
