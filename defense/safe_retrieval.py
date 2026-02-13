# Backward-compatibility re-export
# SafeIndex has moved to rag/safe_retrieval.py
from rag.safe_retrieval import SafeIndex  # noqa: F401

__all__ = ["SafeIndex"]
