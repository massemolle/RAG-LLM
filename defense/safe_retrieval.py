# Backward-compatibility re-export
# SafeIndex has moved to rag_defense/safe_retrieval.py
from rag_defense.safe_retrieval import SafeIndex  # noqa: F401

__all__ = ["SafeIndex"]
