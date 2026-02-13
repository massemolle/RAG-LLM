"""
Retrieval Anomaly Monitor

Tracks per-document retrieval frequency in a sliding window and
flags documents whose access rate exceeds a configurable threshold.
This helps detect targeted retrieval attacks or embedding collisions.
"""

import logging
import time
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


@dataclass
class AnomalyAlert:
    doc_sha: str
    doc_name: str
    retrieval_count: int
    window_average: float
    threshold_multiplier: float
    message: str


class RetrievalMonitor:
    """
    In-memory tracker for per-document retrieval frequency.

    Maintains a sliding window of retrieval events and computes
    anomaly scores based on deviation from the moving average.
    """

    def __init__(
        self,
        window_seconds: float = 3600.0,   # 1 hour sliding window
        max_events: int = 1000,            # cap in-memory events
        threshold_multiplier: float = 3.0, # flag if freq > 3x average
    ):
        self.window_seconds = window_seconds
        self.max_events = max_events
        self.threshold_multiplier = threshold_multiplier

        # List of (timestamp, doc_sha, doc_name) tuples
        self._events: List[tuple] = []

    def record(self, results: List[Dict[str, Any]]) -> None:
        """
        Record retrieval results.

        Args:
            results: list of dicts as returned by SafeIndex.query(),
                     each having a "meta" dict with "doc_sha" and "doc".
        """
        now = time.time()
        for r in results:
            meta = r.get("meta", {})
            doc_sha = meta.get("doc_sha", "unknown")
            doc_name = meta.get("doc", "unknown")
            self._events.append((now, doc_sha, doc_name))

        # Prune old events and cap size
        self._prune(now)

    def _prune(self, now: float) -> None:
        """Remove events outside the sliding window or exceeding max_events."""
        cutoff = now - self.window_seconds
        self._events = [
            e for e in self._events if e[0] >= cutoff
        ]
        # Cap total events
        if len(self._events) > self.max_events:
            self._events = self._events[-self.max_events:]

    def check_anomalies(self) -> List[AnomalyAlert]:
        """
        Check for documents with anomalously high retrieval frequency.

        Returns:
            List of AnomalyAlert for documents exceeding the threshold.
        """
        now = time.time()
        self._prune(now)

        if not self._events:
            return []

        # Count per doc_sha
        counts: Dict[str, int] = defaultdict(int)
        names: Dict[str, str] = {}
        for _, doc_sha, doc_name in self._events:
            counts[doc_sha] += 1
            names[doc_sha] = doc_name

        if not counts:
            return []

        # Compute average retrieval count per document
        total_retrievals = sum(counts.values())
        unique_docs = len(counts)
        avg_per_doc = total_retrievals / unique_docs

        alerts = []
        threshold = avg_per_doc * self.threshold_multiplier

        for doc_sha, count in counts.items():
            if count > threshold and count >= 5:  # minimum 5 to avoid noise
                alerts.append(AnomalyAlert(
                    doc_sha=doc_sha,
                    doc_name=names.get(doc_sha, "?"),
                    retrieval_count=count,
                    window_average=round(avg_per_doc, 2),
                    threshold_multiplier=self.threshold_multiplier,
                    message=(
                        f"Document '{names.get(doc_sha, '?')}' retrieved {count} times "
                        f"in the last {self.window_seconds/60:.0f}min "
                        f"(avg: {avg_per_doc:.1f}, threshold: {threshold:.1f})"
                    ),
                ))

        if alerts:
            for a in alerts:
                logger.warning(f"Retrieval anomaly: {a.message}")

        return alerts

    def get_stats(self) -> Dict[str, Any]:
        """Get current monitor statistics."""
        now = time.time()
        self._prune(now)

        counts: Dict[str, int] = defaultdict(int)
        for _, doc_sha, _ in self._events:
            counts[doc_sha] += 1

        return {
            "total_events": len(self._events),
            "unique_docs": len(counts),
            "window_seconds": self.window_seconds,
            "top_docs": sorted(
                counts.items(), key=lambda x: x[1], reverse=True
            )[:5],
        }


# Global singleton monitor
_monitor: Optional[RetrievalMonitor] = None


def get_retrieval_monitor() -> RetrievalMonitor:
    """Get or create the global retrieval monitor."""
    global _monitor
    if _monitor is None:
        _monitor = RetrievalMonitor()
    return _monitor
