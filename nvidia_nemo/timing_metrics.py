"""
Timing Metrics for Guardrails Pipeline

This module provides comprehensive timing measurement for each layer
of the guardrails pipeline, enabling latency optimization and monitoring.
"""

import time
import logging
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, field
from contextlib import contextmanager
from datetime import datetime
from collections import defaultdict

logger = logging.getLogger(__name__)


@dataclass
class LayerTiming:
    """Timing information for a single layer"""
    layer_name: str
    start_time: float
    end_time: float
    duration_ms: float
    was_skipped: bool = False
    was_cached: bool = False
    result: Optional[str] = None  # ALLOWED, BLOCKED, REVIEW, ESCALATE
    details: Dict = field(default_factory=dict)


@dataclass
class PipelineTiming:
    """Complete timing for the entire guardrails pipeline"""
    request_id: str
    query_preview: str  # First 50 chars of query for identification
    start_time: datetime
    end_time: Optional[datetime]
    layers: List[LayerTiming]
    total_ms: float
    was_blocked: bool
    final_result: str
    
    def to_dict(self) -> Dict:
        """Convert to dictionary for logging/UI display"""
        return {
            "request_id": self.request_id,
            "query_preview": self.query_preview,
            "total_ms": round(self.total_ms, 2),
            "was_blocked": self.was_blocked,
            "final_result": self.final_result,
            "layers": {
                layer.layer_name: {
                    "duration_ms": round(layer.duration_ms, 2),
                    "skipped": layer.was_skipped,
                    "cached": layer.was_cached,
                    "result": layer.result,
                    "details": layer.details if layer.details else {}
                }
                for layer in self.layers
            },
            "layer_breakdown": self._get_breakdown()
        }
    
    def _get_breakdown(self) -> Dict:
        """Get percentage breakdown by layer"""
        if self.total_ms == 0:
            return {}
        return {
            layer.layer_name: round((layer.duration_ms / self.total_ms) * 100, 1)
            for layer in self.layers
            if not layer.was_skipped
        }
    
    def to_ui_display(self) -> str:
        """Format timing for UI display"""
        lines = [
            f"**Total Time:** {self.total_ms:.1f}ms",
            "",
            "**Layer Breakdown:**"
        ]
        
        for layer in self.layers:
            status = ""
            if layer.was_skipped:
                status = " (skipped)"
            elif layer.was_cached:
                status = " (cached)"
            
            result_icon = ""
            if layer.result == "BLOCKED":
                result_icon = " [BLOCKED]"
            elif layer.result == "ESCALATE":
                result_icon = " [ESCALATE]"
            elif layer.result == "REVIEW":
                result_icon = " [REVIEW]"
            
            lines.append(f"- {layer.layer_name}: {layer.duration_ms:.1f}ms{status}{result_icon}")
        
        return "\n".join(lines)


class GuardrailsTimer:
    """
    Context manager based timer for measuring guardrails pipeline latency.
    
    Usage:
        timer = GuardrailsTimer(query="what is an embedding")
        
        with timer.time_layer("embedding_similarity"):
            result = check_embedding_similarity(query)
        
        with timer.time_layer("llm_guard"):
            result = run_llm_guard(query)
        
        summary = timer.get_summary()
    """
    
    def __init__(self, query: str, request_id: Optional[str] = None):
        """
        Initialize timer for a query.
        
        Args:
            query: The user query being processed
            request_id: Optional unique ID for this request
        """
        import uuid
        self.request_id = request_id or str(uuid.uuid4())[:8]
        self.query_preview = query[:50] + "..." if len(query) > 50 else query
        self.start_time = datetime.now()
        self._pipeline_start = time.perf_counter()
        self.layers: List[LayerTiming] = []
        self._current_layer: Optional[str] = None
        self._final_result = "ALLOWED"
        self._was_blocked = False
    
    @contextmanager
    def time_layer(self, layer_name: str, skip: bool = False, cached: bool = False):
        """
        Context manager to time a specific layer.
        
        Args:
            layer_name: Name of the layer (e.g., "embedding_similarity")
            skip: If True, mark this layer as skipped
            cached: If True, mark this layer as using cached result
        """
        self._current_layer = layer_name
        start = time.perf_counter()
        
        layer_timing = LayerTiming(
            layer_name=layer_name,
            start_time=start,
            end_time=0,
            duration_ms=0,
            was_skipped=skip,
            was_cached=cached
        )
        
        try:
            if skip:
                layer_timing.duration_ms = 0
                layer_timing.result = "SKIPPED"
            yield layer_timing
        finally:
            end = time.perf_counter()
            layer_timing.end_time = end
            layer_timing.duration_ms = (end - start) * 1000
            self.layers.append(layer_timing)
            self._current_layer = None
    
    def mark_result(self, result: str):
        """
        Mark the result of the current layer.
        
        Args:
            result: One of "ALLOWED", "BLOCKED", "REVIEW", "ESCALATE"
        """
        if self.layers:
            self.layers[-1].result = result
            
            if result == "BLOCKED":
                self._was_blocked = True
                self._final_result = "BLOCKED"
            elif result == "REVIEW" and self._final_result != "BLOCKED":
                self._final_result = "REVIEW"
            elif result == "ESCALATE" and self._final_result not in ["BLOCKED", "REVIEW"]:
                self._final_result = "ESCALATE"
    
    def add_layer_detail(self, key: str, value: Any):
        """Add additional detail to the current layer"""
        if self.layers:
            self.layers[-1].details[key] = value
    
    def get_summary(self) -> PipelineTiming:
        """Get complete timing summary"""
        end_time = datetime.now()
        total_ms = (time.perf_counter() - self._pipeline_start) * 1000
        
        return PipelineTiming(
            request_id=self.request_id,
            query_preview=self.query_preview,
            start_time=self.start_time,
            end_time=end_time,
            layers=self.layers,
            total_ms=total_ms,
            was_blocked=self._was_blocked,
            final_result=self._final_result
        )
    
    def get_layer_time(self, layer_name: str) -> float:
        """Get the time for a specific layer in milliseconds"""
        for layer in self.layers:
            if layer.layer_name == layer_name:
                return layer.duration_ms
        return 0.0
    
    def get_total_time(self) -> float:
        """Get total elapsed time in milliseconds"""
        return (time.perf_counter() - self._pipeline_start) * 1000


class TimingAggregator:
    """
    Aggregates timing metrics across multiple requests for analysis.
    """
    
    def __init__(self, max_history: int = 1000):
        """
        Initialize aggregator.
        
        Args:
            max_history: Maximum number of timing records to keep
        """
        self.max_history = max_history
        self._history: List[PipelineTiming] = []
        self._layer_totals: Dict[str, float] = defaultdict(float)
        self._layer_counts: Dict[str, int] = defaultdict(int)
        self._blocked_count = 0
        self._total_count = 0
    
    def record(self, timing: PipelineTiming):
        """Record a pipeline timing"""
        self._history.append(timing)
        if len(self._history) > self.max_history:
            self._history.pop(0)
        
        self._total_count += 1
        if timing.was_blocked:
            self._blocked_count += 1
        
        for layer in timing.layers:
            if not layer.was_skipped:
                self._layer_totals[layer.layer_name] += layer.duration_ms
                self._layer_counts[layer.layer_name] += 1
    
    def get_averages(self) -> Dict[str, float]:
        """Get average times per layer"""
        averages = {}
        for layer_name, total in self._layer_totals.items():
            count = self._layer_counts[layer_name]
            if count > 0:
                averages[layer_name] = round(total / count, 2)
        return averages
    
    def get_statistics(self) -> Dict:
        """Get comprehensive statistics"""
        if not self._history:
            return {"error": "No timing data recorded"}
        
        total_times = [t.total_ms for t in self._history]
        
        return {
            "total_requests": self._total_count,
            "blocked_requests": self._blocked_count,
            "block_rate": round(self._blocked_count / self._total_count * 100, 2) if self._total_count > 0 else 0,
            "average_latency_ms": round(sum(total_times) / len(total_times), 2),
            "min_latency_ms": round(min(total_times), 2),
            "max_latency_ms": round(max(total_times), 2),
            "p50_latency_ms": round(sorted(total_times)[len(total_times) // 2], 2),
            "p95_latency_ms": round(sorted(total_times)[int(len(total_times) * 0.95)], 2) if len(total_times) >= 20 else None,
            "layer_averages": self.get_averages(),
            "history_size": len(self._history)
        }
    
    def get_recent(self, n: int = 10) -> List[Dict]:
        """Get the N most recent timing records"""
        return [t.to_dict() for t in self._history[-n:]]
    
    def clear(self):
        """Clear all recorded data"""
        self._history.clear()
        self._layer_totals.clear()
        self._layer_counts.clear()
        self._blocked_count = 0
        self._total_count = 0


# Global aggregator instance
_timing_aggregator: Optional[TimingAggregator] = None


def get_timing_aggregator() -> TimingAggregator:
    """Get or create the global timing aggregator"""
    global _timing_aggregator
    if _timing_aggregator is None:
        _timing_aggregator = TimingAggregator()
    return _timing_aggregator


def record_timing(timing: PipelineTiming):
    """Record timing to the global aggregator and emit OCSF event"""
    get_timing_aggregator().record(timing)
    # Emit OCSF Detection Finding to local JSONL log
    try:
        from observability.ocsf_mapper import pipeline_timing_obj_to_ocsf, emit_ocsf_event
        emit_ocsf_event(pipeline_timing_obj_to_ocsf(timing))
    except Exception:
        pass  # OCSF emission is best-effort


def get_timing_stats() -> Dict:
    """Get timing statistics from the global aggregator"""
    return get_timing_aggregator().get_statistics()
