"""
Production Hardening: Caching, Rate Limiting, Model Routing
"""

import hashlib
import time
import logging
from typing import Dict, Optional, Tuple, Any
from datetime import datetime, timedelta
from collections import defaultdict
from functools import lru_cache

logger = logging.getLogger(__name__)


class GuardrailsCache:
    """
    Cache for guardrails evaluations to reduce redundant checks
    """
    
    def __init__(self, ttl_seconds: int = 3600):
        self.cache: Dict[str, Tuple[Any, float]] = {}
        self.ttl = ttl_seconds
    
    def _make_key(self, query: str, guard_type: str) -> str:
        """Create cache key from query and guard type"""
        normalized = query.lower().strip()
        return hashlib.sha256(f"{guard_type}:{normalized}".encode()).hexdigest()
    
    def get(self, query: str, guard_type: str) -> Optional[Any]:
        """Get cached result if available and not expired"""
        key = self._make_key(query, guard_type)
        if key in self.cache:
            result, timestamp = self.cache[key]
            if time.time() - timestamp < self.ttl:
                return result
            else:
                # Expired, remove
                del self.cache[key]
        return None
    
    def set(self, query: str, guard_type: str, result: Any):
        """Cache a result"""
        key = self._make_key(query, guard_type)
        self.cache[key] = (result, time.time())
    
    def clear(self):
        """Clear all cached entries"""
        self.cache.clear()
    
    def cleanup_expired(self):
        """Remove expired entries"""
        now = time.time()
        expired_keys = [
            key for key, (_, timestamp) in self.cache.items()
            if now - timestamp >= self.ttl
        ]
        for key in expired_keys:
            del self.cache[key]


class RateLimiter:
    """
    Rate limiter for expensive operations (LLM self-check, tool execution)
    """
    
    def __init__(self, max_calls: int = 10, window_seconds: int = 60):
        self.max_calls = max_calls
        self.window = window_seconds
        self.calls: Dict[str, list] = defaultdict(list)  # key -> list of timestamps
    
    def _make_key(self, user_id: Optional[str], operation: str) -> str:
        """Create rate limit key"""
        return f"{operation}:{user_id or 'anonymous'}"
    
    def check(self, user_id: Optional[str], operation: str) -> Tuple[bool, str]:
        """
        Check if operation is allowed
        
        Returns:
            Tuple of (allowed, reason)
        """
        key = self._make_key(user_id, operation)
        now = time.time()
        
        # Clean old entries
        self.calls[key] = [
            ts for ts in self.calls[key]
            if now - ts < self.window
        ]
        
        # Check limit
        if len(self.calls[key]) >= self.max_calls:
            return False, f"Rate limit exceeded: {self.max_calls} calls per {self.window} seconds"
        
        # Record call
        self.calls[key].append(now)
        return True, "Rate limit OK"
    
    def reset(self, user_id: Optional[str], operation: str):
        """Reset rate limit for a user/operation"""
        key = self._make_key(user_id, operation)
        if key in self.calls:
            del self.calls[key]


class ModelRouter:
    """
    Model routing: Cheap detectors first, expensive LLM judges later
    """
    
    def __init__(self):
        self.stats = {
            "layer_a": {"count": 0, "blocked": 0, "time_ms": []},
            "layer_b": {"count": 0, "blocked": 0, "time_ms": []},
            "layer_c": {"count": 0, "blocked": 0, "time_ms": []},
        }
    
    def should_run_layer_b(self, layer_a_result: Tuple[Any, str, list]) -> bool:
        """
        Decide if Layer B (heuristics) should run based on Layer A result
        
        Returns:
            True if Layer B should run
        """
        severity, _, _ = layer_a_result
        # Run Layer B if Layer A didn't block (to catch things Layer A missed)
        # But skip if Layer A already blocked (save time)
        return severity.value != "blocked"
    
    def should_run_layer_c(self, layer_a_result: Tuple[Any, str, list],
                           layer_b_result: Tuple[Any, str]) -> bool:
        """
        Decide if Layer C (LLM judge) should run based on Layer A and B results
        
        Returns:
            True if Layer C should run
        """
        layer_a_sev, _, _ = layer_a_result
        layer_b_sev, _ = layer_b_result
        
        # Run Layer C only if:
        # 1. Layer A didn't block (already blocked, skip)
        # 2. Layer B is suspicious or blocked (need LLM confirmation)
        if layer_a_sev.value == "blocked":
            return False
        
        if layer_b_sev.value in ["blocked", "review"]:
            return True
        
        # If both A and B are clean, skip expensive LLM check
        return False
    
    def record_timing(self, layer: str, time_ms: float, blocked: bool):
        """Record timing statistics"""
        if layer in self.stats:
            self.stats[layer]["count"] += 1
            self.stats[layer]["time_ms"].append(time_ms)
            if blocked:
                self.stats[layer]["blocked"] += 1
    
    def get_stats(self) -> Dict[str, Any]:
        """Get routing statistics"""
        stats = {}
        for layer, data in self.stats.items():
            if data["count"] > 0:
                avg_time = sum(data["time_ms"]) / len(data["time_ms"])
                stats[layer] = {
                    "count": data["count"],
                    "blocked": data["blocked"],
                    "block_rate": data["blocked"] / data["count"],
                    "avg_time_ms": avg_time
                }
        return stats


# Global instances
_guardrails_cache: Optional[GuardrailsCache] = None
_rate_limiter: Optional[RateLimiter] = None
_model_router: Optional[ModelRouter] = None


def get_guardrails_cache() -> GuardrailsCache:
    """Get or create global cache instance"""
    global _guardrails_cache
    if _guardrails_cache is None:
        _guardrails_cache = GuardrailsCache(ttl_seconds=3600)  # 1 hour TTL
    return _guardrails_cache


def get_rate_limiter() -> RateLimiter:
    """Get or create global rate limiter instance"""
    global _rate_limiter
    if _rate_limiter is None:
        _rate_limiter = RateLimiter(max_calls=10, window_seconds=60)
    return _rate_limiter


def get_model_router() -> ModelRouter:
    """Get or create global model router instance"""
    global _model_router
    if _model_router is None:
        _model_router = ModelRouter()
    return _model_router
