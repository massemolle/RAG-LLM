"""
Production Hardening: Caching, Rate Limiting, Model Routing

Enhanced 2026 features:
- Global rate limiting per IP/session
- Suspicious pattern detection (model extraction defense)
- Query logging with hash tracking
- Escalation-based throttling
"""

import hashlib
import time
import logging
from typing import Dict, Optional, Tuple, Any, List
from datetime import datetime, timedelta
from collections import defaultdict
from functools import lru_cache
from dataclasses import dataclass, field

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


# ========== ENHANCED RATE LIMITING FOR MODEL EXTRACTION DEFENSE ==========

@dataclass
class SessionMetrics:
    """Track metrics for a single session"""
    query_count: int = 0
    blocked_count: int = 0
    escalated_count: int = 0
    query_hashes: List[str] = field(default_factory=list)
    first_seen: float = field(default_factory=time.time)
    last_seen: float = field(default_factory=time.time)
    
    def update(self, query_hash: str, was_blocked: bool, was_escalated: bool):
        self.query_count += 1
        self.last_seen = time.time()
        if was_blocked:
            self.blocked_count += 1
        if was_escalated:
            self.escalated_count += 1
        # Keep last 100 hashes
        self.query_hashes.append(query_hash)
        if len(self.query_hashes) > 100:
            self.query_hashes.pop(0)
    
    def get_risk_score(self) -> float:
        """Calculate risk score for this session"""
        score = 0.0
        
        # High block rate increases risk
        if self.query_count > 5:
            block_rate = self.blocked_count / self.query_count
            score += block_rate * 50
        
        # High escalation rate increases risk
        if self.query_count > 3:
            escalate_rate = self.escalated_count / self.query_count
            score += escalate_rate * 30
        
        # Repeated queries (potential probing)
        unique_ratio = len(set(self.query_hashes)) / max(len(self.query_hashes), 1)
        if unique_ratio < 0.5:  # Less than 50% unique queries
            score += (1 - unique_ratio) * 40
        
        # Rapid-fire queries
        session_duration = self.last_seen - self.first_seen
        if session_duration > 0:
            queries_per_second = self.query_count / session_duration
            if queries_per_second > 0.5:  # More than 1 query per 2 seconds
                score += min(queries_per_second * 20, 30)
        
        return min(score, 100)


class GlobalRateLimiter:
    """
    Global rate limiter with per-IP and per-session tracking.
    Implements model extraction defense.
    """
    
    def __init__(
        self,
        global_max_per_minute: int = 60,
        suspicious_max_per_minute: int = 5,
        llm_judge_max_per_minute: int = 10,
        session_ttl_seconds: int = 3600
    ):
        self.global_max = global_max_per_minute
        self.suspicious_max = suspicious_max_per_minute
        self.llm_judge_max = llm_judge_max_per_minute
        self.session_ttl = session_ttl_seconds
        
        # Tracking structures
        self.sessions: Dict[str, SessionMetrics] = {}
        self.global_calls: List[float] = []
        self.suspicious_calls: Dict[str, List[float]] = defaultdict(list)
        self.llm_judge_calls: Dict[str, List[float]] = defaultdict(list)
    
    def _clean_old_calls(self, calls: List[float], window_seconds: int = 60) -> List[float]:
        """Remove calls older than window"""
        cutoff = time.time() - window_seconds
        return [t for t in calls if t > cutoff]
    
    def _get_session_id(self, user_id: Optional[str], session_id: Optional[str], ip: Optional[str]) -> str:
        """Generate session identifier"""
        parts = [user_id or "", session_id or "", ip or ""]
        return hashlib.md5(":".join(parts).encode()).hexdigest()[:16]
    
    def check_global_limit(self) -> Tuple[bool, str]:
        """Check global rate limit"""
        self.global_calls = self._clean_old_calls(self.global_calls)
        
        if len(self.global_calls) >= self.global_max:
            return False, f"Global rate limit exceeded: {self.global_max}/minute"
        
        self.global_calls.append(time.time())
        return True, "OK"
    
    def check_suspicious_limit(self, session_key: str) -> Tuple[bool, str]:
        """Check rate limit for suspicious sessions"""
        self.suspicious_calls[session_key] = self._clean_old_calls(
            self.suspicious_calls[session_key]
        )
        
        if len(self.suspicious_calls[session_key]) >= self.suspicious_max:
            return False, f"Suspicious activity rate limit: {self.suspicious_max}/minute"
        
        self.suspicious_calls[session_key].append(time.time())
        return True, "OK"
    
    def check_llm_judge_limit(self, session_key: str) -> Tuple[bool, str]:
        """Check rate limit for LLM judge calls"""
        self.llm_judge_calls[session_key] = self._clean_old_calls(
            self.llm_judge_calls[session_key]
        )
        
        if len(self.llm_judge_calls[session_key]) >= self.llm_judge_max:
            return False, f"LLM judge rate limit: {self.llm_judge_max}/minute"
        
        self.llm_judge_calls[session_key].append(time.time())
        return True, "OK"
    
    def record_query(
        self,
        query: str,
        user_id: Optional[str] = None,
        session_id: Optional[str] = None,
        ip: Optional[str] = None,
        was_blocked: bool = False,
        was_escalated: bool = False
    ) -> Dict:
        """
        Record a query and return session risk metrics.
        
        Returns:
            Dict with session metrics and risk score
        """
        session_key = self._get_session_id(user_id, session_id, ip)
        query_hash = hashlib.md5(query.lower().strip().encode()).hexdigest()[:8]
        
        # Clean expired sessions
        self._cleanup_sessions()
        
        # Get or create session
        if session_key not in self.sessions:
            self.sessions[session_key] = SessionMetrics()
        
        session = self.sessions[session_key]
        session.update(query_hash, was_blocked, was_escalated)
        
        # Check for repeated queries (potential probing)
        recent_hashes = session.query_hashes[-10:]
        repeated_count = recent_hashes.count(query_hash)
        
        return {
            "session_key": session_key,
            "query_count": session.query_count,
            "risk_score": session.get_risk_score(),
            "blocked_rate": session.blocked_count / max(session.query_count, 1),
            "repeated_query": repeated_count > 2,
            "is_suspicious": session.get_risk_score() > 50
        }
    
    def _cleanup_sessions(self):
        """Remove expired sessions"""
        cutoff = time.time() - self.session_ttl
        expired = [
            key for key, session in self.sessions.items()
            if session.last_seen < cutoff
        ]
        for key in expired:
            del self.sessions[key]
    
    def get_session_metrics(self, user_id: Optional[str] = None,
                           session_id: Optional[str] = None,
                           ip: Optional[str] = None) -> Optional[Dict]:
        """Get metrics for a specific session"""
        session_key = self._get_session_id(user_id, session_id, ip)
        
        if session_key in self.sessions:
            session = self.sessions[session_key]
            return {
                "query_count": session.query_count,
                "blocked_count": session.blocked_count,
                "escalated_count": session.escalated_count,
                "risk_score": session.get_risk_score(),
                "session_duration_seconds": session.last_seen - session.first_seen
            }
        return None
    
    def get_all_stats(self) -> Dict:
        """Get overall rate limiting statistics"""
        return {
            "active_sessions": len(self.sessions),
            "global_calls_last_minute": len(self._clean_old_calls(self.global_calls)),
            "total_queries": sum(s.query_count for s in self.sessions.values()),
            "total_blocked": sum(s.blocked_count for s in self.sessions.values()),
            "high_risk_sessions": sum(1 for s in self.sessions.values() if s.get_risk_score() > 70)
        }


# Global enhanced rate limiter instance
_global_rate_limiter: Optional[GlobalRateLimiter] = None


def get_global_rate_limiter() -> GlobalRateLimiter:
    """Get or create global enhanced rate limiter"""
    global _global_rate_limiter
    if _global_rate_limiter is None:
        _global_rate_limiter = GlobalRateLimiter()
    return _global_rate_limiter
