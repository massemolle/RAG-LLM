"""
NeMo Guardrails Custom Actions
Python actions for RAG integration
"""

import os
import json
import logging
from datetime import datetime
from typing import Dict, List, Optional, Any
from pathlib import Path

from nemoguardrails.actions import action

# Setup logging
logger = logging.getLogger(__name__)

# Import RAG components
try:
    import sys
    project_root = Path(__file__).parent.parent.parent
    sys.path.insert(0, str(project_root))
    
    from RagV2 import safe_idx
    from defense.guards import POLICY, looks_like_injection
except ImportError as e:
    logger.warning(f"Could not import RAG components: {e}")


# Global RAG instance (set by integration)
_rag_instance = None
_rate_limit_store: Dict[str, Dict[str, Any]] = {}


def set_rag_instance(rag_instance):
    """Set the RAG instance"""
    global _rag_instance
    _rag_instance = rag_instance


@action
async def retrieve_from_rag(query: str) -> Dict[str, Any]:
    """
    Retrieve documents from RAG system
    
    Args:
        query: User query string
    
    Returns:
        Dictionary with retrieval results
    """
    global _rag_instance
    
    if _rag_instance is None:
        return {
            "error": "RAG instance not initialized",
            "has_documents": False
        }
    
    try:
        # Use safe index if available
        if safe_idx.records:
            params = POLICY.get("retrieval", {})
            top = safe_idx.query(
                query,
                k=_rag_instance.k,
                min_rel=float(params.get("min_rel", 0.35)),
                min_kw=int(params.get("min_keyword_hits", 1)),
                max_chunks=int(params.get("max_chunks", 4)),
            )
            
            chunks = [t["text"] for t in top]
            metas = [t["meta"] for t in top]
            scores = [t["meta"].get("score", 0.0) for t in top]
            
            return {
                "error": None,
                "has_documents": len(chunks) > 0,
                "chunks": chunks,
                "metas": metas,
                "scores": scores
            }
        else:
            # Fallback to legacy retrieval
            if _rag_instance.model is not None:
                ret = _rag_instance.model.retrieve(query, path=_rag_instance.path)
                chunks = ret.get('doc') or ret.get('documents') or []
                scores = ret.get('score', [0.0] * len(chunks))
                metas = [{"doc": "(legacy)", "chunk": i, "collection": "legacy"}
                         for i in range(len(chunks))]
                
                return {
                    "error": None,
                    "has_documents": len(chunks) > 0,
                    "chunks": chunks,
                    "metas": metas,
                    "scores": scores
                }
            else:
                return {
                    "error": None,
                    "has_documents": False,
                    "chunks": [],
                    "metas": [],
                    "scores": []
                }
    except Exception as e:
        logger.error(f"Retrieval error: {e}")
        return {
            "error": str(e),
            "has_documents": False
        }


@action
async def validate_chunk(chunk: str) -> bool:
    """
    Validate a retrieved chunk for safety
    
    Args:
        chunk: Text chunk to validate
    
    Returns:
        True if chunk is safe, False otherwise
    """
    try:
        if looks_like_injection(chunk):
            return False
        return True
    except Exception:
        return False


@action
async def log_interaction(
    event: str,
    data: Optional[Dict[str, Any]] = None,
    **kwargs
) -> None:
    """
    Log interactions and events
    """
    log_path = POLICY.get("logging", {}).get("path", "./logs/audit.jsonl")
    log_entry = {
        "timestamp": datetime.utcnow().isoformat() + "Z",
        "event": event,
        **kwargs
    }
    if data:
        log_entry["data"] = data

    try:
        from utils.audit_logger import log_audit
        log_audit(log_entry, log_path=log_path)
    except ImportError:
        os.makedirs(os.path.dirname(log_path), exist_ok=True)
        with open(log_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(log_entry, ensure_ascii=False) + "\n")

    logger.info(f"Logged event: {event}")


@action
async def write_security_log(entry: Dict[str, Any]) -> None:
    """Write to security log — delegates to shared utils.audit_logger."""
    security_log_path = os.path.join(
        Path(__file__).parent.parent, "logs", "security.jsonl"
    )
    try:
        from utils.audit_logger import log_security
        log_security(entry, log_path=security_log_path)
    except ImportError:
        os.makedirs(os.path.dirname(security_log_path), exist_ok=True)
        with open(security_log_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")
    logger.warning(f"Security event logged: {entry.get('event', 'unknown')}")


@action
async def write_audit_log(entry: Dict[str, Any]) -> None:
    """Write to audit log — delegates to shared utils.audit_logger."""
    audit_log_path = os.path.join(
        Path(__file__).parent.parent, "logs", "audit.jsonl"
    )
    try:
        from utils.audit_logger import log_audit
        log_audit(entry, log_path=audit_log_path)
    except ImportError:
        os.makedirs(os.path.dirname(audit_log_path), exist_ok=True)
        with open(audit_log_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")


@action
async def send_alert(level: str, message: str, details: Optional[Dict[str, Any]] = None) -> None:
    """Send security alert"""
    alert = {
        "timestamp": datetime.utcnow().isoformat() + "Z",
        "level": level,
        "message": message,
        "details": details or {}
    }
    
    await write_security_log(entry=alert)
    logger.critical(f"ALERT [{level}]: {message}")


def get_rate_limit_count(key: str) -> int:
    """Get current rate limit count"""
    if key not in _rate_limit_store:
        return 0
    
    entry = _rate_limit_store[key]
    if (datetime.now() - entry["timestamp"]).total_seconds() > 60:
        _rate_limit_store[key] = {
            "count": 0,
            "timestamp": datetime.now()
        }
        return 0
    
    return entry["count"]


def increment_rate_limit(key: str) -> None:
    """Increment rate limit count"""
    if key not in _rate_limit_store:
        _rate_limit_store[key] = {
            "count": 0,
            "timestamp": datetime.now()
        }
    
    _rate_limit_store[key]["count"] += 1

