"""
NVIDIA NeMo Guardrails Integration for RAG System
Connects guardrails with existing RAG pipeline
"""

import os
import sys
import json
import logging
from datetime import datetime
from typing import Dict, List, Optional, Any
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from nemoguardrails import LLMRails, RailsConfig
from nemoguardrails.actions import action
from nemoguardrails.llm.helpers import get_llm_instance_wrapper

# Import existing RAG components
from RagV2 import RAG, safe_idx
from defense.guards import POLICY, redact as redact_legacy
from defense.safe_retrieval import SafeIndex

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Global RAG instance (will be initialized)
_rag_instance: Optional[RAG] = None
_rate_limit_store: Dict[str, Dict[str, Any]] = {}


def initialize_guardrails(rag_instance: RAG, config_path: Optional[str] = None) -> LLMRails:
    """
    Initialize NeMo Guardrails with RAG integration
    
    Args:
        rag_instance: The RAG instance to use for retrieval
        config_path: Path to guardrails config (default: ./config/config.yml)
    
    Returns:
        Initialized LLMRails instance
    """
    global _rag_instance
    _rag_instance = rag_instance
    
    # Default config path
    if config_path is None:
        config_path = os.path.join(
            os.path.dirname(__file__),
            "config",
            "config.yml"
        )
    
    # Load configuration from directory
    config_dir = os.path.dirname(config_path)
    config = RailsConfig.from_path(config_dir)
    
    # Set RAG instance in actions module
    from nvidia_nemo.config import actions
    actions.set_rag_instance(rag_instance)
    
    # Initialize guardrails
    # Note: For local LLM, we'll need to configure it properly
    rails = LLMRails(config=config)
    
    logger.info("NeMo Guardrails initialized successfully")
    return rails


# ============================================================================
# Custom Actions for RAG Integration
# ============================================================================

@action
async def retrieve_from_rag(query: str) -> Dict[str, Any]:
    """
    Custom action: Retrieve documents from RAG system
    
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
    Custom action: Validate a retrieved chunk for safety
    
    Args:
        chunk: Text chunk to validate
    
    Returns:
        True if chunk is safe, False otherwise
    """
    from defense.guards import looks_like_injection
    
    # Check for prompt injection
    if looks_like_injection(chunk):
        return False
    
    # Additional safety checks can be added here
    # - Check for malicious content
    # - Verify relevance
    # - Check for PII leakage
    
    return True


@action
async def log_interaction(
    event: str,
    data: Optional[Dict[str, Any]] = None,
    **kwargs
) -> None:
    """
    Custom action: Log interactions and events
    
    Args:
        event: Event type
        data: Event data dictionary
        **kwargs: Additional event fields
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
        os.path.dirname(__file__), "logs", "security.jsonl"
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
        os.path.dirname(__file__), "logs", "audit.jsonl"
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
    """
    Custom action: Send security alert
    
    Args:
        level: Alert level (low, medium, high, critical)
        message: Alert message
        details: Additional details
    """
    alert = {
        "timestamp": datetime.utcnow().isoformat() + "Z",
        "level": level,
        "message": message,
        "details": details or {}
    }
    
    # Log alert
    await write_security_log(entry=alert)
    
    # In production, this could send to:
    # - Email
    # - Slack/Teams
    # - PagerDuty
    # - SIEM system
    
    logger.critical(f"ALERT [{level}]: {message}")


def get_rate_limit_count(key: str) -> int:
    """Get current rate limit count for a key"""
    if key not in _rate_limit_store:
        return 0
    
    entry = _rate_limit_store[key]
    # Reset if expired (1 minute window)
    if (datetime.now() - entry["timestamp"]).total_seconds() > 60:
        _rate_limit_store[key] = {
            "count": 0,
            "timestamp": datetime.now()
        }
        return 0
    
    return entry["count"]


def increment_rate_limit(key: str) -> None:
    """Increment rate limit count for a key"""
    if key not in _rate_limit_store:
        _rate_limit_store[key] = {
            "count": 0,
            "timestamp": datetime.now()
        }
    
    _rate_limit_store[key]["count"] += 1


# ============================================================================
# Guarded RAG Wrapper
# ============================================================================

class GuardedRAG:
    """
    Wrapper class that adds NeMo Guardrails to the RAG system
    """
    
    def __init__(self, rag_instance: RAG, config_path: Optional[str] = None):
        """
        Initialize guarded RAG
        
        Args:
            rag_instance: The RAG instance to guard
            config_path: Path to guardrails config
        """
        self.rag = rag_instance
        self.rails = initialize_guardrails(rag_instance, config_path)
        logger.info("GuardedRAG initialized")
    
    async def answer(
        self,
        query: str,
        user_id: Optional[str] = None,
        session_id: Optional[str] = None,
        role: str = "analyst"
    ) -> str:
        """
        Get answer with guardrails protection
        
        Args:
            query: User query
            user_id: User identifier
            session_id: Session identifier
            role: User role
        
        Returns:
            Guarded response
        """
        try:
            # Generate response through guardrails
            response = await self.rails.generate_async(
                messages=[{"role": "user", "content": query}],
                user_id=user_id or "anonymous",
                session_id=session_id or "default"
            )
            
            return response
        except Exception as e:
            logger.error(f"Error in guarded response: {e}")
            # Fallback to direct RAG (without guardrails)
            return self.rag.answer(query, role=role)
    
    def answer_sync(
        self,
        query: str,
        user_id: Optional[str] = None,
        session_id: Optional[str] = None,
        role: str = "analyst"
    ) -> str:
        """
        Synchronous version of answer (for compatibility)
        
        Args:
            query: User query
            user_id: User identifier
            session_id: Session identifier
            role: User role
        
        Returns:
            Guarded response
        """
        import asyncio
        
        try:
            loop = asyncio.get_event_loop()
        except RuntimeError:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
        
        return loop.run_until_complete(
            self.answer(query, user_id, session_id, role)
        )

