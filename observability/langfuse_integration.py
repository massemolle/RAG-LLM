"""
Langfuse Observability Integration
Tracing, metrics, and monitoring for RAG system
"""

import os
import re
import logging
from typing import Dict, List, Optional, Any
from datetime import datetime
from functools import wraps

try:
    from langfuse import Langfuse
    try:
        from langfuse.decorators import observe, langfuse_context
    except ImportError:
        # Fallback: create dummy decorator
        def observe(*args, **kwargs):
            def decorator(func):
                return func
            return decorator
        langfuse_context = None
    LANGFUSE_AVAILABLE = True
except ImportError as e:
    LANGFUSE_AVAILABLE = False
    # Create dummy decorator if not available
    def observe(*args, **kwargs):
        def decorator(func):
            return func
        return decorator
    langfuse_context = None

logger = logging.getLogger(__name__)

# Global Langfuse client
_langfuse_client: Optional[Any] = None


def initialize_langfuse(
    public_key: Optional[str] = None,
    secret_key: Optional[str] = None,
    host: Optional[str] = None
) -> bool:
    """
    Initialize Langfuse client
    
    Args:
        public_key: Langfuse public key (or LANGFUSE_PUBLIC_KEY env var)
        secret_key: Langfuse secret key (or LANGFUSE_SECRET_KEY env var)
        host: Langfuse host (or LANGFUSE_HOST env var, default: https://cloud.langfuse.com)
    
    Returns:
        True if initialized successfully
    """
    global _langfuse_client
    
    if not LANGFUSE_AVAILABLE:
        logger.warning("Langfuse not available - install with: pip install langfuse")
        return False
    
    try:
        public_key = public_key or os.getenv("LANGFUSE_PUBLIC_KEY")
        secret_key = secret_key or os.getenv("LANGFUSE_SECRET_KEY")
        host = host or os.getenv("LANGFUSE_HOST", "https://cloud.langfuse.com")
        
        if not public_key or not secret_key:
            logger.warning("Langfuse keys not configured - set LANGFUSE_PUBLIC_KEY and LANGFUSE_SECRET_KEY")
            return False
        
        _langfuse_client = Langfuse(
            public_key=public_key,
            secret_key=secret_key,
            host=host
        )
        
        logger.info("Langfuse initialized successfully")
        return True
    except Exception as e:
        logger.error(f"Failed to initialize Langfuse: {e}")
        return False


def get_langfuse_client():
    """Get Langfuse client instance"""
    return _langfuse_client


def anonymize_pii(text: str) -> str:
    """
    Anonymize PII in text before logging.

    Delegates to the shared ``utils.pii`` module for a consistent superset
    of PII patterns across the entire codebase.

    Args:
        text: Text to anonymize.

    Returns:
        Anonymized text with PII replaced by ``[REDACTED_<TYPE>]``.
    """
    if not text:
        return text
    try:
        from utils.pii import redact_pii_regex
        return redact_pii_regex(text)
    except ImportError:
        # Inline fallback (should not happen in normal operation)
        _fallback = [
            ("credit_card", r'\b\d{4}[\s.-]?\d{4}[\s.-]?\d{4}[\s.-]?\d{4}\b'),
            ("email", r'\b[A-Z0-9._%+-]+@[A-Z0-9.-]+\.[A-Z]{2,}\b'),
        ]
        result = text
        for pii_type, pattern in _fallback:
            result = re.sub(pattern, f"[REDACTED_{pii_type.upper()}]", result, flags=re.IGNORECASE)
        return result


def create_trace(
    name: str,
    user_id: Optional[str] = None,
    session_id: Optional[str] = None,
    tags: Optional[List[str]] = None,
    metadata: Optional[Dict[str, Any]] = None,
    release: Optional[str] = None,
    version: Optional[str] = None
):
    """
    Create a Langfuse trace
    
    Args:
        name: Trace name (e.g., "rag_query", "document_ingest")
        user_id: User identifier
        session_id: Session identifier
        tags: List of tags for filtering
        metadata: Additional metadata
        release: Release identifier
        version: Version identifier
    
    Returns:
        Trace object or None
    """
    if not _langfuse_client:
        return None
    
    try:
        # Anonymize metadata
        safe_metadata = {}
        if metadata:
            for key, value in metadata.items():
                if isinstance(value, str):
                    safe_metadata[key] = anonymize_pii(value)
                else:
                    safe_metadata[key] = value
        
        trace = _langfuse_client.trace(
            name=name,
            user_id=user_id,
            session_id=session_id,
            tags=tags or [],
            metadata=safe_metadata,
            release=release,
            version=version
        )
        
        return trace
    except Exception as e:
        logger.error(f"Failed to create trace: {e}")
        return None


def log_generation(
    trace,
    name: str,
    model: str,
    input_text: str,
    output_text: str,
    metadata: Optional[Dict[str, Any]] = None,
    start_time: Optional[datetime] = None,
    end_time: Optional[datetime] = None
):
    """
    Log an LLM generation to Langfuse
    
    Args:
        trace: Langfuse trace object
        name: Generation name
        model: Model name
        input_text: Input prompt (will be anonymized)
        output_text: Output text (will be anonymized)
        metadata: Additional metadata
        start_time: Start timestamp
        end_time: End timestamp
    """
    if not trace:
        return
    
    try:
        # Anonymize input and output
        safe_input = anonymize_pii(input_text)
        safe_output = anonymize_pii(output_text)
        
        # Calculate latency
        latency_ms = None
        if start_time and end_time:
            latency_ms = (end_time - start_time).total_seconds() * 1000
        
        generation = trace.generation(
            name=name,
            model=model,
            input=safe_input,
            output=safe_output,
            metadata=metadata or {},
            start_time=start_time,
            end_time=end_time,
            latency=latency_ms
        )
        
        return generation
    except Exception as e:
        logger.error(f"Failed to log generation: {e}")


def log_retrieval(
    trace,
    name: str,
    query: str,
    documents: List[str],
    scores: Optional[List[float]] = None,
    metadata: Optional[Dict[str, Any]] = None
):
    """
    Log a retrieval operation to Langfuse
    
    Args:
        trace: Langfuse trace object
        name: Retrieval name
        query: User query (will be anonymized)
        documents: Retrieved documents (will be anonymized)
        scores: Relevance scores
        metadata: Additional metadata
    """
    if not trace:
        return
    
    try:
        # Anonymize query and documents
        safe_query = anonymize_pii(query)
        safe_documents = [anonymize_pii(doc) for doc in documents]
        
        span = trace.span(
            name=name,
            type="retriever",
            input={"query": safe_query},
            output={"documents": safe_documents, "count": len(safe_documents)},
            metadata={
                **(metadata or {}),
                "scores": scores if scores else [],
                "document_count": len(safe_documents)
            }
        )
        
        return span
    except Exception as e:
        logger.error(f"Failed to log retrieval: {e}")


def log_guardrails_evaluation(
    trace,
    guard_results: List[Any],
    metadata: Optional[Dict[str, Any]] = None
):
    """
    Log guardrails evaluation to Langfuse
    
    Args:
        trace: Langfuse trace object
        guard_results: List of GuardResult objects
        metadata: Additional metadata
    """
    if not trace:
        return
    
    try:
        # Convert guard results to safe format
        guard_data = []
        for result in guard_results:
            guard_data.append({
                "guard_name": result.guard_name,
                "severity": result.severity.value,
                "reason": anonymize_pii(result.reason),
                "triggered": result.triggered
            })
        
        span = trace.span(
            name="guardrails_evaluation",
            type="guard",
            input={"guard_count": len(guard_results)},
            output={"guards": guard_data},
            metadata=metadata or {}
        )
        
        return span
    except Exception as e:
        logger.error(f"Failed to log guardrails: {e}")


def log_score(
    trace,
    name: str,
    value: float,
    comment: Optional[str] = None
):
    """
    Log a score/metric to Langfuse
    
    Args:
        trace: Langfuse trace object
        name: Score name (e.g., "quality", "relevance")
        value: Score value
        comment: Optional comment
    """
    if not trace:
        return
    
    try:
        trace.score(
            name=name,
            value=value,
            comment=comment
        )
    except Exception as e:
        logger.error(f"Failed to log score: {e}")


def log_tool_call(
    trace,
    tool_name: str,
    tool_input: Dict[str, Any],
    tool_output: Any,
    metadata: Optional[Dict[str, Any]] = None
):
    """
    Log a tool/action call to Langfuse
    
    Args:
        trace: Langfuse trace object
        tool_name: Name of the tool
        tool_input: Tool input parameters (will be anonymized)
        tool_output: Tool output (will be anonymized)
        metadata: Additional metadata
    """
    if not trace:
        return
    
    try:
        # Anonymize tool input/output
        safe_input = {}
        for key, value in tool_input.items():
            if isinstance(value, str):
                safe_input[key] = anonymize_pii(value)
            else:
                safe_input[key] = value
        
        safe_output = anonymize_pii(str(tool_output)) if isinstance(tool_output, str) else tool_output
        
        span = trace.span(
            name=f"tool_{tool_name}",
            type="tool",
            input=safe_input,
            output=safe_output,
            metadata=metadata or {}
        )
        
        return span
    except Exception as e:
        logger.error(f"Failed to log tool call: {e}")

