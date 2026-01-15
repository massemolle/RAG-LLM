"""
Retrieval Rails Integration
Monkey-patches RAG retrieval to apply sanitization
"""

import re
from typing import List, Tuple
from defense.safe_retrieval import SafeIndex
from RagV2 import safe_idx


def sanitize_chunk(chunk: str) -> Tuple[str, List[str]]:
    """
    Sanitize a single chunk against prompt injection
    Returns: (sanitized_chunk, warnings)
    """
    warnings = []
    sanitized = chunk
    
    # Check for instruction patterns
    instruction_patterns = [
        (r"(?i)\b(ignore|disregard|forget)\s+(previous|prior|above)\s+(instructions?|prompts?)", "instruction_override"),
        (r"(?i)\b(system|developer|admin)\s+(prompt|instructions|mode)", "system_extraction"),
        (r"(?i)\b(new\s+)?(context|instructions?|rules?)\s*:", "context_injection"),
    ]
    
    for pattern, category in instruction_patterns:
        if re.search(pattern, sanitized, re.IGNORECASE):
            # Strip or annotate the pattern
            sanitized = re.sub(pattern, "[SANITIZED]", sanitized, flags=re.IGNORECASE)
            warnings.append(f"{category} pattern detected and sanitized")
    
    # Check for secret patterns
    secret_patterns = [
        (r"(?i)\b(api\s+key|token|secret|password|credential)\s*[:=]\s*[\w\-]+", "secret_leakage"),
        (r"(?i)\b(show|print|display)\s+(config|settings|policy)", "config_extraction"),
    ]
    
    for pattern, category in secret_patterns:
        if re.search(pattern, sanitized, re.IGNORECASE):
            # Redact the pattern
            sanitized = re.sub(pattern, "[REDACTED_SECRET]", sanitized, flags=re.IGNORECASE)
            warnings.append(f"{category} pattern detected and redacted")
    
    return sanitized, warnings


def apply_retrieval_rails_to_chunks(chunks: List[str]) -> Tuple[List[str], List[str]]:
    """
    Apply retrieval rails to a list of chunks
    Returns: (sanitized_chunks, all_warnings)
    """
    sanitized_chunks = []
    all_warnings = []
    
    for i, chunk in enumerate(chunks):
        sanitized, warnings = sanitize_chunk(chunk)
        sanitized_chunks.append(sanitized)
        
        if warnings:
            all_warnings.extend([f"Chunk {i}: {w}" for w in warnings])
    
    return sanitized_chunks, all_warnings


# Monkey-patch SafeIndex.query to apply retrieval rails
_original_query = SafeIndex.query

def _query_with_retrieval_rails(self, q: str, k=5, min_rel=0.35, min_kw=1, max_chunks=4):
    """
    Wrapped query method that applies retrieval rails
    """
    # Get original results
    results = _original_query(self, q, k, min_rel, min_kw, max_chunks)
    
    # Apply retrieval rails to sanitize chunks
    chunks = [r["text"] for r in results]
    sanitized_chunks, warnings = apply_retrieval_rails_to_chunks(chunks)
    
    # Update results with sanitized chunks
    for i, sanitized in enumerate(sanitized_chunks):
        if i < len(results):
            results[i]["text"] = sanitized
    
    # Log warnings if any
    if warnings:
        import logging
        logger = logging.getLogger(__name__)
        for warning in warnings:
            logger.warning(f"Retrieval rails: {warning}")
    
    return results


# Apply monkey-patch
SafeIndex.query = _query_with_retrieval_rails

