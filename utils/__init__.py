"""
Shared Utility Library
======================

Reusable functions consolidated from across the codebase.

Modules:
    pii                 – Unified PII detection and redaction (Presidio + regex)
    injection_patterns  – Canonical prompt-injection pattern registry
    text                – Text sanitization, hashing, file I/O, chunking
    audit_logger        – Structured JSONL audit / security logging
"""

from utils.pii import detect_pii, redact_pii, PII_PATTERNS
from utils.injection_patterns import check_injection, INJECTION_REGISTRY
from utils.text import (
    sanitize_llm_output,
    strip_role_markers,
    sha256_hash,
    md5_hash,
    read_document,
    chunk_text,
)
from utils.audit_logger import log_audit, log_security

__all__ = [
    # PII
    "detect_pii", "redact_pii", "PII_PATTERNS",
    # Injection
    "check_injection", "INJECTION_REGISTRY",
    # Text
    "sanitize_llm_output", "strip_role_markers",
    "sha256_hash", "md5_hash",
    "read_document", "chunk_text",
    # Logging
    "log_audit", "log_security",
]
