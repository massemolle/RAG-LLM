"""
Unified PII Detection and Redaction
====================================

Single source of truth for all PII regex patterns and redaction logic.
Wraps ``nvidia_nemo.pii_detection.PIIDetector`` (Presidio-capable) and
extends it with additional patterns (IP address, passport) gathered from
across the codebase.

Usage::

    from utils.pii import detect_pii, redact_pii, PII_PATTERNS

    entities, redacted = detect_pii("Send to john@acme.com")
    clean = redact_pii("Card: 4111-1111-1111-1111")
"""

from __future__ import annotations

import re
import logging
from typing import Dict, List, Tuple, Optional

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Canonical PII pattern registry (superset of all patterns in the codebase)
# Order matters: more-specific patterns first to avoid partial matches.
# Each entry is (name, compiled_regex).
# ---------------------------------------------------------------------------

_RAW_PATTERNS: List[Tuple[str, str, int]] = [
    # Credit card: 4 groups of 4 digits
    ("credit_card", r"\b\d{4}[\s.-]?\d{4}[\s.-]?\d{4}[\s.-]?\d{4}\b", 0),
    # IBAN: 2-letter country code + 2 check digits + up to 30 alphanumeric
    ("iban", r"\b[A-Z]{2}\d{2}[A-Z0-9]{4}\d{7}([A-Z0-9]?){0,16}\b", 0),
    # Luxembourg phone
    ("phone_lux", r"\b(\+352)?\s?\d{3}[\s.-]?\d{3}[\s.-]?\d{3}\b", 0),
    # International phone
    ("phone_intl", r"\+\d{1,3}[\s.-]?\d{1,4}[\s.-]?\d{1,4}[\s.-]?\d{1,9}\b", 0),
    # US Social Security Number
    ("ssn", r"\b\d{3}-\d{2}-\d{4}\b", 0),
    # Email address
    ("email", r"\b[A-Z0-9._%+-]+@[A-Z0-9.-]+\.[A-Z]{2,}\b", re.IGNORECASE),
    # IP address (v4)
    ("ip_address", r"\b(?:\d{1,3}\.){3}\d{1,3}\b", 0),
    # Passport (1-2 letters + 6-9 digits)
    ("passport", r"\b[A-Z]{1,2}\d{6,9}\b", 0),
    # API key / secret / token / password / credential
    ("api_key", r"(?i)(api[_-]?key|secret[_-]?key|access[_-]?token|bearer|password|credential)\s*[=:]\s*[\w\-]{16,}", re.IGNORECASE),
]

# Pre-compiled version for fast scanning
PII_PATTERNS: List[Tuple[str, "re.Pattern[str]"]] = [
    (name, re.compile(pattern, flags)) for name, pattern, flags in _RAW_PATTERNS
]


# ---------------------------------------------------------------------------
# Lightweight regex-only helpers (no Presidio dependency)
# ---------------------------------------------------------------------------

def detect_pii_regex(text: str) -> Tuple[List[Dict], str]:
    """
    Detect and redact PII using the canonical regex patterns.

    Returns:
        (entities, redacted_text) where *entities* is a list of dicts with
        keys ``type``, ``start``, ``end``, ``score``, ``text``.
    """
    entities: List[Dict] = []
    redacted = text
    for pii_type, pattern in PII_PATTERNS:
        for match in pattern.finditer(text):
            entities.append({
                "type": pii_type,
                "start": match.start(),
                "end": match.end(),
                "score": 1.0,
                "text": match.group(),
            })
        redacted = pattern.sub(f"[REDACTED_{pii_type.upper()}]", redacted)
    return entities, redacted


def redact_pii_regex(text: str) -> str:
    """Redact PII from *text* using regex only (no Presidio)."""
    for pii_type, pattern in PII_PATTERNS:
        text = pattern.sub(f"[REDACTED_{pii_type.upper()}]", text)
    return text


# ---------------------------------------------------------------------------
# Full-featured API (delegates to PIIDetector when available)
# ---------------------------------------------------------------------------

_PIIDetector = None  # lazy import to avoid circular deps


def _get_detector():
    """Lazy-load PIIDetector from nvidia_nemo.pii_detection."""
    global _PIIDetector
    if _PIIDetector is None:
        try:
            from nvidia_nemo.pii_detection import PIIDetector
            _PIIDetector = PIIDetector
        except ImportError:
            _PIIDetector = False  # sentinel: unavailable
    return _PIIDetector


# Singleton detector instance (created on first call)
_detector_instance: Optional[object] = None


def _ensure_detector():
    global _detector_instance
    if _detector_instance is not None:
        return _detector_instance
    cls = _get_detector()
    if cls and cls is not False:
        try:
            inst = cls(use_presidio=True)
            # Inject our extra patterns into the detector's regex fallback
            for name, _, flags in _RAW_PATTERNS:
                if name not in inst.regex_patterns:
                    raw = [r for n, r, _f in _RAW_PATTERNS if n == name][0]
                    inst.regex_patterns[name] = raw
            _detector_instance = inst
            return inst
        except Exception as exc:
            logger.warning("PIIDetector init failed (%s), using regex fallback", exc)
    _detector_instance = False  # sentinel
    return False


def detect_pii(text: str) -> Tuple[List[Dict], str]:
    """
    Detect PII in *text*.

    Tries Presidio via ``PIIDetector`` first; falls back to regex.

    Returns:
        (entities, redacted_text)
    """
    det = _ensure_detector()
    if det and det is not False:
        try:
            return det.detect(text)
        except Exception:
            pass
    return detect_pii_regex(text)


def redact_pii(text: str) -> str:
    """
    Redact PII from *text*.

    Tries Presidio via ``PIIDetector`` first; falls back to regex.
    """
    det = _ensure_detector()
    if det and det is not False:
        try:
            return det.redact(text)
        except Exception:
            pass
    return redact_pii_regex(text)
